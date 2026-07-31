"""Training loop for Q^2 quantized YOLOv5 on PASCAL VOC.

Uses the OFFICIAL yolov5 ComputeLoss (no reimplementation) for box/obj/cls
losses, and adds Q-ADA distillation on top when --qada is enabled.

Hyperparameters follow Q^2 Appendix 8.1:
  SGD, lr=0.00334, OneCycleLR (final_ratio=0.15135), momentum=0.74832,
  weight_decay=0.00025, batch=64 (effective), seed=0, Q-ADA weight=0.01.

Q-GBFusion closed-loop update runs after loss.backward() / before
optimizer.step().
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data.voc import VOCDataset, collate_detection, prepare_voc
from .models.yolov5 import QYOLOv5


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def targets_to_yolo(targets, img_size=640):
    """Convert VOCDataset labels to the ComputeLoss target format.

    yolov5 ComputeLoss.build_targets expects targets of shape (N,6) with
    columns (image_idx, class, cx, cy, w, h) in NORMALISED [0,1] coordinates
    (it multiplies by grid-space gain internally).

    VOCDataset returns [cls, cx, cy, w, h] in letterboxed pixels, so we divide
    by img_size.
    """
    out = []
    for i, t in enumerate(targets):
        if t.numel() == 0:
            continue
        cls = t[:, 0:1]
        cx, cy, w, h = t[:, 1], t[:, 2], t[:, 3], t[:, 4]
        img = torch.full((t.shape[0], 1), float(i))
        row = torch.cat([img, cls,
                         (cx / img_size)[:, None], (cy / img_size)[:, None],
                         (w / img_size)[:, None], (h / img_size)[:, None]], 1)
        out.append(row)
    return torch.cat(out, 0) if out else torch.zeros((0, 6))


def build_optimizer(model, lr, momentum, weight_decay):
    """SGD Nesterov with two param-groups.

    Quantizer parameters (LSQ step, PACT alpha, N2UQ thresholds/range) go in a
    separate group with weight_decay=0: applying weight decay to the LSQ step
    drives it negative / explodes it, which collapses the objectness head
    (issue #4). LSQ's own gradient scale ``1/sqrt(Qp*N)`` already balances its
    update magnitude against weights (Esser 2020 Sec. 2.2).
    """
    quant_param_ids = {id(p) for p in model.quantizer_params() if p.requires_grad}
    quant_params, weight_params = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (quant_params if id(p) in quant_param_ids else weight_params).append(p)
    groups = [
        {"params": weight_params, "weight_decay": weight_decay},
        {"params": quant_params, "weight_decay": 0.0},
    ]
    return torch.optim.SGD(groups, lr=lr, momentum=momentum,
                           weight_decay=weight_decay, nesterov=True)


@torch.no_grad()
def project_quant_steps(model) -> None:
    """Keep every quantizer step-size / scale strictly positive after an update.

    LSQ uses ``step.clamp(min=1e-8)`` in forward/backward, so a negative or zero
    raw parameter is masked there; but the raw value can still drift far below
    zero (SGD + momentum), wasting capacity and making the lazily-initialised
    value meaningless. Clamp the raw parameter in place so it stays in a sane
    range. N2UQ/PACT have no such positivity constraint and are left alone.
    """
    from .quantizers.lsq import LSQ
    for m in model.modules():
        if isinstance(m, LSQ):
            m.step.data.clamp_(min=1e-8)


def train_one_epoch(model, loader, optimizer, scheduler, compute_loss,
                    qada_loss, device, epoch, qada_weight, img_size=640,
                    log_every=50, qada_teacher=None, use_amp=False,
                    scaler=None, clip_grad=0.0, monitor_obj=True, grad_accum=1):
    model.train()
    t0 = time.time()
    running = {"loss": 0.0, "box": 0.0, "obj": 0.0, "cls": 0.0, "qada": 0.0}
    nb = 0

    # Q-ADA feature-supervision point = the SPPF output (yolov5s layer 9), the
    # last backbone feature before the neck. Per Eq.13-16 the distortion is
    # Delta = |X - Q(X)| on the SAME feature, so we need, for the student:
    #   X    = its FP feature at the supervision point (= SPPF output),
    #   Q(X) = that feature after the neck's first quantized conv quantises it.
    # We grab the teacher's FP feature at the same point (its Delta is 0).
    teacher_feats = {}
    student_supervision = None  # set to the quantized conv right after SPPF

    def _make_hook(store, key):
        def hook(_mod, _inp, out):
            store[key] = out
        return hook

    handles = []
    if qada_loss is not None and qada_teacher is not None:
        handles.append(qada_teacher.model[9].register_forward_hook(
            _make_hook(teacher_feats, "sppf")))
        # Student: hook SPPF output for X_fp; ask the following quantized conv
        # to capture its pre/post-quant input (which IS the SPPF feature).
        handles.append(model.model[9].register_forward_hook(
            _make_hook(teacher_feats, "student_sppf")))
        student_supervision = model.model[10]
        student_supervision.capture_acts = True

    try:
      for it, (imgs, targets) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        tg = targets_to_yolo(targets, img_size).to(device)

        # Teacher FP pass (no grad) to emit supervision features.
        if qada_loss is not None and qada_teacher is not None:
            with torch.no_grad():
                _ = qada_teacher(imgs)

        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                preds = model(imgs)
                loss, items = compute_loss(preds, tg)
                total = loss
        else:
            preds = model(imgs)  # list of 3 scale tensors (training mode)
            loss, items = compute_loss(preds, tg)
            total = loss

        # Q-ADA (Eq. 13-16): align teacher and student saliency at the shared
        # SPPF supervision point. Delta = |X - Q(X)| on the SAME feature:
        #   teacher  : FP feature x_t (Delta=0).
        #   student  : X = its FP feature (SPPF out), Q(X) = the quantized
        #              version captured by the neck's first quantized conv.
        qada_val = 0.0
        if (qada_loss is not None and qada_teacher is not None
                and "sppf" in teacher_feats and "student_sppf" in teacher_feats
                and student_supervision is not None
                and student_supervision._last_x_q is not None):
            x_t = teacher_feats["sppf"]                       # teacher FP feature
            x_s_fp = teacher_feats["student_sppf"]            # student FP feature (X)
            x_s_q = student_supervision._last_x_q             # student Q(X)
            loss_qada = qada_loss(x_t, x_s_fp, x_s_q)
            total = total + qada_weight * loss_qada
            qada_val = float(loss_qada.detach())

        # Gradient accumulation: scale the loss so the accumulated gradient
        # matches a single step at the effective (batch*accum) batch size.
        is_accum_step = ((it + 1) % grad_accum == 0) or (it + 1 == len(loader))
        if is_accum_step:
            optimizer.zero_grad()
        scaled_total = total / grad_accum
        if use_amp and scaler is not None:
            scaler.scale(scaled_total).backward()
        else:
            scaled_total.backward()

        # Only step the optimizer / scheduler at the end of an accumulation
        # window, so the OneCycleLR advances once per effective batch.
        if is_accum_step:
            # Q-GBFusion closed-loop dual update (Eq. 8-10)
            if model.qgb_nodes:
                energies = model.step_qgb()

            # Gradient clipping (safety net against the large first-step under
            # quantization that previously collapsed the objectness head, #4).
            if clip_grad and clip_grad > 0:
                if use_amp and scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            if use_amp and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            # Keep LSQ step sizes strictly positive (see project_quant_steps).
            project_quant_steps(model)
            scheduler.step()

        running["loss"] += float(loss.detach())
        box, obj, cls = items
        running["box"] += float(box)
        running["obj"] += float(obj)
        running["cls"] += float(cls)
        running["qada"] += qada_val
        nb += 1

        if (it % log_every) == 0:
            # Objectness-head health probe: raw obj-sigmoid max on EACH of the 3
            # detection scales. A true collapse (issue #4) drives ALL three to
            # ~0.001/0.005/0.106 (NMS keeps nothing -> mAP=0); a healthy head
            # keeps at least the small-object scale (idx 0) well above 0.1 as
            # training proceeds. NOTE: a fresh Detect head is randomly init'd
            # (pretrained transfer skips det.model.24.* because COCO nc=80 != VOC
            # nc=20), so the max-obj starts LOW everywhere and only recovers as
            # the head adapts — a low early omax is normal, a PERMANENTLY near-
            # zero omax across all scales is the failure signature.
            # yolov5 train output per scale: [B, na, ny, nx, 5+nc]; obj at [...,4].
            objtag = ""
            if monitor_obj and isinstance(preds, (list, tuple)) and len(preds) >= 1:
                try:
                    omax = []
                    for pi in preds:
                        if pi.dim() >= 2:
                            omax.append(f"{float(pi.detach()[..., 4].sigmoid().max()):.3f}")
                        else:
                            omax.append("nan")
                    objtag = " omax=" + "/".join(omax)
                except Exception:
                    pass
            print(f"[ep {epoch} it {it}/{len(loader)}] "
                  f"loss={float(loss):.3f} box={float(box):.3f} "
                  f"obj={float(obj):.3f} cls={float(cls):.3f} "
                  f"lr={optimizer.param_groups[0]['lr']:.5f}{objtag} "
                  f"({(time.time()-t0)/(it+1):.2f}s/it)", flush=True)
    finally:
        for h in handles:
            h.remove()
        if student_supervision is not None:
            student_supervision.capture_acts = False

    n = max(nb, 1)
    for k in running:
        running[k] /= n
    running["time"] = time.time() - t0
    return running


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voc", default="/mnt/hdd2/datasets/voc")
    ap.add_argument("--data", default="/mnt/hdd2/qat_run/voc_yolo")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.00334)
    ap.add_argument("--final-ratio", type=float, default=0.15135)
    ap.add_argument("--momentum", type=float, default=0.74832)
    ap.add_argument("--weight-decay", type=float, default=0.00025)
    ap.add_argument("--quant", default=None, choices=[None, "lsq", "pact", "n2uq"])
    ap.add_argument("--wbits", type=int, default=4)
    ap.add_argument("--abits", type=int, default=4)
    ap.add_argument("--qgb", action="store_true", help="enable Q-GBFusion")
    ap.add_argument("--qada", action="store_true", help="enable Q-ADA distillation")
    ap.add_argument("--qada-weight", type=float, default=0.01)
    ap.add_argument("--clip-grad", type=float, default=10.0,
                    help="max grad norm (0 disables); safety net against the "
                         "large first-step under quantization (issue #4)")
    ap.add_argument("--nc", type=int, default=20)
    ap.add_argument("--out", default="/mnt/hdd2/qat_run/run1")
    ap.add_argument("--pretrained",
                    default="/mnt/hdd2/qat_run/weights/yolov5s.pt",
                    help="path to yolov5s.pt (COCO-pretrained body weights)")
    ap.add_argument("--init-ckpt", default=None,
                    help="a VOC-FP QYOLOv5 checkpoint to init QAT from (the paper's "
                         "'full-precision pretrained checkpoint'). Takes priority "
                         "over --pretrained and transfers the (warm) Detect head too.")
    ap.add_argument("--limit", type=int, default=0, help="limit train batches (debug)")
    ap.add_argument("--log-every", type=int, default=50, help="print every N iters")
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="gradient accumulation steps; effective batch = --batch * this "
                         "(paper uses batch 64; on 1 GPU set --batch 32 --grad-accum 2)")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the detection forward (Blackwell speedup)")
    ap.add_argument("--amp", action="store_true",
                    help="mixed-precision autocast (faster, small accuracy impact)")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    # Prepare VOC -> YOLO format (idempotent)
    summary = prepare_voc(args.voc, args.data)
    print("VOC prepared:", summary, flush=True)

    train_list = str(Path(args.data) / "train.list")
    test_list = str(Path(args.data) / "test.list")
    train_ds = VOCDataset(train_list, img_size=args.img_size, augment=True)
    test_ds = VOCDataset(test_list, img_size=args.img_size, augment=False)
    # Optionally cap the number of training samples (debug / smoke runs). The
    # Q^2 paper trains on the full 07+12 split (~16.5k imgs); --limit rounds
    # down to a multiple of --batch (drop_last=True needs that).
    if args.limit and args.limit > 0:
        n = (args.limit // args.batch) * args.batch
        n = max(n, args.batch)
        train_ds = torch.utils.data.Subset(train_ds, list(range(min(n, len(train_ds)))))
        print(f"[limit] using {len(train_ds)} train samples", flush=True)

    def seed_worker(_):
        np.random.seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, collate_fn=collate_detection,
                              worker_init_fn=seed_worker, drop_last=True, pin_memory=True)

    model = QYOLOv5(nc=args.nc, quant=args.quant, wbits=args.wbits, abits=args.abits,
                    use_qgb=args.qgb).to(device)

    # Weight init priority (Q^2 Appendix 8.1 "initialized from full-precision
    # pretrained checkpoint"):
    #   1. --init-ckpt : a VOC-FP QYOLOv5 checkpoint (warm head) — the faithful
    #      reading of the paper; transfers the Detect head too.
    #   2. --pretrained : official COCO yolov5s.pt (body only; head stays cold).
    if args.init_ckpt and os.path.exists(args.init_ckpt):
        from .pretrained import load_qyolo_state_dict
        load_qyolo_state_dict(model, args.init_ckpt)
    elif args.pretrained and os.path.exists(args.pretrained):
        from .pretrained import load_yolov5s_pretrained
        load_yolov5s_pretrained(model, args.pretrained)

    # Initialise the lazy quantizer parameters on a REAL batch, not a dummy of
    # zeros. The Q^2 paper enables W4A4 from step 0 (no FP warmup), so the LSQ
    # step sizes must reflect the activation distribution at the very first
    # step; zero-input init gave nonsense scales. One real batch materialises
    # the per-channel weight scales and per-tensor activation scales, paying
    # the memory cost before training so the first real step does not OOM.
    if args.quant:
        with torch.no_grad():
            imgs0, _ = next(iter(DataLoader(
                train_ds, batch_size=1, shuffle=False,
                collate_fn=collate_detection, num_workers=0)))
            model.init_quantizers_from(imgs0.to(device))
        print(f"[quant] fake-quant ON from step 0 (paper schedule); "
              f"LSQ scales initialised on a real batch", flush=True)

    # Scale the official YOLOv5 loss gains to the actual class count / image
    # size (the package defaults are tuned for COCO 80 classes at 640px). The
    # cls gain scales with the label space; the obj gain scales with the grid
    # count, keeping the per-anchor objectness signal at a comparable strength.
    box_gain = 0.05
    cls_gain = 0.5 * (args.nc / 80.0)
    obj_gain = 1.0 * (args.img_size / 640.0) ** 2
    model.det.hyp["box"] = box_gain
    model.det.hyp["cls"] = cls_gain
    model.det.hyp["obj"] = obj_gain

    # Official YOLOv5 loss (reused, not reimplemented)
    from yolov5.utils.loss import ComputeLoss
    compute_loss = ComputeLoss(model.det, autobalance=False)

    qada_loss = None
    qada_teacher = None
    if args.qada:
        # Q-ADA teacher: a FROZEN full-precision model used only to emit
        # supervision signals (paper Sec. 4.5, line 291: "a frozen pretrained
        # full-precision model, which does not participate in quantized
        # parameter updates"). We build it from the same pretrained weights as
        # the student but without any quantization, then supervise the student
        # at shared feature-supervision points (the backbone SPPF output, the
        # last feature before the neck, is a stable single-point proxy for the
        # per-fusion-node Q-ADA described in Sec. 3.3).
        from .qada import QADALoss
        qada_loss = QADALoss(divergence="js")
        qada_teacher = QYOLOv5(nc=args.nc, quant=None, wbits=args.wbits,
                               abits=args.abits, use_qgb=False).to(device)
        # Teacher uses the SAME warm init as the student so it is a true FP-VOC
        # teacher (paper: "frozen pretrained full-precision model").
        if args.init_ckpt and os.path.exists(args.init_ckpt):
            from .pretrained import load_qyolo_state_dict
            load_qyolo_state_dict(qada_teacher, args.init_ckpt, verbose=False)
        elif args.pretrained and os.path.exists(args.pretrained):
            from .pretrained import load_yolov5s_pretrained
            load_yolov5s_pretrained(qada_teacher, args.pretrained, verbose=False)
        qada_teacher.eval()
        for p in qada_teacher.parameters():
            p.requires_grad_(False)
        print("[qada] frozen FP teacher built (parameter-free supervision)", flush=True)

    optimizer = build_optimizer(model, args.lr, args.momentum, args.weight_decay)

    if args.compile:
        # torch.compile the detection forward for speed (Blackwell benefits).
        # Q-GBFusion hooks run outside the compiled region (they probe grads),
        # so they are unaffected. reduce-overhead mode cuts CPU overhead.
        print("[compile] torch.compile(model, mode='reduce-overhead') ...", flush=True)
        model.det.model = torch.compile(model.det.model, mode="reduce-overhead")
        model._compiled = True

    # With gradient accumulation the scheduler advances once per EFFECTIVE
    # batch: every grad_accum iters, PLUS a trailing step on the last iter of
    # the epoch if len(loader) is not a multiple of accum. So steps_per_epoch is
    # ceil(len/accum), not floor (floor caused an off-by-one that stepped the
    # OneCycleLR past its total on the final epoch).
    import math as _math
    ga = max(args.grad_accum, 1)
    steps_per_epoch = max(_math.ceil(len(train_loader) / ga), 1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps_per_epoch,
        final_div_factor=1.0 / args.final_ratio, pct_start=0.1,
    )

    print(f"Model: quant={args.quant} wbits={args.wbits} abits={args.abits} "
          f"qgb={args.qgb} qada={args.qada}", flush=True)
    print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)

    scaler = torch.amp.GradScaler("cuda") if (args.amp and torch.cuda.is_available()) else None
    for epoch in range(args.epochs):
        r = train_one_epoch(model, train_loader, optimizer, scheduler,
                            compute_loss, qada_loss, device, epoch,
                            args.qada_weight, img_size=args.img_size,
                            qada_teacher=qada_teacher, use_amp=args.amp,
                            scaler=scaler, clip_grad=args.clip_grad,
                            log_every=args.log_every, grad_accum=args.grad_accum)
        print(f"== epoch {epoch} avg: loss={r['loss']:.3f} "
              f"box={r['box']:.3f} obj={r['obj']:.3f} cls={r['cls']:.3f} "
              f"({r['time']:.0f}s)", flush=True)
        ckpt = os.path.join(args.out, f"ckpt_ep{epoch}.pt")
        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "args": vars(args)}, ckpt)

    print("Training complete.", flush=True)


if __name__ == "__main__":
    main()
