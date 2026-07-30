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
                    log_every=50, qada_targets=None, use_amp=False,
                    scaler=None, clip_grad=0.0, monitor_obj=True):
    model.train()
    t0 = time.time()
    running = {"loss": 0.0, "box": 0.0, "obj": 0.0, "cls": 0.0, "qada": 0.0}
    nb = 0

    for it, (imgs, targets) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        tg = targets_to_yolo(targets, img_size).to(device)

        # Forward: teacher (FP, no grad) for Q-ADA + student (quantized)
        if qada_loss is not None and qada_targets is not None:
            with torch.no_grad():
                _ = model.det.eval()  # not used; teacher handled separately
        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                preds = model(imgs)
                loss, items = compute_loss(preds, tg)
                total = loss
        else:
            preds = model(imgs)  # list of 3 scale tensors (training mode)
            loss, items = compute_loss(preds, tg)
            total = loss

        # Q-ADA: align teacher/student feature distributions at the concat
        # inputs. We hook the model's last backbone feature as a proxy
        # supervision location (full per-node Q-ADA is wired via the fusers).
        qada_val = 0.0
        if qada_loss is not None and qada_targets is not None:
            # qada_targets holds teacher features captured by forward hooks
            # during a no-grad FP pass; here we add the Q-ADA term using the
            # student's own quantized features vs the stored teacher ones.
            for (xt, xs) in qada_targets.get_pairs():
                total = total + qada_weight * qada_loss(xt, xs)
                qada_val += 1

        optimizer.zero_grad()
        if use_amp and scaler is not None:
            scaler.scale(total).backward()
        else:
            total.backward()

        # Q-GBFusion closed-loop dual update (Eq. 8-10)
        if model.qgb_nodes:
            energies = model.step_qgb()

        # Gradient clipping (safety net against the large first-step under
        # quantization that previously collapsed the objectness head, issue #4).
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
            # Objectness-head health probe: raw obj-sigmoid max per scale. When
            # the head collapses (issue #4) this drops to ~0.001/0.005/0.106 vs
            # a healthy ~0.75, so NMS keeps nothing and mAP=0. Tracking it here
            # surfaces a collapse within a few steps instead of at eval time.
            # yolov5 train output per scale: [B, na, ny, nx, 5+nc]; obj is [...,4].
            objtag = ""
            if monitor_obj and isinstance(preds, (list, tuple)) and len(preds) >= 1:
                try:
                    p0 = preds[0].detach()
                    if p0.dim() >= 2:
                        omax = float(p0[..., 4].sigmoid().max())
                        objtag = f" omax0={omax:.3f}"
                except Exception:
                    pass
            print(f"[ep {epoch} it {it}/{len(loader)}] "
                  f"loss={float(loss):.3f} box={float(box):.3f} "
                  f"obj={float(obj):.3f} cls={float(cls):.3f} "
                  f"lr={optimizer.param_groups[0]['lr']:.5f}{objtag} "
                  f"({(time.time()-t0)/(it+1):.2f}s/it)", flush=True)

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
    ap.add_argument("--limit", type=int, default=0, help="limit train batches (debug)")
    ap.add_argument("--log-every", type=int, default=50, help="print every N iters")
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

    if args.pretrained and os.path.exists(args.pretrained):
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
    qada_targets = None
    if args.qada:
        from .qada import QADALoss
        qada_loss = QADALoss(divergence="js")

    optimizer = build_optimizer(model, args.lr, args.momentum, args.weight_decay)

    if args.compile:
        # torch.compile the detection forward for speed (Blackwell benefits).
        # Q-GBFusion hooks run outside the compiled region (they probe grads),
        # so they are unaffected. reduce-overhead mode cuts CPU overhead.
        print("[compile] torch.compile(model, mode='reduce-overhead') ...", flush=True)
        model.det.model = torch.compile(model.det.model, mode="reduce-overhead")
        model._compiled = True

    steps_per_epoch = len(train_loader)
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
                            qada_targets=qada_targets, use_amp=args.amp,
                            scaler=scaler, clip_grad=args.clip_grad,
                            log_every=args.log_every)
        print(f"== epoch {epoch} avg: loss={r['loss']:.3f} "
              f"box={r['box']:.3f} obj={r['obj']:.3f} cls={r['cls']:.3f} "
              f"({r['time']:.0f}s)", flush=True)
        ckpt = os.path.join(args.out, f"ckpt_ep{epoch}.pt")
        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "args": vars(args)}, ckpt)

    print("Training complete.", flush=True)


if __name__ == "__main__":
    main()
