"""Train YOLOv5s LSQ W4A4 ± FPQ (SFP + CSD) on PASCAL VOC.

Reuses ``methods/q2`` QYOLOv5 + VOC loaders + official yolov5 ComputeLoss.
Does not reimplement the detector or task loss.

FPQ (arXiv:2503.11159):
  * SFP — stochastic feature noise on quantized-conv activations (Eq. 9, 14)
  * CSD — channel-wise standardized feature distillation from an FP teacher
    (Eq. 16–17); total loss = task loss + λ · mean-CSD (Eq. 22 adapted)

Usage (Docker ``qat-repro`` on the GPU server)::

    PYTHONPATH=/workspace/methods/sota_qat/fpq:/workspace/methods/q2 \\
      python -u methods/sota_qat/fpq/train_detect.py \\
        --quant lsq --wbits 4 --abits 4 --fpq \\
        --epochs 50 --batch 16 --out /mnt/hdd2/qat_run/fpq_lsq_w4a4

LSQ-only baseline (same script, no ``--fpq``)::

    PYTHONPATH=... python -u methods/sota_qat/fpq/train_detect.py \\
        --quant lsq --wbits 4 --abits 4 \\
        --epochs 50 --batch 16 --out /mnt/hdd2/qat_run/lsq_w4a4_baseline

Smoke::

    ... train_detect.py --limit 8 --epochs 1 --batch 4 --fpq \\
        --out /mnt/hdd2/qat_run/fpq_smoke
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
for p in (_HERE, _REPO / "methods" / "q2"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from fpq import csd_loss  # noqa: E402
from sfp_inject import (  # noqa: E402
    DEFAULT_CSD_LAYERS,
    attach_csd_hooks,
    copy_body_weights,
    disable_sfp,
    enable_sfp,
    freeze_teacher,
)
from qat.data.voc import VOCDataset, collate_detection, prepare_voc  # noqa: E402
from qat.models.yolov5 import QYOLOv5  # noqa: E402
from qat.pretrained import load_yolov5s_pretrained  # noqa: E402


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def targets_to_yolo(targets, img_size=640):
    """VOCDataset labels → ComputeLoss (N,6): img, cls, cx,cy,w,h ∈ [0,1]."""
    out = []
    for i, t in enumerate(targets):
        if t.numel() == 0:
            continue
        cls = t[:, 0:1]
        cx, cy, w, h = t[:, 1], t[:, 2], t[:, 3], t[:, 4]
        img = torch.full((t.shape[0], 1), float(i))
        row = torch.cat(
            [
                img,
                cls,
                (cx / img_size)[:, None],
                (cy / img_size)[:, None],
                (w / img_size)[:, None],
                (h / img_size)[:, None],
            ],
            1,
        )
        out.append(row)
    return torch.cat(out, 0) if out else torch.zeros((0, 6))


def mean_csd(student_feats, teacher_feats) -> torch.Tensor:
    """CSD (Eq. 17) scaled to a mean-squared form for stable detection losses.

    ``csd_loss`` sums squared diffs then divides by batch; we further divide
    by elements-per-image so the term is O(1) like YOLO box/obj/cls losses.
    λ is applied by the caller (``csd_weight``).
    """
    if not student_feats:
        return torch.tensor(0.0)
    raw = csd_loss(student_feats, teacher_feats)  # already /B
    n_el = sum(int(f.numel()) for f in student_feats)
    b = int(student_feats[0].shape[0])
    # mean over all feature elements = (sum / B) * B / n_el
    return raw * (b / max(n_el, 1))


def train_one_epoch(
    model,
    teacher,
    s_hook,
    t_hook,
    loader,
    optimizer,
    scheduler,
    compute_loss,
    device,
    epoch,
    img_size=640,
    log_every=20,
    limit=0,
    use_fpq=False,
    csd_weight=1.0,
    use_amp=False,
    scaler=None,
):
    model.train()
    if teacher is not None:
        teacher.eval()

    t0 = time.time()
    running = {"loss": 0.0, "box": 0.0, "obj": 0.0, "cls": 0.0, "csd": 0.0}
    nb = 0

    for it, (imgs, targets) in enumerate(loader):
        if limit and it >= limit:
            break
        imgs = imgs.to(device, non_blocking=True)
        tg = targets_to_yolo(targets, img_size).to(device)

        if s_hook is not None:
            s_hook.clear()
        if t_hook is not None:
            t_hook.clear()

        # FP teacher forward (no grad) for CSD targets
        if use_fpq and teacher is not None and t_hook is not None:
            with torch.no_grad():
                _ = teacher(imgs)

        if use_amp and scaler is not None:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                preds = model(imgs)
                det_loss, items = compute_loss(preds, tg)
                csd_val = torch.zeros((), device=device)
                if use_fpq and s_hook is not None and t_hook is not None:
                    csd_val = mean_csd(s_hook.feats, t_hook.feats)
                    total = det_loss + csd_weight * csd_val
                else:
                    total = det_loss
        else:
            preds = model(imgs)
            det_loss, items = compute_loss(preds, tg)
            csd_val = torch.zeros((), device=device)
            if use_fpq and s_hook is not None and t_hook is not None:
                csd_val = mean_csd(s_hook.feats, t_hook.feats)
                total = det_loss + csd_weight * csd_val
            else:
                total = det_loss

        if not torch.isfinite(total):
            print(
                f"[ep {epoch} it {it}] NON-FINITE total={float(total)} "
                f"det={float(det_loss)} csd={float(csd_val)} — abort step",
                flush=True,
            )
            return {**running, "finite": False, "time": time.time() - t0, "steps": nb}

        optimizer.zero_grad(set_to_none=True)
        if use_amp and scaler is not None:
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        box, obj, cls = items
        running["loss"] += float(det_loss.detach())
        running["box"] += float(box)
        running["obj"] += float(obj)
        running["cls"] += float(cls)
        running["csd"] += float(csd_val.detach()) if torch.is_tensor(csd_val) else float(csd_val)
        nb += 1

        if (it % log_every) == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"[ep {epoch} it {it}/{len(loader)}] "
                f"det={float(det_loss):.3f} csd={float(csd_val):.4f} "
                f"box={float(box):.3f} obj={float(obj):.3f} cls={float(cls):.3f} "
                f"lr={lr:.5f} ({(time.time() - t0) / (it + 1):.2f}s/it)",
                flush=True,
            )

    n = max(nb, 1)
    for k in ("loss", "box", "obj", "cls", "csd"):
        running[k] /= n
    running["finite"] = True
    running["time"] = time.time() - t0
    running["steps"] = nb
    return running


def main():
    ap = argparse.ArgumentParser(description="YOLOv5s LSQ ± FPQ on VOC")
    ap.add_argument("--voc", default="/mnt/hdd2/datasets/voc")
    ap.add_argument("--data", default="/mnt/hdd2/qat_run/voc_yolo")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.00334)
    ap.add_argument("--final-ratio", type=float, default=0.15135)
    ap.add_argument("--momentum", type=float, default=0.74832)
    ap.add_argument("--weight-decay", type=float, default=0.00025)
    ap.add_argument("--quant", default="lsq", choices=[None, "lsq", "pact", "n2uq"])
    ap.add_argument("--wbits", type=int, default=4)
    ap.add_argument("--abits", type=int, default=4)
    ap.add_argument("--nc", type=int, default=20)
    ap.add_argument("--out", default="/mnt/hdd2/qat_run/fpq_lsq_w4a4")
    ap.add_argument(
        "--pretrained",
        default="/mnt/hdd2/qat_run/weights/yolov5s.pt",
        help="yolov5s.pt COCO body weights",
    )
    ap.add_argument("--limit", type=int, default=0, help="max steps/epoch (0=full)")
    ap.add_argument("--fpq", action="store_true", help="enable SFP + CSD (FPQ)")
    ap.add_argument("--sfp-p", type=float, default=0.1,
                    help="SFP Bernoulli p (paper Tab.3 peak ~0.1)")
    ap.add_argument("--csd-weight", type=float, default=1.0,
                    help="λ on mean-CSD term (Eq. 22)")
    ap.add_argument("--no-sfp", action="store_true", help="CSD only (ablation)")
    ap.add_argument("--no-csd", action="store_true", help="SFP only (ablation)")
    ap.add_argument(
        "--csd-layers",
        default=",".join(str(i) for i in DEFAULT_CSD_LAYERS),
        help="comma-separated yolov5s stage indices for CSD hooks",
    )
    ap.add_argument("--amp", action="store_true", help="CUDA AMP autocast")
    args = ap.parse_args()

    use_sfp = args.fpq and not args.no_sfp
    use_csd = args.fpq and not args.no_csd
    use_fpq = use_sfp or use_csd

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    summary = prepare_voc(args.voc, args.data)
    print("VOC prepared:", summary, flush=True)

    train_list = str(Path(args.data) / "train.list")
    train_ds = VOCDataset(train_list, img_size=args.img_size, augment=True)

    def seed_worker(_):
        np.random.seed(args.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_detection,
        worker_init_fn=seed_worker,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )

    quant = None if args.quant in (None, "None", "none") else args.quant
    model = QYOLOv5(
        nc=args.nc, quant=quant, wbits=args.wbits, abits=args.abits, use_qgb=False,
    ).to(device)

    if args.pretrained and os.path.exists(args.pretrained):
        load_yolov5s_pretrained(model, args.pretrained)
    else:
        print(f"[pretrained] skip (missing {args.pretrained})", flush=True)

    n_sfp = 0
    if use_sfp:
        n_sfp = enable_sfp(model, p=args.sfp_p)
        print(f"[fpq] SFP enabled on {n_sfp} quantized convs (p={args.sfp_p})", flush=True)

    teacher = None
    s_hook = t_hook = None
    csd_idxs = []
    if use_csd:
        teacher = QYOLOv5(
            nc=args.nc, quant=None, wbits=args.wbits, abits=args.abits, use_qgb=False,
        ).to(device)
        if args.pretrained and os.path.exists(args.pretrained):
            load_yolov5s_pretrained(teacher, args.pretrained)
        else:
            n_copy = copy_body_weights(model, teacher)
            print(f"[fpq] teacher body copied from student ({n_copy} tensors)", flush=True)
        freeze_teacher(teacher)
        layer_idxs = [int(x) for x in args.csd_layers.split(",") if x.strip()]
        s_hook, t_hook, csd_idxs = attach_csd_hooks(model, teacher, layer_idxs)
        print(f"[fpq] CSD hooks on layers {csd_idxs} (λ={args.csd_weight})", flush=True)

    from yolov5.utils.loss import ComputeLoss

    compute_loss = ComputeLoss(model.det, autobalance=False)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    steps_per_epoch = args.limit if args.limit else len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=max(args.epochs, 1),
        steps_per_epoch=max(steps_per_epoch, 1),
        final_div_factor=1.0 / args.final_ratio,
        pct_start=0.1,
    )

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(
        f"Model: quant={quant} W{args.wbits}A{args.abits} fpq={args.fpq} "
        f"sfp={use_sfp}(n={n_sfp}) csd={use_csd} params={n_params:.2f}M "
        f"device={device}",
        flush=True,
    )

    # Dummy forward initialises lazy LSQ steps and checks shapes
    with torch.no_grad():
        dummy = torch.zeros(1, 3, args.img_size, args.img_size, device=device)
        model.eval()
        if teacher is not None:
            teacher.eval()
            _ = teacher(dummy)
        _ = model(dummy)
        model.train()
        print("[smoke-fwd] ok", flush=True)

    scaler = (
        torch.amp.GradScaler("cuda")
        if (args.amp and torch.cuda.is_available())
        else None
    )

    history = []
    for epoch in range(args.epochs):
        r = train_one_epoch(
            model,
            teacher,
            s_hook,
            t_hook,
            train_loader,
            optimizer,
            scheduler,
            compute_loss,
            device,
            epoch,
            img_size=args.img_size,
            limit=args.limit,
            use_fpq=use_csd,  # CSD term only when hooks live; SFP is in forward
            csd_weight=args.csd_weight if use_csd else 0.0,
            use_amp=args.amp,
            scaler=scaler,
        )
        print(
            f"== epoch {epoch} avg: det={r['loss']:.3f} csd={r['csd']:.4f} "
            f"box={r['box']:.3f} obj={r['obj']:.3f} cls={r['cls']:.3f} "
            f"steps={r.get('steps', 0)} finite={r['finite']} ({r['time']:.0f}s)",
            flush=True,
        )
        history.append(r)
        ckpt = os.path.join(args.out, f"ckpt_ep{epoch}.pt")
        # Unwrap SFP so keys match plain QYOLOv5 (qat.eval_detect / no nested a_quant).
        if use_sfp:
            disable_sfp(model)
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "args": vars(args),
                "metrics": r,
                "csd_layers": csd_idxs,
                "n_sfp": n_sfp,
            },
            ckpt,
        )
        if use_sfp:
            enable_sfp(model, p=args.sfp_p)
        if not r["finite"]:
            print("Stopping early due to non-finite loss.", flush=True)
            break

    summary_path = os.path.join(args.out, "train_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"quant={quant} W{args.wbits}A{args.abits}\n")
        f.write(f"fpq={args.fpq} sfp={use_sfp} csd={use_csd} sfp_p={args.sfp_p}\n")
        f.write(f"csd_weight={args.csd_weight} csd_layers={csd_idxs}\n")
        f.write(f"device={device}\n")
        for i, r in enumerate(history):
            f.write(
                f"ep{i}: det={r['loss']:.4f} csd={r['csd']:.4f} "
                f"box={r['box']:.4f} obj={r['obj']:.4f} cls={r['cls']:.4f} "
                f"finite={r['finite']}\n"
            )
    if s_hook is not None:
        s_hook.close()
    if t_hook is not None:
        t_hook.close()
    print(f"Training complete. summary → {summary_path}", flush=True)


if __name__ == "__main__":
    main()
