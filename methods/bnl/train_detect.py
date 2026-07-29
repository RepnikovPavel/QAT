"""Train YOLOv5s + BNL (W1 body) on PASCAL VOC.

Reuses the Q^2 VOC data pipeline and the official yolov5 ComputeLoss.
Does not reimplement detection architecture or loss.

Usage (inside Docker qat-repro on the GPU server)::

    PYTHONPATH=/workspace/methods/bnl:/workspace/methods/q2 \\
      python -u methods/bnl/train_detect.py \\
        --epochs 50 --batch 16 --out /mnt/hdd2/qat_run/bnl_voc

Smoke (few steps)::

    PYTHONPATH=/workspace/methods/bnl:/workspace/methods/q2 \\
      python -u methods/bnl/train_detect.py --limit 8 --epochs 1 --batch 4 \\
        --out /mnt/hdd2/qat_run/bnl_smoke
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

# Package roots: bnl (this method) + qat (VOC data helpers from methods/q2)
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
for p in (_HERE, _REPO / "methods" / "q2"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from bnl.yolo_bnl import BNLYOLOv5  # noqa: E402
from qat.data.voc import VOCDataset, collate_detection, prepare_voc  # noqa: E402


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def targets_to_yolo(targets, img_size=640):
    """VOCDataset labels → ComputeLoss format (N,6): img, cls, cx,cy,w,h ∈ [0,1]."""
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


def load_pretrained_body(model: BNLYOLOv5, ckpt_path: str, verbose: bool = True) -> dict:
    """Transfer COCO-pretrained yolov5s body weights where shapes match.

    BNL layers keep the same ``weight`` shape as the original Conv2d, so stem
    and any still-FP modules transfer cleanly. Binary shadow weights that
    match shape also get the pretrained init (STE then binarises them).
    Detect head (nc=20 vs COCO 80) is left as-is when shapes differ.
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    src_sd = ck["model"].float().state_dict()
    dst_sd = model.state_dict()

    transferred = 0
    new_sd = {}
    for k, v in dst_sd.items():
        src_key = k[len("det.") :] if k.startswith("det.") else k
        if src_key in src_sd and src_sd[src_key].shape == v.shape:
            new_sd[k] = src_sd[src_key]
            transferred += 1
        else:
            new_sd[k] = v
    model.load_state_dict(new_sd, strict=False)
    if verbose:
        print(
            f"[pretrained] transferred {transferred}/{len(dst_sd)} tensors "
            f"from {ckpt_path}",
            flush=True,
        )
    return {"transferred": transferred, "total": len(dst_sd)}


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    compute_loss,
    device,
    epoch,
    img_size=640,
    log_every=20,
    limit=0,
):
    model.train()
    t0 = time.time()
    running = {"loss": 0.0, "box": 0.0, "obj": 0.0, "cls": 0.0}
    nb = 0

    for it, (imgs, targets) in enumerate(loader):
        if limit and it >= limit:
            break
        imgs = imgs.to(device, non_blocking=True)
        tg = targets_to_yolo(targets, img_size).to(device)

        preds = model(imgs)
        loss, items = compute_loss(preds, tg)

        if not torch.isfinite(loss):
            print(
                f"[ep {epoch} it {it}] NON-FINITE loss={loss.item()} — aborting step",
                flush=True,
            )
            return {**running, "finite": False, "time": time.time() - t0}

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        box, obj, cls = items
        running["loss"] += float(loss.detach())
        running["box"] += float(box)
        running["obj"] += float(obj)
        running["cls"] += float(cls)
        nb += 1

        if (it % log_every) == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"[ep {epoch} it {it}/{len(loader)}] "
                f"loss={float(loss):.3f} box={float(box):.3f} "
                f"obj={float(obj):.3f} cls={float(cls):.3f} "
                f"lr={lr:.5f} ({(time.time() - t0) / (it + 1):.2f}s/it)",
                flush=True,
            )

    n = max(nb, 1)
    for k in list(running):
        running[k] /= n
    running["finite"] = True
    running["time"] = time.time() - t0
    running["steps"] = nb
    return running


def main():
    ap = argparse.ArgumentParser(description="YOLOv5s + BNL W1 on VOC")
    ap.add_argument("--voc", default="/mnt/hdd2/datasets/voc")
    ap.add_argument("--data", default="/mnt/hdd2/qat_run/voc_yolo")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="Adam lr (paper CNN uses 1e-4; detection default)")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--opt", default="adam", choices=["adam", "sgd"])
    ap.add_argument("--momentum", type=float, default=0.937)
    ap.add_argument("--nc", type=int, default=20)
    ap.add_argument("--out", default="/mnt/hdd2/qat_run/bnl_voc")
    ap.add_argument(
        "--pretrained",
        default="/mnt/hdd2/qat_run/weights/yolov5s.pt",
        help="yolov5s.pt COCO body weights (optional)",
    )
    ap.add_argument("--limit", type=int, default=0,
                    help="max train steps per epoch (0 = full; smoke use 8)")
    ap.add_argument("--fp32", action="store_true",
                    help="FP32 baseline (no BNL injection)")
    ap.add_argument("--no-stem-fp", action="store_true",
                    help="also binarise stem Conv (layer 0)")
    args = ap.parse_args()

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

    model = BNLYOLOv5(
        nc=args.nc,
        keep_stem_fp=not args.no_stem_fp,
        binary_body=not args.fp32,
    ).to(device)

    if args.pretrained and os.path.exists(args.pretrained):
        load_pretrained_body(model, args.pretrained)
    else:
        print(f"[pretrained] skip (missing {args.pretrained})", flush=True)

    from yolov5.utils.loss import ComputeLoss

    compute_loss = ComputeLoss(model.det, autobalance=False)

    params = [p for p in model.parameters() if p.requires_grad]
    if args.opt == "adam":
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    else:
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
        pct_start=0.1,
    )

    n_bnl = model.n_bnl
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(
        f"Model: BNL body={not args.fp32} n_bnl_convs={n_bnl} "
        f"stem_fp={not args.no_stem_fp} params={n_params:.2f}M device={device}",
        flush=True,
    )

    # One dummy forward to confirm shapes / finite activations
    with torch.no_grad():
        dummy = torch.zeros(1, 3, args.img_size, args.img_size, device=device)
        model.eval()
        out = model(dummy)
        model.train()
        print(f"[smoke-fwd] outputs={len(out) if isinstance(out, (list, tuple)) else 1}",
              flush=True)

    history = []
    for epoch in range(args.epochs):
        r = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            compute_loss,
            device,
            epoch,
            img_size=args.img_size,
            limit=args.limit,
        )
        print(
            f"== epoch {epoch} avg: loss={r['loss']:.3f} box={r['box']:.3f} "
            f"obj={r['obj']:.3f} cls={r['cls']:.3f} steps={r.get('steps', 0)} "
            f"finite={r['finite']} ({r['time']:.0f}s)",
            flush=True,
        )
        history.append(r)
        ckpt = os.path.join(args.out, f"ckpt_ep{epoch}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "args": vars(args),
                "n_bnl": n_bnl,
                "metrics": r,
            },
            ckpt,
        )
        if not r["finite"]:
            print("Stopping early due to non-finite loss.", flush=True)
            break

    # Write a tiny run summary next to checkpoints (for RESULTS.md)
    summary_path = os.path.join(args.out, "train_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"n_bnl={n_bnl}\n")
        f.write(f"device={device}\n")
        for i, r in enumerate(history):
            f.write(
                f"ep{i}: loss={r['loss']:.4f} box={r['box']:.4f} "
                f"obj={r['obj']:.4f} cls={r['cls']:.4f} finite={r['finite']}\n"
            )
    print(f"Training complete. summary → {summary_path}", flush=True)


if __name__ == "__main__":
    main()
