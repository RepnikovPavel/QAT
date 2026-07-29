"""M0: reproduce the gradient-imbalance diagnostic from Q^2 Fig. 1(b).

Tracks the per-branch gradient energy ``G_i = ||dL/dF~_i||_2`` at a feature-
fusion (concat) node of YOLOv5 over training steps, with and without Q-GBFusion.

Expected outcome (paper claim):
* Under low-bit QAT, the deep branch dominates -> G_1 >> G_0 (large ratio).
* With Q-GBFusion, the two energies converge -> ratio -> 1.

This is a cheap, fast experiment (a few hundred steps on real VOC batches),
used as a sanity check before the full mAP runs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data.voc import VOCDataset, collate_detection, prepare_voc
from .models.yolov5 import QYOLOv5
from .train_detect import targets_to_yolo, build_optimizer
from .imbalance import BranchGradientProbe


def attach_probe_to_concat(model, concat_layer_idx=16):
    """Attach a 2-branch gradient probe to the inputs of a Concat/QGBConcat node.

    The yolov5 forward feeds the concat a list of branch tensors; we wrap the
    node so that, in training, each branch tensor retains its grad and is
    logged by the probe.
    """
    probe = BranchGradientProbe(num_branches=2)
    node = model.model[concat_layer_idx]

    original_forward = node.forward

    def patched_forward(branches):
        if isinstance(branches, torch.Tensor):
            return original_forward(branches)
        if model.training:
            probe.reset()
            wrapped = []
            for i, b in enumerate(branches):
                b = b.detach().requires_grad_(True)
                b.retain_grad()
                b.register_hook(probe.hook(i))
                wrapped.append(b)
            branches = wrapped
        return original_forward(branches)

    node.forward = patched_forward
    return probe


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prepare_voc(args.voc, args.data)
    train_list = str(Path(args.data) / "train.list")
    ds = VOCDataset(train_list, img_size=args.img_size, augment=True)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, collate_fn=collate_detection,
                        drop_last=True)

    def build(qgb):
        m = QYOLOv5(nc=args.nc, quant=args.quant, wbits=args.wbits,
                    abits=args.abits, use_qgb=qgb).to(device)
        if args.pretrained:
            from .pretrained import load_yolov5s_pretrained
            load_yolov5s_pretrained(m, args.pretrained, verbose=False)
        from yolov5.utils.loss import ComputeLoss
        loss = ComputeLoss(m.det)
        opt = build_optimizer(m, args.lr, 0.9, 1e-4)
        return m, loss, opt

    results = {}
    for tag, qgb in [("baseline", False), ("qgbfusion", True)]:
        if args.mode == "baseline_only":
            if qgb:
                continue
        model, compute_loss, opt = build(qgb)
        probe = attach_probe_to_concat(model, args.concat_idx)
        model.train()
        history = []
        it = 0
        for imgs, targets in loader:
            if it >= args.steps:
                break
            imgs = imgs.to(device)
            tg = targets_to_yolo(targets).to(device)
            try:
                preds = model(imgs)
                loss, _ = compute_loss(preds, tg)
                opt.zero_grad()
                loss.backward()
                row = probe.log(it)
                history.append(row)
                opt.step()
                if model.qgb_nodes:
                    model.step_qgb()
                if it % 20 == 0:
                    print(f"[{tag} it {it}] G_0={row['G_0']:.4g} "
                          f"G_1={row['G_1']:.4g} ratio={row['ratio']:.3f}", flush=True)
            except Exception as e:
                print(f"[{tag} it {it}] skip: {repr(e)[:120]}", flush=True)
            it += 1
        results[tag] = history
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # summarise: mean ratio over last 20% of steps
    summary = {}
    for tag, hist in results.items():
        if not hist:
            summary[tag] = None
            continue
        n = len(hist)
        tail = hist[max(0, n - max(5, n // 5)):]
        ratios = [r["ratio"] for r in tail if not np.isnan(r["ratio"])]
        g0 = np.mean([r["G_0"] for r in tail])
        g1 = np.mean([r["G_1"] for r in tail])
        summary[tag] = {
            "mean_ratio_last20pct": float(np.mean(ratios)) if ratios else float("nan"),
            "mean_G0": float(g0), "mean_G1": float(g1),
            "steps": n,
        }
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "imbalance.json")
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "raw": results}, f, indent=2, default=str)
    print("=== M0 SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print("Saved", out_path)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voc", default="/mnt/hdd2/datasets/voc")
    ap.add_argument("--data", default="/mnt/hdd2/qat_run/voc_yolo")
    ap.add_argument("--quant", default="lsq")
    ap.add_argument("--wbits", type=int, default=4)
    ap.add_argument("--abits", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--lr", type=float, default=0.00334)
    ap.add_argument("--nc", type=int, default=20)
    ap.add_argument("--concat-idx", type=int, default=16)
    ap.add_argument("--mode", default="both", choices=["both", "baseline_only"])
    ap.add_argument("--pretrained", default="/mnt/hdd2/qat_run/weights/yolov5s.pt",
                    help="path to yolov5s.pt (COCO-pretrained body weights)")
    ap.add_argument("--out", default="/mnt/hdd2/qat_run/m0")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
