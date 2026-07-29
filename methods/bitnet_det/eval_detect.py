"""mAP evaluation for YOLOv5s + BitNet on PASCAL VOC.

Loads a checkpoint from ``train_detect.py``, runs VOC test in eval (decoded)
mode, applies official yolov5 NMS, and reports mAP@0.5 (VOC primary metric).

Usage (Docker qat-repro)::

    PYTHONPATH=/workspace/methods/bitnet_det:/workspace/methods/q2 \\
      python -u methods/bitnet_det/eval_detect.py \\
        --ckpt /mnt/hdd2/qat_run/bitnet_voc/ckpt_ep49.pt \\
        --data /mnt/hdd2/qat_run/voc_yolo
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
for p in (_HERE, _REPO / "methods" / "q2"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from yolo_bitnet import BitNetYOLOv5  # noqa: E402
from qat.data.voc import VOCDataset, collate_detection  # noqa: E402


def targets_to_yolo(targets, img_size=640):
    """VOCDataset labels → (N,6) img, cls, cx,cy,w,h ∈ [0,1]."""
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


def _iou_matrix(boxes1, boxes2):
    """boxes: (N,4) xyxy, (M,4) xyxy -> (N,M) IoU."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]))
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(0) * (
        boxes1[:, 3] - boxes1[:, 1]
    ).clamp(0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(0) * (
        boxes2[:, 3] - boxes2[:, 1]
    ).clamp(0)
    inter = (
        torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
        - torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    ).clamp(0) * (
        torch.min(boxes1[:, None, 3], boxes2[None, :, 3])
        - torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    ).clamp(0)
    return inter / (area1[:, None] + area2[None, :] - inter + 1e-7)


def evaluate(
    model,
    loader,
    device,
    nc=20,
    iou_thres=0.5,
    conf_thres=0.001,
    nms_thres=0.6,
    max_det=300,
    img_size=640,
):
    """Compute mAP@0.5 over the loader."""
    from yolov5.utils.general import non_max_suppression
    from yolov5.utils.metrics import ap_per_class

    model.eval()
    stats = []
    seen = 0
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            out = model(imgs)
            if isinstance(out, (list, tuple)) and len(out) == 2:
                decoded = out[0]
            elif isinstance(out, (list, tuple)) and len(out) == 1:
                decoded = out[0]
            else:
                decoded = out
            pred = non_max_suppression(
                decoded, conf_thres, nms_thres, max_det=max_det
            )
            tg = targets_to_yolo(targets, img_size=img_size)
            for i, det in enumerate(pred):
                raw = tg[tg[:, 0] == i]
                nl = raw.shape[0]
                tcls = raw[:, 1].long().tolist() if nl else []
                if nl:
                    cx, cy, w, h = (raw[:, 2:6] * img_size).T
                    tbox = torch.stack(
                        [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1
                    ).to(device)
                    labels_cls = raw[:, 1:2].to(device)
                else:
                    tbox = torch.zeros((0, 4), device=device)
                    labels_cls = torch.zeros((0, 1), device=device)
                if det is None or len(det) == 0:
                    if nl:
                        stats.append(
                            (
                                torch.zeros(0, 1),
                                torch.zeros(0),
                                torch.zeros(0),
                                torch.tensor(tcls),
                            )
                        )
                    continue
                det = det.to(device)
                dbox = det[:, :4]
                dconf = det[:, 4]
                dcls = det[:, 5].long()
                if nl:
                    correct = torch.zeros(
                        (dbox.shape[0], 1), dtype=torch.float32, device=device
                    )
                    iou = _iou_matrix(dbox, tbox)
                    for c in torch.unique(torch.tensor(tcls, device=device)):
                        ti = (labels_cls[:, 0] == c).nonzero().view(-1)
                        di = (dcls == c).nonzero().view(-1)
                        if len(ti) and len(di):
                            m = iou[di][:, ti].cpu()
                            if m.numel():
                                x = torch.where(m >= iou_thres)
                                if x[0].numel():
                                    matched = set()
                                    for j, k in zip(*x):
                                        if k.item() not in matched:
                                            correct[di[j], 0] = 1.0
                                            matched.add(k.item())
                                            break
                    stats.append(
                        (
                            correct.cpu(),
                            dconf.cpu(),
                            dcls.cpu(),
                            torch.tensor(tcls),
                        )
                    )
                else:
                    stats.append(
                        (
                            torch.zeros((dbox.shape[0], 1)),
                            dconf.cpu(),
                            dcls.cpu(),
                            torch.tensor([]),
                        )
                    )
                seen += 1

    if not stats:
        return {"mAP@0.5": 0.0, "n_images": 0}
    tp = torch.cat([s[0] for s in stats]).float()
    conf = torch.cat([s[1] for s in stats]).float()
    pcls = torch.cat([s[2] for s in stats])
    tcls = torch.cat([s[3] for s in stats])
    if tcls.numel() == 0:
        return {"mAP@0.5": 0.0, "n_images": seen}
    _tp, _fp, p, r, _f1, ap, _uc = ap_per_class(
        tp, conf, pcls, tcls, names={}, eps=1e-16
    )
    ap = np.asarray(ap)
    p = np.asarray(p)
    r = np.asarray(r)
    map50 = float(ap[:, 0].mean()) if ap.size else 0.0
    map_ = float(ap.mean()) if ap.size else 0.0
    return {
        "mAP@0.5": map50,
        "mAP@0.5:0.95": map_,
        "precision": float(p.mean()) if p.size else 0.0,
        "recall": float(r.mean()) if r.size else 0.0,
        "n_images": seen,
    }


def build_model_from_ckpt(ckpt_path: str, device: torch.device) -> BitNetYOLOv5:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ck.get("args") or {}
    nc = int(args.get("nc", 20))
    fp32 = bool(args.get("fp32", False))
    no_stem_fp = bool(args.get("no_stem_fp", False))
    mode = str(args.get("mode", "bitnet"))
    model = BitNetYOLOv5(
        nc=nc,
        keep_stem_fp=not no_stem_fp,
        binary_body=not fp32,
        mode=mode,
    )
    model.load_state_dict(ck["model"], strict=True)
    model.to(device)
    return model


def main():
    ap = argparse.ArgumentParser(description="Eval YOLOv5s+BitNet mAP@0.5 on VOC")
    ap.add_argument("--ckpt", required=True, help="checkpoint from train_detect.py")
    ap.add_argument("--data", default="/mnt/hdd2/qat_run/voc_yolo")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--nc", type=int, default=20)
    ap.add_argument(
        "--out",
        default="",
        help="optional path to write RESULT line (default: next to ckpt)",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_ckpt(args.ckpt, device)
    print(
        f"Loaded {args.ckpt} n_bit={model.n_bit} mode={model.mode} device={device}",
        flush=True,
    )

    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, args.img_size, args.img_size, device=device)
        _ = model(dummy)

    test_list = str(Path(args.data) / "test.list")
    if not os.path.exists(test_list):
        raise SystemExit(f"missing test list: {test_list}")
    ds = VOCDataset(test_list, img_size=args.img_size, augment=False)
    loader = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_detection,
        pin_memory=torch.cuda.is_available(),
    )
    res = evaluate(model, loader, device, nc=args.nc, img_size=args.img_size)
    print("RESULT:", res, flush=True)

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.ckpt)), "eval_result.txt"
    )
    with open(out_path, "w") as f:
        f.write(f"ckpt={args.ckpt}\n")
        for k, v in res.items():
            f.write(f"{k}={v}\n")
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
