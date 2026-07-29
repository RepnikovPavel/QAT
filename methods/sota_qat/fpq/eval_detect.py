"""mAP@0.5 eval for FPQ / LSQ YOLOv5s checkpoints on PASCAL VOC.

Reuses Q^2 VOC dataset + yolov5 NMS/metrics. SFP is training-only (eval
disables noise via module.training=False); no teacher needed at eval.

Usage::

    PYTHONPATH=/workspace/methods/sota_qat/fpq:/workspace/methods/q2 \\
      python -u methods/sota_qat/fpq/eval_detect.py \\
        --ckpt /mnt/hdd2/qat_run/fpq_lsq_w4a4/ckpt_ep49.pt \\
        --quant lsq --wbits 4 --abits 4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
for p in (_HERE, _REPO / "methods" / "q2"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from qat.data.voc import VOCDataset, collate_detection  # noqa: E402
from qat.eval_detect import evaluate  # noqa: E402
from qat.models.yolov5 import QYOLOv5  # noqa: E402
from sfp_inject import enable_sfp  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="/mnt/hdd2/qat_run/voc_yolo")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--quant", default="lsq")
    ap.add_argument("--wbits", type=int, default=4)
    ap.add_argument("--abits", type=int, default=4)
    ap.add_argument("--nc", type=int, default=20)
    ap.add_argument(
        "--sfp-p",
        type=float,
        default=0.0,
        help="If >0, wrap acts with SFP (eval still no-ops noise; only for ckpt compat)",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    quant = None if args.quant in (None, "None", "none") else args.quant
    model = QYOLOv5(
        nc=args.nc, quant=quant, wbits=args.wbits, abits=args.abits, use_qgb=False,
    ).to(device)
    # Match train-time SFP wrappers so state_dict keys for a_quant.* align when
    # the checkpoint was saved with --fpq (SFPActWrapper nests a_quant).
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    ck_args = ck.get("args") or {}
    want_sfp = float(args.sfp_p) > 0 or bool(ck_args.get("fpq")) and not ck_args.get("no_sfp", False)
    if want_sfp:
        p = float(args.sfp_p) if args.sfp_p > 0 else float(ck_args.get("sfp_p", 0.1))
        enable_sfp(model, p=p)

    model.init_quantizers(img_size=args.img_size)
    model.load_state_dict(ck["model"], strict=False)
    print("Loaded", args.ckpt, flush=True)

    test_list = str(Path(args.data) / "test.list")
    ds = VOCDataset(test_list, img_size=args.img_size, augment=False)
    loader = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_detection,
    )
    res = evaluate(model, loader, device, nc=args.nc, img_size=args.img_size)
    print("RESULT:", res, flush=True)

    out_dir = os.path.dirname(os.path.abspath(args.ckpt))
    out_path = os.path.join(out_dir, "eval_map.txt")
    with open(out_path, "w") as f:
        f.write(f"ckpt={args.ckpt}\n")
        for k, v in res.items():
            f.write(f"{k}={v}\n")
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
