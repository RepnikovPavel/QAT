"""Load official yolov5s COCO-pretrained weights into a QYOLOv5 model.

The pretrained checkpoint (yolov5s.pt) is a pickled ``models.yolo.DetectionModel``
trained on COCO (nc=80). We transfer all backbone+neck weights (everything
except ``model.24.*`` Detect head) into our VOC model (nc=20), leaving the
head freshly initialised. This matches the Q^2 Appendix 8.1 protocol of
starting QAT from a pretrained checkpoint.
"""

from __future__ import annotations

import sys
from typing import Optional

import torch


def load_yolov5s_pretrained(qyolo, ckpt_path: str, strict: bool = False,
                            verbose: bool = True) -> dict:
    """Load COCO-pretrained body weights into ``qyolo`` (a QYOLOv5).

    Returns a dict with 'transferred' and 'skipped' key counts.
    """
    import yolov5
    import os
    yp = os.path.dirname(yolov5.__file__)
    if yp not in sys.path:
        sys.path.insert(0, yp)

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    src_sd = ck["model"].float().state_dict()
    dst_sd = qyolo.state_dict()

    transferred = 0
    skipped = []
    new_sd = {}
    for k, v in dst_sd.items():
        # QYOLOv5 stores the layer list under self.det, so dst keys are
        # 'det.model.<i>...'. Pretrained keys are 'model.<i>...'. Strip the
        # 'det.' prefix to align them.
        src_key = k[len("det."):] if k.startswith("det.") else k
        if src_key in src_sd and src_sd[src_key].shape == v.shape:
            new_sd[k] = src_sd[src_key]
            transferred += 1
        else:
            new_sd[k] = v  # keep existing init
            if not k.startswith("det.model.24."):
                skipped.append(k)
    qyolo.load_state_dict(new_sd, strict=False)
    if verbose:
        print(f"[pretrained] transferred {transferred}/{len(dst_sd)} tensors; "
              f"kept {(len(dst_sd)-transferred)} (incl. quantizer params + head)")
        if skipped[:6]:
            print(f"[pretrained] sample non-transferred model keys: {skipped[:6]}")
    return {"transferred": transferred, "total": len(dst_sd), "skipped": skipped}


def load_qyolo_state_dict(qyolo, ckpt_path: str, verbose: bool = True) -> dict:
    """Load a previously-trained QYOLOv5 checkpoint (our own format) into ``qyolo``.

    Unlike ``load_yolov5s_pretrained`` (which reads an official COCO yolov5s.pt
    DetectionModel and must skip the nc=80 head), this loads a checkpoint saved
    by our own ``train_detect`` (``{"model": qyolo.state_dict(), ...}``). Such a
    checkpoint already has a VOC (nc=20) Detect head, so it transfers the head
    too — this is the "initialized from full-precision pretrained checkpoint"
    step of Q^2 Appendix 8.1, applied to a VOC-FP checkpoint.

    Quantizer parameters that exist in the target but not the source (e.g. when
    the source was trained FP) are left at their fresh init.
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    src_sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    if hasattr(src_sd, "state_dict"):
        src_sd = src_sd.state_dict()
    dst_sd = qyolo.state_dict()
    transferred = 0
    new_sd = {}
    for k, v in dst_sd.items():
        if k in src_sd and src_sd[k].shape == v.shape:
            new_sd[k] = src_sd[k]
            transferred += 1
        else:
            new_sd[k] = v
    qyolo.load_state_dict(new_sd, strict=False)
    if verbose:
        print(f"[init-ckpt] transferred {transferred}/{len(dst_sd)} tensors "
              f"(incl. Detect head when shapes match)", flush=True)
    return {"transferred": transferred, "total": len(dst_sd)}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--quant", default="lsq")
    ap.add_argument("--wbits", type=int, default=4)
    ap.add_argument("--abits", type=int, default=4)
    args = ap.parse_args()
    from .models.yolov5 import QYOLOv5
    m = QYOLOv5(nc=20, quant=args.quant, wbits=args.wbits, abits=args.abits)
    load_yolov5s_pretrained(m, args.ckpt)
