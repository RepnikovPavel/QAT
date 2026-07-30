#!/usr/bin/env python
"""Standalone eval for a checkpoint, auto-detecting quant/qgb from the saved args."""
import sys, torch, argparse
sys.path.insert(0, ".")
from qat.models.yolov5 import QYOLOv5
from qat.eval_detect import evaluate
from qat.data.voc import VOCDataset, collate_detection
from torch.utils.data import DataLoader

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--n", type=int, default=0, help="num test imgs (0=all)")
ap.add_argument("--batch", type=int, default=16)
args = ap.parse_args()

dev = torch.device("cuda")
ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
ca = ck.get("args", {})
quant = ca.get("quant")
qgb = ca.get("qgb", False)
wbits = ca.get("wbits", 4); abits = ca.get("abits", 4)
print(f"ckpt: quant={quant} qgb={qgb} wbits={wbits} abits={abits} epoch={ck.get('epoch')}")
m = QYOLOv5(nc=20, quant=quant, wbits=wbits, abits=abits, use_qgb=qgb).to(dev)
m.init_quantizers(640)
m.load_state_dict(ck["model"])
ds = VOCDataset("/mnt/hdd2/qat_run/voc_yolo/test.list", img_size=640, augment=False)
idx = list(range(min(args.n, len(ds)))) if args.n else list(range(len(ds)))
loader = DataLoader(torch.utils.data.Subset(ds, idx), batch_size=args.batch, collate_fn=collate_detection)
print("RESULT:", evaluate(m, loader, dev, nc=20, img_size=640))
