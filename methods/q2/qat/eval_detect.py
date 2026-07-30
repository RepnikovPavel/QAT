"""mAP evaluation for quantized YOLOv5 on PASCAL VOC.

Runs the trained model in eval (decoded) mode, applies NMS via the official
yolov5 non_max_suppression, matches predictions to ground truth, and computes
mAP@0.5 (VOC primary metric, as in Q^2 Table 1) using yolov5.utils.metrics.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data.voc import VOCDataset, collate_detection, VOC_CLASSES
from .models.yolov5 import QYOLOv5
from .train_detect import targets_to_yolo


def _iou_matrix(boxes1, boxes2):
    """boxes: (N,4) xyxy, (M,4) xyxy -> (N,M) IoU."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]))
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(0)
    inter = (torch.min(boxes1[:, None, 2], boxes2[None, :, 2]) -
             torch.max(boxes1[:, None, 0], boxes2[None, :, 0])).clamp(0) * \
            (torch.min(boxes1[:, None, 3], boxes2[None, :, 3]) -
             torch.max(boxes1[:, None, 1], boxes2[None, :, 1])).clamp(0)
    return inter / (area1[:, None] + area2[None, :] - inter + 1e-7)


def evaluate(model, loader, device, nc=20, iou_thres=0.5, conf_thres=0.001,
             nms_thres=0.6, max_det=300, img_size=640):
    """Compute mAP@0.5 over the loader."""
    from yolov5.utils.general import non_max_suppression
    from yolov5.utils.metrics import ap_per_class

    model.eval()
    stats = []
    seen = 0
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            out, _ = model(imgs)  # decoded (N, n_anchors*nx*ny, 5+nc) in eval
            # out is (N, num_boxes, 5+nc): [xyxy, conf, classes...]
            pred = non_max_suppression(out, conf_thres, iou_thres,
                                       max_det=max_det)
            # build target boxes per image in PIXEL xyxy (predictions decode to
            # pixel xyxy, so we convert normalised cxcywh targets accordingly).
            tg = targets_to_yolo(targets, img_size=img_size)  # norm cxcywh
            for i, det in enumerate(pred):
                raw = tg[tg[:, 0] == i]  # (L,6) [img,cls,cx,cy,w,h] normalised
                nl = raw.shape[0]
                tcls = raw[:, 1].long().tolist() if nl else []
                if nl:
                    cx, cy, w, h = (raw[:, 2:6] * img_size).T
                    tbox = torch.stack([cx - w / 2, cy - h / 2,
                                        cx + w / 2, cy + h / 2], 1).to(device)
                    labels_cls = raw[:, 1:2].to(device)
                else:
                    tbox = torch.zeros((0, 4), device=device)
                    labels_cls = torch.zeros((0, 1), device=device)
                if det is None or len(det) == 0:
                    if nl:
                        stats.append((torch.zeros(0), torch.zeros(0),
                                      torch.zeros(0), torch.tensor(tcls)))
                    continue
                det = det.to(device)
                dbox = det[:, :4]
                dconf = det[:, 4]
                dcls = det[:, 5].long()
                # match
                if nl:
                    # tp shape (n_pred, 1) as expected by yolov5 ap_per_class
                    correct = torch.zeros((dbox.shape[0], 1), dtype=torch.float32, device=device)
                    iou = _iou_matrix(dbox, tbox)
                    # for each class
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
                    stats.append((correct.cpu(), dconf.cpu(), dcls.cpu(),
                                  torch.tensor(tcls)))
                else:
                    stats.append((torch.zeros((dbox.shape[0], 1)),
                                  dconf.cpu(), dcls.cpu(), torch.tensor([])))
                seen += 1

    if not stats:
        return {"mAP@0.5": 0.0, "n_images": 0}
    # ap_per_class expects (tp, conf, pred_cls, target_cls)
    tp = torch.cat([s[0] for s in stats]).float()
    conf = torch.cat([s[1] for s in stats]).float()
    pcls = torch.cat([s[2] for s in stats])
    tcls = torch.cat([s[3] for s in stats])
    if tcls.numel() == 0:
        return {"mAP@0.5": 0.0, "n_images": seen}
    # ap_per_class returns: tp, fp, p, r, f1, ap, unique_classes
    # where p/r/f1 have shape (nc, 1000) (per-conf-threshold) and ap has (nc, 10)
    # for IoU 0.5..0.95. mAP@0.5 = ap[:, 0].mean().
    _tp, _fp, p, r, f1, ap, _uc = ap_per_class(tp, conf, pcls, tcls, names={},
                                               eps=1e-16)
    def _m(x):
        x = np.asarray(x)
        return float(x.mean()) if x.size else 0.0

    # p/r at the operating point = value at the F1-optimal confidence threshold
    # (column where f1 is max). Use per-class best f1's conf column, averaged.
    f1_arr = np.asarray(f1)  # (nc, 1000) but may be (1000,) if single class
    p_arr = np.asarray(p)
    r_arr = np.asarray(r)
    if f1_arr.ndim == 1:
        f1_arr = f1_arr[None, :]
        p_arr = p_arr[None, :]
        r_arr = r_arr[None, :]
    if f1_arr.size and f1_arr.shape[1]:
        best = f1_arr.argmax(axis=1)  # (nc,)
        precision = float(np.mean([p_arr[i, best[i]] for i in range(p_arr.shape[0])]))
        recall = float(np.mean([r_arr[i, best[i]] for i in range(r_arr.shape[0])]))
    else:
        precision = recall = 0.0

    ap_arr = np.asarray(ap)
    if ap_arr.ndim == 1:
        ap_arr = ap_arr[None, :]
    map50 = float(ap_arr[:, 0].mean()) if ap_arr.size else 0.0
    return {
        "mAP@0.5": map50,
        "mAP@0.5:0.95": _m(ap),
        "precision": precision,
        "recall": recall,
        "n_images": seen,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="/mnt/hdd2/qat_run/voc_yolo")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--quant", default=None)
    ap.add_argument("--wbits", type=int, default=4)
    ap.add_argument("--abits", type=int, default=4)
    ap.add_argument("--qgb", action="store_true")
    ap.add_argument("--nc", type=int, default=20)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QYOLOv5(nc=args.nc, quant=args.quant, wbits=args.wbits, abits=args.abits,
                    use_qgb=args.qgb).to(device)
    model.init_quantizers(img_size=args.img_size)
    ck = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ck["model"])
    print("Loaded", args.ckpt, flush=True)

    test_list = str(Path(args.data) / "test.list")
    ds = VOCDataset(test_list, img_size=args.img_size, augment=False)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, collate_fn=collate_detection)
    res = evaluate(model, loader, device, nc=args.nc, img_size=args.img_size)
    print("RESULT:", res, flush=True)


if __name__ == "__main__":
    main()
