# BNL (arXiv:2509.07025) — results

Decision: **adapt to 2D detection** (see `METHOD.md`). Paper itself reports
only Food-101 classification and WikiText-103 LM — those numbers are copied
below for reference. Detection numbers are filled after VOC runs.

## Paper numbers (classification / LM) — reference only

### Food-101 val accuracy (paper Table 2)

| Model | Train loss | Val loss | Train acc | Val acc |
| --- | --- | --- | --- | --- |
| FP32 3×3 | 0.0369 | 1.37 | 0.989 | 0.703 |
| Binary 3×3 | 1.46 | 1.55 | 0.670 | 0.637 |
| FP32 5×5 | 0.0495 | 1.48 | 0.986 | 0.679 |
| Binary 5×5 | 0.836 | 1.35 | 0.834 | 0.686 |

### WikiText-103 (paper Table 5)

| Model | Val loss | Val acc | Val ppl |
| --- | --- | --- | --- |
| FP32 small | 1.94 | 0.664 | 7.47 |
| Binary small (154M) | 1.99 | 0.659 | 7.92 |
| Binary large (333M) | 1.91 | 0.666 | 7.47 |

## Detection adaptation (VOC mAP@0.5)

| Run | mAP@0.5 (ours) | paper |
| --- | --- | --- |
| YOLOv5s FP32 | — (pending full train) | n/a (paper has no detection) |
| YOLOv5s + BNL W1 (body, stem+Detect FP32) | — (pending full train) | n/a |

### Smoke (finite-loss check, not mAP)

| Item | Value |
| --- | --- |
| Date | 2026-07-29 |
| Hardware | server GPU1 RTX 5060 Ti, Docker `qat-repro` |
| Config | img=320, batch=2, limit=4–6 steps, Adam lr=1e-4, pretrained yolov5s.pt |
| n_bnl_convs | 56 |
| pretrained transfer | 63/125 tensors (body kernels; head nc differs) |
| loss (smoke2 it0 → avg) | 1.630 → 1.148 |
| finite | **True** |
| Log | `/mnt/hdd2/qat_run/bnl_smoke2/` |

Full VOC train (50 ep, mAP@0.5) still pending — both GPUs were occupied by q2 M1 runs at smoke time.

Hardware target: GPU server, Docker `qat-repro`, datasets `/mnt/hdd2/datasets/voc`.
