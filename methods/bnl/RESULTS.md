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

## Detection adaptation (VOC mAP@0.5) — planned

| Run | ours | paper |
| --- | --- | --- |
| YOLOv5s FP32 | — | n/a (paper has no detection) |
| YOLOv5s + BNL W1 (body) | — | n/a |

Hardware target: GPU server, Docker `qat-repro`, datasets `/mnt/hdd2/datasets/voc`.
