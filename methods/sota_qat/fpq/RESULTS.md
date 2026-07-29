# FPQ results — YOLOv5s LSQ W4A4 ± SFP/CSD on PASCAL VOC

Paper: arXiv:2503.11159 (classification only; no published VOC/COCO numbers).
Adaptation: `methods/sota_qat/fpq/` on top of `methods/q2` LSQ YOLOv5s.

Hardware: GPU server 2× RTX 5060 Ti, Docker `qat-repro`.
Metric: mAP@0.5 on VOC07 test (Q² Table 1 protocol).

## Smoke (finite-loss check)

| Setting | epochs | limit steps | finite | det loss | csd | notes |
| --- | --- | --- | --- | --- | --- | --- |
| LSQ+FPQ W4A4 | 1 | 2 | **True** | 1.026 | 1.77 | CPU Docker smoke, img=320, batch=2; SFP on 57 convs, CSD layers 4/6/9/13/17/20/23 |

## VOC mAP@0.5 (W4A4)

| Method | SFP p | CSD λ (mean) | epochs | batch | mAP@0.5 | notes |
| --- | --- | --- | --- | --- | --- | --- |
| LSQ baseline | — | — | 50 | 16 | — | `fpq_lsq_w4a4_base` |
| LSQ + FPQ (SFP+CSD) | 0.1 | 1.0 | 50 | 16 | — | `fpq_lsq_w4a4` |

Q² Table 1 LSQ W4A4 YOLOv5s VOC reference: **76.9** mAP@0.5 (different setup;
compare LSQ vs LSQ+FPQ under this recipe first). Paper CIFAR-10 ResNet-18 W2A4:
LSQ 88.36 → FPQ 89.92.
