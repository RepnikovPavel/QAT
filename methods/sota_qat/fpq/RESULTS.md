# FPQ results — YOLOv5s LSQ W4A4 ± SFP/CSD on PASCAL VOC

Paper: arXiv:2503.11159 (classification only; no published VOC/COCO numbers).
Adaptation: `methods/sota_qat/fpq/` on top of `methods/q2` LSQ YOLOv5s.

Metric: mAP@0.5 on VOC07 test (same as Q² Table 1 protocol).

## Smoke (finite-loss check)

| Setting | epochs | limit steps | finite | det loss | csd | notes |
| --- | --- | --- | --- | --- | --- | --- |
| *(pending GPU)* | — | — | — | — | — | GPUs busy at wire time |

## VOC mAP@0.5 (W4A4)

| Method | SFP p | CSD λ | epochs | batch | mAP@0.5 | notes |
| --- | --- | --- | --- | --- | --- | --- |
| LSQ baseline | — | — | 50 | 16 | — | pending |
| LSQ + FPQ (SFP+CSD) | 0.1 | 0.01 | 50 | 16 | — | pending |

Q² paper Table 1 LSQ W4A4 YOLOv5s VOC reference: **76.9** mAP@0.5 (different train
setup/hardware; compare LSQ vs LSQ+FPQ under this recipe first).
