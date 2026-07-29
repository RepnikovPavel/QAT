# Reproduction results

Live results as experiments complete on the GPU server (2× RTX 5060 Ti).
Targets are from the Q² paper, Table 1 (VOC, mAP@0.5).

## M0 — gradient imbalance at a PANet concat (Fig. 1b)

Paper claim: under W4A4 the deep branch gradient energy dominates (ratio ≫ 1);
Q-GBFusion equalises the two branches (ratio → 1).

| Setting | ‖∂L/∂F₀‖₂ (shallow) | ‖∂L/∂F₁‖₂ (deep) | ratio (deep/shallow) |
| --- | --- | --- | --- |
| Baseline (LSQ W4A4) | _pending_ | _pending_ | _pending_ |
| + Q-GBFusion | _pending_ | _pending_ | _pending_ |

## M1 — YOLOv5s, LSQ, W4A4, PASCAL VOC

| Run | mAP@0.5 (ours) | mAP@0.5 (paper) |
| --- | --- | --- |
| Baseline (LSQ W4A4) | _pending_ | 76.9 |
| + Q² (Q-GBFusion + Q-ADA) | _pending_ | 78.9 (+2.0) |

## M2 — YOLOv5s, N2UQ, W4A4, PASCAL VOC

| Run | mAP@0.5 (ours) | mAP@0.5 (paper) |
| --- | --- | --- |
| Baseline (N2UQ W4A4) | _pending_ | 82.1 |
| + Q² (Q-GBFusion + Q-ADA) | _pending_ | 84.2 (+2.1) |

_Update policy: each cell is filled the moment the corresponding run finishes,
with a timestamped pointer to the checkpoint. Empty = not yet run._
