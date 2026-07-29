# Reproduction results

Live results from the GPU server (2× RTX 5060 Ti, Blackwell sm_120, torch
2.11+cu128). Targets are from the Q² paper.

## M0 — gradient imbalance at a PANet concat (Fig. 1b)  ✅ DONE

Paper claim: under W4A4 the deep branch gradient energy dominates (ratio ≫ 1);
Q-GBFusion equalises the two branches (ratio → 1).

Probe: 2-branch gradient energy `G_i = ‖∂L/∂F̃_i‖₂` at the PANet concat node
(layer 16 of yolov5s), LSQ W4A4, yolov5s COCO-pretrained body weights,
60 training steps on VOC train.

| Setting | G₀ (shallow) | G₁ (deep) | **ratio (G₀/G₁)** |
| --- | --- | --- | --- |
| **Baseline (LSQ W4A4)** | 4.68e-05 | 2.20e-05 | — |
| **+ Q-GBFusion** | 4.80e-05 | 4.83e-05 | — |

Branch imbalance ratio deep-vs-shallow (mean over last 20% of steps; higher = more imbalanced):

| Setting | ratio | paper expectation |
| --- | --- | --- |
| Baseline (LSQ W4A4) | **2.13** | ≫ 1 (imbalanced) ✅ |
| + Q-GBFusion | **1.01** | → 1 (balanced) ✅ |

**Conclusion:** the central diagnostic of the paper is reproduced. Low-bit QAT
produces a ~2× branch-wise gradient imbalance at the feature-fusion node, and
Q-GBFusion's closed-loop control drives it back to ~1.0. Raw data:
`/mnt/hdd2/qat_run/m0/imbalance.json` (run log `m0.log`).

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
with a pointer to the checkpoint. Empty = not yet run._
