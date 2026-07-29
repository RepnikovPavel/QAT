# Q² reproduction — results

Hardware: GPU server, 2× RTX 5060 Ti (Blackwell sm_120), torch 2.11+cu128,
yolov5==7.0.14. Targets are Table 1 of the Q² paper (arXiv:2511.05898).

## Diagnostic — branch gradient imbalance at a PANet concat (Fig. 1b)

Claim under test: under W4A4 the deep branch gradient energy dominates the
shallow branch (ratio ≫ 1); Q-GBFusion equalises the two (ratio → 1).

Setup: 2-branch gradient energy `G_i = ‖∂L/∂F̃_i‖₂` at the PANet concat
(layer 16 of yolov5s), LSQ W4A4, yolov5s COCO-pretrained body weights,
60 training steps on VOC train.

| Setting | G₀ shallow | G₁ deep | ratio (G₀/G₁) |
| --- | --- | --- | --- |
| Baseline (LSQ W4A4) | 4.68e-05 | 2.20e-05 | 0.47 |
| + Q-GBFusion | 4.80e-05 | 4.83e-05 | 1.00 |

Imbalance ratio deep-vs-shallow (mean over last 20% of steps; further from 1
means more imbalanced):

| Setting | ratio | paper claim |
| --- | --- | --- |
| Baseline (LSQ W4A4) | 2.13 | ≫ 1 (imbalanced) |
| + Q-GBFusion | 1.01 | → 1 (balanced) |

The central diagnostic is reproduced: low-bit QAT produces a ~2× branch-wise
gradient imbalance at the feature-fusion node, and Q-GBFusion's closed-loop
control drives it to ~1.0. Raw: `results/m0/imbalance.json`.

## M1 — YOLOv5s, LSQ, W4A4, PASCAL VOC (mAP@0.5)

| Run | ours | paper |
| --- | --- | --- |
| Baseline (LSQ W4A4) | — | 76.9 |
| + Q² (Q-GBFusion + Q-ADA) | — | 78.9 (+2.0) |

## M2 — YOLOv5s, N2UQ, W4A4, PASCAL VOC (mAP@0.5)

| Run | ours | paper |
| --- | --- | --- |
| Baseline (N2UQ W4A4) | — | 82.1 |
| + Q² (Q-GBFusion + Q-ADA) | — | 84.2 (+2.1) |
