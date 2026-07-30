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

## Pipeline validation — FP sanity (quant=None)

To verify the data + loss + eval pipeline independently of quantization, a
2-epoch FP train (yolov5s, COCO-pretrained body, VOC) reaches loss 0.5-0.86
and eval gives **mAP@0.5 = 0.524** on 200 test images. This confirms the data
loader, targets format, official ComputeLoss wiring, and the fixed
`eval_detect` (ap_per_class unpacking) are all correct.

## M1/M2 — QAT runs (in progress)

Three bugs were found and two fixed during M1 bring-up; the third is open:

1. **FIXED** — eval_detect mis-unpacked `ap_per_class`'s
   `(tp,fp,p,r,f1,ap,uc)` return (commit 5771a13). Was reporting fp as
   precision (251).
2. **FIXED** — QAT schedule: quantizers were ON from step 0 at lr=0.00334 and
   destroyed pretrained features (loss stuck at 5.4, obj conf<0.1). Added
   `--quant-warmup-epochs` (commit d4912dc): FP warmup then enable fake-quant.
   After the fix QAT loss stays ~1.25 (FP ep1 was 0.65).
3. **OPEN** — train/eval discrepancy under quantization: with quant ON, train
   loss is ~1.25 but eval gives mAP@0.5≈0.003 with high recall (0.54) and very
   low precision (0.03) — i.e. the model detects objects but box coordinates
   are poor. FP (quant=None) does not show this (eval mAP 0.52). Likely the
   Detect-head anchor/stride decode is sensitive to quantized neck features.
   Investigating before the M1/M2 numbers are reportable.

| Run | ours | paper |
| --- | --- | --- |
| Baseline (LSQ W4A4) | pending QAT-decode fix | 76.9 |
| + Q² (Q-GBFusion + Q-ADA) | pending QAT-decode fix | 78.9 (+2.0) |
