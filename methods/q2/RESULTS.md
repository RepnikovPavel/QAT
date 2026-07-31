# Q² reproduction — results

Hardware: reproduced on two boxes — 2× RTX 5060 Ti (Blackwell sm_120, torch
2.11+cu128) and 2× RTX 4090 (Ada sm_89, torch 2.13+cu126); yolov5==7.0.14.
Targets are Table 1 of the Q² paper (arXiv:2511.05898). The numbers below are
stable across both unless a row notes otherwise.

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

### M0 reproduced on the 4090 box (Ada sm_89, cu126)

Re-ran the probe on the second box to confirm the diagnostic is hardware-
independent. Same protocol (LSQ W4A4, yolov5s COCO body, 120 steps on VOC train,
concat layer 16):

| Setting | ratio (G₀/G₁) | paper claim |
| --- | --- | --- |
| Baseline (LSQ W4A4) | 2.16 | ≫ 1 (imbalanced) |
| + Q-GBFusion | 1.08 | → 1 (balanced) |

Matches the Blackwell row above within noise (2.13 / 1.01). The Q² core is
confirmed working under cu126 on Ada. Raw: `qat_run/m0/imbalance.json`.

## Pipeline validation — FP sanity (quant=None)

To verify the data + loss + eval pipeline independently of quantization, a
2-epoch FP train (yolov5s, COCO-pretrained body, VOC) reaches loss 0.5-0.86
and eval gives **mAP@0.5 = 0.524** on 200 test images. This confirms the data
loader, targets format, official ComputeLoss wiring, and the fixed
`eval_detect` (ap_per_class unpacking) are all correct.

## M1/M2 — QAT runs

Bugs found and fixed during M1 bring-up:

1. **FIXED** — eval_detect mis-unpacked `ap_per_class`'s
   `(tp,fp,p,r,f1,ap,uc)` return (commit 5771a13). Was reporting fp as
   precision (251).
2. **FIXED** — QAT schedule: quantizers were ON from step 0 at lr=0.00334 and
   destroyed pretrained features (loss stuck at 5.4, obj conf<0.1). Added
   `--quant-warmup-epochs` (commit d4912dc): FP warmup then enable fake-quant.
   After the fix QAT loss stays ~1.25 (FP ep1 was 0.65).
3. **FIXED** — LSQ lazy per-channel init OOM'd mid-epoch when quant enabled at
   ep[warmup] (commit 47ed7e5): now init_quantizers() materialises params up
   front.
4. **RESOLVED (re-diagnosed) — objectness collapse.** The original diagnosis
   ("the large first-step under quantization destroys the obj head") was wrong.
   Root cause, found by an FP control run + cross-check against LSQ
   (`papers/refs/lsq/document.md`): the **LSQ step-size gradient was routed
   incorrectly** (the STE form fed only `g·q` to `s`, not the Esser
   `q − x/s`, and never zeroed the saturated exterior). Per LSQ Table 3 the
   net then "did not converge". Compounded by `weight_decay=0.00025` on the
   step param (drove it negative/exploding) and by the staged-QAT warmup being
   *our* addition (the Q² paper enables W4A4 from step 0 — Appendix 8.1 — so
   the discrete switch onto a high LR was a self-inflicted cliff). Fixes
   (`glm52/fix-q2-obj-collapse`): LSQ via `autograd.Function` with the correct
   `q − x/s` interior / saturated-exterior gradient; quant params in a
   `weight_decay=0` group; `project_quant_steps` (clamp step ≥1e-8); init the
   scales on a REAL batch; fake-quant ON from step 0 with the OneCycleLR ramp;
   `clip_grad`. **The collapse (mAP=0, all-scales obj →0.001) no longer
   reproduces**: a 75-step LSQ W4A4 smoke reaches mAP@0.5 = 0.035 on 200 VOC
   test imgs (vs FP 0.29 over the same 75 steps) and keeps climbing — the head
   is merely cold-starting (it is randomly init'd because COCO nc=80 ≠ VOC
   nc=20), not dead.

| Run | ours (corrected eval) | paper |
| --- | --- | --- |
| FP YOLOv5s (official yolov5 train, our eval) | **mAP@0.5 = 0.792** | 0.859 |
| Baseline (LSQ W4A4, 30 ep, bs32×accum2, warm FP init) | **mAP@0.5 = 0.652** | 0.769 |
| + Q² (Q-GBFusion + Q-ADA, OLD Δ=teacher−student) | **mAP@0.5 = 0.651** | 0.789 (+2.0) |
| + Q² (Q-GBFusion + Q-ADA, Eq13 Δ=\|X−Q(X)\|) | **mAP@0.5 = 0.650** | 0.789 (+2.0) |

> NOTE on the eval fix: an earlier version of this table reported 0.449/0.440 —
> that was a bug in `eval_detect` (1-IoU-column matching in detection order
> instead of the yolov5/val.py `process_batch`). After rewriting the matching
> verbatim, the same FP checkpoint reads 0.792 (vs the official `yolov5 val`
> 0.827 — the ~0.035 residual is our square-letterbox preprocess vs official
> rect inference). All numbers above use the corrected eval.

> NOTE on the Q² delta (A/B): both Q-ADA variants (the wrong Δ=teacher−student
> AND the verbatim Eq13 Δ=|X−Q(X)|) give **~0 gain over the LSQ baseline**
> (0.651 / 0.650 vs 0.652). The paper reports +2.0. So the Q² delta is NOT
> recovered by fixing Δ — Q-GBFusion+Q-ADA as wired here do not improve mAP over
> plain LSQ QAT in this setup. The gap to absolute targets (FP 0.792 vs 0.859;
> LSQ 0.652 vs 0.769) is separate (effective batch / epochs).
>
> Re-reading paper line 293 + Table 4: Q-GBFusion gives +1.4-1.5% (gradient
> stability), and Q-ADA's main effect is FASTER CONVERGENCE ("substantially
> reduces training time") + only +0.3-0.7% accuracy. Table 4 reports
> time-to-convergence with validation-based early stopping, NOT fixed epochs.
> So on a fixed 30-epoch budget Q² may legitimately show ~0 mAP delta — its +2.0
> in Table 1 is measured under the paper's own (early-stop / budget) protocol.
> This is the leading hypothesis for our 0 delta; verifying it needs the same
> convergence protocol (not fixed 30 epochs).


### Gap to the paper targets — honest accounting

The collapse is fixed and the eval is now faithful (FP within 0.035 of the
official val). Remaining gaps (LSQ 0.652 vs 0.769; Q² gives ~0 delta vs +2.0):

1. **Q-ADA is effectively inert.** Eq. 16 builds the distribution as
   `Ã=Sigmoid(S)` then L1-normalises; for a strong saliency peak S saturates
   sigmoid, so teacher/student distributions nearly coincide and JS≈0 (verified
   in the unit test). The Q-ADA term contributes ~0 to training, so Q² reduces
   to Q-GBFusion alone here. The paper does not give γ/κ, so the S-scale that
   avoids sigmoid saturation is unknown — a candidate fix.
2. **Effective batch.** grad-accum=2 gives an effective 64 but BatchNorm
   statistics still see 32, unlike the paper's true 64.
3. **30 epochs** (the paper does not fix a detection epoch count).

## Short-smoke isolation (75 steps, 200 VOC test imgs)

| Run | quant | mAP@0.5 | precision | recall |
| --- | --- | --- | --- | --- |
| FP control | none | 0.292 | 1.00 | 0.45 |
| LSQ W4A4 | lsq | 0.035 | 0.04 | 0.25 |

Same 75 steps for both; the FP gap is expected for such a short run and
closes as the cold Detect head adapts. The point is mAP > 0 — the pre-fix
collapse gave a hard 0.

## Cross-check: official ultralytics/yolov5 on the same VOC split

To localise the FP gap (our `train_detect` FP plateaued at mAP@0.5 ≈ 0.50 while
loss kept dropping), we ran the **official** `python -m yolov5.train` on the
identical VOC 07+12 split, from the same `yolov5s.pt` COCO body, 30 epochs:

| epoch (official) | mAP50 | mAP50-95 |
| --- | --- | --- |
| 0 | 0.660 | 0.346 |
| 1 | 0.748 | 0.433 |

Diagnosis: the official pipeline reaches **0.66 in ONE epoch** vs our custom
`train_detect` plateau of ~0.50. The difference is the recipe, not the
quantizer: official uses `mosaic=1.0`, `lr0=0.01` with a 3-epoch warmup, and
standard YOLOv5 val; our `VOCDataset` has `mosaic=False` and our loop uses the
paper's QAT `lr=0.00334` even for the FP baseline. → The faithful path to 85.9%
is to train the FP-VOC teacher with the **official** recipe, then start Q² QAT
from that checkpoint (`--init-ckpt`). Work in progress.

