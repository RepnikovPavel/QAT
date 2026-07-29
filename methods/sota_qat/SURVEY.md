# SOTA QAT survey (2D detection focus)

Entries are grounded in parsed paper text under `papers/<arxiv_id>/`.
Decisions: **reproduce** (full method on VOC/COCO) / **adapt** (port core idea into
our YOLO LSQ stack) / **skip**.

---

## 2503.11159 — Feature-Perturbed Quantization (FPQ)

- **Method:** Stabilizing QAT by Implicit-Regularization on Hessian Matrix
  (Pang & Cai, 2025). Builds on LSQ-style fake-quant; injects **Stochastic Feature
  Perturbations (SFP)** on each conv-layer input with probability `p`, where
  δ ~ U[−s/2, s/2] and `s` is the activation quant step (Eq. 9, 14). Pairs SFP
  with **Channel-wise Standardization Distillation (CSD)** (Eq. 16–17) from an FP
  teacher. Total loss: `CE + L_CSD` (Eq. 22). Algorithm 1 in the paper.
- **Core idea:** Feature noise + feature distillation implicitly regularizes the
  Hessian trace → flatter loss landscape → more stable ultra-low-bit QAT
  (W4A4 / W2A4). Ablations on CIFAR-10 ResNet-18 W2A4: baseline LSQ 88.36% →
  +SFP 89.30% → +CSD 89.66% → FPQ (both) 89.92%.
- **Applicability to 2D detection:** High. Architecture-agnostic add-on to any
  LSQ QAT (no DETR/ViT-specific modules). Our `methods/q2` already has LSQ;
  SFP+CSD can wrap quantized YOLO convs. Paper evaluates classification only
  (CIFAR-10/100, ResNet/MobileNetV2) — no COCO/VOC numbers published.
- **Decision: adapt** → implement SFP+CSD under `methods/sota_qat/fpq/` on top of
  LSQ, benchmark W4A4 YOLO on VOC vs LSQ-only. Promote to reproduce once VOC
  mAP is measured.

---

## 2506.11784 — GPLQ (General, Practical, Lightning QAT for ViTs)

- **Method:** NeurIPS 2025. Two-stage “activation-first, weights-later”:
  (1) **Act-QAT** — 1 epoch, weights stay FP32, only activation scales (LSQ) trained
  with a PCA-based feature-mimicking loss (TCS-inspired); (2) **Weight-PTQ**
  (percentile per-channel 4-bit + QwT compensation). Claims ~100× faster than full
  QAT and memory ≤ FP32 training.
- **Core idea:** Activation quantization is the accuracy bottleneck; full QAT
  leaves the FP32 “basin” and hurts downstream transfer. Short Act-QAT keeps the
  model in the same basin, then cheap PTQ finishes weights.
- **Applicability to 2D detection:** Medium–high for *idea*, medium for *recipe*.
  Paper reports COCO box/mask AP with Swin-T/S (Table 2: detection/instance seg);
  design “conveniently applied to object detection.” But primary stack is ViT/Swin;
  our detectors are YOLO/CNN. Act-first + feature-mimic is portable; QwT weight
  PTQ is ViT-oriented.
- **Decision: adapt** — port Act-QAT (1-epoch activation LSQ + feature mimic) onto
  YOLO backbone; skip full QwT unless we move to a ViT detector. Lower priority
  than FPQ for immediate VOC CNN runs.

---

## 2603.05964 — QATMA (Curriculum QAT + multimodal alignment for OVOD)

- **Method:** Quantization-Aware Training with Multimodal Alignment (Park et al.,
  2026; arXiv v3 title; earlier drafts/CR-QAT naming). For **open-vocabulary**
  detectors (YOLO-World, OmDet-Turbo). Two parts: **CQAT** — partition detector
  into functional modules (backbone → neck → head) and progressively expand the
  quantization scope to suppress error accumulation; **TPSD** — text-anchored
  pairwise similarity distillation for region–text and region–region alignment.
  Gains up to +4.3 AP (LVIS) / +7.6 AP (COCO zero-shot) at extreme low-bit
  (e.g. W4A4 / 4-4-8).
- **Core idea:** Ultra-low-bit breaks multimodal alignment that OVOD depends on;
  curriculum QAT stabilizes training so KD can actually transfer alignments.
- **Applicability to 2D detection:** High for detection *pipelines*, medium for
  this repo’s closed-vocab VOC setup. CQAT module-wise progressive quant is
  architecture-agnostic and valuable for YOLO; TPSD needs a text encoder /
  region-text joint space (YOLO-World), which we do not currently train.
- **Decision: adapt** — extract CQAT progressive schedule for closed-vocab YOLO
  W4A4; defer TPSD until an OVOD baseline exists. Not first reproduce target.

---

## 2402.03666 — QuEST (diffusion model selective finetuning)

- **Method:** ICCV 2025. Low-bit **diffusion** quantization via efficient selective
  finetuning: fix imbalanced activation distributions by weight finetuning; identify
  temporal-embedding-critical and bit-sensitive layers; finetune only those under
  local+global supervision. Data-free / parameter-efficient (~7% params). Code:
  https://github.com/hatchetProject/QuEST.
- **Core idea:** Selective layer finetuning for quantized U-Net / DiT denoisers,
  not a general vision QAT stabilizer.
- **Applicability to 2D detection:** Low. Entire formulation (time embeddings,
  multi-step denoising, FID/IS metrics) is generation-specific. No detector
  experiments.
- **Decision: skip** for 2D detection QAT. Keep parsed for cross-domain ideas only
  (activation-distribution diagnosis).

---

## Priority order for next implementation

1. **FPQ** (adapt → VOC W4A4 vs LSQ) — smallest surface area, reuses `methods/q2` LSQ.
2. **GPLQ Act-QAT idea** — activation-only short finetune + feature mimic on YOLO.
3. **QATMA CQAT** — progressive backbone→neck→head quant schedule.
4. QuEST — no further work for detection.
