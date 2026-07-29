# Method: Binary Normalized Layers (BNL)

Source: Cabral, Pirozelli, Driemeier — *1 bit is all we need: binary
normalized neural networks*, arXiv:2509.07025v1 (2025-09-07).

Parsed paper: `papers/2509.07025/document.md`.

## Decision

**Adapt to 2D detection** (not full paper reproduction, not skip).

| Option | Verdict | Reason |
| --- | --- | --- |
| Reproduce fully | No | Paper has **no** VOC/COCO detection numbers. Tasks are Food-101 classification and WikiText-103 language modelling only. No public code. |
| Skip | No | Core idea (1-bit `{0,1}` weights + mandatory post-linear normalize) is detection-relevant and distinct from `bitnet_det` (BitNet ±1 / ternary + absmax). |
| **Adapt** | **Yes** | Port BNL conv/linear into a YOLOv5 backbone on VOC; report mAP@0.5 vs FP32 and vs `bitnet_det`. |

## Paper extract (anti-hallucination)

### Task / models / bit-width

| Field | Paper |
| --- | --- |
| Tasks | Food-101 multiclass classification; WikiText-103 next-token LM |
| Models | Small CNN (BNCVL + BNFCL); decoder transformer (BEMBL + BTFB/BATL) |
| Bit-width | **W1** weights **and biases** ∈ `{0, 1}`; activations stay full precision |
| Code public? | No (arxiv has no code link; GitHub search empty as of 2026-07-29) |

### Quantization (Eq. 1)

Each parameter exists in two forms during training: full-precision `p` (updated
by the optimiser) and binary `p_b` (used in the forward pass):

```
p_b = 1  if p > p_mean
p_b = 0  if p ≤ p_mean
```

`p_mean` = mean of the parameters of **that layer** (kernel and bias each
quantized with their own mean).

### STE dual representation (Alg. 1 / Alcorn-style)

```
W_q = W + stopgrad(Quant(W) − W)   # forward = Quant(W); backward → W
```

Same for bias. After training, keep only `W_q, b_q`.

### Critical fix: post-linear normalize (Alg. 1 steps 8–10)

```
z = W_q x + b_q          # or Conv(W_q, x) + b_q
z = Normalize(z)         # per-example zero mean, unit std
a = Activation(z)
```

Without this normalize, the paper (and Cabral & Driemeier 2025) report that
pure binary-parameter nets fail to train. Normalize is **not** a learnable
LayerNorm — fixed zero-mean / unit-std over the features of each example.

### Reported numbers (paper Tables 2 & 5) — not detection

**Food-101 (Table 2), val accuracy:**

| Model | Train acc | Val acc |
| --- | --- | --- |
| FP32 3×3 | 0.989 | 0.703 |
| Binary 3×3 | 0.670 | 0.637 |
| FP32 5×5 | 0.986 | 0.679 |
| Binary 5×5 | 0.834 | **0.686** |

**WikiText-103 (Table 5), val perplexity:**

| Model | Val ppl | Val acc |
| --- | --- | --- |
| FP32 small | 7.47 | 0.664 |
| Binary small (154M) | 7.92 | 0.659 |
| Binary large (333M) | 7.47 | 0.666 |

CNN hparams (Table 1): Adam, lr_max=1e-4, warmup 20 / decay 1100 steps,
batch 64, 1000 epochs, 256×256, CE loss. LM (Table 4): AdamW, lr=1e-5,
100 epochs.

### Key references (most important for the method)

| Ref | Why it matters | Local parse |
| --- | --- | --- |
| Alcorn 2023 (arXiv:2301.08838) | STE identity used in Alg. 1 | not yet (ocrc model error) |
| Hubara et al. 2016/2017 BNN/QNN | classic ±1 binary nets | not yet |
| Rastegari et al. XNOR-Net | scale factor alternative to normalize | cited only |
| Choi et al. PACT | QAT baseline in lit review | `papers/refs/pact/` |
| Jacob et al. 2018 | QAT dual fp/quant training | cited only |
| Cabral & Driemeier 2025 | prior claim pure binary fails | Neural Networks journal |

Full ref re-parse deferred: ocrc worker is paused (`model_state=error`);
paper formulas above are taken from `papers/2509.07025/document.md`.

## Paper → code map

| Paper component | Code |
| --- | --- |
| Eq. 1 mean-threshold quant | `bnl/quant.py::mean_threshold_quantize` |
| Alg. 1 STE wrap | `bnl/quant.py::ste_quantize` |
| Per-example Normalize | `bnl/normalize.py::per_example_normalize` |
| BNFCL (Alg. 1) | `bnl/layers.py::BinaryNormalizedLinear` |
| BNCVL (Alg. 3 / Eq. 2) | `bnl/layers.py::BinaryNormalizedConv2d` |
| Detection adapter | `bnl/yolo_bnl.py::BNLYOLOv5` — mid-body YOLOv5 `Conv` → `BinaryNormalizedConv2d`; stem + Detect FP32 |
| VOC train | `train_detect.py` + `recipes/bnl_voc.md` |

## Detection adaptation plan

1. ~~Unit-test BNL layers (forward binary ∈ {0,1}, STE grads flow, normalize
   stats).~~ done (12 tests in `qat-repro`).
2. ~~Swap YOLOv5s body convs for `BinaryNormalizedConv2d` (stem + Detect head
   stay FP32).~~ done (`bnl/yolo_bnl.py`, 56 BNL convs). VOC smoke finite.
   Full train + mAP@0.5 still pending.
3. Compare to FP32 YOLOv5s and to `methods/bitnet_det` when available.
4. Numbers go in `RESULTS.md` (paper classification table filled; detection
   smoke logged; mAP table after full runs).
