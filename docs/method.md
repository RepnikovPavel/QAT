# Method reference: Q² mapped to code

This documents how each component of the Q² paper (arXiv:2511.05898) maps to
code in this repo. Every formula referenced by number comes from the paper;
the implementation file is given next to it.

## 1. Quantizers (baseline, Sec. 2.1, Appendix 8.1)

The paper plugs in three conv-oriented quantizers as baselines (PACT, LSQ,
N2UQ). We re-implemented each from its **original paper** (cross-checked via
the ocrc-parsed PDFs in `qat_pdf/refs/`):

| Quantizer | Source paper | Code | Key formula |
| --- | --- | --- | --- |
| LSQ | Esser et al., ICLR 2020 (arXiv:1902.08153) | `qat/quantizers/lsq.py` | `xq = s·clip(round(x/s), Qn, Qp)`; step init `s = 2·mean(|v|)/√Qp`; grad scale `1/√(Qp·N)` |
| PACT | Choi et al., 2018 (arXiv:1805.06085) | `qat/quantizers/pact.py` | learnable clip α: `clip(x,0,α)` then uniform quantize |
| N2UQ | Liu et al., CVPR 2022 (arXiv:2111.14826) | `qat/quantizers/n2uq.py` | learnable INPUT thresholds, UNIFORM output levels, G-STE backward (Theorem 1) |

Each quantizer returns `(xq, step)` where `step` is the quantization step `s_c`
needed by Q-ADA (Eq. 14).

## 2. Q-GBFusion (Sec. 3.2, Eq. 5–10)  → `qat/qgbfusion.py`

A Q-GBFusion node replaces a PANet `Concat` in the YOLOv5 neck (layers
12/16/19/22 in yolov5s).

- **Allocation (Eq. 5):** `α = Softmax(λ)`, `F'_i = α_i·F̃_i`, then concat.
- **Post-fusion LayerNorm (Eq. 6):** per-channel LN over the fused feature.
- **EMA of branch gradient energy (Eq. 8):** `Ḡ_i ← (1-β)Ḡ_i + β·G_i`,
  `G_i = ‖∂L/∂F̃_i‖₂`.
- **Log-energy deviation (Eq. 9):** `e_i = log(Ḡ_i+ε) − mean_j log(Ḡ_j+ε) − τ_i`.
- **Closed-loop dual update (Eq. 10):** `λ_i ← λ_i − η·e_i`, called from
  `QYOLOv5.step_qgb()` after `loss.backward()` and before `optimizer.step()`.

α is a simplex by construction (`test_qgbfusion_alpha_is_simplex`). The closed
loop is verified to pull dominant-branch energies down
(`test_qgbfusion_reduces_imbalance`).

## 3. Q-ADA (Sec. 3.3, Eq. 13–16)  → `qat/qada.py`

Parameter-free distillation loss between a frozen FP teacher and the
quantized student, applied at feature-supervision points.

- **Quantization distortion (Eq. 13):** `Δ_{c,ij} = |X_{c,ij} − X̂_{c,ij}|`.
- **Saliency statistics (Eq. 14):** `z = |X−μ_c|/(σ_c+κ)`, `r = Δ/(s_c+κ)`.
- **Saliency score (Eq. 15):** `S = log(1+z²) + γ·log(1+r²)`.
- **Distribution alignment (Eq. 16):** Jensen–Shannon divergence between the
  spatial softmax distributions of teacher/student saliency (KL option too,
  Sec. 4.5 ablation). Loss weight = 0.01 (Appendix 8).

Note on the activation: the paper writes `Ã = Sigmoid(S)` for the attention
*weight*; for the probability distribution that JS operates on we softmax the
raw score `S` (softmax(S) yields sharp, non-saturated distributions, whereas
softmax(sigmoid(S)) collapses to near-uniform for strong distortion — see
`test_qada_monotone_in_distortion`).

## 4. Integration into YOLOv5  → `qat/models/yolov5.py`

`QYOLOv5` builds the **official** `yolov5.DetectionModel` (yolov5==7.0.14) and
injects Q² by walking the parsed layer list — no hand-rolled architecture:

- Each `Conv` → `_QuantizedConv` (fake-quant weight + input activation).
- Each PANet `Concat` (layers 12/16/19/22) → `QGBFusion` when `use_qgb=True`.
- Routing metadata `i/f/type/np` is preserved so the official `_forward_once`
  still drives the graph.
- The official `Detect` head and `ComputeLoss` are reused as-is.

## 5. Reproduced targets (Table 1, VOC)

| Model | Quant | BW | Baseline | +Q² | Gain |
| --- | --- | --- | --- | --- | --- |
| YOLOv5s | N2UQ | W4A4 | 82.1 | 84.2 | +2.1 |
| YOLOv5s | LSQ | W4A4 | 76.9 | 78.9 | +2.0 |
| YOLOv5s | PACT | W4A4 | 79.1 | 80.6 | +1.5 |

Our reproduction results land in `docs/results.md` as runs complete.
