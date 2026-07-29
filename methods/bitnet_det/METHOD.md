# BitNet b1.58 for 2D detection

## Sources (anti-hallucination)

| Source | Role |
| --- | --- |
| microsoft/BitNet `gpu/model.py` | Inference BitLinear (A8 absmax; weights offline ternary) |
| microsoft/unilm `bitnet/` FAQ PDF Figure 3 | **Official training** BitLinear (STE, absmean W, absmax A, RMSNorm) |
| arXiv:2402.17764 | BitNet b1.58 paper |
| arXiv:1603.05279 | XNOR-Net (binary W1A1, XNOR+popcount) |

microsoft/BitNet itself is an **inference** framework (bitnet.cpp + GPU kernels).
The training-time module is the FAQ Figure 3 code; this method ports that.

## Formulas (FAQ Figure 3)

```text
activation_quant(x):   # per-token A8
  scale = 127 / max(|x|, dim=-1)
  y = round(x * scale).clamp(-128, 127) / scale

weight_quant(w):       # per-tensor W1.58
  scale = 1 / mean(|w|)
  u = round(w * scale).clamp(-1, 1) / scale   # ∈ {-γ, 0, +γ}, γ=mean(|w|)

BitLinear.forward(x):
  x_norm = RMSNorm(x)
  x_q = x_norm + (activation_quant(x_norm) - x_norm).detach()   # STE
  w_q = w + (weight_quant(w) - w).detach()
  y = linear(x_q, w_q)
```

## Detector mapping

| YOLO piece | Treatment |
| --- | --- |
| Stem Conv (layer 0) | FP32 |
| Mid-body Conv (C3 / SPPF / standalone) | `BitConv2d` (W1.58 A8 + channel RMSNorm) |
| Detect head | FP32 |
| Optional `--mode xnor` | `XNORConv2d` W1A1 instead of BitConv2d |

`BitConv2d` applies the same quantizers on NCHW maps (RMSNorm + absmax over
channels per spatial location). BN is dropped (norm is inside the bit layer),
SiLU is kept.

## Files

| Path | Role |
| --- | --- |
| `bitlinear.py` | `BitLinear`, `BitConv2d`, quant helpers |
| `xnor_conv.py` | `XNORConv2d` + popcount reference |
| `yolo_bitnet.py` | YOLOv5s injection |
| `train_detect.py` | VOC train loop |
| `tests/` | shape, ternary, popcount bit-exactness, YOLO inject |
