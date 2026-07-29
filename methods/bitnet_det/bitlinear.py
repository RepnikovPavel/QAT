"""BitLinear / BitConv2d — 1.58-bit ternary QAT from Microsoft BitNet b1.58.

Ported from the official training code in:
  microsoft/unilm/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf
  (Figure 3; also described in arXiv:2402.17764)

Training path (STE, FP matmul):
  1. RMSNorm on activations (built into BitLinear; replace pre-norm in LLM stacks)
  2. Per-token absmax 8-bit activation quant
  3. Per-tensor absmean ternary weight quant → values effectively in {-γ, 0, +γ}
  4. Straight-through estimator via ``x + (q(x) - x).detach()``
  5. ``F.linear`` / ``F.conv2d`` in full precision

Inference kernels live in microsoft/BitNet (gpu/model.py BitLinearKernel); this
module is the training-time fake-quant counterpart.

Detector adaptation:
  :class:`BitConv2d` applies the same quantizers to 4-D feature maps, treating
  the channel axis as the per-token feature dim (RMSNorm + absmax over C).
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def rms_norm(x: Tensor, eps: float = 1e-6) -> Tensor:
    """RMSNorm over the last dimension (official BitLinear built-in norm)."""
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


def rms_norm_channels(x: Tensor, eps: float = 1e-6) -> Tensor:
    """RMSNorm over channels for NCHW feature maps (vision analogue of per-token)."""
    # x: [N, C, H, W] → normalize along C
    return x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + eps)


def activation_quant(x: Tensor) -> Tensor:
    """Per-token quantization to 8 bits (FAQ Figure 3).

    Args:
        x: activation ``[..., d]`` (last dim = feature)
    Returns:
        Fake-quantized activation, same shape (dequantized to original scale).
    """
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y


def activation_quant_channels(x: Tensor) -> Tensor:
    """Per-spatial-location absmax 8-bit quant over channels (NCHW)."""
    scale = 127.0 / x.abs().amax(dim=1, keepdim=True).clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y


def weight_quant(w: Tensor) -> Tensor:
    """Per-tensor absmean quantization to 1.58 bits (FAQ Figure 3).

    Args:
        w: weight tensor (any shape; scale is global over all elements)
    Returns:
        Dequantized weights: ternary codes ``{-1,0,1}`` scaled by mean(|w|).
    """
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    u = (w * scale).round().clamp_(-1, 1) / scale
    return u


def weight_ternary_codes(w: Tensor) -> Tensor:
    """Return pure ternary codes in ``{-1, 0, 1}`` (no dequant scale)."""
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    return (w * scale).round().clamp_(-1, 1)


def ste_quantize(x: Tensor, q: Tensor) -> Tensor:
    """Straight-through estimator: forward uses ``q``, backward treats as identity on ``x``."""
    return x + (q - x).detach()


class BitLinear(nn.Linear):
    """BitNet b1.58 training BitLinear (FAQ Figure 3).

    Built-in RMSNorm + W1.58 absmean + A8 absmax with STE.
    Drop-in for ``nn.Linear`` during QAT; kernel optimization is for inference.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        eps: float = 1e-6,
    ) -> None:
        # Official BitNet Linear layers typically have no bias
        super().__init__(in_features, out_features, bias=bias)
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight
        x_norm = rms_norm(x, eps=self.eps)
        x_quant = ste_quantize(x_norm, activation_quant(x_norm))
        w_quant = ste_quantize(w, weight_quant(w))
        return F.linear(x_quant, w_quant, self.bias)

    @torch.no_grad()
    def ternary_weight(self) -> Tensor:
        """Pure ``{-1,0,1}`` codes of the current full-precision shadow weights."""
        return weight_ternary_codes(self.weight)


class BitConv2d(nn.Module):
    """1.58-bit ternary convolution for detectors (BitLinear quant scheme on Conv2d).

    Forward:
      1. RMSNorm over channels
      2. Per-spatial absmax 8-bit activation quant (STE)
      3. Per-tensor absmean ternary weight quant (STE)
      4. ``F.conv2d`` in full precision
      5. Optional bias + activation (e.g. SiLU from YOLOv5 Conv)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int], str] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = False,
        eps: float = 1e-6,
        activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.eps = eps
        self.activation = activation

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight
        x_norm = rms_norm_channels(x, eps=self.eps)
        x_quant = ste_quantize(x_norm, activation_quant_channels(x_norm))
        w_quant = ste_quantize(w, weight_quant(w))
        y = F.conv2d(
            x_quant,
            w_quant,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        if self.activation is not None:
            y = self.activation(y)
        return y

    @torch.no_grad()
    def ternary_weight(self) -> Tensor:
        return weight_ternary_codes(self.weight)
