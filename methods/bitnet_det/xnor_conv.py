"""XNOR + popcount convolution (XNOR-Net, arXiv:1603.05279).

Provides:
  1. Fake-quant binary conv (sign weights + sign activations + STE) that trains
     with standard FP GEMM — bit-exact (up to float rounding) with the integer
     XNOR-popcount path for ±1 tensors.
  2. A pure-Python / torch reference popcount kernel used only for verification.

XNOR-Net §3.1–3.2 (binary approx of real-valued convolution):
  W ≈ α · B_w ,  A ≈ K · B_a
  where B_* ∈ {+1, −1}, α = mean(|W|) per filter (or per-tensor),
  and K is the average of absolute activations over the receptive field
  (optional scaling; we expose a simpler per-tensor / per-channel α form).

For vectors of ±1 of length K:
  <a, w> = (# agreements) − (# disagreements) = 2 · popcount(XNOR) − K
  where bits encode +1→1, −1→0 and XNOR is bitwise equality.

This module is the binary (W1) sibling of BitNet's ternary BitConv2d; BitNet
b1.58 uses {-1,0,1}, XNOR-Net uses {-1,+1}.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def binary_sign(x: Tensor) -> Tensor:
    """Map reals to ``{+1, −1}`` (zeros → +1, matching common BNN practice)."""
    return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))


def ste_binary_sign(x: Tensor) -> Tensor:
    """Sign with STE (forward binary, backward identity)."""
    return x + (binary_sign(x) - x).detach()


def weight_alpha(w: Tensor, per_channel: bool = True) -> Tensor:
    """XNOR-Net filter-wise scale α = mean(|W|) over kernel spatial + in-ch.

    Args:
        w: ``[out_c, in_c/groups, kH, kW]``
        per_channel: if True, one α per output channel; else global scalar.
    """
    if per_channel:
        dims = tuple(range(1, w.ndim))
        return w.abs().mean(dim=dims).view(-1, *([1] * (w.ndim - 1)))
    return w.abs().mean()


def binary_weight(w: Tensor, per_channel: bool = True) -> Tuple[Tensor, Tensor]:
    """Return (B_w ∈ {±1}, α) such that W ≈ α · B_w."""
    alpha = weight_alpha(w, per_channel=per_channel)
    b = binary_sign(w)
    return b, alpha


def xnor_dot_popcount(a_bits: Tensor, w_bits: Tensor) -> Tensor:
    """Integer XNOR-popcount dot product for ±1 vectors packed as bits.

    Args:
        a_bits: bool/uint8 tensor ``[..., K]`` where True means +1, False −1
        w_bits: bool/uint8 tensor ``[..., K]`` same encoding
    Returns:
        Integer tensor ``[...]`` equal to ``(2 * agreements - K)`` = float ±1 dot.
    """
    # agreements = XNOR count
    agreements = (a_bits == w_bits).sum(dim=-1)
    k = a_bits.shape[-1]
    return 2 * agreements - k


def float_pm1_dot(a: Tensor, w: Tensor) -> Tensor:
    """Reference float dot for ±1 tensors (same as XNOR-popcount result)."""
    return (a * w).sum(dim=-1)


def conv2d_xnor_popcount_ref(
    x_pm1: Tensor,
    w_pm1: Tensor,
    stride: Tuple[int, int] = (1, 1),
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Tuple[int, int] = (1, 1),
    groups: int = 1,
) -> Tensor:
    """Reference XNOR-popcount conv via unfold + integer popcount path.

    ``x_pm1`` / ``w_pm1`` must be exactly ``{+1, −1}``. Zero-padding breaks
    pure popcount (pads are 0, not ±1); for ``padding != 0`` this routine uses
    a masked popcount that skips pad positions so the result stays bit-exact
    with ``F.conv2d(x_pm1, w_pm1, padding=...)``.

    Output equals ``F.conv2d(x_pm1.float(), w_pm1.float(), ...)``.
    """
    assert groups == 1, "popcount ref supports groups=1 only"
    n, c_in, h, w = x_pm1.shape
    c_out, c_w, kh, kw = w_pm1.shape
    assert c_w == c_in

    if isinstance(padding, int):
        pad_h = pad_w = padding
    else:
        pad_h, pad_w = int(padding[0]), int(padding[1])

    # Validity mask: 1 on real pixels, 0 on pads (F.pad default 0)
    ones = torch.ones(n, c_in, h, w, device=x_pm1.device, dtype=torch.float32)
    if pad_h or pad_w:
        x_pad = F.pad(x_pm1.float(), (pad_w, pad_w, pad_h, pad_h), value=0.0)
        mask_pad = F.pad(ones, (pad_w, pad_w, pad_h, pad_h), value=0.0)
    else:
        x_pad = x_pm1.float()
        mask_pad = ones

    # Unfold without extra padding (already applied)
    patches = F.unfold(
        x_pad,
        kernel_size=(kh, kw),
        dilation=dilation,
        padding=0,
        stride=stride,
    )  # [N, C*kh*kw, L]
    masks = F.unfold(
        mask_pad,
        kernel_size=(kh, kw),
        dilation=dilation,
        padding=0,
        stride=stride,
    )  # [N, K, L]

    n, ck, l = patches.shape
    # Encode ±1 → bool (+1=True, −1=False); pads are 0 → treat as False but masked out
    patches_bits = patches > 0  # [N, K, L]
    valid = masks > 0.5  # [N, K, L]
    w_bits = (w_pm1 > 0).reshape(c_out, ck)  # [Cout, K]

    # Reshape for broadcasting: patches [N*L, K], valid [N*L, K]
    pb = patches_bits.permute(0, 2, 1).reshape(n * l, ck)
    vb = valid.permute(0, 2, 1).reshape(n * l, ck)

    # agreements on valid positions only; K_eff = valid count
    # agr[i, o] = sum_k 1[pb[i,k]==w[o,k] and vb[i,k]]
    # disagreements contribute negatively: dot = agr - (K_eff - agr) = 2*agr - K_eff
    # where agr only counts valid matches; invalid never agree
    # For invalid k: product is 0 in float conv, so exclude from both agr and K_eff
    eq = pb.unsqueeze(1) == w_bits.unsqueeze(0)  # [N*L, Cout, K]
    vb_e = vb.unsqueeze(1)  # [N*L, 1, K]
    agr = (eq & vb_e).sum(dim=-1)  # [N*L, Cout]
    k_eff = vb.sum(dim=-1, keepdim=True).expand_as(agr)  # [N*L, Cout]
    dots = (2 * agr - k_eff).to(torch.float32)

    h_out = (h + 2 * pad_h - dilation[0] * (kh - 1) - 1) // stride[0] + 1
    w_out = (w + 2 * pad_w - dilation[1] * (kw - 1) - 1) // stride[1] + 1
    out = dots.view(n, h_out, w_out, c_out).permute(0, 3, 1, 2).contiguous()
    return out


class XNORConv2d(nn.Module):
    """Binary (W1 A1) convolution with XNOR-Net scaling and STE training.

    Forward (training / fake-quant):
      B_a = sign(x)           # STE
      B_w, α = binary_weight(W)
      y = conv(B_a, B_w) * α  (+ optional bias / act)

    The integer path :func:`conv2d_xnor_popcount_ref` is bit-exact with
    ``conv(B_a, B_w)`` for ±1 inputs (verified in unit tests).
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
        per_channel_alpha: bool = True,
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
        self.per_channel_alpha = per_channel_alpha
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
        b_a = ste_binary_sign(x)
        b_w, alpha = binary_weight(self.weight, per_channel=self.per_channel_alpha)
        # STE on weights: forward uses b_w, backward through full-precision W
        b_w_ste = ste_binary_sign(self.weight)
        # Use scaled path: conv(Ba, Bw) * α  ≡  conv(Ba, α·Bw)
        # Prefer scaled weights for a single conv (alpha broadcast on out-ch)
        w_scaled = b_w_ste * alpha
        y = F.conv2d(
            b_a,
            w_scaled,
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
    def binary_weight_codes(self) -> Tuple[Tensor, Tensor]:
        return binary_weight(self.weight, per_channel=self.per_channel_alpha)
