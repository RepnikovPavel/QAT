"""BNFCL / BNCVL layers (arXiv:2509.07025 Alg. 1 and Alg. 3).

Forward (training and inference):
    1. Quantize kernel and bias to {0,1} via mean-threshold + STE
    2. Linear / Conv with binary params
    3. Per-example Normalize (zero mean, unit std)
    4. Optional activation
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .normalize import per_example_normalize
from .quant import ste_quantize


class BinaryNormalizedLinear(nn.Module):
    """Binary normalized fully connected layer (BNFCL, Algorithm 1)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Paper §3.2: Glorot Uniform weights, zero bias
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        w_q = ste_quantize(self.weight)
        b_q = ste_quantize(self.bias) if self.bias is not None else None
        z = F.linear(x, w_q, b_q)
        z = per_example_normalize(z)
        if self.activation is not None:
            z = self.activation(z)
        return z


class BinaryNormalizedConv2d(nn.Module):
    """Binary normalized convolutional layer (BNCVL, Algorithm 3 / Eq. 2)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] | str = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        activation: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
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
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        w_q = ste_quantize(self.weight)
        b_q = ste_quantize(self.bias) if self.bias is not None else None
        # Eq. 2: z = Conv(W_q, x) + b_q
        z = F.conv2d(
            x,
            w_q,
            b_q,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        z = per_example_normalize(z)
        if self.activation is not None:
            z = self.activation(z)
        return z
