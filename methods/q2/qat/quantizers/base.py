"""Base classes and helpers for fake-quantization modules.

All quantizers here are *fake-quant* (a.k.a. quantization-aware): they keep
parameters in FP but emulate the rounding/clipping behaviour of integer
arithmetic via the straight-through estimator (STE), so that downstream conv
layers can be bit-accurately converted at deployment.

The Q^2 paper (arXiv:2511.05898) builds on top of three conv-oriented
quantizers - PACT, LSQ and N2UQ - so we expose a uniform interface::

    q = Quantizer(bit_width=4, signed=...)
    xq, step, info = q(x, mode="weight"|"activation")

Every quantizer returns the quantized tensor and reports its quantization
step size ``s`` (needed by Q-ADA, Eq. 14, where ``r = dX / (s + k)``).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def ste_round(x: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest with the straight-through estimator.

    Gradient flows through unchanged; forward is integer rounding.
    """
    return (x.round() - x).detach() + x


class QuantizerBase(nn.Module):
    """Common interface implemented by every fake-quant module."""

    # subclasses set these
    bit_width: int = 4
    signed: bool = True
    # when False, forward is identity (used for FP warmup before QAT kicks in)
    enabled: bool = True

    def enable(self, flag: bool = True) -> None:
        self.enabled = flag

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (quantized tensor, step-size tensor matching x shape broadcast)."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not getattr(self, "enabled", True):
            return x  # FP warmup: pass-through
        xq, _ = self.quantize(x)
        return xq

    def step_size(self, x: torch.Tensor) -> torch.Tensor:
        """Per-tensor (or per-channel) quantization step ``s`` used by Q-ADA."""
        _, s = self.quantize(x)
        return s
