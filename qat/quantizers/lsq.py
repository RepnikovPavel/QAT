"""Learnable Step-size Quantization (LSQ) [Bunde et al., CVPR 2020].

LSQ learns a single scale ``s`` per tensor (or per output-channel for weights).
Forward::

    xq = s * clip(round(x / s), -Qn, Qp)

The scale ``s`` is a learnable parameter; its gradient comes from the data
path (the original paper rescales it by ``sqrt(1/#weights)`` / ``sqrt(1/#acts)``
for balanced updates — we apply the same ``grad_factor`` correction).

This is one of the conv quantizers used in the Q^2 paper (Table 1, "LSQ").
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .base import QuantizerBase, ste_round


class LSQ(QuantizerBase):
    def __init__(
        self,
        bit_width: int = 4,
        signed: bool = True,
        per_channel: bool = False,
        channel_dim: int = 0,  # weight layout: out-ch on dim 0
        init_step: float = 2e-3,
    ) -> None:
        super().__init__()
        self.bit_width = bit_width
        self.signed = signed
        self.per_channel = per_channel
        self.channel_dim = channel_dim

        if signed:
            self.Qn = -(2 ** (bit_width - 1))
            self.Qp = 2 ** (bit_width - 1) - 1
        else:
            self.Qn = 0
            self.Qp = 2 ** bit_width - 1

        self.step = nn.Parameter(torch.tensor(float(init_step)))
        self._initialised = False

    def _grad_factor(self, x: torch.Tensor) -> torch.Tensor:
        """LSQ step-size gradient scale (Esser et al. 2020, Sec. 2.2).

        Balances the magnitude of step-size updates against weight updates:
        ``g = 1/sqrt(Qp * N)`` where N is the number of quantized elements.
        """
        return 1.0 / math.sqrt(max(self.Qp, 1) * x.numel())

    def _broadcast_step(self, x: torch.Tensor) -> torch.Tensor:
        s = self.step.abs()
        if self.per_channel:
            shape = [1] * x.dim()
            shape[self.channel_dim] = x.shape[self.channel_dim]
            s = s[: x.shape[self.channel_dim]].reshape(shape)
        return s

    def quantize(self, x: torch.Tensor):
        if not self._initialised:
            with torch.no_grad():
                # Canonical LSQ init (Esser 2020, Sec. 2.1):
                #   s = 2 * <|v|> / sqrt(Qp)
                if self.per_channel:
                    dims = [d for d in range(x.dim()) if d != self.channel_dim]
                    mean_abs = x.abs().mean(dim=dims)
                    nc = x.shape[self.channel_dim]
                    init = (2 * mean_abs / math.sqrt(max(self.Qp, 1))).clamp(min=1e-6)
                    self.step.data = init.detach().clone()
                else:
                    init = (2 * x.abs().mean() / math.sqrt(max(self.Qp, 1))).clamp(min=1e-6)
                    self.step.data = init.detach().clone()
            self._initialised = True

        s = self._broadcast_step(x)
        # LSQ gradient trick: rescale the term feeding the scale parameter so
        # that s receives a balanced gradient regardless of tensor size. We keep
        # the forward value identical to  q*s  but route the gradient to s via a
        # (q - g*q.detach())*s + g*q.detach()*s.detach()  form.
        grad_factor = self._grad_factor(x)
        q = x / (s + 1e-12)
        q = ste_round(q)
        q = torch.clamp(q, self.Qn, self.Qp)
        xq = (q - grad_factor * q.detach()) * s + grad_factor * q.detach() * s.detach()
        return xq, s
