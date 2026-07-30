"""PACT: Parameterized Clipping Activation for QAT [Choi et al., 2018].

PACT learns an upper clipping bound ``alpha`` for activations (the lower bound
is 0 for ReLU'd activations, or symmetric -alpha for signed). Weights are
quantized symmetric with a learnable scale (LSQ-style) on top of the same
clipping. This is the conv quantizer "PACT" used in Q^2 (Table 1).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import QuantizerBase, ste_round


class PACT(QuantizerBase):
    def __init__(
        self,
        bit_width: int = 4,
        signed: bool = True,
        per_channel: bool = False,
        channel_dim: int = 0,
        init_alpha: float = 8.0,
    ) -> None:
        super().__init__()
        self.bit_width = bit_width
        self.signed = signed
        self.per_channel = per_channel
        self.channel_dim = channel_dim

        # PACT: activations clipped to [0, alpha] (relu) or [-alpha, alpha].
        # For signed weights we use symmetric [-alpha, alpha].
        levels = 2 ** bit_width
        if signed:
            self.Qn = -(levels // 2)
            self.Qp = levels // 2 - 1
        else:
            self.Qn = 0
            self.Qp = levels - 1

        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))
        # Persist the init flag across state_dict so lazy per-channel init is
        # NOT re-run after loading a trained checkpoint mid-session.
        self.register_buffer("_initialised", torch.tensor(False), persistent=True)

    def _broadcast(self, v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.per_channel:
            shape = [1] * x.dim()
            shape[self.channel_dim] = x.shape[self.channel_dim]
            return v[: x.shape[self.channel_dim]].reshape(shape)
        return v

    def quantize(self, x: torch.Tensor):
        if self.per_channel and not bool(self._initialised.item()):
            nc = x.shape[self.channel_dim]
            with torch.no_grad():
                dims = [d for d in range(x.dim()) if d != self.channel_dim]
                a = x.abs().amax(dim=dims).clamp(min=1e-3)
                self.alpha.data = a.detach().clone()
            self._initialised.fill_(True)

        alpha = self.alpha.abs()
        a = self._broadcast(alpha, x)
        # clip then uniform quantize within [-alpha, alpha] (or [0, alpha])
        if self.signed:
            xc = torch.clamp(x, -a, a)
            s = (2 * a) / (self.Qp - self.Qn)
        else:
            xc = torch.clamp(x, torch.zeros_like(a), a)
            s = a / self.Qp
        q = ste_round(xc / (s + 1e-12))
        q = torch.clamp(q, self.Qn, self.Qp)
        xq = q * s
        return xq, s
