"""N2UQ: Non-uniform Quantization via learnable breakpoints.

Non-Uniform Quantization (N2UQ, Liu et al., NeurIPS 2022) learns *where* to
place the quantization thresholds (breakpoints) instead of using evenly spaced
levels, so it can allocate more levels where the data is dense. The original
paper parameterises the non-uniform quantizer as a piecewise-linear function
implemented via a combination of learnable thresholds + a learnable "softmax"
relaxation of the rounding.

We implement a faithful, self-contained N2UQ variant:

* learnable output levels ``t = cumsum(softmax(theta))`` mapped onto a learned
  dynamic range ``[-L, L]``;
* each input ``x`` is quantized to the nearest level via differentiable
  soft-assignment (softmax over distances, temperature-annealed) in training
  and hard-nearest at eval.

N2UQ is the strongest conv quantizer in Q^2 (Table 1): YOLOv5s W4A4 82.1%.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import QuantizerBase


class N2UQ(QuantizerBase):
    def __init__(
        self,
        bit_width: int = 4,
        signed: bool = True,
        per_channel: bool = False,
        channel_dim: int = 0,
        init_range: float = 8.0,
        temp: float = 1.0,
    ) -> None:
        super().__init__()
        self.bit_width = bit_width
        self.signed = signed
        self.per_channel = per_channel
        self.channel_dim = channel_dim
        self.levels = 2 ** bit_width
        self.temp = temp

        # learnable log-spacings between breakpoints -> non-uniform placement
        # theta has length = levels - 1; softmax gives positive gaps.
        self.theta = nn.Parameter(torch.zeros(self.levels - 1))
        # learnable full dynamic range (symmetric)
        self.log_range = nn.Parameter(torch.tensor(float(init_range)).log())
        self._initialised = False

    def _levels_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Return the non-uniform quantization levels, shaped for broadcasting.

        Output shape is ``(levels, 1, 1, ..., 1)`` so it broadcasts against
        ``x`` of shape ``(N, C, *spatial)`` along a leading "levels" axis.
        Levels span [-L, L] symmetrically, placed at cumulative softmax gaps.
        """
        L = self.log_range.exp()  # scalar (per-tensor) or (nc,) per-channel
        gaps = F.softmax(self.theta, dim=0)  # (levels-1,)
        cum = torch.cumsum(gaps, dim=0)  # in (0, 1]
        if self.signed:
            interior = 2 * L * (cum - 0.5)
            edge = -L if not isinstance(L, float) else torch.tensor(-float(L))
        else:
            interior = L * cum
            edge = torch.zeros_like(L) if isinstance(L, torch.Tensor) else torch.tensor(0.0)

        # Per-tensor path (L scalar): build a 1D (levels,) vector, then expand.
        if not isinstance(L, torch.Tensor) or L.dim() == 0:
            L0 = float(L) if not isinstance(L, torch.Tensor) else L.item()
            gaps_sum = gaps.sum()  # ~1
            # distribute levels over [-L0, L0]
            lvl = torch.cat(
                [torch.tensor([-L0], device=gaps.device, dtype=gaps.dtype), interior]
            )
        else:
            # per-channel L: shape (nc,). levels per channel -> (nc, levels).
            edge_col = edge.reshape(-1, 1)
            interior_t = interior if interior.dim() == 2 else interior.reshape(-1, 1)
            lvl = torch.cat([edge_col, interior_t], dim=1)  # (nc, levels)

        # Reshape for broadcasting against x along a leading "levels" axis.
        x_ndim = x.dim()
        if lvl.dim() == 1:
            shape = [lvl.shape[0]] + [1] * x_ndim
            return lvl.view(*shape)
        # per-channel (nc, levels) -> (1, nc, levels) then we expand per spatial
        # We handle per-channel by indexing later; simplest: (levels, nc, 1, ...)
        lvl = lvl.t()  # (levels, nc)
        shape = [lvl.shape[0], lvl.shape[1]] + [1] * (x_ndim - 1)
        return lvl.view(*shape)

    def quantize(self, x: torch.Tensor):
        if self.per_channel and not self._initialised:
            nc = x.shape[self.channel_dim]
            with torch.no_grad():
                dims = [d for d in range(x.dim()) if d != self.channel_dim]
                r = x.abs().amax(dim=dims).clamp(min=1e-3)
                self.log_range.data = r.log().detach().clone()
            self._initialised = True

        L = self.log_range.exp()
        # effective step size reported to Q-ADA (Eq.14): range / (levels-1).
        # For per-channel L has shape (nc,); we broadcast it to x afterwards.
        s = (2 * L / (self.levels - 1)) if self.signed else (L / (self.levels - 1))

        lvl = self._levels_tensor(x)  # (levels, *broadcast)
        x_exp = x.unsqueeze(0)  # (1, *x-shape)
        d2 = (x_exp - lvl) ** 2  # (levels, *x-shape)
        if self.training:
            w = F.softmax(-d2 / (self.temp + 1e-12), dim=0)
            xq = (w * lvl).sum(dim=0)
        else:
            idx = d2.argmin(dim=0)
            xq = lvl.gather(0, idx.unsqueeze(0)).squeeze(0)

        s_full = s.detach()
        if s_full.dim() == 0:
            s_full = s_full.expand_as(x)
        else:  # per-channel: reshape to broadcast over channel_dim
            shape = [1] * x.dim()
            shape[self.channel_dim] = x.shape[self.channel_dim]
            s_full = s_full.reshape(shape).expand_as(x)
        return xq, s_full
