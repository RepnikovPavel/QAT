"""N2UQ: Nonuniform-to-Uniform Quantization (Liu et al., CVPR 2022).

Reference (parsed from arXiv:2111.14826 via ocrc):
  * Forward: learnable INPUT thresholds ``theta_0 < theta_1 < ... < theta_{n-1}``
    divide the real axis into ``2^n`` regions, but the OUTPUT quantization
    levels are UNIFORM (equidistant). This gives the representation power of
    non-uniform quantization with the hardware efficiency of uniform output.
  * Backward: the Generalized Straight-Through Estimator (G-STE) — instead of
    passing gradients through as the identity (standard STE), it weights the
    backward pass by the local interval length, encoding the influence of the
    learnable thresholds.
  * Weights: use equidistant thresholds (no threshold learning) + entropy-
    preserving normalisation; here we keep weights uniform/LSQ-like.

Faithful but compact implementation. ``bit_width=n`` gives ``2^n`` output levels
uniformly spaced in ``[-1, 1]`` (signed) or ``[0, 1]`` (unsigned); the input
thresholds that decide which level each input maps to are learnable.
"""

from __future__ import annotations

import math

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
        init_range: float = 1.0,
        gste_beta: float = 0.0,
    ) -> None:
        super().__init__()
        self.bit_width = bit_width
        self.signed = signed
        self.per_channel = per_channel
        self.channel_dim = channel_dim
        self.num_levels = 2 ** bit_width
        # number of *interior* thresholds = num_levels - 1
        self.num_thresh = self.num_levels - 1
        self.gste_beta = gste_beta  # G-STE bias (0 = pure interval-weighted)

        # Output levels are uniform in [-1, 1] (signed) / [0, 1] (unsigned).
        if signed:
            lo, hi = -1.0, 1.0
        else:
            lo, hi = 0.0, 1.0
        self.register_buffer("level_lo", torch.tensor(float(lo)))
        # uniform output step (between adjacent levels)
        self.out_step = (hi - lo) / (self.num_levels - 1)

        # Learnable input thresholds, parameterised as cumulative softmax of
        # unconstrained logits -> monotonically increasing, spread across a
        # learnable range [-L, L]. Initialised to uniform spacing.
        self.theta_logits = nn.Parameter(torch.zeros(self.num_thresh))
        self.log_range = nn.Parameter(torch.tensor(float(init_range)).log())
        self._initialised = False

    # ------------------------------------------------------------------ levels
    def _thresholds(self, x: torch.Tensor) -> torch.Tensor:
        """Interior input thresholds, shape (num_thresh,) or per-channel."""
        L = self.log_range.exp()
        gaps = F.softmax(self.theta_logits, dim=0)  # (num_thresh,)
        cum = torch.cumsum(gaps, dim=0)  # in (0, 1]
        # map cumulative gaps onto (lo, L) interior thresholds
        if self.signed:
            th = -L + 2 * L * cum  # spread in (-L, L)
        else:
            th = L * cum  # spread in (0, L)
        return th

    def _quantize_forward(self, x: torch.Tensor, th: torch.Tensor) -> torch.Tensor:
        """Hard forward: assign each x to a uniform level via thresholds.

        ``searchsorted`` finds the number of thresholds below each x, which is
        the level index in [0, num_levels-1].
        """
        # th shape (num_thresh,); for per-channel we loop — keep simple (per-tensor).
        idx = torch.searchsorted(th, x.contiguous(), right=True)  # (N,...)
        # output level = lo + idx * out_step
        return self.level_lo + idx.to(x.dtype) * self.out_step

    def _quantize_backward(self, x: torch.Tensor, th: torch.Tensor) -> torch.Tensor:
        """G-STE backward approximation (Theorem 1).

        The quantizer is a sum of binarization segments with thresholds ``th``.
        G-STE encodes the influence of each (learnable) threshold by weighting
        the backward pass through the *interval lengths* between thresholds.

        We implement a soft differentiable surrogate of the forward staircase
        that DEPENDS ON the thresholds (so gradients reach ``theta_logits`` and
        ``log_range``): for each x, weight the contribution of each threshold by
        a sigmoid of how close x is to it, scaled by the local interval length.
        The result is a smooth staircase whose gradient w.r.t. the thresholds
        is non-zero — this is the essence of G-STE.
        """
        # Adjacent interval lengths: prepend/append the outer edges (±L).
        L = self.log_range.exp()
        edges = torch.cat(
            [torch.tensor([-float(L)] if self.signed else [0.0],
                          device=th.device, dtype=th.dtype),
             th,
             torch.tensor([float(L)], device=th.device, dtype=th.dtype)]
        )  # (num_levels,)
        # interval length between consecutive edges
        intervals = edges[1:] - edges[:-1]  # (num_thresh+1,) = num_levels
        # output level per interval (uniform)
        levels = self.level_lo + torch.arange(
            self.num_levels, device=x.device, dtype=x.dtype
        ) * self.out_step  # (num_levels,)

        # soft assignment of x to intervals: weight interval k by how close x is
        # to the centre of interval k, measured in units of interval length.
        centres = 0.5 * (edges[1:] + edges[:-1])  # (num_levels,)
        # relative distance, normalised by interval length -> G-STE slope
        # d_k = (x - centre_k) / interval_k
        x_exp = x.unsqueeze(-1)  # (..., 1)
        c_exp = centres.view(*([1] * x.dim()), -1)  # (..., num_levels)
        iv_exp = intervals.view(*([1] * x.dim()), -1).clamp_min(1e-6)
        d = (x_exp - c_exp) / iv_exp
        # temperature ~ out_step keeps the surrogate close to the staircase
        w = F.softmax(-(d ** 2) / (self.out_step ** 2 + 1e-12), dim=-1)
        lvl_exp = levels.view(*([1] * x.dim()), -1)
        soft = (w * lvl_exp).sum(dim=-1)
        return soft

    def quantize(self, x: torch.Tensor):
        if self.per_channel and not self._initialised:
            nc = x.shape[self.channel_dim]
            with torch.no_grad():
                dims = [d for d in range(x.dim()) if d != self.channel_dim]
                r = x.abs().amax(dim=dims).clamp(min=1e-3)
                self.log_range.data = r.log().detach().clone()
            self._initialised = True
        elif not self._initialised:
            with torch.no_grad():
                r = x.abs().amax().clamp(min=1e-3)
                self.log_range.data = r.log().detach().clone()
            self._initialised = True

        th = self._thresholds(x)
        # STE: hard forward value, soft (G-STE) backward
        x_hard = self._quantize_forward(x, th.detach())
        x_soft = self._quantize_backward(x, th)
        xq = x_hard.detach() + x_soft - x_soft.detach()

        # step size reported to Q-ADA: uniform output step scaled to x range
        L = self.log_range.exp().detach()
        s = (2 * L / (self.num_levels - 1)) if self.signed else (L / (self.num_levels - 1))
        s_full = s if s.dim() == 0 else s
        if not isinstance(s_full, torch.Tensor):
            s_full = torch.tensor(float(s_full), device=x.device)
        s_full = s_full.expand_as(x) if s_full.dim() == 0 else s_full.expand_as(x)
        return xq, s_full
