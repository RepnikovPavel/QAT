"""Learned Step-size Quantization (LSQ) [Esser et al., ICLR 2020].

LSQ learns a positive scale ``s`` per tensor (or per output-channel for weights).
Forward::

    xq = s * clip(round(x / s), Qn, Qp)

The scale ``s`` is a learnable parameter. Its gradient is the data-path
derivative ``d xq/d s = q - x/s`` (interior; Qn/Qp in the saturated exterior),
scaled by the paper's factor ``g = 1/sqrt(Qp * N)`` so that step updates stay
balanced with weight updates (Esser 2020 Sec. 2.2, Eq. 4).

This is one of the conv quantizers used in the Q^2 paper (Table 1, "LSQ").
Cross-checked against ``papers/refs/lsq/document.md`` (Sec. 2.1 init, Sec. 2.2
gradient scale, Eq. 5 STE).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .base import QuantizerBase


class _LSQFunction(torch.autograd.Function):
    """Fake-quant with the correct LSQ scale gradient (Esser et al. 2020).

    Forward: ``xq = s * clip(round(x/s), Qn, Qp)``.

    Backward (Eq. 5 of the LSQ paper): the straight-through estimator passes the
    input gradient through where ``x/s`` is *inside* the quant range and zero in
    the saturated exterior; the scale gradient is ``q - x/s`` interior and the
    saturated level (Qn/Qp) in the exterior.
    """

    @staticmethod
    def forward(ctx, x, step, Qn, Qp, grad_factor):
        s = step
        inv_s = 1.0 / s.clamp(min=1e-12)
        v = x * inv_s
        q = v.round().clamp(Qn, Qp)
        ctx.save_for_backward(x, s, q)
        ctx.Qn = Qn
        ctx.Qp = Qp
        ctx.grad_factor = grad_factor
        return q * s

    @staticmethod
    def backward(ctx, grad_output):
        x, s, q = ctx.saved_tensors
        Qn, Qp = ctx.Qn, ctx.Qp
        g = ctx.grad_factor
        inv_s = 1.0 / s.clamp(min=1e-12)
        v = x * inv_s
        # STE w.r.t. x: pass gradient where x/s is inside the quant range.
        # Outside the clip region round+clip saturates -> zero local grad.
        interior = (v > Qn) & (v < Qp)
        grad_x = grad_output * interior.to(dtype=grad_output.dtype)
        # Esser d xq/d s: interior q - x/s; exterior the saturated level.
        ds = q - v
        ds = torch.where(v <= Qn, torch.full_like(ds, Qn), ds)
        ds = torch.where(v >= Qp, torch.full_like(ds, Qp), ds)
        grad_s = grad_output * ds * g
        # Reduce grad_s back to the shape of the (broadcast) step parameter.
        while grad_s.dim() > s.dim():
            grad_s = grad_s.sum(dim=0)
        for dim in range(s.dim()):
            if s.shape[dim] == 1 and grad_s.shape[dim] > 1:
                grad_s = grad_s.sum(dim=dim, keepdim=True)
        return grad_x, grad_s, None, None, None


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

        # Positive step parameter (projected to >= 1e-8 after each optim step).
        # Initialised small; overwritten on the first real forward by Esser init.
        self.step = nn.Parameter(torch.tensor(float(init_step)))
        # Persist the init flag across state_dict so lazy init is NOT re-run
        # after loading a trained checkpoint mid-session.
        self.register_buffer("_initialised", torch.tensor(False), persistent=True)

    def _grad_factor(self, x: torch.Tensor) -> float:
        """LSQ step-size gradient scale (Esser et al. 2020, Sec. 2.2).

        ``g = 1/sqrt(Qp * N)`` where N is the number of quantized elements.
        Without it the step-size updates are 2-3 orders of magnitude larger than
        weight updates and the network fails to converge (LSQ Table 3).
        """
        return 1.0 / math.sqrt(max(self.Qp, 1) * x.numel())

    def _positive_step(self) -> torch.Tensor:
        """Always-positive step used in forward/backward.

        Prefer ``clamp`` over ``abs`` so the gradient never flips sign through a
        negative parameter value (which previously oscillated/exploded).
        """
        return self.step.clamp(min=1e-8)

    def _broadcast_step(self, x: torch.Tensor) -> torch.Tensor:
        s = self._positive_step()
        if self.per_channel:
            shape = [1] * x.dim()
            shape[self.channel_dim] = x.shape[self.channel_dim]
            # Expand a still-scalar placeholder to the channel count.
            if s.numel() == 1:
                s = s.expand(x.shape[self.channel_dim]).clone()
            s = s[: x.shape[self.channel_dim]].reshape(shape)
        return s

    def quantize(self, x: torch.Tensor):
        if not bool(self._initialised.item()):
            with torch.no_grad():
                # Canonical LSQ init (Esser 2020, Sec. 2.1):
                #   s = 2 * <|v|> / sqrt(Qp)
                if self.per_channel:
                    dims = [d for d in range(x.dim()) if d != self.channel_dim]
                    mean_abs = x.abs().mean(dim=dims)
                    init = (2 * mean_abs / math.sqrt(max(self.Qp, 1))).clamp(min=1e-6)
                    # Replace the Parameter so its shape matches out-channels
                    # (in-place .data shape change is fragile across torch).
                    self.step = nn.Parameter(init.detach().clone())
                else:
                    init = (2 * x.abs().mean() / math.sqrt(max(self.Qp, 1))).clamp(min=1e-6)
                    self.step.data.fill_(float(init))
            self._initialised.fill_(True)

        s = self._broadcast_step(x)
        g = self._grad_factor(x)
        xq = _LSQFunction.apply(x, s, float(self.Qn), float(self.Qp), float(g))
        return xq, s
