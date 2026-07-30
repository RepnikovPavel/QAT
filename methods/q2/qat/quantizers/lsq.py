"""Learned Step-size Quantization (LSQ) [Esser et al., ICLR 2020].

LSQ learns a positive scale ``s`` per tensor (or per output-channel for weights).
Forward::

    xq = s * clip(round(x / s), Qn, Qp)

The scale ``s`` is a learnable parameter. Its gradient is the data-path
derivative ``Σ ∂L/∂xq · (q − x/s)`` scaled by the paper's factor
``g = 1/sqrt(Qp · N)`` so that step updates stay balanced with weight updates.

This is one of the conv quantizers used in the Q^2 paper (Table 1, "LSQ").
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .base import QuantizerBase


class _LSQFunction(torch.autograd.Function):
    """Fake-quant with correct LSQ scale gradient (Esser et al. 2020)."""

    @staticmethod
    def forward(ctx, x, step, Qn, Qp, grad_factor):
        # step is already positive and broadcast-shaped to match x (or channel)
        s = step
        # Save for backward: use the pre-round ratio for ∂xq/∂s = q - x/s
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
        # Outside the clip region the round+clip saturates → zero local grad.
        interior = (v > Qn) & (v < Qp)
        grad_x = grad_output * interior.to(dtype=grad_output.dtype)
        # Esser Eq. for ∂xq/∂s:
        #   interior:  q − x/s
        #   v <= Qn:   Qn
        #   v >= Qp:   Qp
        # (q is already clamp(round(v), Qn, Qp), so exterior needs override.)
        ds = q - v
        ds = torch.where(v <= Qn, torch.full_like(ds, Qn), ds)
        ds = torch.where(v >= Qp, torch.full_like(ds, Qp), ds)
        grad_s = grad_output * ds * g
        # Reduce grad_s back to the shape of the step Parameter.
        # step was broadcast to x; sum over the expanded dims.
        while grad_s.dim() > s.dim():
            grad_s = grad_s.sum(dim=0)
        # If s was broadcast along spatial dims (per-channel), sum those dims.
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

        # Positive step parameter (projected after each optim step in train).
        # Initialised small; overwritten on first real forward by Esser init.
        self.step = nn.Parameter(torch.tensor(float(init_step)))
        # Persist init flag across state_dict so dummy-forward init is not
        # re-run after loading a trained checkpoint mid-session.
        self.register_buffer("_initialised", torch.tensor(False), persistent=True)

    def _grad_factor(self, x: torch.Tensor) -> float:
        """LSQ step-size gradient scale (Esser et al. 2020, Sec. 2.2).

        ``g = 1/sqrt(Qp * N)`` where N is the number of quantized elements.
        """
        return 1.0 / math.sqrt(max(self.Qp, 1) * x.numel())

    def _positive_step(self) -> torch.Tensor:
        """Always-positive step used in forward/backward.

        Prefer ``clamp`` over ``abs`` so the gradient never flips sign through
        a negative parameter value (which previously oscillated/exploded).
        """
        return self.step.clamp(min=1e-8)

    def _broadcast_step(self, x: torch.Tensor) -> torch.Tensor:
        s = self._positive_step()
        if self.per_channel:
            shape = [1] * x.dim()
            shape[self.channel_dim] = x.shape[self.channel_dim]
            # Pad/truncate if step was still a scalar placeholder.
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
                    # Replace Parameter so shape matches out-channels (not a
                    # scalar placeholder) — in-place .data shape change is
                    # fragile across torch versions.
                    self.step = nn.Parameter(init.detach().clone())
                else:
                    init = (2 * x.abs().mean() / math.sqrt(max(self.Qp, 1))).clamp(min=1e-6)
                    self.step.data.fill_(float(init))
            self._initialised.fill_(True)

        s = self._broadcast_step(x)
        g = self._grad_factor(x)
        xq = _LSQFunction.apply(x, s, float(self.Qn), float(self.Qp), float(g))
        return xq, s
