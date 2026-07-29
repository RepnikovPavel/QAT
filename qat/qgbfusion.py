"""Q-GBFusion: Quantization-aware Gradient Balancing Fusion.

Implements the core training-time module from Q^2 (arXiv:2511.05898, Sec. 3.2).

A Q-GBFusion node replaces a feature-fusion op (e.g. ``torch.cat``) at a neck
concatenation point of a detection/segmentation network. For ``K`` quantized
branch features ``F~_i`` it:

1. Maintains an unconstrained dual state ``lambda in R^K`` and produces a
   simplex allocation via ``alpha = softmax(lambda)`` (Eq. 5).
2. Rescales each branch ``F'_i = alpha_i * F~_i`` then applies the original
   fusion op (concat by default) (Eq. 5).
3. Applies a per-channel LayerNorm over the fused feature (Eq. 6) to stabilise
   gradient propagation under quantization noise.
4. Updates ``lambda`` via a closed-loop feedback law driven by the per-branch
   gradient energy ``G_i = ||dL/dF~_i||_2`` (Eq. 8-10):
       Gbar_i <- (1-beta) Gbar_i + beta G_i            (EMA)
       e_i    <- log(Gbar_i+eps) - mean_j log(Gbar_j+eps) - tau_i
       lambda_i <- lambda_i - eta * e_i

At deployment the closed loop is disabled, ``alpha`` is frozen, and the
LayerNorm is folded into the following layer via calibration (see ``fold``).
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class QGBFusion(nn.Module):
    """Gradient-energy-balanced feature fusion node (training-time controller)."""

    def __init__(
        self,
        num_branches: int,
        num_channels: int,
        eta: float = 0.05,
        beta: float = 0.1,
        eps: float = 1e-6,
        tau: Optional[List[float]] = None,
        ln_eps: float = 1e-5,
        ln_affine: bool = True,
    ) -> None:
        """Args:
        num_branches: number ``K`` of fused feature branches.
        num_channels: TOTAL number of channels of the fused feature, i.e. the
            sum of branch channel counts (LayerNorm is applied over this).
        """
        super().__init__()
        self.K = num_branches
        self.C = num_channels
        self.eta = eta
        self.beta = beta
        self.eps = eps
        self.ln_eps = ln_eps

        if tau is None:
            tau = [0.0] * num_branches
        assert len(tau) == num_branches
        self.register_buffer("tau", torch.tensor(tau, dtype=torch.float32))

        # dual logits -> softmax gives the simplex allocation alpha
        self.lam = nn.Parameter(torch.zeros(num_branches))

        # EMA of per-branch gradient energy (not a learned parameter)
        self.register_buffer(
            "Gbar", torch.full((num_branches,), -1.0, dtype=torch.float32)
        )
        self.register_buffer("step_count", torch.zeros(1))

        # post-fusion LayerNorm over the channel dimension (per spatial location)
        self.ln = nn.LayerNorm(num_channels, eps=ln_eps, elementwise_affine=ln_affine)

        # gradient-energy probe: register hooks on the *rescaled* branch features
        # so that backward through F'_i captures the post-gating gradient. The
        # paper defines G_i = ||dL/dF~_i||_2 with dF~_i = alpha_i dF'_i; we
        # equivalently probe F'_i and add log(alpha_i) per Eq.(7).
        self._probe_handles: List[torch.utils.hooks.RemovableHandle] = []
        self._grad_samples: List[Optional[torch.Tensor]] = []

    # ------------------------------------------------------------------ utils
    @property
    def alpha(self) -> torch.Tensor:
        return F.softmax(self.lam, dim=0)

    def _probe_grad(self, idx: int):
        def hook(grad: torch.Tensor) -> None:
            # store detached norm; keep per-call to avoid retaining graph
            self._grad_samples[idx] = grad.detach()

        return hook

    # -------------------------------------------------------------- forward
    def forward(self, branches: List[torch.Tensor]) -> torch.Tensor:
        assert len(branches) == self.K, f"expected {self.K} branches"
        alpha = self.alpha  # (K,)

        scaled_branches: List[torch.Tensor] = []
        self._grad_samples = [None] * self.K
        for i, feat in enumerate(branches):
            fp = feat * alpha[i]
            if self.training:
                fp.retain_grad()
                fp.register_hook(self._probe_grad(i))
            scaled_branches.append(fp)

        fused = torch.cat(scaled_branches, dim=1)  # concat over channels
        # LayerNorm expects channels-last; fused is (N, C_total, ...). Apply LN
        # per-spatial-location across channels: move channels to last dim.
        fused = self._channel_layernorm(fused)
        return fused

    def _channel_layernorm(self, x: torch.Tensor) -> torch.Tensor:
        """LayerNorm over the channel dim at each spatial location (Eq. 6).

        Input shape (N, C, *spatial). We permute channels to the end, LN, then
        permute back.
        """
        if x.dim() == 2:  # (N, C)
            return self.ln(x)
        spatial_dims = list(range(2, x.dim()))
        x_perm = x.permute(0, *spatial_dims, 1)  # (N, *spatial, C)
        x_perm = self.ln(x_perm)
        # inverse perm: channel axis (now last) goes back to dim 1.
        # after permute the axis order is [N, s0, s1, ..., C]; we want [N, C, s0, ...]
        inv = [0, x_perm.dim() - 1] + list(range(1, x_perm.dim() - 1))
        return x_perm.permute(*inv).contiguous()

    # ------------------------------------------------- closed-loop update
    @torch.no_grad()
    def update_dual(self) -> Optional[torch.Tensor]:
        """Closed-loop feedback update of ``lambda`` (Eq. 8-10).

        Call this after ``loss.backward()`` and before ``optimizer.step()``.
        Returns the per-branch gradient energy tensor ``G`` for logging, or
        ``None`` if no gradients were captured (e.g. eval mode).
        """
        if not self.training:
            return None
        # any captured gradient?
        if all(g is None for g in self._grad_samples):
            return None

        G = torch.zeros(self.K, device=self.lam.device, dtype=self.lam.dtype)
        alpha = self.alpha
        for i, g in enumerate(self._grad_samples):
            if g is None:
                gi_norm = torch.zeros((), device=self.lam.device)
            else:
                gi_norm = g.norm(2)
            # branch gradient w.r.t. F~_i = alpha_i * grad w.r.t. F'_i (Eq.7)
            gi = (alpha[i].detach().abs() + self.eps) * gi_norm
            G[i] = gi

        # EMA update of Gbar (Eq. 8); initialise on first call
        if (self.Gbar < 0).all():
            self.Gbar.copy_(G)
        else:
            self.Gbar.mul_(1 - self.beta).add_(G, alpha=self.beta)

        # log-energy deviations from the mean (Eq. 9)
        log_G = torch.log(self.Gbar + self.eps)
        e = log_G - log_G.mean() - self.tau.to(log_G.device)

        # first-order feedback law (Eq. 10)
        self.lam.add_(-self.eta * e)

        self.step_count += 1
        return self.Gbar.detach().clone()
