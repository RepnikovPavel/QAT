"""Q-ADA: Quantization-aware Attention Distribution Alignment.

Implements the parameter-free distillation loss from Q^2 (arXiv:2511.05898,
Sec. 3.3, Eq. 13-16).

Given a full-precision teacher feature ``X`` and the quantized student feature
``Xb = Q(X)`` (shape ``C x H x W`` per sample), Q-ADA builds a *saliency* map
that combines statistical saliency with quantization-vulnerability::

    mu_c, sigma_c   = per-channel spatial mean / std
    dX_{c,ij}       = |X_{c,ij} - Xb_{c,ij}|            (Eq. 13)
    s_c             = per-channel quantization step
    z_{c,ij} = |X_{c,ij} - mu_c| / (sigma_c + k)         (Eq. 14)
    r_{c,ij} = dX_{c,ij} / (s_c + k)                     (Eq. 14)
    S_{c,ij} = log(1 + z^2) + gamma * log(1 + r^2)       (Eq. 15)
    A~_{c,ij} = Sigmoid(S_{c,ij})                        (attention map)

The teacher and student attention maps are spatially normalised to
probability distributions and aligned via Jensen-Shannon divergence::

    L_ADA = JS( P_teacher || R_student )                 (Eq. 16)

with ``P = A~_t / sum A~_t`` over spatial positions (A~ = Sigmoid(S)),
``R = A~_s / sum A~_s`` — following Eq. 16 (paper line 215) verbatim.

Default loss weight: 0.01 (Appendix 8).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _per_channel_stats(x: torch.Tensor):
    """Return (mean_c, std_c) computed over spatial dims of (N, C, *spatial)."""
    if x.dim() == 2:
        return x.mean(dim=0), x.std(dim=0, unbiased=False)
    spatial = list(range(2, x.dim()))
    mu = x.mean(dim=spatial)  # (N, C)
    sigma = x.std(dim=spatial, unbiased=False)
    return mu, sigma


def _broadcast_channels(v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    shape = [v.shape[0], v.shape[1]] + [1] * (x.dim() - 2)
    return v.reshape(shape)


def raw_saliency(
    x: torch.Tensor,
    xb: torch.Tensor,
    step: Optional[torch.Tensor] = None,
    gamma: float = 1.0,
    k: float = 1e-3,
) -> torch.Tensor:
    """Raw (pre-activation) saliency score ``S`` (Eq. 15), no sigmoid."""
    mu, sigma = _per_channel_stats(x)
    mu_b = _broadcast_channels(mu, x)
    sg_b = _broadcast_channels(sigma, x)

    z = (x - mu_b).abs() / (sg_b + k)  # statistical saliency
    dX = (x - xb).abs()  # quantization distortion
    if step is not None:
        if step.dim() <= 2:
            s_b = _broadcast_channels(step, x) if step.dim() == 2 else step
        else:
            s_b = step
        denom = s_b + k
    else:
        denom = sg_b + k  # fall back to sigma when step unknown

    r = dX / denom
    return torch.log1p(z.pow(2)) + gamma * torch.log1p(r.pow(2))


def saliency_map(
    x: torch.Tensor,
    xb: torch.Tensor,
    step: Optional[torch.Tensor] = None,
    gamma: float = 1.0,
    k: float = 1e-3,
    temp: float = 1.0,
) -> torch.Tensor:
    """Compute the attention map ``A~ = Sigmoid(S)`` (Eq. 13-15).

    The sigmoid form matches the paper, but because ``S`` saturates sigmoid for
    strong quantization distortion, the *distribution* used by Q-ADA is built
    by softmax over the raw ``S`` (see ``QADALoss``), which yields sharp,
    informative spatial distributions.
    """
    S = raw_saliency(x, xb, step, gamma, k)
    return torch.sigmoid(S / max(temp, 1e-6))


def _spatial_softmax(attn: torch.Tensor) -> torch.Tensor:
    """Normalise attention to a spatial probability distribution per channel."""
    flat = attn.flatten(start_dim=2)  # (N, C, HW)
    return F.softmax(flat, dim=2).view_as(attn)


def _spatial_normalize(attn: torch.Tensor) -> torch.Tensor:
    """Normalise an attention weight to a spatial prob. distribution (Eq. 16).

    The paper (Sec. 3.3, line 215) builds the alignment distribution as
    ``P_{c,ij} = A~_{c,ij} / sum_{i',j'} A~_{c,i'j'}`` with
    ``A~ = Sigmoid(S)`` — i.e. L1-normalise the sigmoid attention map, NOT a
    softmax over the raw score. We follow the paper verbatim here.
    """
    flat = attn.flatten(start_dim=2)  # (N, C, HW)
    z = flat.sum(dim=2, keepdim=True).clamp_min(1e-12)
    return (flat / z).view_as(attn)


def js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Jensen-Shannon divergence between two spatial prob. distributions.

    Returns the mean JS divergence over batch & channels.
    """
    m = 0.5 * (p + q).clamp_min(1e-12)
    # KL(p||m) = sum p log(p/m); base-e -> divide by ln2 for bits? Keep nats.
    kl_pm = (p * (torch.log(p.clamp_min(1e-12)) - torch.log(m))).sum(dim=2)
    kl_qm = (q * (torch.log(q.clamp_min(1e-12)) - torch.log(m))).sum(dim=2)
    js = 0.5 * kl_pm + 0.5 * kl_qm
    return js.mean()


def kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Alternative KL divergence option (paper Sec. 4.5 compares KL vs JS)."""
    m = q  # student distribution
    kl = (p * (torch.log(p.clamp_min(1e-12)) - torch.log(m.clamp_min(1e-12)))).sum(dim=2)
    return kl.mean()


class QADALoss(nn.Module):
    """Q-ADA distillation loss (Eq. 13-16)."""

    def __init__(
        self,
        gamma: float = 1.0,
        k: float = 1e-3,
        divergence: str = "js",
        temp: float = 1.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.k = k
        self.temp = temp
        self.divergence = divergence

    def forward(
        self,
        x_teacher: torch.Tensor,
        x_student_fp: torch.Tensor,
        x_student_q: torch.Tensor,
        step: Optional[torch.Tensor] = None,
        temp: Optional[float] = None,
    ) -> torch.Tensor:
        """Q-ADA loss (Eq. 13-16), verbatim interpretation.

        For a feature X the saliency S(X) uses the quantization distortion
        Delta = |X - Q(X)| of THAT SAME feature (Eq. 13), not a teacher-vs-
        student difference. Hence:
          * teacher (FP): its feature is never quantized, so Q(X)=X and
            Delta=0 -> S uses only the statistical z-term (Eq. 15).
          * student (quant): X = its FP feature, Q(X) = its quantized feature,
            both captured at the same supervision point.

        Args:
            x_teacher: FP teacher feature (detached; its own distortion is 0).
            x_student_fp: student FP feature (detached; supplies mu/sigma/z/Delta).
            x_student_q: student quantized feature (the Q(X) above; carries grad).
            step: per-channel quantization step ``s_c`` (Eq. 14). May be None.
            temp: sigmoid temperature for the attention map.
        """
        # Eq. 16 (paper line 215): A~ = Sigmoid(S); distribution P = A~/sum(A~).
        # Teacher Delta = 0 (FP), so Q(X)=X.
        x_teacher = x_teacher.detach()
        x_student_fp = x_student_fp.detach()
        S_t = raw_saliency(x_teacher, x_teacher, step, self.gamma, self.k)
        # Student: X = x_student_fp, Q(X) = x_student_q.
        S_s = raw_saliency(x_student_fp, x_student_q, step, self.gamma, self.k)
        t = self.temp if temp is None else temp
        A_t = torch.sigmoid(S_t / max(t, 1e-6))
        A_s = torch.sigmoid(S_s / max(t, 1e-6))
        P = _spatial_normalize(A_t)
        R = _spatial_normalize(A_s)
        if self.divergence == "kl":
            return kl_divergence(P, R)
        return js_divergence(P, R)
