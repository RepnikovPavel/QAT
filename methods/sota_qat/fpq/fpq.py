"""Feature-Perturbed Quantization (FPQ) — arXiv:2503.11159.

Cross-checked against ``papers/2503.11159/document.md``:

* **SFP** (Eq. 9, 14): with probability ``p``, add ``δ ~ U[-s/2, s/2]`` to a
  layer's input feature, where ``s`` is that layer's activation quant step.
* **CSD** (Eq. 16–17): channel-wise standardize student/teacher features, then
  sum squared L2 distances over layers and channels.
* **Loss** (Eq. 22): task loss + ``L_CSD`` (classification uses CE; detection
  substitutes the detector's multi-task loss).

FPQ is an *add-on* to any LSQ-style QAT stack: it does not replace the
quantizer, it regularizes training via feature noise + FP teacher distillation.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def uniform_feature_noise(
    x: torch.Tensor,
    step: Union[float, torch.Tensor],
) -> torch.Tensor:
    """Sample δ ~ U[-s/2, s/2] broadcastable to ``x`` (Eq. 9).

    ``step`` may be a scalar or a tensor broadcastable to ``x`` (e.g. per-channel
    LSQ step with shape suitable for activations).
    """
    if not torch.is_tensor(step):
        s = torch.as_tensor(step, device=x.device, dtype=x.dtype)
    else:
        s = step.to(device=x.device, dtype=x.dtype).abs()
    # half-width of the uniform interval
    half = 0.5 * s
    # rand_like in [0,1) → map to [-half, half]
    u = torch.rand_like(x)
    return u * (2.0 * half) - half


def stochastic_feature_perturb(
    x: torch.Tensor,
    step: Union[float, torch.Tensor],
    p: float = 0.5,
    training: bool = True,
) -> torch.Tensor:
    """Apply SFP (Eq. 14): inject Eq. 9 noise with probability ``p``.

    When not training, or if a uniform draw ≥ p, returns ``x`` unchanged.
    A single Bernoulli is drawn per forward (not per-element) so a whole
    layer is either fully perturbed or left clean, matching Algorithm 1.
    """
    if (not training) or p <= 0.0:
        return x
    if p < 1.0 and torch.rand((), device=x.device).item() >= p:
        return x
    return x + uniform_feature_noise(x, step)


def channel_standardize(feat: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Channel-wise standardization for CSD (Eq. 16 ``Norm``).

    Expects NCHW (or N×C×*) features. Mean/std are computed over batch and
    all non-channel dims so each channel is zero-mean unit-variance.
    """
    if feat.dim() < 2:
        raise ValueError(f"expected at least N×C, got shape {tuple(feat.shape)}")
    # dims to reduce: all except channel (dim 1)
    reduce_dims = tuple(d for d in range(feat.dim()) if d != 1)
    mean = feat.mean(dim=reduce_dims, keepdim=True)
    var = feat.var(dim=reduce_dims, keepdim=True, unbiased=False)
    return (feat - mean) / (var.sqrt() + eps)


def csd_loss(
    student_feats: Sequence[torch.Tensor],
    teacher_feats: Sequence[torch.Tensor],
    eps: float = 1e-5,
) -> torch.Tensor:
    """Channel-wise Standardization Distillation loss (Eq. 17).

    ``L_CSD = Σ_layers Σ_channels || z̃_t^{c} - z̃_s^{c} ||_2^2`` after
    channel-wise standardization of each feature map.
    """
    if len(student_feats) != len(teacher_feats):
        raise ValueError(
            f"feature count mismatch: student={len(student_feats)} "
            f"teacher={len(teacher_feats)}"
        )
    if len(student_feats) == 0:
        return torch.tensor(0.0)

    total = None
    for zs, zt in zip(student_feats, teacher_feats):
        if zs.shape != zt.shape:
            # spatial mismatch: interpolate student to teacher size
            if zs.dim() == 4 and zt.dim() == 4 and zs.shape[1] == zt.shape[1]:
                zs = F.interpolate(zs, size=zt.shape[-2:], mode="bilinear",
                                   align_corners=False)
            else:
                raise ValueError(
                    f"incompatible features student={tuple(zs.shape)} "
                    f"teacher={tuple(zt.shape)}"
                )
        zs_n = channel_standardize(zs, eps=eps)
        zt_n = channel_standardize(zt.detach(), eps=eps)
        # sum of squared diffs over all dims (batch, channel, spatial)
        layer_loss = (zs_n - zt_n).pow(2).sum()
        total = layer_loss if total is None else total + layer_loss
    # normalise by batch size of the first map for scale stability
    b = student_feats[0].shape[0]
    return total / max(b, 1)


class StochasticFeaturePerturb(nn.Module):
    """nn.Module wrapper for SFP (Eq. 9, 14).

    Parameters
    ----------
    p : float
        Probability of injecting noise (paper default search around 0.1–0.5;
        CIFAR ablations peak near 0.1–0.3).
    step : float
        Fallback scalar quant step when none is passed to ``forward``.
    """

    def __init__(self, p: float = 0.5, step: float = 1.0) -> None:
        super().__init__()
        self.p = float(p)
        self.step = float(step)

    def forward(
        self,
        x: torch.Tensor,
        step: Optional[Union[float, torch.Tensor]] = None,
    ) -> torch.Tensor:
        s = self.step if step is None else step
        return stochastic_feature_perturb(x, s, p=self.p, training=self.training)


class FeatureHook:
    """Collect intermediate feature maps from named modules (for CSD)."""

    def __init__(self, modules: Iterable[nn.Module]) -> None:
        self.feats: List[torch.Tensor] = []
        self._handles = []
        for m in modules:
            self._handles.append(m.register_forward_hook(self._hook))

    def _hook(self, module, inputs, output) -> None:
        # store activation *output* of the module (post-conv / post-block)
        if isinstance(output, torch.Tensor):
            self.feats.append(output)
        elif isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
            self.feats.append(output[0])

    def clear(self) -> None:
        self.feats = []

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        self.feats = []


def fpq_regularizer(
    student_feats: Sequence[torch.Tensor],
    teacher_feats: Sequence[torch.Tensor],
    csd_weight: float = 1.0,
) -> torch.Tensor:
    """Weighted CSD term to add to the task loss (Eq. 22 second term)."""
    if csd_weight == 0.0:
        device = student_feats[0].device if student_feats else "cpu"
        return torch.zeros((), device=device)
    return csd_weight * csd_loss(student_feats, teacher_feats)
