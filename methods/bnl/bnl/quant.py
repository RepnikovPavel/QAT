"""Mean-threshold 1-bit quantisation + STE (arXiv:2509.07025 Eq. 1, Alg. 1).

Paper Eq. 1:
    p_b = 1 if p > p_mean else 0
where p_mean is the mean of the parameters of the layer tensor.

STE (Alg. 1, Alcorn-style):
    W_q = W + stopgrad(Quant(W) - W)
so the forward uses binary weights and the backward treats Quant as identity.
"""

from __future__ import annotations

import torch
from torch import Tensor


def mean_threshold_quantize(p: Tensor) -> Tensor:
    """Eq. 1: hard threshold at the mean of ``p`` → {0, 1}."""
    p_mean = p.mean()
    return (p > p_mean).to(dtype=p.dtype)


def ste_quantize(p: Tensor) -> Tensor:
    """Alg. 1 lines 2–3 / 5–6: binary forward, identity STE backward.

    Training and inference both use the binary values in the forward pass.
    Gradients flow to the full-precision ``p`` via the STE identity.
    """
    p_b = mean_threshold_quantize(p)
    # W_q = W + NoGradient(Quant(W) - W)
    return p + (p_b - p).detach()
