"""Per-example feature normalize (arXiv:2509.07025 Alg. 1 step 9).

Paper: "Normalize is the normalization function … normalizes the features of
each example so that it has zero mean and unit standard deviation."

This is fixed (non-learnable) zero-mean / unit-std over all non-batch axes,
applied after the binary linear/conv transform and before the activation.
"""

from __future__ import annotations

import torch
from torch import Tensor


def per_example_normalize(z: Tensor, eps: float = 1e-5) -> Tensor:
    """Zero-mean unit-std over every axis except batch (dim 0).

    Works for both N×C (linear) and N×C×H×W (conv) tensors.
    """
    if z.dim() < 2:
        raise ValueError(f"expected rank ≥ 2, got shape {tuple(z.shape)}")
    reduce_dims = tuple(range(1, z.dim()))
    mean = z.mean(dim=reduce_dims, keepdim=True)
    var = z.var(dim=reduce_dims, keepdim=True, unbiased=False)
    return (z - mean) / torch.sqrt(var + eps)
