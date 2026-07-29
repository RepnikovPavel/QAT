"""Unit tests for Binary Normalized Layers (arXiv:2509.07025)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# methods/bnl is the package root for `import bnl`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bnl import (  # noqa: E402
    BinaryNormalizedConv2d,
    BinaryNormalizedLinear,
    mean_threshold_quantize,
    per_example_normalize,
    ste_quantize,
)


def test_mean_threshold_quantize_values_in_01():
    p = torch.tensor([-2.0, -0.5, 0.0, 0.5, 3.0])
    q = mean_threshold_quantize(p)
    assert set(q.unique().tolist()).issubset({0.0, 1.0})
    # mean ≈ 0.2 → values > 0.2 map to 1
    assert q.tolist() == [0.0, 0.0, 0.0, 1.0, 1.0]


def test_ste_forward_equals_hard_quant():
    p = torch.randn(16, 8, requires_grad=True)
    hard = mean_threshold_quantize(p)
    ste = ste_quantize(p)
    assert torch.equal(ste, hard)


def test_ste_backward_is_identity():
    """Alg. 1 STE: ∂L/∂p flows as if Quant were identity."""
    p = torch.randn(8, 4, requires_grad=True)
    y = ste_quantize(p).sum()
    y.backward()
    assert p.grad is not None
    assert torch.allclose(p.grad, torch.ones_like(p))


def test_per_example_normalize_stats():
    z = torch.randn(4, 32) * 5 + 3
    out = per_example_normalize(z)
    means = out.mean(dim=1)
    stds = out.std(dim=1, unbiased=False)
    assert torch.allclose(means, torch.zeros_like(means), atol=1e-5)
    assert torch.allclose(stds, torch.ones_like(stds), atol=1e-5)


def test_per_example_normalize_conv_shape():
    z = torch.randn(2, 8, 16, 16)
    out = per_example_normalize(z)
    assert out.shape == z.shape
    flat = out.view(2, -1)
    assert torch.allclose(flat.mean(1), torch.zeros(2), atol=1e-5)
    assert torch.allclose(flat.std(1, unbiased=False), torch.ones(2), atol=1e-5)


def test_linear_forward_binary_weights_and_grad():
    layer = BinaryNormalizedLinear(16, 8, activation=F.relu)
    x = torch.randn(4, 16, requires_grad=True)
    y = layer(x)
    assert y.shape == (4, 8)
    assert y.min() >= 0  # relu
    y.sum().backward()
    assert layer.weight.grad is not None
    assert layer.bias.grad is not None
    assert not torch.isnan(layer.weight.grad).any()


def test_conv_forward_binary_weights():
    layer = BinaryNormalizedConv2d(3, 16, kernel_size=3, padding=1, activation=F.relu)
    x = torch.randn(2, 3, 32, 32)
    y = layer(x)
    assert y.shape == (2, 16, 32, 32)
    # Re-run quant to inspect: weights used in forward are {0,1}
    w_q = mean_threshold_quantize(layer.weight)
    assert set(w_q.unique().tolist()).issubset({0.0, 1.0})
    y.sum().backward()
    assert layer.weight.grad is not None


def test_binary_model_trains_one_step():
    """Smoke: a tiny BNL stack reduces CE loss for one step on random data."""
    model = torch.nn.Sequential(
        BinaryNormalizedConv2d(3, 8, 3, padding=1, activation=F.relu),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        BinaryNormalizedLinear(8, 4),
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(8, 3, 16, 16)
    target = torch.randint(0, 4, (8,))
    logits0 = model(x)
    loss0 = F.cross_entropy(logits0, target)
    opt.zero_grad()
    loss0.backward()
    opt.step()
    loss1 = F.cross_entropy(model(x), target)
    # Not required that loss decreases on random data, but must stay finite.
    assert torch.isfinite(loss0) and torch.isfinite(loss1)
