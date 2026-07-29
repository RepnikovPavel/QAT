"""Unit tests for BitNet b1.58 BitLinear / BitConv2d (FAQ Figure 3)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bitlinear import (  # noqa: E402
    BitConv2d,
    BitLinear,
    activation_quant,
    rms_norm,
    ste_quantize,
    weight_quant,
    weight_ternary_codes,
)


def test_weight_ternary_codes_in_pm1_0():
    w = torch.randn(32, 64) * 0.5
    codes = weight_ternary_codes(w)
    uniq = set(codes.unique().tolist())
    assert uniq.issubset({-1.0, 0.0, 1.0})
    assert codes.abs().max() <= 1.0


def test_weight_quant_is_scaled_ternary():
    w = torch.randn(16, 32)
    u = weight_quant(w)
    scale = w.abs().mean().clamp(min=1e-5)
    codes = weight_ternary_codes(w)
    # u = codes / (1/scale) = codes * scale  (FAQ: / scale after round)
    assert torch.allclose(u, codes * scale, atol=1e-5)


def test_activation_quant_range_and_ste():
    x = torch.randn(4, 64) * 3
    y = activation_quant(x)
    # dequantized: finite, same shape
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    # STE path
    x2 = x.clone().requires_grad_(True)
    q = ste_quantize(x2, activation_quant(x2))
    q.sum().backward()
    assert x2.grad is not None
    assert torch.allclose(x2.grad, torch.ones_like(x2))


def test_rms_norm_unit_rms():
    x = torch.randn(3, 128) * 5
    y = rms_norm(x)
    rms = y.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)


def test_bitlinear_shape_and_ternary_property():
    layer = BitLinear(64, 32, bias=False)
    x = torch.randn(8, 64, requires_grad=True)
    y = layer(x)
    assert y.shape == (8, 32)
    y.sum().backward()
    assert layer.weight.grad is not None
    assert not torch.isnan(layer.weight.grad).any()
    codes = layer.ternary_weight()
    assert set(codes.unique().tolist()).issubset({-1.0, 0.0, 1.0})


def test_bitlinear_matches_manual_faq_forward():
    """Forward equals hand-rolled FAQ Figure 3 path."""
    torch.manual_seed(0)
    layer = BitLinear(16, 8, bias=False)
    x = torch.randn(4, 16)
    y = layer(x)

    w = layer.weight
    x_norm = rms_norm(x)
    x_q = x_norm + (activation_quant(x_norm) - x_norm).detach()
    w_q = w + (weight_quant(w) - w).detach()
    y_ref = F.linear(x_q, w_q)
    assert torch.allclose(y, y_ref, atol=1e-6)


def test_bitconv2d_shape_grad_ternary():
    layer = BitConv2d(3, 16, kernel_size=3, padding=1, bias=False, activation=None)
    x = torch.randn(2, 3, 32, 32, requires_grad=True)
    y = layer(x)
    assert y.shape == (2, 16, 32, 32)
    y.sum().backward()
    assert layer.weight.grad is not None
    assert not torch.isnan(layer.weight.grad).any()
    codes = layer.ternary_weight()
    assert set(codes.unique().tolist()).issubset({-1.0, 0.0, 1.0})


def test_bitconv2d_with_silu():
    act = torch.nn.SiLU()
    layer = BitConv2d(8, 16, kernel_size=1, activation=act)
    x = torch.randn(1, 8, 16, 16)
    y = layer(x)
    assert y.shape == (1, 16, 16, 16)
    assert torch.isfinite(y).all()
