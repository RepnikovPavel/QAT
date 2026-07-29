"""Unit tests for XNOR-popcount convolution reference vs fake-quant."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xnor_conv import (  # noqa: E402
    XNORConv2d,
    binary_sign,
    conv2d_xnor_popcount_ref,
    float_pm1_dot,
    ste_binary_sign,
    xnor_dot_popcount,
)


def test_xnor_dot_equals_float_pm1():
    torch.manual_seed(0)
    a = binary_sign(torch.randn(32, 64))
    w = binary_sign(torch.randn(32, 64))
    a_bits = a > 0
    w_bits = w > 0
    pc = xnor_dot_popcount(a_bits, w_bits).float()
    ref = float_pm1_dot(a, w)
    assert torch.equal(pc, ref)


def test_conv_popcount_matches_float_conv():
    torch.manual_seed(1)
    x = binary_sign(torch.randn(2, 4, 8, 8))
    w = binary_sign(torch.randn(6, 4, 3, 3))
    y_pc = conv2d_xnor_popcount_ref(x, w, stride=(1, 1), padding=1)
    y_f = F.conv2d(x, w, stride=1, padding=1)
    assert y_pc.shape == y_f.shape
    assert torch.allclose(y_pc, y_f, atol=1e-5)


def test_ste_binary_sign_grad_identity():
    x = torch.randn(8, 4, requires_grad=True)
    y = ste_binary_sign(x)
    assert set(y.detach().unique().tolist()).issubset({-1.0, 1.0})
    y.sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x))


def test_xnor_conv2d_forward_grad():
    layer = XNORConv2d(3, 8, kernel_size=3, padding=1, bias=False)
    x = torch.randn(2, 3, 16, 16, requires_grad=True)
    y = layer(x)
    assert y.shape == (2, 8, 16, 16)
    y.sum().backward()
    assert layer.weight.grad is not None
    assert not torch.isnan(layer.weight.grad).any()
    b, alpha = layer.binary_weight_codes()
    assert set(b.unique().tolist()).issubset({-1.0, 1.0})
    assert (alpha > 0).all()


def test_xnor_conv_unscaled_matches_popcount():
    """conv(Ba, Bw) bit-exact with popcount ref (before α scale)."""
    torch.manual_seed(2)
    layer = XNORConv2d(4, 5, kernel_size=3, padding=1, bias=False, per_channel_alpha=True)
    x = torch.randn(1, 4, 7, 7)
    with torch.no_grad():
        ba = binary_sign(x)
        bw, _ = layer.binary_weight_codes()
        y_f = F.conv2d(ba, bw, padding=1)
        y_pc = conv2d_xnor_popcount_ref(ba, bw, padding=1)
    assert torch.allclose(y_f, y_pc, atol=1e-5)
