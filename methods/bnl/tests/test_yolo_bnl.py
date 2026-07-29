"""Tests for BNL ↔ YOLOv5s injection (skipped if yolov5 is not installed)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

yolov5 = pytest.importorskip("yolov5")

from bnl.layers import BinaryNormalizedConv2d  # noqa: E402
from bnl.yolo_bnl import (  # noqa: E402
    BNLYOLOv5,
    conv_module_to_bnl,
    count_bnl_convs,
)


def _make_fake_yolo_conv(cin=16, cout=32, k=3, s=1):
    """Minimal stand-in for yolov5.models.common.Conv."""

    class FakeConv(nn.Module):
        def __init__(self):
            super().__init__()
            p = k // 2
            self.conv = nn.Conv2d(cin, cout, k, s, p, bias=False)
            self.bn = nn.BatchNorm2d(cout)
            self.act = nn.SiLU()

        def forward(self, x):
            return self.act(self.bn(self.conv(x)))

    return FakeConv()


def test_conv_module_to_bnl_shapes_and_binary():
    src = _make_fake_yolo_conv(8, 16, 3, 1)
    bnl = conv_module_to_bnl(src)
    assert isinstance(bnl, BinaryNormalizedConv2d)
    x = torch.randn(2, 8, 16, 16)
    y = bnl(x)
    assert y.shape == (2, 16, 16, 16)
    assert torch.isfinite(y).all()
    y.sum().backward()
    assert bnl.weight.grad is not None


def test_bnl_yolov5_injects_and_keeps_stem_head_fp():
    model = BNLYOLOv5(nc=20, keep_stem_fp=True, binary_body=True)
    assert model.n_bnl > 0
    assert count_bnl_convs(model) == model.n_bnl

    stem, head = model.fp_stem_and_head()
    assert stem is not None
    assert type(stem).__name__ == "Conv"  # not BinaryNormalizedConv2d
    assert head is not None
    assert type(head).__name__ == "Detect"
    # Detect subtree must stay free of BNL
    assert count_bnl_convs(head) == 0


def test_bnl_yolov5_forward_and_loss_finite():
    model = BNLYOLOv5(nc=20, keep_stem_fp=True, binary_body=True)
    model.train()
    x = torch.randn(2, 3, 320, 320)  # smaller than 640 for CPU speed
    out = model(x)
    assert isinstance(out, (list, tuple))
    assert len(out) == 3
    for o in out:
        assert torch.isfinite(o).all()

    from yolov5.utils.loss import ComputeLoss

    compute_loss = ComputeLoss(model.det, autobalance=False)
    # empty-ish targets: one box on image 0
    tg = torch.tensor([[0.0, 0.0, 0.5, 0.5, 0.2, 0.2]])
    loss, items = compute_loss(out, tg)
    assert torch.isfinite(loss)
    loss.backward()
    # at least one BNL weight received a gradient
    grads = [m.weight.grad for m in model.bnl_modules() if m.weight.grad is not None]
    assert len(grads) > 0


def test_fp32_baseline_has_zero_bnl():
    model = BNLYOLOv5(nc=20, binary_body=False)
    assert model.n_bnl == 0
    assert count_bnl_convs(model) == 0
