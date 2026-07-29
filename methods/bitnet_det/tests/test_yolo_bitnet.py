"""Tests for BitNetYOLOv5 injection (requires yolov5 package)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("yolov5")

from bitlinear import BitConv2d  # noqa: E402
from xnor_conv import XNORConv2d  # noqa: E402
from yolo_bitnet import BitNetYOLOv5, count_bit_convs  # noqa: E402


def test_bitnet_yolo_inject_and_forward():
    model = BitNetYOLOv5(nc=20, keep_stem_fp=True, binary_body=True, mode="bitnet")
    assert model.n_bit > 0
    assert count_bit_convs(model) == model.n_bit
    # stem is still official Conv
    assert type(model.model[0]).__name__ == "Conv"
    # at least one BitConv2d in body
    assert any(isinstance(m, BitConv2d) for m in model.modules())
    # Detect present and not replaced
    assert any(type(m).__name__ == "Detect" for m in model.model)

    x = torch.randn(1, 3, 320, 320)
    model.eval()
    with torch.no_grad():
        out = model(x)
    assert out is not None
    # training forward for loss path returns multi-scale list
    model.train()
    out_t = model(x)
    assert isinstance(out_t, (list, tuple))
    assert all(torch.isfinite(o).all() for o in out_t)


def test_xnor_mode_inject():
    model = BitNetYOLOv5(nc=20, mode="xnor", binary_body=True)
    assert model.n_bit > 0
    assert any(isinstance(m, XNORConv2d) for m in model.modules())
    assert not any(isinstance(m, BitConv2d) for m in model.modules())


def test_fp32_baseline_no_inject():
    model = BitNetYOLOv5(nc=20, binary_body=False)
    assert model.n_bit == 0
    assert count_bit_convs(model) == 0
