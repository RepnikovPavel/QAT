"""Unit tests for SFP/CSD injection helpers (no GPU required)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sfp_inject import (  # noqa: E402
    SFPActWrapper,
    disable_sfp,
    enable_sfp,
)


class _FakeLSQ(nn.Module):
    def __init__(self, step: float = 0.25):
        super().__init__()
        self.step = nn.Parameter(torch.tensor(step))

    def forward(self, x):
        return x  # identity for inject tests


class _FakeQConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.a_quant = _FakeLSQ(0.5)
        self.w_quant = _FakeLSQ(0.1)

    def forward(self, x):
        x = self.a_quant(x)
        return x


def test_enable_sfp_wraps_once():
    net = nn.Sequential(_FakeQConv(), _FakeQConv())
    n1 = enable_sfp(net, p=0.3)
    assert n1 == 2
    assert isinstance(net[0].a_quant, SFPActWrapper)
    assert net[0].a_quant.p == 0.3
    # idempotent: re-enable only updates p
    n2 = enable_sfp(net, p=0.1)
    assert n2 == 2
    assert isinstance(net[0].a_quant.a_quant, _FakeLSQ)
    assert net[0].a_quant.p == 0.1


def test_disable_sfp_unwraps():
    net = nn.Sequential(_FakeQConv())
    enable_sfp(net, p=0.5)
    n = disable_sfp(net)
    assert n == 1
    assert isinstance(net[0].a_quant, _FakeLSQ)


def test_sfp_wrapper_uses_step_and_p1():
    torch.manual_seed(0)
    aq = _FakeLSQ(2.0)
    wrap = SFPActWrapper(aq, p=1.0)
    wrap.train()
    x = torch.zeros(2, 4, 8, 8)
    y = wrap(x)
    # noise half-width = 1.0; must differ almost surely
    assert not torch.equal(y, x)
    assert (y - x).abs().max() <= 1.0 + 1e-5


def test_sfp_wrapper_eval_identity():
    aq = _FakeLSQ(1.0)
    wrap = SFPActWrapper(aq, p=1.0)
    wrap.eval()
    x = torch.randn(1, 2, 3, 3)
    assert torch.equal(wrap(x), x)


def test_enable_sfp_skips_plain_modules():
    net = nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU())
    assert enable_sfp(net, p=0.5) == 0
