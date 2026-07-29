"""Unit tests for FPQ (arXiv:2503.11159) core ops — no GPU required."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

# allow `python -m pytest methods/sota_qat/fpq/tests` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpq import (  # noqa: E402
    StochasticFeaturePerturb,
    channel_standardize,
    csd_loss,
    fpq_regularizer,
    stochastic_feature_perturb,
    uniform_feature_noise,
)


def test_uniform_noise_bounds():
    x = torch.zeros(2, 3, 4, 4)
    step = 0.5
    delta = uniform_feature_noise(x, step)
    assert delta.shape == x.shape
    # U[-s/2, s/2] = U[-0.25, 0.25]
    assert delta.min() >= -0.25 - 1e-6
    assert delta.max() <= 0.25 + 1e-6


def test_sfp_identity_when_p0():
    x = torch.randn(1, 2, 3, 3)
    y = stochastic_feature_perturb(x, step=1.0, p=0.0, training=True)
    assert torch.equal(y, x)


def test_sfp_always_when_p1():
    torch.manual_seed(0)
    x = torch.zeros(4, 8, 8, 8)
    y = stochastic_feature_perturb(x, step=2.0, p=1.0, training=True)
    # must differ (noise almost surely non-zero on this many elements)
    assert not torch.equal(y, x)
    assert (y - x).abs().max() <= 1.0 + 1e-5  # half-width = 1.0


def test_sfp_eval_no_noise():
    m = StochasticFeaturePerturb(p=1.0, step=1.0)
    m.eval()
    x = torch.randn(1, 1, 2, 2)
    assert torch.equal(m(x), x)


def test_channel_standardize_stats():
    torch.manual_seed(1)
    x = torch.randn(4, 8, 16, 16) * 3 + 2
    z = channel_standardize(x)
    # mean ≈ 0, std ≈ 1 per channel over batch+spatial
    mean = z.mean(dim=(0, 2, 3))
    std = z.std(dim=(0, 2, 3), unbiased=False)
    assert mean.abs().max() < 1e-4
    assert (std - 1.0).abs().max() < 1e-3


def test_csd_zero_when_identical():
    f = torch.randn(2, 4, 8, 8)
    loss = csd_loss([f], [f.clone()])
    assert loss.item() < 1e-8


def test_csd_positive_when_different():
    a = torch.randn(2, 4, 8, 8)
    b = a + 1.0
    loss = csd_loss([a], [b])
    assert loss.item() > 0.0


def test_fpq_regularizer_weight():
    a = torch.randn(2, 3, 4, 4)
    b = torch.randn(2, 3, 4, 4)
    l1 = fpq_regularizer([a], [b], csd_weight=1.0)
    l0 = fpq_regularizer([a], [b], csd_weight=0.0)
    assert l0.item() == 0.0
    assert l1.item() > 0.0


def test_hook_collects_features():
    from fpq import FeatureHook

    net = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU())
    hook = FeatureHook([net[0]])
    x = torch.randn(1, 3, 16, 16)
    _ = net(x)
    assert len(hook.feats) == 1
    assert hook.feats[0].shape == (1, 8, 16, 16)
    hook.close()


class _FakeActQuant(nn.Module):
    """Minimal LSQ-like act quant for SFP wrapper tests."""

    def __init__(self, step: float = 0.25) -> None:
        super().__init__()
        self.step = nn.Parameter(torch.tensor(step))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.step.abs()
        return (x / s).round().clamp(-8, 7) * s


class _FakeQuantConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w_quant = nn.Identity()
        self.a_quant = _FakeActQuant(0.5)
        self.conv = nn.Conv2d(3, 4, 1)

    def forward(self, x):
        x = self.a_quant(x)
        return self.conv(x)


def test_sfp_act_wrapper_eval_identity_path():
    from sfp_inject import SFPActWrapper

    aq = _FakeActQuant(1.0)
    wrap = SFPActWrapper(aq, p=1.0)
    wrap.eval()
    x = torch.randn(2, 3, 4, 4)
    # eval: no SFP; output equals bare quant
    y_w = wrap(x)
    y_a = aq(x)
    assert torch.equal(y_w, y_a)


def test_sfp_act_wrapper_train_perturbs():
    from sfp_inject import SFPActWrapper

    class SpyQuant(nn.Module):
        """Records pre-quant input; noise is U[-s/2,s/2] so may not flip bins."""

        def __init__(self) -> None:
            super().__init__()
            self.step = nn.Parameter(torch.tensor(1.0))
            self.last_x = None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.last_x = x.detach().clone()
            return x

    torch.manual_seed(0)
    spy = SpyQuant()
    wrap = SFPActWrapper(spy, p=1.0)
    wrap.train()
    x = torch.zeros(2, 4, 8, 8)
    _ = wrap(x)
    assert spy.last_x is not None
    # SFP must have added noise before quant (Eq. 9/14)
    assert not torch.equal(spy.last_x, x)
    assert (spy.last_x - x).abs().max() <= 0.5 + 1e-5


def test_enable_sfp_wraps_and_disable():
    from sfp_inject import SFPActWrapper, disable_sfp, enable_sfp

    net = nn.Sequential(_FakeQuantConv(), _FakeQuantConv())
    n = enable_sfp(net, p=0.2)
    assert n == 2
    for m in net:
        assert isinstance(m.a_quant, SFPActWrapper)
        assert m.a_quant.p == 0.2
    n2 = enable_sfp(net, p=0.3)  # idempotent update
    assert n2 == 2
    assert net[0].a_quant.p == 0.3
    nu = disable_sfp(net)
    assert nu == 2
    assert not isinstance(net[0].a_quant, SFPActWrapper)
