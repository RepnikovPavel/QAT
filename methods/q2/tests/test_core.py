"""Tensor-level correctness tests for the Q^2 core modules.

Run with:  python -m pytest tests/ -q
These are intentionally CPU-only and fast (no GPU / training needed).
"""

import math

import pytest
import torch

from qat import QGBFusion, QADALoss, LSQ, PACT, N2UQ, saliency_map
from qat.imbalance import BranchGradientProbe


# ---------------------------------------------------------- quantizers
@pytest.mark.parametrize("QuantCls", [LSQ, PACT, N2UQ])
@pytest.mark.parametrize("bw", [3, 4])
def test_quantizer_shape_and_range(QuantCls, bw):
    torch.manual_seed(0)
    q = QuantCls(bit_width=bw, signed=True)
    x = torch.randn(2, 8, 6, 6) * 3
    xq, s = q.quantize(x)
    assert xq.shape == x.shape
    assert s.shape == x.shape or s.dim() == 0
    # output must be finite
    assert torch.isfinite(xq).all()
    # gradient flows (STE)
    xq.sum().backward()


def test_lsq_step_reduces_error():
    """More bits -> closer to full precision."""
    torch.manual_seed(0)
    x = torch.randn(4, 16, 8, 8) * 2
    errs = {}
    for bw in [2, 4, 8]:
        q = LSQ(bit_width=bw, signed=True)
        # initialise step sensibly
        with torch.no_grad():
            q.step.fill_(2 * x.std() / (2 ** (bw - 1)))
            q._initialised.fill_(True)
        xq, _ = q.quantize(x)
        errs[bw] = (xq - x).abs().mean().item()
    assert errs[8] < errs[4] < errs[2]


def test_lsq_step_grad_is_scaled_and_finite():
    """Scale gradients must stay finite and not explode without g."""
    torch.manual_seed(0)
    q = LSQ(bit_width=4, signed=True)
    x = torch.randn(8, 16, 32, 32, requires_grad=True) * 2
    xq, s = q.quantize(x)
    xq.pow(2).mean().backward()
    assert q.step.grad is not None
    assert torch.isfinite(q.step.grad).all()
    # With g = 1/sqrt(Qp*N), |grad| should be small relative to unscaled.
    assert q.step.grad.abs().max().item() < 1.0


def test_lsq_step_stable_under_sgd():
    """A few SGD steps on a large activation tensor must not explode s.

    Regression for M1: the old STE scale-path produced first-layer act steps
    of O(1e3) within an epoch; with the corrected Esser gradient + g-factor
    the step should stay O(init).
    """
    torch.manual_seed(0)
    q = LSQ(bit_width=4, signed=True)
    # Init on a real tensor, then optimise a trivial reconstruction loss.
    x0 = torch.randn(4, 32, 64, 64) * 2
    with torch.no_grad():
        q.quantize(x0)  # triggers Esser init
    s0 = float(q.step.detach())
    opt = torch.optim.SGD([q.step], lr=0.01)
    for _ in range(50):
        x = torch.randn(4, 32, 64, 64) * 2
        xq, _ = q.quantize(x)
        loss = (xq - x).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            q.step.clamp_(min=1e-8)
    s1 = float(q.step.detach())
    assert math.isfinite(s1)
    # Stay within ~10x of the initialised step (not 1000x).
    assert s1 < 10.0 * s0 + 1e-3
    assert s1 > 0.0


# ---------------------------------------------------------- Q-GBFusion
def test_qgbfusion_alpha_is_simplex():
    f = QGBFusion(num_branches=3, num_channels=8)
    a = f.alpha
    assert a.shape == (3,)
    assert torch.allclose(a.sum(), torch.tensor(1.0), atol=1e-6)
    assert (a >= 0).all()


def test_qgbfusion_forward_shape():
    f = QGBFusion(num_branches=2, num_channels=8)  # total fused channels = 4+4
    branches = [torch.randn(2, 4, 8, 8, requires_grad=True) for _ in range(2)]
    out = f(branches)
    # concat of two 4-channel branches -> 8 channels
    assert out.shape == (2, 8, 8, 8)


def test_qgbfusion_reduces_imbalance():
    """Closed-loop update should pull branch energies towards equal."""
    torch.manual_seed(0)
    f = QGBFusion(num_branches=2, num_channels=8, eta=0.1, beta=0.5)
    f.train()
    f.Gbar.fill_(-1.0)

    # simulate strongly imbalanced branches: branch 1 has ~10x the energy
    for _ in range(30):
        b0 = torch.randn(2, 4, 4, 4, requires_grad=True)
        b1 = torch.randn(2, 4, 4, 4, requires_grad=True) * 10  # dominant
        out = f([b0, b1])
        out.sum().backward()
        G = f.update_dual()
        assert G is not None

    a = f.alpha
    # the under-dominant branch (0) should get a LARGER allocation to boost it
    assert a[0] > a[1]


# ---------------------------------------------------------- Q-ADA
def test_saliency_shape():
    x = torch.randn(2, 8, 10, 10)
    xb = x + torch.randn_like(x) * 0.1
    a = saliency_map(x, xb, step=torch.tensor(0.1))
    assert a.shape == x.shape
    assert (a >= 0).all() and (a <= 1).all()


def test_qada_zero_when_identical():
    """Teacher == student feature -> near-zero divergence."""
    x = torch.randn(2, 8, 12, 12)
    loss = QADALoss()(x, x.clone(), step=torch.tensor(0.5))
    assert loss.item() < 1e-4


def test_qada_positive_when_distorted():
    # Feature with a clear saliency peak (a bright region) so the attention
    # distribution is *non-uniform* spatially; distortion should then move the
    # student distribution away from the teacher -> measurable JS divergence.
    x = torch.randn(2, 8, 12, 12) * 0.1
    x[:, :, 4:8, 4:8] += 5.0  # salient block
    xb = x.clone()
    xb[:, :, 4:8, 4:8] *= 0.3  # quantization suppresses the salient region
    xb += torch.randn_like(xb) * 0.05
    loss = QADALoss()(x, xb, step=torch.tensor(0.5))
    assert loss.item() > 0.01


def test_qada_monotone_in_distortion():
    """Stronger distortion -> larger Q-ADA loss."""
    x = torch.randn(2, 8, 12, 12) * 0.1
    x[:, :, 4:8, 4:8] += 5.0
    losses = []
    for scale in [0.0, 0.5, 2.0]:
        xb = x.clone()
        xb[:, :, 4:8, 4:8] *= (1.0 - 0.3 * scale)
        xb += torch.randn_like(xb) * (0.05 + scale)
        losses.append(QADALoss()(x, xb, step=torch.tensor(0.5)).item())
    assert losses[0] < losses[1] < losses[2]


def test_qada_js_vs_kl():
    x = torch.randn(2, 8, 12, 12)
    xb = x + torch.randn_like(x)
    l_js = QADALoss(divergence="js")(x, xb)
    l_kl = QADALoss(divergence="kl")(x, xb)
    assert l_js.item() > 0 and l_kl.item() > 0


# ---------------------------------------------------------- imbalance probe
def test_probe_records_imbalance():
    probe = BranchGradientProbe(num_branches=2)
    b0 = torch.randn(4, requires_grad=True)
    b1 = (torch.randn(4) * 5).requires_grad_(True)
    b0.retain_grad()
    b1.retain_grad()
    b0.register_hook(probe.hook(0))
    b1.register_hook(probe.hook(1))
    # asymmetric loss so gradient magnitudes differ across branches
    (b0.sum() * 0.1 + b1.sum() * 2.0).backward()
    row = probe.log(step=0)
    assert row["ratio"] > 1.0
    assert row["G_1"] > row["G_0"]
