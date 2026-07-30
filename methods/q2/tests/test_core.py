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
        xq, _ = q.quantize(x)
        errs[bw] = (xq - x).abs().mean().item()
    assert errs[8] < errs[4] < errs[2]


def test_lsq_forward_matches_canonical_formula():
    """LSQ forward must equal s * clip(round(x/s), Qn, Qp) exactly (match test).

    Guards against any drift between the autograd.Function forward and the
    paper's definition (Esser 2020, Eq. 1) — bit-exact on a fixed seed.
    """
    torch.manual_seed(0)
    q = LSQ(bit_width=4, signed=True)
    x = torch.randn(2, 8, 6, 6) * 3
    with torch.no_grad():
        q.step.fill_(0.1)          # fixed positive step
        q._initialised.fill_(True)
    xq, s = q.quantize(x)
    # canonical reference computed directly from the formula
    v = x / s.clamp(min=1e-12)
    ref = v.round().clamp(q.Qn, q.Qp) * s
    assert torch.allclose(xq, ref, atol=1e-6)


def test_lsq_step_grad_matches_esser_formula():
    """The step-size gradient must be the Esser d xq/d s = q - x/s (interior).

    Check it numerically against autograd at an interior point (x/s strictly
    inside [Qn, Qp]), where the analytic formula is exact (no saturation).
    """
    torch.manual_seed(0)
    q = LSQ(bit_width=4, signed=True)   # Qn=-8, Qp=7
    # craft x so every element is interior at the chosen step
    s0 = 0.05
    x = (torch.rand(100) * 12 - 6) * s0   # x/s in [-6, 6] ⊂ (-8, 7)
    with torch.no_grad():
        q.step.fill_(s0)
        q._initialised.fill_(True)
    xq, s = q.quantize(x)
    xq.sum().backward()
    # analytic: d xq/d s summed over all elements = sum(q - x/s), times g
    import math
    g = 1.0 / math.sqrt(q.Qp * x.numel())
    v = x / s0
    qv = v.round().clamp(q.Qn, q.Qp)
    expected = (qv - v).sum() * g
    assert torch.allclose(q.step.grad.sum(), expected, atol=1e-5)


def test_lsq_step_grad_is_scaled_and_finite():
    """Without grad_factor the step gradient would be ~sqrt(N) too large.

    The Esser scale g = 1/sqrt(Qp*N) must keep the step gradient within an order
    of magnitude of the per-element input gradient (regression for the LSQ
    rewrite; the pre-fix formula routed only g*q to s and was ~1000x off).
    """
    torch.manual_seed(0)
    q = LSQ(bit_width=4, signed=True)
    x = torch.randn(4, 16, 8, 8)
    xq, _ = q.quantize(x)
    xq.sum().backward()
    gstep = q.step.grad.abs().item()
    assert math.isfinite(gstep)
    # a single-element loss sum gives a step grad of order g * N ~ sqrt(N/Qp);
    # a correctly-scaled grad stays modest (the broken form exploded >1e4).
    assert gstep < 100.0


def test_lsq_step_stable_under_sgd():
    """A few SGD steps must NOT explode the LSQ step (issue #4 root cause).

    With weight_decay=0 on the step param-group + positivity clamp, the step
    stays within 10x of its init. The pre-fix path could blow it up ~1000x.
    """
    torch.manual_seed(0)
    q = LSQ(bit_width=4, signed=True)
    x = torch.randn(4, 16, 8, 8)
    q.quantize(x)                      # trigger Esser init
    s0 = float(q.step.detach().abs())
    opt = torch.optim.SGD([q.step], lr=0.01, momentum=0.9)
    for _ in range(20):
        opt.zero_grad()
        xq, _ = q.quantize(x)
        xq.sum().backward()
        opt.step()
        with torch.no_grad():
            q.step.clamp_(min=1e-8)    # project_quant_steps
    s1 = float(q.step.detach().abs())
    assert s1 < 10.0 * s0


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
    # student distribution away from the teacher -> a positive JS divergence.
    #
    # NOTE: with the paper's exact Eq.16 form (A~ = Sigmoid(S), then L1-normalise)
    # a strongly salient region saturates Sigmoid -> the teacher/student
    # distributions both concentrate on the same block and the JS is small but
    # strictly positive. We assert positivity (>0) and rely on
    # test_qada_monotone_in_distortion for the "more distortion -> more loss"
    # ordering. The previous >0.01 threshold matched the softmax(S) form, which
    # we changed to follow the paper verbatim (see qada.py / SPEC_AUDIT.md #5).
    torch.manual_seed(0)
    x = torch.randn(2, 8, 12, 12) * 0.1
    x[:, :, 4:8, 4:8] += 5.0  # salient block
    xb = x.clone()
    xb[:, :, 4:8, 4:8] *= 0.3  # quantization suppresses the salient region
    xb += torch.randn_like(xb) * 0.05
    loss = QADALoss()(x, xb, step=torch.tensor(0.5))
    assert loss.item() > 0.0


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
