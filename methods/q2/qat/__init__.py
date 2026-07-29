"""Q^2 reference implementation — Quantization-aware Gradient Balancing Fusion
(Q-GBFusion) and Attention Distribution Alignment (Q-ADA).

Paper: "Q2: Quantization-Aware Gradient Balancing and Attention Alignment for
Low-Bit Quantization", arXiv:2511.05898 (2026).
"""

from .qgbfusion import QGBFusion
from .qada import QADALoss, saliency_map, js_divergence, kl_divergence
from .imbalance import BranchGradientProbe
from .quantizers.lsq import LSQ
from .quantizers.pact import PACT
from .quantizers.n2uq import N2UQ

__all__ = [
    "QGBFusion",
    "QADALoss",
    "saliency_map",
    "js_divergence",
    "kl_divergence",
    "BranchGradientProbe",
    "LSQ",
    "PACT",
    "N2UQ",
]
