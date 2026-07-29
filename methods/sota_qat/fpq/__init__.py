"""FPQ (arXiv:2503.11159) — feature perturbation + CSD for stable QAT."""

from .fpq import (
    FeatureHook,
    StochasticFeaturePerturb,
    channel_standardize,
    csd_loss,
    fpq_regularizer,
    stochastic_feature_perturb,
    uniform_feature_noise,
)

__all__ = [
    "FeatureHook",
    "StochasticFeaturePerturb",
    "channel_standardize",
    "csd_loss",
    "fpq_regularizer",
    "stochastic_feature_perturb",
    "uniform_feature_noise",
]
