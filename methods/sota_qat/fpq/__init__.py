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

# sfp_inject imports `from fpq import ...` when used as a script package root;
# re-export inject helpers only when available as a package submodule.
try:
    from .sfp_inject import (  # noqa: F401
        DEFAULT_CSD_LAYERS,
        SFPActWrapper,
        attach_csd_hooks,
        disable_sfp,
        enable_sfp,
        freeze_teacher,
    )
except ImportError:
    pass

__all__ = [
    "FeatureHook",
    "StochasticFeaturePerturb",
    "channel_standardize",
    "csd_loss",
    "fpq_regularizer",
    "stochastic_feature_perturb",
    "uniform_feature_noise",
    "DEFAULT_CSD_LAYERS",
    "SFPActWrapper",
    "attach_csd_hooks",
    "disable_sfp",
    "enable_sfp",
    "freeze_teacher",
]
