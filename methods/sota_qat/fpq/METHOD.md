# FPQ — Feature-Perturbed Quantization

Paper: *Stabilizing Quantization-Aware Training by Implicit-Regularization on
Hessian Matrix* (arXiv:2503.11159). Parsed text: `papers/2503.11159/`.

## Idea

QAT instability comes from a sharp loss landscape under quantization noise.
FPQ smooths that landscape by:

1. **SFP** — stochastic uniform noise on layer inputs, scaled to the activation
   quant step `s` (Eq. 9, 14).
2. **CSD** — channel-wise standardize student vs FP teacher features and
   penalise their L2 gap (Eq. 16–17).

Task loss (CE on CIFAR; detector multi-task loss on VOC) + `L_CSD` (Eq. 22).

## Repo mapping

| Paper | Code |
| --- | --- |
| Eq. 9 uniform noise | `uniform_feature_noise` |
| Eq. 14 Bernoulli gate | `stochastic_feature_perturb` / `StochasticFeaturePerturb` |
| Eq. 16–17 CSD | `channel_standardize` + `csd_loss` |
| Eq. 22 regulariser | `fpq_regularizer` |

## Status

Core ops + unit tests. VOC W4A4 LSQ+FPQ vs LSQ-only benchmark is the next
measurement step (GPU was busy at scaffold time).
