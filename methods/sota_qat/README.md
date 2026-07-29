# SOTA QAT for 2D object detection

Track and reproduce state-of-the-art quantization-aware training methods,
adapted to 2D object detection benchmarks (VOC, COCO).

## Candidates under evaluation

| Method | arXiv | Key idea | Status |
| --- | --- | --- | --- |
| Q² | 2511.05898 | gradient balancing fusion + attention alignment | reproduced (see ../q2) |
| BNL | 2509.07025 | binary {0,1} weights + post-linear normalize | adapt to detection (see ../bnl) |
| QuEST | Hadamard norm + MSE-opt fit + trust-gradient for 1-bit | LLM-focused | under review |
| GPLQ | activation-first weights-later + feature-mimicking loss | ViT-focused | under review |
| FPQ | stochastic feature perturbation + feature distillation | regularizes Hessian | under review |

Method code + RESULTS.md land here as each is reproduced.
