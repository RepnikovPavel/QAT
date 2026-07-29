# SOTA QAT for 2D object detection

Track and reproduce state-of-the-art quantization-aware training methods,
adapted to 2D object detection benchmarks (VOC, COCO).

## Candidates under evaluation

| Method | arXiv | Key idea | Status |
| --- | --- | --- | --- |
| Q² | 2511.05898 | gradient balancing fusion + attention alignment | reproduced (see ../q2) |
| BNL | 2509.07025 | binary {0,1} weights + post-linear normalize | adapt to detection (see ../bnl) |
| FPQ | 2503.11159 | SFP + CSD (Hessian-smoothing QAT regularizer) | adapt — core ops in `fpq/`; VOC bench pending |
| GPLQ | 2506.11784 | activation-first QAT then weight PTQ (ViT; COCO AP reported) | adapt (Act-QAT idea) |
| QATMA | 2603.05964 | curriculum module QAT + text-anchored pairwise KD (OVOD) | adapt (CQAT schedule) |
| QuEST | 2402.03666 | selective finetune for low-bit diffusion | skip (not detection) |

See `SURVEY.md` for grounded triage. Method code + RESULTS.md land here as each is reproduced.
