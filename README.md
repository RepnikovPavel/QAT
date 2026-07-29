# QAT — quantization-aware training for 2D object detection

Reproducing and benchmarking SOTA quantization-aware training (QAT) methods on
standard 2D detection benchmarks (PASCAL VOC, COCO). Each method lives under
`methods/<name>/` with its own code, METHOD write-up and RESULTS table. Recipes
(how to run, dataset prep) live under `recipes/`.

Scope: ultra-low-bit (≤4-bit, down to binary/ternary) QAT for detectors — the
regime where accuracy degrades hardest and where recent methods add the most.

## Layout

```
methods/
  q2/          # Q^2 (arXiv:2511.05898): gradient-balanced fusion + attention alignment
    qat/       #   implementation (quantizers, Q-GBFusion, Q-ADA, YOLOv5 wrapper)
    tests/     #   16 tensor-level unit tests
    METHOD.md  #   paper<->code mapping
    RESULTS.md #   reproduced numbers
  bitnet_det/  # BitNet-style 1.58-bit / XNOR+popcount detection (in progress)
  sota_qat/    # QuEST / GPLQ / FPQ etc., adapted to detection (in progress)
recipes/       # dataset prep + how-to-run for each method
docker/        # CUDA 12.8 / Blackwell sm_120 environment
scripts/       # docker_run.sh + server launch helpers
```

## Methods & benchmarks

| Method | Source | Bits | Benchmark | Status |
| --- | --- | --- | --- | --- |
| Q² | arXiv:2511.05898 | W4A4 / W3A3 | VOC, COCO | core reproduced; mAP runs in progress |
| BitNet-det | microsoft/BitNet | 1.58 (ternary) | VOC, COCO | porting |
| paper 2509.07025 | arXiv:2509.07025 | TBD | VOC | verifying |
| QuEST / GPLQ / FPQ | various | 1–4 bit | VOC, COCO | surveying |

## Environment (GPU server)

2× RTX 5060 Ti (Blackwell sm_120), reached via ssh (see recipes). All compute
runs in Docker (`docker/Dockerfile`, torch 2.11 + cu128). Datasets are
pre-staged at `/mnt/hdd2/datasets/{voc,coco,busi}` on the server.

## License

MIT — see [LICENSE](LICENSE).
