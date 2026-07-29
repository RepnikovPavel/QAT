# QAT — Q² reproduction

Reproduction of **Q²: Quantization-Aware Gradient Balancing and Attention
Alignment for Low-Bit Quantization** (arXiv:2511.05898), built from scratch
since the paper released **no official code**.

Q² is a training-time-only, plug-and-play add-on for quantization-aware
training (QAT) of detectors/segmenters. Two components:

- **Q-GBFusion** — closed-loop gradient balancing at feature-fusion (concat)
  nodes, addressing the branch-wise gradient imbalance that low-bit
  quantization induces at multi-scale fusion stages.
- **Q-ADA** — parameter-free attention-distribution distillation that
  emphasises quantization-sensitive regions to accelerate QAT convergence.

Paper headline results (Table 1, PASCAL VOC, mAP@0.5):

| Model | Quant | BW | Baseline | +Q² | Gain |
| --- | --- | --- | --- | --- | --- |
| YOLOv5s | N2UQ | W4A4 | 82.1 | 84.2 | **+2.1** |
| YOLOv5s | LSQ  | W4A4 | 76.9 | 78.9 | **+2.0** |
| YOLOv5s | PACT | W4A4 | 79.1 | 80.6 | +1.5 |

## What's implemented

```
qat/
├── quantizers/          # LSQ, PACT, N2UQ — cross-checked vs the original papers
│   ├── base.py          #   (arXiv:1902.08153 / 1805.06085 / 2111.14826)
│   ├── lsq.py  pact.py  n2uq.py
├── qgbfusion.py         # Q-GBFusion: Eq.5-10 (softmax allocation, LN, closed-loop dual update)
├── qada.py              # Q-ADA: Eq.13-16 (saliency stats, JS/KL distillation)
├── imbalance.py         # per-branch gradient-energy probe (Fig.1b diagnostic)
├── models/yolov5.py     # QYOLOv5 — wraps the OFFICIAL ultralytics DetectionModel,
│                        #   injects fake-quant + QGBFusion at the 4 PANet concats
├── data/voc.py          # VOC→YOLO converter + loader
├── train_detect.py      # training loop (official ComputeLoss, Appendix 8.1 hyperparams)
├── eval_detect.py       # mAP@0.5 (official non_max_suppression + ap_per_class)
├── pretrained.py        # load yolov5s COCO-pretrained body weights
└── probe_imbalance.py   # M0: reproduce Fig.1b gradient imbalance
tests/test_core.py       # 16 tensor-level unit tests (all pass)
docker/Dockerfile        # CUDA 12.8 / Blackwell sm_120 env
scripts/docker_run.sh    # build + run on the GPU server
docs/method.md           # paper<->code mapping
docs/repro.md            # how to run
```

**Anti-hallucination policy:** the model architecture, the detection loss,
NMS and mAP all come from the `yolov5==7.0.14` package — nothing is
re-implemented where a reference exists. The three baseline quantizers were
each cross-checked against the original paper, parsed via **ocrc** from
arXiv (PDFs cached under `qat_pdf/refs/`).

## Status

| Component | State |
| --- | --- |
| Core library (Q-GBFusion, Q-ADA, quantizers) | ✅ done, 16/16 tests pass |
| YOLOv5 detection pipeline | ✅ done, forward/backward verified |
| Docker env (cu128, Blackwell) | ⏳ building |
| M0 imbalance probe | ✅ code, pending GPU run |
| M1 (YOLOv5s LSQ W4A4) | pending |
| M2 (YOLOv5s N2UQ W4A4) | pending |

Reproduction results land in [docs/results.md](docs/results.md) as runs complete.

## Datasets (already on the GPU server)

`scripts/download_datasets.sh` fetches VOC + COCO + BUSI. On the server they
live at `/mnt/hdd2/datasets/{voc,coco,busi}`.

## Quick start (on the GPU server)

```sh
scripts/docker_run.sh python -m pytest tests/ -q                       # unit tests
scripts/docker_run.sh python -m qat.probe_imbalance --quant lsq         # M0 (Fig.1b)
scripts/docker_run.sh python -m qat.train_detect --quant lsq --qgb --qada   # M1
```

Full instructions: [docs/repro.md](docs/repro.md). Method→code mapping:
[docs/method.md](docs/method.md).

## License

MIT — see [LICENSE](LICENSE).

## Author

Reproduction by **ZCode agent (GLM-5.2)**.
