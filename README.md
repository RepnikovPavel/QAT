# QAT

Working repository for **QAT** (Quantization-Aware Training) experiments.

This repo is a workspace — code, configs, scripts and notes related to
quantization-aware training and related low-precision neural network work.
The scope is intentionally open: training loops, quantizer implementations,
benchmarks, paper reproductions and ablations all belong here.

## Status

Bootstrapped; content will be added as the work progresses.

## Datasets (Q² paper, arXiv:2511.05898)

`scripts/download_datasets.sh` fetches the three datasets the paper evaluates
on, with byte-size verification and resume. Full per-dataset notes
(sources, expected sizes, on-disk layout): [docs/datasets.md](docs/datasets.md).

```sh
scripts/download_datasets.sh /mnt/hdd2/datasets          # all three
scripts/download_datasets.sh /mnt/hdd2/datasets voc      # only one
scripts/download_datasets.sh /mnt/hdd2/datasets --no-extract
```

| Dataset | Task | Size | Status on `/mnt/hdd2/datasets/` |
| --- | --- | --- | --- |
| PASCAL VOC 2007+2012 | object detection | ~5.6 GB | ✅ downloaded + extracted |
| COCO 2017 (train/val/test/ann) | object detection | ~52 GB | ✅ downloaded + extracted |
| BUSI (breast ultrasound) | segmentation | ~450 MB | ✅ downloaded + extracted |

**Infra note.** On hosts where AWS S3 throttles the source IP (200 headers,
~0 body bytes/sec — observed on cocodataset.org from our server), download on
a non-throttled workstation and `rsync -a --partial` over the LAN. BUSI and
PASCAL VOC were not affected by this throttle on the same host.

## Layout

```
QAT/
├── README.md
├── LICENSE
├── .gitignore
├── docs/
│   └── datasets.md         # dataset sources, sizes, layout, throttle note
└── scripts/
    └── download_datasets.sh  # idempotent fetcher for VOC + COCO + BUSI
```

## License

MIT — see [LICENSE](LICENSE).

## Author

Bootstrap by **ZCode agent (GLM-5.2)**.
