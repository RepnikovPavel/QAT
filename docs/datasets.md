# Datasets for the Q² paper (arXiv:2511.05898)

The Q² paper "Quantization-Aware Gradient Balancing and Attention Alignment for
Low-Bit Quantization" evaluates on **three datasets** across two tasks. This
file lists them, their canonical sources, on-disk sizes and how to fetch them
via `scripts/download_datasets.sh`.

Source: §4 of the paper and bibliography entries [1], [9], [17].

| Dataset | Task | Model(s) | Metric | Approx. size |
| --- | --- | --- | --- | --- |
| **PASCAL VOC** 2007 + 2012 [9] | object detection | YOLOv5, YOLOv11, RT-DETR | mAP | ~3 GB |
| **COCO 2017** [17] | object detection | YOLOv5 (appendix 10.1) | mAP₅₀, mAP₅₀₋₉₅ | ~33 GB |
| **BUSI** [1] | medical image segmentation | MK-UNet | mDICE | ~200 MB |

## Quick start

```sh
# default: all three, into the given root, with extraction
scripts/download_datasets.sh /mnt/hdd2/datasets

# only one dataset
scripts/download_datasets.sh /mnt/hdd2/datasets voc

# keep the archives, don't extract
scripts/download_datasets.sh /mnt/hdd2/datasets --no-extract

# or via env var
QAT_DATASETS=/mnt/hdd2/datasets scripts/download_datasets.sh
```

The script is idempotent: re-running resumes partial downloads (wget `-c`)
and skips files that already match the expected byte size. Expected sizes are
hardcoded from the upstream mirrors as of July 2026.

## Sources

### PASCAL VOC — `voc/`

- **VOC 2012 trainval** (1999639040 bytes)
  `https://thor.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar`
  (the original `host.robots.ox.ac.uk` host 301-redirects here)
- **VOC 2007 trainval** (4600320000 bytes)
  `https://data.pjreddie.com/files/VOCtrainval_06-Nov-2007.tar`
- **VOC 2007 test** (4510208000 bytes)
  `https://data.pjreddie.com/files/VOCtest_06-Nov-2007.tar`

YOLO experiments typically train on VOC2007 trainval + VOC2012 trainval and
evaluate on VOC2007 test (the "07+12" split). The script downloads all three
so any combination is reproducible.

### COCO 2017 — `coco/`

| File | Bytes | URL |
| --- | --- | --- |
| `train2017.zip` | 19336861798 (~18 GB) | `http://images.cocodataset.org/zips/train2017.zip` |
| `val2017.zip` | 815585330 | `http://images.cocodataset.org/zips/val2017.zip` |
| `test2017.zip` | 6646970404 (~6.4 GB) | `http://images.cocodataset.org/zips/test2017.zip` |
| `annotations_trainval2017.zip` | 252907541 | `http://images.cocodataset.org/annotations/annotations_trainval2017.zip` |

Mirror: the official `images.cocodataset.org` CDN. No auth required.

### BUSI — `busi/`

- **Dataset of breast ultrasound images** (15716425 bytes)
  `https://ars.els-cdn.com/content/image/1-s2.0-S2352340919312181-mmc1.zip`

This is the supplementary archive attached to the Data in Brief article
[Al-Dhabyani et al., 2020] (DOI 10.1016/j.dib.2019.104863), which is exactly
reference [1] in the paper. It is the same content as the Kaggle dataset
`aryashah2k/breast-ultrasound-images-dataset`, but served from Elsevier's CDN
with **no Kaggle token required** — important because no Kaggle credentials
ship with this repo.

The archive extracts to a `Dataset_BUSI/` tree with three classes:
`benign/`, `malignant/`, `normal/`, ~780 images total.

## On-disk layout after a full run

```
<mnt>/
├── voc/
│   ├── VOCtrainval_11-May-2012.tar
│   ├── VOCtrainval_06-Nov-2007.tar
│   ├── VOCtest_06-Nov-2007.tar
│   ├── VOCdevkit/VOC2012/        (extracted)
│   ├── VOCdevkit/VOC2007/        (extracted, trainval)
│   └── VOCdevkit/VOC2007_test/   (extracted, test)
├── coco/
│   ├── train2017.zip  val2017.zip  test2017.zip  annotations_trainval2017.zip
│   ├── train2017/  val2017/  test2017/
│   └── annotations/
└── busi/
    ├── Dataset_BUSI.zip
    └── Dataset_BUSI/{benign,malignant,normal}/
```

## Verification

`scripts/download_datasets.sh` checks each downloaded file against its expected
byte size from the table above and warns on mismatch. For full integrity
verification, the upstream publishers do not publish checksums; if needed,
record them after a known-good download with:

```sh
cd /mnt/hdd2/datasets && sha256sum */*.zip */*.tar > sha256sums.txt
# later: sha256sum -c sha256sums.txt
```
