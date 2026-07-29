# Binary Normalized Layers (BNL)

Port of Cabral et al. arXiv:2509.07025 binary normalized layers for use in
2D object detection QAT experiments.

- Paper parse: `papers/2509.07025/document.md`
- Method map + decision: `METHOD.md`
- Numbers: `RESULTS.md`
- Core: `bnl/` (`BinaryNormalizedLinear`, `BinaryNormalizedConv2d`)
- Detection adapter: `bnl/yolo_bnl.py` (`BNLYOLOv5`)
- Train: `train_detect.py` (VOC; recipe `recipes/bnl_voc.md`)

```sh
# unit tests (core; yolo tests skip if yolov5 missing)
cd methods/bnl && PYTHONPATH=. python -m pytest tests/ -q

# GPU server smoke (Docker qat-repro) — see recipes/bnl_voc.md
PYTHONPATH=methods/bnl:methods/q2 python -u methods/bnl/train_detect.py \
  --limit 8 --epochs 1 --batch 4 --out /mnt/hdd2/qat_run/bnl_smoke
```
