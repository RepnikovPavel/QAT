# BNL (arXiv:2509.07025) on VOC — reproduction guide

Binary Normalized Layers adapted to YOLOv5s detection. Paper has no VOC/COCO
numbers; this recipe trains the detection adaptation on PASCAL VOC.

## Layout

| Path | Role |
| --- | --- |
| `methods/bnl/bnl/` | core quant / normalize / BNCVL / BNFCL |
| `methods/bnl/bnl/yolo_bnl.py` | YOLOv5s injection (mid-body → BNL, stem+Detect FP32) |
| `methods/bnl/train_detect.py` | VOC train loop (reuses `methods/q2` VOC data) |
| `papers/2509.07025/document.md` | parsed paper |

## 0. Prerequisites (GPU server)

Same as Q²: Docker image `qat-repro`, datasets at `/mnt/hdd2/datasets/voc`.

```sh
# on user@192.168.1.68 (or 192.168.0.1 via eno1)
cd ~/QAT && git pull
docker image inspect qat-repro >/dev/null 2>&1 || \
  docker build -t qat-repro -f docker/Dockerfile .
```

## 1. Unit tests (CPU)

```sh
docker run --rm -v "$PWD":/workspace -w /workspace \
  -e PYTHONPATH=/workspace/methods/bnl qat-repro \
  python -m pytest methods/bnl/tests/ -q
```

## 2. Smoke train (few steps, confirm finite loss)

```sh
docker run --rm --gpus all --shm-size=16g \
  -v "$PWD":/workspace -v /mnt/hdd2:/mnt/hdd2 -w /workspace \
  -e PYTHONPATH=/workspace/methods/bnl:/workspace/methods/q2 \
  -e WANDB_MODE=disabled \
  qat-repro \
  python -u methods/bnl/train_detect.py \
    --limit 8 --epochs 1 --batch 4 --workers 2 \
    --out /mnt/hdd2/qat_run/bnl_smoke
```

Expect: `finite=True`, loss is a finite float, `n_bnl_convs > 0`.

## 3. Full VOC train (YOLOv5s + BNL W1 body)

```sh
docker run --rm --gpus '"device=0"' --shm-size=16g --name bnl_voc \
  -v "$PWD":/workspace -v /mnt/hdd2:/mnt/hdd2 -w /workspace \
  -e PYTHONPATH=/workspace/methods/bnl:/workspace/methods/q2 \
  -e WANDB_MODE=disabled \
  qat-repro \
  python -u methods/bnl/train_detect.py \
    --epochs 50 --batch 16 --workers 4 --lr 1e-4 --opt adam \
    --out /mnt/hdd2/qat_run/bnl_voc \
  > /mnt/hdd2/qat_run/bnl_voc.log 2>&1
```

FP32 baseline (no BNL injection):

```sh
# same as above plus --fp32 --out /mnt/hdd2/qat_run/bnl_voc_fp32
```

## 4. Hyperparameters

| Param | Value | Source |
| --- | --- | --- |
| Body quant | W1 `{0,1}` mean-threshold + STE | paper Eq. 1 / Alg. 1 |
| Activations | FP32 | paper |
| Post-linear | per-example Normalize | paper Alg. 1 step 9 |
| Stem / Detect | FP32 | detection adapter convention |
| Optimizer | Adam, lr=1e-4 | paper CNN (Table 1) |
| Epochs / batch | 50 / 16 | practical (2×16GB GPUs) |
| Data | VOC07+12 trainval, VOC07 test | same as Q² |
| Pretrained | yolov5s.pt COCO body if present | Q² Appendix 8.1 style |

## 5. Eval mAP@0.5

```sh
docker run --rm --gpus '"device=0"' --shm-size=16g --name bnl_eval \
  -v "$PWD":/workspace -v /mnt/hdd2:/mnt/hdd2 -w /workspace \
  -e PYTHONPATH=/workspace/methods/bnl:/workspace/methods/q2 \
  -e WANDB_MODE=disabled \
  qat-repro \
  python -u methods/bnl/eval_detect.py \
    --ckpt /mnt/hdd2/qat_run/bnl_voc/ckpt_ep49.pt \
    --data /mnt/hdd2/qat_run/voc_yolo \
    --batch 16 --workers 4
```

Writes `eval_result.txt` next to the checkpoint. Copy mAP into
`methods/bnl/RESULTS.md`.

## 6. Results

Fill `methods/bnl/RESULTS.md` detection table after full runs (mAP@0.5).
Smoke only confirms finite training loss, not mAP.

## 7. Wait-and-launch (when GPUs busy)

If both GPUs are occupied, poll until free then start train:

```sh
# on GPU server, from ~/QAT_main after git pull
bash recipes/bnl_wait_launch.sh
```
