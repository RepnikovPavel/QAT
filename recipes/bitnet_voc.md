# BitNet b1.58 (YOLOv5s) on VOC — recipe

Ternary weight (W1.58) + 8-bit absmax activations on the YOLOv5s mid-body.
Official training BitLinear from microsoft/unilm BitNet FAQ Figure 3; inference
kernels reference microsoft/BitNet.

## Layout

| Path | Role |
| --- | --- |
| `methods/bitnet_det/bitlinear.py` | BitLinear / BitConv2d |
| `methods/bitnet_det/xnor_conv.py` | XNOR-popcount binary conv |
| `methods/bitnet_det/yolo_bitnet.py` | YOLOv5s injection |
| `methods/bitnet_det/train_detect.py` | VOC train |
| `papers/2402.17764/` | BitNet b1.58 paper (when parsed) |
| `papers/1603.05279/` | XNOR-Net paper (when parsed) |

## 0. Prerequisites (GPU server)

```sh
cd ~/QAT && git pull
docker image inspect qat-repro >/dev/null 2>&1 || \
  docker build -t qat-repro -f docker/Dockerfile .
```

## 1. Unit tests

```sh
docker run --rm -v "$PWD":/workspace -w /workspace \
  -e PYTHONPATH=/workspace/methods/bitnet_det qat-repro \
  python -m pytest methods/bitnet_det/tests/ -q
```

## 2. Smoke train (finite loss check)

```sh
docker run --rm --gpus all --shm-size=16g \
  -v "$PWD":/workspace -v /mnt/hdd2:/mnt/hdd2 -w /workspace \
  -e PYTHONPATH=/workspace/methods/bitnet_det:/workspace/methods/q2 \
  -e WANDB_MODE=disabled \
  qat-repro \
  python -u methods/bitnet_det/train_detect.py \
    --limit 8 --epochs 1 --batch 4 --workers 2 \
    --out /mnt/hdd2/qat_run/bitnet_smoke
```

Expect: `finite=True`, `n_bit_convs > 0`.

## 3. Full VOC train (BitNet W1.58A8 body)

```sh
docker run --rm --gpus '"device=0"' --shm-size=16g --name bitnet_voc \
  -v "$PWD":/workspace -v /mnt/hdd2:/mnt/hdd2 -w /workspace \
  -e PYTHONPATH=/workspace/methods/bitnet_det:/workspace/methods/q2 \
  -e WANDB_MODE=disabled \
  qat-repro \
  python -u methods/bitnet_det/train_detect.py \
    --epochs 50 --batch 16 --workers 4 --lr 1e-3 --opt adam \
    --out /mnt/hdd2/qat_run/bitnet_voc \
  > /mnt/hdd2/qat_run/bitnet_voc.log 2>&1
```

FP32 baseline:

```sh
# add --fp32 --out /mnt/hdd2/qat_run/bitnet_voc_fp32
```

XNOR-Net binary body:

```sh
# add --mode xnor --out /mnt/hdd2/qat_run/xnor_voc
```

## 4. Hyperparameters

| Param | Value | Source |
| --- | --- | --- |
| Body quant | W1.58 absmean + A8 absmax + STE | FAQ Fig. 3 |
| Norm | channel RMSNorm inside BitConv2d | FAQ (built-in RMSNorm) |
| Stem / Detect | FP32 | detection convention |
| Optimizer | Adam, lr=1e-3 | BitNet FAQ (higher LR than FP) |
| Epochs / batch | 50 / 16 | practical (2× RTX 5060 Ti) |
| Data | VOC07+12 trainval | same as Q² / BNL |
| Pretrained | yolov5s.pt COCO body if present | optional warm-start |
