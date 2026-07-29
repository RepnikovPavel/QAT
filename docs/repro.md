# Reproduction guide

How to run the Q² reproduction on the GPU server (2× RTX 5060 Ti, Blackwell
sm_120) using Docker.

## 0. Prerequisites (server)

Datasets already on disk at `/mnt/hdd2/datasets/{voc,coco,busi}`. Docker +
NVIDIA Container Toolkit installed and verified (`docker run --rm --gpus all
nvidia/cuda:... nvidia-smi -L` shows both GPUs).

## 1. Clone + build the image

```sh
git clone https://github.com/RepnikovPavel/QAT.git ~/QAT
cd ~/QAT
docker build -t qat-repro -f docker/Dockerfile .
```

The image uses `torch==2.9.1 + torchvision==0.9.1` from the cu128 wheel index
(stable Blackwell support). The `yolov5==7.0.14` package provides the
official `DetectionModel` / `ComputeLoss` / `non_max_suppression` that the
pipeline reuses.

## 2. Verify GPU + sm_120

```sh
scripts/docker_run.sh python -c "import torch; \
  print(torch.__version__, torch.cuda.is_available(), \
        torch.cuda.get_device_capability(0))"
# expect: 2.9.1+cu128 True (12, 0)
```

## 3. Unit tests (CPU, fast)

```sh
scripts/docker_run.sh python -m pytest tests/ -q
```

## 4. Prepare VOC → YOLO format (idempotent)

Done automatically on the first training run, or explicitly:

```sh
scripts/docker_run.sh python -c "from qat.data.voc import prepare_voc; \
  print(prepare_voc('/mnt/hdd2/datasets/voc', '/mnt/hdd2/qat_run/voc_yolo'))"
```

## 5. M0 — gradient-imbalance probe (Fig. 1b, fast)

```sh
scripts/docker_run.sh python -m qat.probe_imbalance \
  --quant lsq --wbits 4 --abits 4 --steps 120 \
  --out /mnt/hdd2/qat_run/m0
```

Expected: under W4A4 the deep branch gradient energy dominates (ratio ≫ 1);
with `use_qgb` the two energies converge (ratio → 1).

## 6. Train (M1/M2, long)

YOLOv5s + LSQ, W4A4, with Q² (Q-GBFusion + Q-ADA):

```sh
scripts/docker_run.sh python -m qat.train_detect \
  --quant lsq --wbits 4 --abits 4 --qgb --qada \
  --epochs 50 --batch 32 \
  --out /mnt/hdd2/qat_run/yolov5s_lsq_w4a4_q2
```

For the baseline drop `--qgb --qada`. For N2UQ swap `--quant n2uq`. The
training loop loads the yolov5s COCO-pretrained weights automatically is left
to a one-line call to `qat.pretrained.load_yolov5s_pretrained`).

> Effective batch: the paper uses bs=64 across 8 GPUs. On 2× 16GB GPUs we use
> bs=32×2 via DDP (or bs=32 + grad-accum=2 on one GPU). Adjust `--batch`.

## 7. Evaluate mAP@0.5

```sh
scripts/docker_run.sh python -m qat.eval_detect \
  --ckpt /mnt/hdd2/qat_run/yolov5s_lsq_w4a4_q2/ckpt_ep49.pt \
  --quant lsq --wbits 4 --abits 4 --qgb
```

## Hyperparameters (Appendix 8.1, defaults baked in)

| Param | Value |
| --- | --- |
| Optimizer | SGD (Nesterov) |
| lr / final_ratio | 0.00334 / 0.15135 (OneCycleLR) |
| momentum / weight_decay | 0.74832 / 0.00025 |
| batch (paper) | 64 |
| seed | 0 |
| Q-ADA weight | 0.01 |
| Q-GBFusion η / β | 0.05 / 0.1 (defaults) |
