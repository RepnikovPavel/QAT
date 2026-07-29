# FPQ on YOLOv5s LSQ W4A4 — VOC recipe

Feature-Perturbed Quantization (arXiv:2503.11159) as an add-on to the Q² LSQ
stack: **SFP** on quantized-conv activations + **CSD** from an FP teacher.

Code: `methods/sota_qat/fpq/`.

## Files

| Path | Role |
| --- | --- |
| `methods/sota_qat/fpq/fpq.py` | Eq. 9/14/16–17/22 ops |
| `methods/sota_qat/fpq/sfp_inject.py` | Wrap QYOLOv5 acts + CSD hooks |
| `methods/sota_qat/fpq/train_detect.py` | LSQ ± FPQ VOC train |
| `methods/sota_qat/fpq/eval_detect.py` | mAP@0.5 eval |

## Unit tests (CPU)

```sh
cd /path/to/QAT
PYTHONPATH=methods/sota_qat/fpq python -m pytest methods/sota_qat/fpq/tests/ -q
```

## Smoke (GPU, few steps)

```sh
docker run --rm --gpus all --shm-size=16g \
  -v "$PWD":/workspace -v /mnt/hdd2:/mnt/hdd2 -w /workspace \
  -e PYTHONPATH=/workspace/methods/sota_qat/fpq:/workspace/methods/q2 \
  -e WANDB_MODE=disabled \
  qat-repro \
  python -u methods/sota_qat/fpq/train_detect.py \
    --quant lsq --wbits 4 --abits 4 --fpq \
    --limit 8 --epochs 1 --batch 4 --workers 2 \
    --out /mnt/hdd2/qat_run/fpq_smoke
```

Expect finite `det` / `csd` lines and `ckpt_ep0.pt`.

## Full W4A4 comparison (50 epochs)

LSQ-only baseline:

```sh
docker run --rm --gpus '"device=0"' --shm-size=16g --name fpq_lsq_base \
  -v "$PWD":/workspace -v /mnt/hdd2:/mnt/hdd2 -w /workspace \
  -e PYTHONPATH=/workspace/methods/sota_qat/fpq:/workspace/methods/q2 \
  -e WANDB_MODE=disabled \
  qat-repro \
  python -u methods/sota_qat/fpq/train_detect.py \
    --quant lsq --wbits 4 --abits 4 \
    --epochs 50 --batch 16 --workers 4 \
    --out /mnt/hdd2/qat_run/fpq_lsq_w4a4_base
```

LSQ + FPQ (SFP p=0.1 + CSD):

```sh
docker run --rm --gpus '"device=0"' --shm-size=16g --name fpq_lsq_fpq \
  -v "$PWD":/workspace -v /mnt/hdd2:/mnt/hdd2 -w /workspace \
  -e PYTHONPATH=/workspace/methods/sota_qat/fpq:/workspace/methods/q2 \
  -e WANDB_MODE=disabled \
  qat-repro \
  python -u methods/sota_qat/fpq/train_detect.py \
    --quant lsq --wbits 4 --abits 4 --fpq --sfp-p 0.1 --csd-weight 1.0 \
    --epochs 50 --batch 16 --workers 4 \
    --out /mnt/hdd2/qat_run/fpq_lsq_w4a4
```

Or poll until VRAM free:

```sh
bash recipes/fpq_wait_launch.sh
```

## Eval mAP@0.5

```sh
docker run --rm --gpus all --shm-size=16g \
  -v "$PWD":/workspace -v /mnt/hdd2:/mnt/hdd2 -w /workspace \
  -e PYTHONPATH=/workspace/methods/sota_qat/fpq:/workspace/methods/q2 \
  qat-repro \
  python -u methods/sota_qat/fpq/eval_detect.py \
    --ckpt /mnt/hdd2/qat_run/fpq_lsq_w4a4/ckpt_ep49.pt \
    --quant lsq --wbits 4 --abits 4
```

Write measured mAP into `methods/sota_qat/fpq/RESULTS.md`. If FPQ beats
LSQ-only, promote SURVEY decision adapt → reproduce.

## Hyperparameters

| Param | Value | Source |
| --- | --- | --- |
| Optimizer | SGD Nesterov | Q² App. 8.1 |
| lr / final_ratio | 0.00334 / 0.15135 | Q² |
| momentum / wd | 0.74832 / 0.00025 | Q² |
| SFP p | 0.1 | FPQ Tab. 3 |
| CSD λ | 1.0 on *mean* CSD | scale-adapted for det |
| CSD stages | 4,6,9,13,17,20,23 | yolov5s C3/SPPF |
