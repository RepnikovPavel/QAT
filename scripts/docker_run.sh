#!/usr/bin/env bash
# Build (if needed) and run a command inside the Q^2 docker container on the
# GPU server. Datasets are mounted read-only from /mnt/hdd2; the repo is
# mounted read-write at /workspace.
#
# Usage:
#   scripts/docker_run.sh <command...>
#   scripts/docker_run.sh python -m pytest tests/ -q
#   scripts/docker_run.sh python qat/train_detect.py --config configs/yolov5s_lsq_voc.yaml
set -euo pipefail

IMAGE=${IMAGE:-qat-repro}
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Build if missing
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "[docker_run] building image $IMAGE ..."
    docker build -t "$IMAGE" -f "$REPO_DIR/docker/Dockerfile" "$REPO_DIR"
fi

docker run --rm --gpus all \
    --shm-size=16g \
    -v "$REPO_DIR:/workspace" \
    -v /mnt/hdd2:/mnt/hdd2 \
    -e WANDB_MODE=disabled \
    -e PYTHONPATH=/workspace \
    -w /workspace \
    "$IMAGE" "$@"
