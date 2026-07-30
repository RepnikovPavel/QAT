#!/usr/bin/env bash
# Build (if needed) and run a command inside the Q^2 docker container on the
# GPU server. The repo is mounted read-write at /workspace.
#
# Select the CUDA build via PLATFORM (controls the image name + Dockerfile):
#   PLATFORM=cu126  -> docker/Dockerfile.cu126, image qat-repro-cu126
#                      Ada/Ampere (RTX 4090 sm_89) + CUDA-13 hosts (default)
#   PLATFORM=blackwell (or cu128) -> docker/Dockerfile, image qat-repro
#                      Blackwell sm_120 (RTX 5060 Ti) only
# Or override everything with IMAGE=<name>.
#
# DATA_DIR points the host dataset mount at the server's dataset disk
# (default /mnt/hdd2; on the 4090 box set DATA_DIR=/mnt/data1/logchecker/qat_repro).
#
# Usage:
#   scripts/docker_run.sh <command...>
#   PLATFORM=cu126 DATA_DIR=/mnt/data1/logchecker/qat_repro scripts/docker_run.sh \
#       python -m pytest tests/ -q
set -euo pipefail

# Map PLATFORM -> (IMAGE, DOCKERFILE).
case "${PLATFORM:-cu126}" in
    cu126)            IMAGE=${IMAGE:-qat-repro-cu126}; DOCKERFILE=docker/Dockerfile.cu126 ;;
    blackwell|cu128)  IMAGE=${IMAGE:-qat-repro};        DOCKERFILE=docker/Dockerfile ;;
    *)                IMAGE=${IMAGE:-$PLATFORM};        DOCKERFILE=docker/Dockerfile ;;
esac
# If IMAGE was explicitly overridden, still need a Dockerfile to build from.
DOCKERFILE=${DOCKERFILE:-docker/Dockerfile}
DATA_DIR=${DATA_DIR:-/mnt/hdd2}
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Build if missing
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "[docker_run] building image $IMAGE from $DOCKERFILE ..."
    docker build -t "$IMAGE" -f "$REPO_DIR/$DOCKERFILE" "$REPO_DIR"
fi

docker run --rm --gpus all \
    --shm-size=16g \
    -v "$REPO_DIR:/workspace" \
    -v "$DATA_DIR:$DATA_DIR" \
    -e WANDB_MODE=disabled \
    -e PYTHONPATH=/workspace \
    -w /workspace \
    "$IMAGE" "$@"
