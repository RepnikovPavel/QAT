#!/usr/bin/env bash
# Poll GPU free memory; when a device has enough free VRAM, launch BitNet VOC train.
# Idempotent: skips if container bitnet_voc is already running or out dir has ckpt_ep0.
set -euo pipefail

REPO="${REPO:-$HOME/QAT_main}"
OUT="${OUT:-/mnt/hdd2/qat_run/bitnet_voc}"
LOG="${LOG:-/mnt/hdd2/qat_run/bitnet_voc.log}"
NEED_MIB="${NEED_MIB:-10000}"   # free MiB required (batch 16 @ 640)
POLL_SEC="${POLL_SEC:-120}"
NAME="${NAME:-bitnet_voc}"
MAX_WAIT_SEC="${MAX_WAIT_SEC:-28800}"  # 8h
EPOCHS="${EPOCHS:-50}"
BATCH="${BATCH:-16}"
LR="${LR:-1e-3}"
MODE="${MODE:-bitnet}"

cd "$REPO"

if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "[bitnet_wait] container $NAME already running — exit"
  exit 0
fi
if [[ -f "$OUT/ckpt_ep0.pt" ]]; then
  echo "[bitnet_wait] $OUT already has ckpt_ep0.pt — exit (train started before)"
  exit 0
fi

echo "[bitnet_wait] polling every ${POLL_SEC}s for GPU with free >= ${NEED_MIB} MiB"
start_ts=$(date +%s)
while true; do
  now=$(date +%s)
  if (( now - start_ts > MAX_WAIT_SEC )); then
    echo "[bitnet_wait] timed out after ${MAX_WAIT_SEC}s"
    exit 2
  fi
  free_line=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | tr -d ' ')
  chosen=""
  while IFS=',' read -r idx free; do
    if (( free >= NEED_MIB )); then
      chosen=$idx
      break
    fi
  done <<< "$free_line"

  if [[ -n "$chosen" ]]; then
    echo "[bitnet_wait] GPU $chosen free enough — launching $NAME"
    mkdir -p "$OUT"
    nohup docker run --rm --gpus "\"device=${chosen}\"" --shm-size=16g --name "$NAME" \
      -v "$REPO":/workspace -v /mnt/hdd2:/mnt/hdd2 -w /workspace \
      -e PYTHONPATH=/workspace/methods/bitnet_det:/workspace/methods/q2 \
      -e WANDB_MODE=disabled \
      qat-repro \
      python -u methods/bitnet_det/train_detect.py \
        --epochs "$EPOCHS" --batch "$BATCH" --img-size 640 --workers 4 \
        --lr "$LR" --opt adam --mode "$MODE" \
        --out "$OUT" \
      > "$LOG" 2>&1 &
    echo "[bitnet_wait] docker launched pid=$! log=$LOG"
    sleep 15
    if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
      echo "[bitnet_wait] OK container up"
      tail -n 20 "$LOG" || true
      exit 0
    fi
    echo "[bitnet_wait] container failed to stay up; tail log:"
    tail -n 40 "$LOG" || true
    exit 1
  fi
  echo "[bitnet_wait] $(date -Is) no free GPU yet: $free_line"
  sleep "$POLL_SEC"
done
