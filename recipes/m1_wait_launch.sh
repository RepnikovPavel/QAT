#!/usr/bin/env bash
# Wait for a free GPU, then launch M1 baseline + Q2 (Q-GBFusion+Q-ADA) LSQ W4A4.
# Uses fixed LSQ (commit 6d00058). Logs under /mnt/hdd2/qat_run/.
set -euo pipefail
REPO=${REPO:-/home/user/QAT}
OUT_BASE=/mnt/hdd2/qat_run/m1_lsq_baseline_v2
OUT_Q2=/mnt/hdd2/qat_run/m1_lsq_q2_v2
LOG_BASE=/mnt/hdd2/qat_run/m1_baseline_v2.log
LOG_Q2=/mnt/hdd2/qat_run/m1_q2_v2.log
IMAGE=qat-repro
EPOCHS=${EPOCHS:-30}
BATCH=${BATCH:-16}
FREE_MB=${FREE_MB:-12000}

need_gpu() {
  # print first GPU index with free memory >= FREE_MB, or empty
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F"," -v need="$FREE_MB" "{gsub(/ /,\"\"); if (\$2+0>=need) {print \$1; exit}}"
}

launch_one() {
  local gpu=$1 name=$2 out=$3 log=$4 extra=$5
  mkdir -p "$out"
  echo "[m1_wait] $(date -Is) launching $name on GPU$gpu -> $out"
  docker run --rm --gpus "device=${gpu}" --shm-size=16g \
    --name "$name" \
    -v "$REPO:/workspace" \
    -v /mnt/hdd2:/mnt/hdd2 \
    -w /workspace/methods/q2 \
    -e PYTHONPATH=/workspace/methods/q2 \
    -e WANDB_MODE=disabled \
    -e YOLO_CONFIG_DIR=/tmp/Ultralytics \
    "$IMAGE" python -u -m qat.train_detect \
      --quant lsq --wbits 4 --abits 4 $extra \
      --epochs "$EPOCHS" --batch "$BATCH" --workers 4 \
      --out "$out" \
    > "$log" 2>&1
  echo "[m1_wait] $(date -Is) finished $name"
}

# Skip if already complete
base_done=0; q2_done=0
if [[ -f "$OUT_BASE/ckpt_ep$((EPOCHS-1)).pt" ]]; then base_done=1; fi
if [[ -f "$OUT_Q2/ckpt_ep$((EPOCHS-1)).pt" ]]; then q2_done=1; fi

echo "[m1_wait] $(date -Is) start; base_done=$base_done q2_done=$q2_done free_need=${FREE_MB}MiB"

while [[ $base_done -eq 0 || $q2_done -eq 0 ]]; do
  # avoid double-launch if container already running
  if docker ps --format "{{.Names}}" | grep -qx qat-m1-base-v2; then
    sleep 60; continue
  fi
  if docker ps --format "{{.Names}}" | grep -qx qat-m1-q2-v2; then
    sleep 60; continue
  fi

  g=$(need_gpu || true)
  if [[ -z "${g:-}" ]]; then
    echo "[m1_wait] $(date -Is) no free GPU (>=${FREE_MB}MiB); sleeping 120s"
    sleep 120
    continue
  fi

  if [[ $base_done -eq 0 ]]; then
    launch_one "$g" qat-m1-base-v2 "$OUT_BASE" "$LOG_BASE" ""
    base_done=1
    continue
  fi
  if [[ $q2_done -eq 0 ]]; then
    launch_one "$g" qat-m1-q2-v2 "$OUT_Q2" "$LOG_Q2" "--qgb --qada"
    q2_done=1
    continue
  fi
done

echo "[m1_wait] $(date -Is) both M1 trains finished"
