#!/usr/bin/env bash
# Poll GPU free memory; launch LSQ baseline + LSQ+FPQ VOC trains when free.
# Prefer two free GPUs (parallel); otherwise serialise on one free device.
# Idempotent: skips a job if its container is up or ckpt_ep0 already exists.
set -euo pipefail

REPO="${REPO:-$HOME/QAT}"
# Prefer synced main clone if present (same pattern as other wait scripts)
if [[ -d "${HOME}/QAT_main/.git" ]]; then
  REPO="${REPO_OVERRIDE:-$HOME/QAT_main}"
fi
NEED_MIB="${NEED_MIB:-9000}"
POLL_SEC="${POLL_SEC:-120}"
MAX_WAIT_SEC="${MAX_WAIT_SEC:-28800}"
EPOCHS="${EPOCHS:-50}"
BATCH="${BATCH:-16}"

BASE_OUT="${BASE_OUT:-/mnt/hdd2/qat_run/fpq_lsq_w4a4_base}"
FPQ_OUT="${FPQ_OUT:-/mnt/hdd2/qat_run/fpq_lsq_w4a4}"
BASE_LOG="${BASE_LOG:-/mnt/hdd2/qat_run/fpq_lsq_base.log}"
FPQ_LOG="${FPQ_LOG:-/mnt/hdd2/qat_run/fpq_lsq_fpq.log}"
BASE_NAME="${BASE_NAME:-fpq_lsq_base}"
FPQ_NAME="${FPQ_NAME:-fpq_lsq_fpq}"

cd "$REPO"

launch_one() {
  local name="$1" out="$2" log="$3" gpu="$4" extra_args="$5"
  if docker ps --format '{{.Names}}' | grep -qx "$name"; then
    echo "[fpq_wait] container $name already running — skip"
    return 0
  fi
  if [[ -f "$out/ckpt_ep0.pt" ]]; then
    echo "[fpq_wait] $out already has ckpt_ep0.pt — skip"
    return 0
  fi
  mkdir -p "$out"
  # shellcheck disable=SC2086
  nohup docker run --rm --gpus "\"device=${gpu}\"" --shm-size=16g --name "$name" \
    -v "$REPO":/workspace -v /mnt/hdd2:/mnt/hdd2 -w /workspace \
    -e PYTHONPATH=/workspace/methods/sota_qat/fpq:/workspace/methods/q2 \
    -e WANDB_MODE=disabled \
    qat-repro \
    python -u methods/sota_qat/fpq/train_detect.py \
      --quant lsq --wbits 4 --abits 4 \
      --epochs "$EPOCHS" --batch "$BATCH" --img-size 640 --workers 4 \
      --out "$out" \
      $extra_args \
    > "$log" 2>&1 &
  echo "[fpq_wait] launched $name on GPU $gpu pid=$! log=$log"
  sleep 12
  if docker ps --format '{{.Names}}' | grep -qx "$name"; then
    echo "[fpq_wait] OK $name up"
    tail -n 15 "$log" || true
    return 0
  fi
  echo "[fpq_wait] $name failed to stay up; log:"
  tail -n 40 "$log" || true
  return 1
}

need_base=1
need_fpq=1
[[ -f "$BASE_OUT/ckpt_ep0.pt" ]] && need_base=0
[[ -f "$FPQ_OUT/ckpt_ep0.pt" ]] && need_fpq=0
docker ps --format '{{.Names}}' | grep -qx "$BASE_NAME" && need_base=0
docker ps --format '{{.Names}}' | grep -qx "$FPQ_NAME" && need_fpq=0

if (( need_base == 0 && need_fpq == 0 )); then
  echo "[fpq_wait] both jobs already started or complete — exit"
  exit 0
fi

echo "[fpq_wait] polling every ${POLL_SEC}s for free >= ${NEED_MIB} MiB"
echo "[fpq_wait] need_base=$need_base need_fpq=$need_fpq repo=$REPO"
start_ts=$(date +%s)
while true; do
  now=$(date +%s)
  if (( now - start_ts > MAX_WAIT_SEC )); then
    echo "[fpq_wait] timed out after ${MAX_WAIT_SEC}s"
    exit 2
  fi

  # collect free GPUs
  mapfile -t free_gpus < <(
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
      | tr -d ' ' \
      | while IFS=',' read -r idx free; do
          if (( free >= NEED_MIB )); then
            echo "$idx"
          fi
        done
  )

  if (( ${#free_gpus[@]} == 0 )); then
    free_line=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tr -d ' ')
    echo "[fpq_wait] $(date -Is) no free GPU yet: $free_line"
    sleep "$POLL_SEC"
    continue
  fi

  # refresh needs (another waiter may have started)
  need_base=1; need_fpq=1
  [[ -f "$BASE_OUT/ckpt_ep0.pt" ]] && need_base=0
  [[ -f "$FPQ_OUT/ckpt_ep0.pt" ]] && need_fpq=0
  docker ps --format '{{.Names}}' | grep -qx "$BASE_NAME" && need_base=0
  docker ps --format '{{.Names}}' | grep -qx "$FPQ_NAME" && need_fpq=0
  if (( need_base == 0 && need_fpq == 0 )); then
    echo "[fpq_wait] both done/started — exit"
    exit 0
  fi

  gi=0
  if (( need_base == 1 && gi < ${#free_gpus[@]} )); then
    launch_one "$BASE_NAME" "$BASE_OUT" "$BASE_LOG" "${free_gpus[$gi]}" "" || true
    gi=$((gi + 1))
    need_base=0
  fi
  if (( need_fpq == 1 && gi < ${#free_gpus[@]} )); then
    launch_one "$FPQ_NAME" "$FPQ_OUT" "$FPQ_LOG" "${free_gpus[$gi]}" \
      "--fpq --sfp-p 0.1 --csd-weight 1.0" || true
    gi=$((gi + 1))
    need_fpq=0
  fi

  # if still need one job but only one GPU freed, loop again after launch
  sleep 5
  still=0
  [[ ! -f "$BASE_OUT/ckpt_ep0.pt" ]] && ! docker ps --format '{{.Names}}' | grep -qx "$BASE_NAME" && still=1
  [[ ! -f "$FPQ_OUT/ckpt_ep0.pt" ]] && ! docker ps --format '{{.Names}}' | grep -qx "$FPQ_NAME" && still=1
  if (( still == 0 )); then
    echo "[fpq_wait] all requested launches done"
    exit 0
  fi
  sleep "$POLL_SEC"
done
