#!/usr/bin/env bash
# Watch M1c training; when both jobs reach 30 epochs, run eval on the full VOC
# test set and append results to methods/q2/RESULTS.md. Runs detached on the
# server host (no agent needed).
set -uo pipefail
REPO=/home/user/QAT
RUN=/mnt/hdd2/qat_run
BASE_LOG=$RUN/m1c_baseline.log
QGB_LOG=$RUN/m1c_qgb.log
TARGET=30
RESULTS=$REPO/methods/q2/M1_RESULTS.md

while :; do
  be=$(ssh -o ConnectTimeout=8 localhost "grep -cE '== epoch' $BASE_LOG" 2>/dev/null || echo 0)
  qe=$(ssh -o ConnectTimeout=8 localhost "grep -cE '== epoch' $QGB_LOG" 2>/dev/null || echo 0)
  echo "$(date +%H:%M) base=$be/30 qgb=$qe/30"
  if [ "$be" -ge "$TARGET" ] && [ "$qe" -ge "$TARGET" ]; then break; fi
  # also stop if both containers are gone but epochs < target (crashed)
  ba=$(ssh -o ConnectTimeout=8 localhost "docker ps --filter name=qat-m1c-base --format '{{.Status}}'" 2>/dev/null)
  qa=$(ssh -o ConnectTimeout=8 localhost "docker ps --filter name=qat-m1c-qgb --format '{{.Status}}'" 2>/dev/null)
  if [ -z "$ba" ] && [ -z "$qa" ] && [ "$be" -lt "$TARGET" ]; then
    echo "both containers gone before target"; break
  fi
  sleep 300
done

cd "$REPO" && git fetch -q origin main && git reset -q --hard origin/main
LAST=$((TARGET-1))
eval_one() {
  local ckpt=$1 flags=$2 gpu=$3
  docker run --rm --gpus "\"device=$gpu\"" -v $REPO:/workspace -v $RUN:/mnt/hdd2 \
    -w /workspace -e PYTHONPATH=/workspace qat-repro \
    bash -c "cd methods/q2 && python scripts/eval_any.py --ckpt $ckpt" 2>&1 | grep RESULT
}
echo "## M1 eval (auto, $(date -I))" > "$RESULTS"
b=$(eval_one $RUN/m1c_lsq_baseline/ckpt_ep${LAST}.pt "" 0)
q=$(eval_one $RUN/m1c_lsq_qgb/ckpt_ep${LAST}.pt "--qgb" 1)
echo "baseline: $b" >> "$RESULTS"
echo "qgb:      $q" >> "$RESULTS"
cat "$RESULTS"
