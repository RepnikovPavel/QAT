#!/usr/bin/env bash
# grok-4.5 supervisor: runs bounded work-chunks on the top open QAT task,
# round-robin, until killed by --max-runtime. Designed to run from cron or
# a long nohup loop. Re-entrant: reads task STATUS from the .md files.
set -uo pipefail
REPO="/home/user/ZCodeProject/QAT"
TASKS="/home/user/.grok/tasks"
LOG="$REPO/logs/supervisor.log"
mkdir -p "$REPO/logs"
MAX_RUNTIME="${MAX_RUNTIME:-3600}"   # default 1h per supervisor invocation
TURNS="${GROK_TURNS:-40}"            # grok agent turns per task chunk
START=$(date +%s)

cd "$REPO"

# ordered task list (priority order, mirrors INDEX.md)
TASK_FILES=(qat_m1_eval.md qat_paper_2509_07025.md qat_bitnet_det.md qat_sota_qat_survey.md)

ensure_tunnel() {
  if curl -s --max-time 3 http://127.0.0.1:18601/api/v1/queue >/dev/null 2>&1; then return; fi
  pkill -f "ssh.*-L 18601:127.0.0.1:8601" 2>/dev/null || true
  sleep 1
  ssh -o StrictHostKeyChecking=no -o ConnectTimeout=8 -f -N \
      -L 18601:127.0.0.1:8601 user@192.168.1.68 2>/dev/null || true
  sleep 2
}

next_open_task() {
  for t in "${TASK_FILES[@]}"; do
    f="$TASKS/$t"
    [ -f "$f" ] || continue
    # STATUS: DONE  -> skip ; STATUS: BLOCKED with cooldown handled elsewhere
    if ! grep -qiE '^STATUS:\s*DONE' "$f"; then
      echo "$f"; return
    fi
  done
  echo ""
}

run_chunk() {
  local taskfile="$1"
  local name="$(basename "$taskfile" .md)"
  echo "[$(date +%H:%M)] chunk: $name (turns=$TURNS)" | tee -a "$LOG"
  # Build the prompt: rules are in AGENTS.md (auto-read); give the task file path.
  local prompt="You are resuming an autonomous research task in this repo.
Read AGENTS.md for hard rules, then read the task file $taskfile and do the
next concrete step it describes. Work up to $TURNS turns, then STOP and write
your progress + the exact next step into the task file's '## Next step'
section. Do not mark STATUS: DONE unless acceptance criteria are fully met
and verified. Push any new artifacts (code/results/recipes) to origin main."
  timeout $((TURNS * 90 + 120)) grok \
      --model grok-4.5 --always-approve --max-turns "$TURNS" \
      --cwd "$REPO" --no-plan \
      -p "$prompt" >> "$REPO/logs/${name}.log" 2>&1 || true
  echo "[$(date +%H:%M)] chunk done: $name" | tee -a "$LOG"
}

# main loop: round-robin open tasks until time budget exhausted
ensure_tunnel
while :; do
  now=$(date +%s)
  [ $((now - START)) -ge "$MAX_RUNTIME" ] && { echo "time budget reached" | tee -a "$LOG"; break; }
  t="$(next_open_task)"
  [ -z "$t" ] && { echo "no open tasks" | tee -a "$LOG"; break; }
  ensure_tunnel
  run_chunk "$t"
  sleep 10
done
echo "=== supervisor exit $(date +%H:%M) ===" | tee -a "$LOG"
