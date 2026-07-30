# Agent rules (read first)

You are an autonomous research agent working on this repository
(quantization-aware training for 2D object detection). Work towards the
current task file you were given.

## Hard rules

1. **Repo = code + results + recipes ONLY.** Never commit status, "todo",
   "in progress", "pending", or chat-style text. Results go into
   `methods/<name>/RESULTS.md` as finished numbers. Code goes under
   `methods/<name>/`. Run-recipes under `recipes/`.
2. **Work on a BRANCH, never push to main.** At the start of each chunk,
   create (if missing) and checkout a branch named `grok/<task>` from
   origin/main (the supervisor tells you the exact branch name). Commit and
   push ONLY that branch (`git push -u origin grok/<task>`). NEVER run
   `git checkout main`, `git merge main`, or push to main — a reviewer merges
   your branch after code review. If main moved, rebase your branch onto it.
3. **Commit identity:** `git config user.name "grok-4.5 agent"`,
   `user.email "grok-4.5@local"`.
4. **Never hallucinate code.** If a paper or upstream repo exists, fetch its
   source (git clone / web) or ocrc-parse the PDF before implementing. The
   ocrc service is available at `http://127.0.0.1:18601` (if down, see
   `recipes/ocrc_tunnel.md`).
5. **All training runs on the GPU server** via ssh `user@192.168.1.68` (see
   `/grok/secrets.md` for the password). Datasets at
   `/mnt/hdd2/datasets/{voc,coco}`. Use Docker image `qat-repro` (build from
   `docker/Dockerfile` if missing) with `--gpus all --shm-size=16g`. Never
   install deps on the host — Docker only.
6. **Anti-hallucination for every method:** cross-check formulas against the
   parsed paper in `papers/<arxivid>/document.md` before coding.
7. **Coordinate GPU usage.** The Q^2 (q2) M1/M2 training+eval and the
   `methods/q2/` RESULTS are owned by the primary agent — do NOT touch
   `/mnt/hdd2/qat_run/m1_*` / `m2_*` runs or modify `methods/q2/`. Use your
   own run dirs like `/mnt/hdd2/qat_run/grok_<task>/`.

## When you finish a work chunk

- Update the task file's "## Progress" section with concrete facts (commands
  run, files changed, numbers measured). Mark `STATUS: DONE` only when the
  task's acceptance criteria are fully met and verified.
- `git add` only artifact dirs (methods/ recipes/ docker/ scripts/ papers/
  results/) and commit+push. Skip logs/queues/.
- Append a 2-line summary to `logs/supervisor.log`.

## When you're blocked or hit your turn budget

- If the blocker is **waiting on something time-based** (a training job still
  running, an ocrc queue, a cooldown), write the estimated ready-time into the
  task file as `STATUS: WAITING until YYYY-MM-DD_HH:MM` (24h, server local
  time) AND STOP. The supervisor will skip this task until that time and move
  to the next open one. Do NOT keep re-checking the same thing every turn.
- Otherwise (real blocker) write the exact blocker + next concrete step into
  the task file's "## Next step" section, then stop. The supervisor resumes you.
