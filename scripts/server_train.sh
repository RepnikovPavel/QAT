#!/usr/bin/env bash
# Launch a detached Q^2 training run on the GPU server.
# Runs INSIDE the server host (not in a container). Usage:
#   scripts/server_train.sh <gpu_id> <name> <out_dir> <train_args...>
set -euo pipefail

GPU=${1:?gpu id}; shift
NAME=${1:?container name}; shift
OUT=${1:?out dir}; shift

REPO=/home/user/QAT
IMAGE=qat-repro
LOG="${OUT}.log"
mkdir -p "$(dirname "$OUT")"

# Write a launcher script to avoid quoting hell, then run it detached.
LAUNCH=/tmp/qat_launch_${NAME}.sh
cat > "$LAUNCH" <<EOF
#!/usr/bin/env bash
docker run --rm --gpus '"device=${GPU}"' --shm-size=16g --name ${NAME} \
  -v ${REPO}:/workspace -v /mnt/hdd2:/mnt/hdd2 -w /workspace \
  -e PYTHONPATH=/workspace ${IMAGE} \
  python -u -m qat.train_detect --out ${OUT} "$@" > ${LOG} 2>&1
EOF
chmod +x "$LAUNCH"
setsid bash "$LAUNCH" </dev/null >/dev/null 2>&1 &
echo "launched ${NAME} on GPU ${GPU}, log: ${LOG}"
sleep 6
docker ps --format "{{.Names}} {{.Status}}" --filter "name=${NAME}" || true
