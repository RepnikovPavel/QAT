# Q^2 reproduction environment.
#
# Host: server1 with 2x NVIDIA RTX 5060 Ti (Blackwell, sm_120). These GPUs
# need CUDA 12.8+ and a recent PyTorch build (nightly cu128). We build from an
# Ubuntu base (the bare cuda images lack python), install Python and PyTorch
# from the nightly cu128 wheel index (Blackwell support).
#
# Build:
#   docker build -t qat-repro -f docker/Dockerfile .
#
# Run (see scripts/docker_run.sh):
#   docker run --rm --gpus all -v /mnt/hdd2:/mnt/hdd2 -v $(pwd):/workspace qat-repro ...

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# System deps + Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv python3-pip \
        git wget curl unzip vim ca-certificates \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
    && python -m pip install --upgrade pip

# PyTorch stable with CUDA 12.8 wheels (supports Blackwell sm_120, RTX 5060 Ti).
# Let pip resolve the compatible torch/torchvision pair for cu128.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Project Python deps (yolov5 pins the official ultralytics/yolov5 package so
# we use the real DetectionModel/parse_model/ComputeLoss, no reimplementation).
RUN pip install \
        numpy scipy opencv-python-headless \
        pycocotools tqdm pyyaml matplotlib pandas requests psutil \
        pytest einops \
        yolov5==7.0.14

WORKDIR /workspace
CMD ["bash"]
