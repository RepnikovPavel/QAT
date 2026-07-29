# Q^2 reproduction environment.
#
# Host: server1 with 2x NVIDIA RTX 5060 Ti (Blackwell, sm_120). These GPUs
# need CUDA 12.8+ and a recent PyTorch build (nightly cu128). We use the
# official NVIDIA CUDA devel image as a base and install PyTorch from the
# nightly cu128 wheel index.
#
# Build:
#   docker build -t qat-repro -f docker/Dockerfile .
#
# Run (see scripts/docker_run.sh):
#   docker run --rm --gpus all \
#     -v /mnt/hdd2:/mnt/hdd2 \
#     -v $(pwd):/workspace qat-repro ...

FROM nvcr.io/nvidia/cuda:13.0.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip python3-venv \
        git wget curl unzip vim \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python3

# PyTorch nightly with CUDA 12.8 wheels (supports Blackwell sm_120).
RUN pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# Project Python deps
RUN pip install \
        numpy scipy scikit-image opencv-python-headless \
        pycocotools albumentations tqdm pyyaml matplotlib pandas \
        pytest einops

WORKDIR /workspace
CMD ["bash"]
