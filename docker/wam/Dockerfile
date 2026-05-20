# WAM inference container.
# Base: CUDA 12.2 + Python 3.12.
# unitree_sdk2_python (DDS) is bind-mounted at /workspace/unitree_sdk2py_src — no pip install.
# Model code is bind-mounted at /workspace/wam — rebuild is NOT needed for code changes.
# Rebuild only when adding new pip packages.

FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Python 3.12 is not in Ubuntu 22.04 default repos — add deadsnakes PPA.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-venv \
        git curl wget \
        libglib2.0-0 libsm6 libxext6 libxrender1 \
        iproute2 \
    && rm -rf /var/lib/apt/lists/*

# Bootstrap pip for 3.12 (deadsnakes doesn't include python3-pip for 3.12).
RUN python3.12 -m ensurepip --upgrade && \
    python3.12 -m pip install --upgrade pip

# Make python3.12 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python  python  /usr/bin/python3.12 1

# Core ML stack — installed into image to keep restart times fast.
# Add model-specific deps here (e.g. transformers, timm, einops).
RUN pip install --no-cache-dir \
    torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128

RUN pip install --no-cache-dir \
    numpy \
    transformers \
    einops \
    scipy \
    pyyaml \
    tqdm

# CycloneDDS Python — used by unitree_sdk2_python (mounted at runtime).
# Pin to match the version bundled in the unitree SDK.
RUN pip install --no-cache-dir cyclonedds==0.10.2

# /workspace/wam      — your model code (bind-mounted, hot-reload)
# /workspace/unitree_sdk2py_src — DDS SDK (bind-mounted read-only)
WORKDIR /workspace/wam

CMD ["python", "main.py"]
