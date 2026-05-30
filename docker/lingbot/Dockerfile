FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip python3.10-venv \
    git curl wget ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    python3 -m pip install --upgrade pip

# PyTorch 2.9.0 + CUDA 12.6
RUN pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
    --index-url https://download.pytorch.org/whl/cu126

# flash-attn must come after torch and before other deps
RUN pip install packaging && \
    pip install flash-attn --no-build-isolation

# LingBot-VA dependencies (minus torch/flash-attn already installed)
RUN pip install \
    diffusers==0.36.0 \
    transformers==4.55.2 \
    accelerate \
    einops \
    easydict \
    "numpy==1.26.4" \
    tqdm \
    "imageio[ffmpeg]" \
    websockets \
    msgpack \
    opencv-python-headless \
    matplotlib \
    ftfy \
    safetensors \
    Pillow \
    huggingface_hub

# CycloneDDS for G1 DDS communication (same version as wam container)
RUN pip install cyclonedds==0.10.2

# Install lingbot-va as a package (bind-mounted at runtime)
# We pre-install deps here; the actual source is mounted
WORKDIR /workspace

# G1 URDF / Pinocchio for EEF → joint IK
RUN pip install pin  # pinocchio Python bindings via robotpkg-py310-pinocchio

CMD ["python", "-m", "wan_va.wan_va_server", "--config-name", "robotwin", "--port", "29056"]
