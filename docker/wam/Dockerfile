# WAM inference container.
# Base: CUDA 12.2 + Python 3.10 (Ubuntu 22.04 system Python).
#
# CRITICAL: Must use cyclonedds==0.10.2 (matching GEAR-SONIC and Isaac Sim).
# cyclonedds 11.x sends XTypes 1.3 discovery packets that crash GEAR-SONIC's
# 0.10.x DDS library with a segfault in ddsi_xt_type_init_impl.
#
# cyclonedds 0.10.2 has no binary wheels — it must be compiled against the
# CycloneDDS C library. We build it from source (releases/0.10.x branch).
#
# unitree_sdk2_python (DDS) is bind-mounted at /workspace/unitree_sdk2py_src — no pip install.
# Model code is bind-mounted at /workspace/wam — rebuild is NOT needed for code changes.
# Rebuild only when adding new pip packages.

FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Python 3.10 is the Ubuntu 22.04 default — no PPA needed.
# cmake + gcc are needed to build CycloneDDS from source.
# Disable GPG verification for repositories (workaround for signature issues in build environment)
RUN mkdir -p /etc/apt/apt.conf.d && \
    echo 'APT::Get::AllowUnauthenticated=true;' > /etc/apt/apt.conf.d/70allow-unsigned && \
    echo 'Acquire::AllowInsecureRepositories true;' >> /etc/apt/apt.conf.d/70allow-unsigned && \
    echo 'Acquire::AllowDowngradeToInsecureRepositories true;' >> /etc/apt/apt.conf.d/70allow-unsigned && \
    apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip \
        cmake gcc g++ make \
        git curl wget \
        libssl-dev \
        libglib2.0-0 libsm6 libxext6 libxrender1 \
        libgl1-mesa-glx \
        iproute2 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python  python  /usr/bin/python3.10 1

RUN pip install --upgrade pip

# Build CycloneDDS 0.10.x C library from source.
# This is required by cyclonedds==0.10.2 Python package (no binary wheels exist).
# 0.10.x does not send XTypes type info — compatible with GEAR-SONIC's DDS.
RUN git clone --depth 1 -b releases/0.10.x \
        https://github.com/eclipse-cyclonedds/cyclonedds.git /tmp/cyclonedds && \
    cmake -S /tmp/cyclonedds -B /tmp/cyclonedds/build \
          -DCMAKE_INSTALL_PREFIX=/opt/cyclonedds \
          -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF && \
    cmake --build /tmp/cyclonedds/build -j$(nproc) && \
    cmake --install /tmp/cyclonedds/build && \
    rm -rf /tmp/cyclonedds

ENV CYCLONEDDS_HOME=/opt/cyclonedds
ENV LD_LIBRARY_PATH="/opt/cyclonedds/lib:${LD_LIBRARY_PATH:-}"

# Core ML stack — installed into image to keep restart times fast.
# Add model-specific deps here (e.g. transformers, timm, einops).
RUN pip install --no-cache-dir \
    torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128

# cyclonedds==0.10.2 builds against /opt/cyclonedds via CYCLONEDDS_HOME.
# Note: ROS2 packages (rclpy, cv_bridge) are available via apt, not pip in ROS containers.
# For now, install OpenCV and other deps; ROS2 packages come from system or bind-mounts.
RUN pip install --no-cache-dir \
    numpy \
    "transformers==4.40.1" \
    einops \
    scipy \
    pyyaml \
    tqdm \
    opencv-python \
    Pillow \
    "cyclonedds==0.10.2" \
    omegaconf \
    "diffusers==0.30.2" \
    accelerate \
    xformers

# Runtime deps for unifolm_wma (loaded via PYTHONPATH from bind-mount at /workspace/unifolm/src).
# unifolm_wma is NOT pip-installed — source is bind-mounted from repos/unifolm at runtime.
# Covers inference path; skips gradio / tensorflow / datasets (training-only).
# NOTE: unifolm pyproject.toml specifies torch==2.3.1; we use 2.7.0 for CUDA 12.2 compat.
#       If torch API breaks arise, downgrade torch/torchvision in the block above.
RUN pip install --no-cache-dir \
    "pytorch-lightning==1.9.3" \
    "open-clip-torch==2.22.0" \
    "timm==0.9.10" \
    "kornia==0.7.3" \
    "decord==0.6.0" \
    "moviepy==1.0.3" \
    "scikit-learn==1.5.1" \
    "termcolor==2.4.0" \
    "fairscale==0.4.13" \
    "draccus==0.11.5"

# /workspace/wam      — your model code (bind-mounted, hot-reload)
# /workspace/unitree_sdk2py_src — DDS SDK (bind-mounted read-only)
# /workspace/unifolm — unifolm_wma repo (bind-mounted from repos/unifolm)
WORKDIR /workspace/wam

CMD ["python", "main.py"]
