# GEAR-SONIC policy container
# Builds the C++ TensorRT inference binary for G1 whole-body control.
#
# Reuses upstream install_deps.sh and setup_env.sh for dependency management
# and environment configuration, ensuring consistency with the upstream
# gear_sonic_deploy build system.
#
# Usage:
#   docker compose -f docker/docker-compose.yml build gear-sonic
#   docker compose -f docker/docker-compose.yml up gear-sonic

ARG TENSORRT_IMAGE=nvcr.io/nvidia/tensorrt:24.12-py3
FROM ${TENSORRT_IMAGE}

USER root
ENV DEBIAN_FRONTEND=noninteractive

# ── Bootstrap packages required by install_deps.sh ──────────────────────────
# Matches upstream Dockerfile.ros2 prerequisites: install_deps.sh uses sudo,
# curl, wget, and lsb_release internally.
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo curl wget lsb-release software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# ── Upstream dependency installation ────────────────────────────────────────
# install_deps.sh installs: cmake, clang, just, eigen3, zmq, msgpack, gtest,
# ONNX Runtime 1.16.3, and other build requirements.
WORKDIR /opt/gear_sonic_deploy
COPY modules/gwbc/gear_sonic_deploy/scripts/install_deps.sh scripts/install_deps.sh
RUN chmod +x scripts/install_deps.sh && scripts/install_deps.sh

# ── Copy source and build ───────────────────────────────────────────────────
COPY modules/gwbc/gear_sonic_deploy/ .

# Source setup_env.sh for LD_LIBRARY_PATH / CMake paths, then build.
# The TensorRT base image already provides TensorRT_ROOT.
RUN bash -c "source scripts/setup_env.sh && just build Release"

# Copy binary outside the deploy dir so a full volume mount won't shadow it.
RUN cp target/release/g1_deploy_onnx_ref /usr/local/bin/g1_deploy_onnx_ref

# ── Runtime environment ─────────────────────────────────────────────────────
# Persist the critical library paths that setup_env.sh would set at runtime.
ENV LD_LIBRARY_PATH=/opt/onnxruntime/lib:${LD_LIBRARY_PATH}

# ── Entrypoint ──────────────────────────────────────────────────────────────
COPY docker/gear-sonic/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
