# # GEAR-SONIC policy container.
# # Uses the pre-built binary from mws-sim-gear-sonic-policy:latest.
# # No source compilation — binary and all runtime deps are already in that image.
# # Rebuild only if the upstream image changes.

# FROM mws-sim-gear-sonic-policy:latest

# # Override entrypoint to fix DDS lib path for x86_64 sim.
# COPY docker/gear-sonic/entrypoint.sh /entrypoint.sh
# RUN chmod +x /entrypoint.sh
# ENTRYPOINT ["/entrypoint.sh"]

# GEAR-SONIC policy container — multi-stage build.
#
# Stage 1 (builder): compiles g1_deploy_onnx_ref against TensorRT headers.
# Stage 2 (runtime): copies the binary into a lean runtime image.
#
# For sim (x86_64):
#   TENSORRT_IMAGE defaults to the full NVIDIA TensorRT dev image, which is
#   used for both stages (no separate runtime image needed).
#
# For real robot (aarch64 / Jetson):
#   Override both ARGs in the compose build section:
#     TENSORRT_IMAGE=nvcr.io/nvidia/l4t-tensorrt:r8.6.2-devel   (build stage)
#     TENSORRT_RUNTIME_IMAGE=nvcr.io/nvidia/l4t-tensorrt:r8.6.2-runtime  (runtime stage)
#
# Usage:
#   docker compose -f deploy/sim/ros2/compose.yml build sim-gear-sonic
#   docker compose -f deploy/real/robot/compose.yml build real-gear-sonic

# Global ARGs — available to all FROM lines.
ARG TENSORRT_IMAGE=nvcr.io/nvidia/tensorrt:24.12-py3
ARG TENSORRT_RUNTIME_IMAGE=nvcr.io/nvidia/tensorrt:24.12-py3

# ── Stage 1: build ────────────────────────────────────────────────────────────
FROM ${TENSORRT_IMAGE} AS builder

USER root
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Bootstrap packages required by install_deps.sh.
# tzdata pre-installed so the interactive timezone prompt never fires
# (install_deps.sh uses sudo which strips DEBIAN_FRONTEND).
# The sudoers rule preserves DEBIAN_FRONTEND for remaining sudo apt calls.
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo curl wget lsb-release software-properties-common tzdata \
    && echo 'Defaults env_keep += "DEBIAN_FRONTEND"' > /etc/sudoers.d/debian_frontend \
    && rm -rf /var/lib/apt/lists/*

# install_deps.sh installs: cmake, clang, just, eigen3, zmq, msgpack, gtest,
# ONNX Runtime 1.16.3, and other build requirements.
WORKDIR /opt/gear_sonic_deploy
COPY modules/gwbc/gear_sonic_deploy/scripts/install_deps.sh scripts/install_deps.sh
# touch /etc/nv_tegra_release ensures install_deps.sh Jetson detection works
# inside BuildKit — the file from the base image sits in a read-only snapshot
# layer and [[ -f ]] may not see it; touching it writes it into the current layer.
RUN touch /etc/nv_tegra_release \
    && chmod +x scripts/install_deps.sh \
    && scripts/install_deps.sh

COPY modules/gwbc/gear_sonic_deploy/ .

# Source setup_env.sh for LD_LIBRARY_PATH / CMake paths, then build.
RUN bash -c "source scripts/setup_env.sh && just build Release"

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM ${TENSORRT_RUNTIME_IMAGE}

USER root
ENV DEBIAN_FRONTEND=noninteractive

# Runtime shared libs the binary dynamically links against. The leaner TRT 10.3
# runtime image does not include libzmq; install it explicitly here.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libzmq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy compiled binary, ONNX Runtime libs, and Unitree-bundled Cyclone DDS libs
# from builder stage. DDS libs ship with unitree_sdk2 (not in apt) so copy them
# verbatim and refresh the linker cache.
COPY --from=builder /opt/gear_sonic_deploy/target/release/g1_deploy_onnx_ref \
                    /usr/local/bin/g1_deploy_onnx_ref
COPY --from=builder /opt/onnxruntime/lib /opt/onnxruntime/lib
COPY --from=builder /opt/gear_sonic_deploy/thirdparty/unitree_sdk2/thirdparty/lib/aarch64/libddsc.so* \
                    /opt/gear_sonic_deploy/thirdparty/unitree_sdk2/thirdparty/lib/aarch64/libddscxx.so* \
                    /usr/local/lib/
RUN ldconfig

ENV LD_LIBRARY_PATH=/opt/onnxruntime/lib:/usr/local/lib:${LD_LIBRARY_PATH}

COPY docker/gear-sonic/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
