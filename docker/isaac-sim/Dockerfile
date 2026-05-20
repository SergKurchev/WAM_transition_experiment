ARG BASE_IMAGE=mws-dimos/isaac-sim-base:unitree-lab-5.1
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        xvfb x11vnc novnc websockify iproute2 procps zenity \
    && rm -rf /var/lib/apt/lists/*

RUN conda run --no-capture-output -n unitree_sim_env \
        pip install --no-cache-dir pyzmq==27.0.0

ENV PYTHONPATH=/workspace/wam-stack:/workspace/gwbc:/workspace/unitree_sdk2py_src

WORKDIR /workspace/wam-stack

CMD ["/bin/bash", "/workspace/wam-stack/scripts/isaac/launch_g1.sh"]
