# WAM Transition Experiment

Isaac Sim + GEAR-SONIC + WAM model pipeline for cross-robot skill transfer.

**Robots:** Unitree G1, UBTech Walker Tienkung  
**Models:** UnifoLM-WMA-0, EVA  
**Goal:** fine-tune on (G1, Act1), (G1, Act2), (UBTech, Act1) → evaluate transfer on (UBTech, Act2)

---

## Architecture

```
WAM model (UnifoLM / EVA)
      │  rt/run_command/cmd  [vx, vy, wz, body_height]
      ▼
GEAR-SONIC  ←──── rt/lowstate ────  Isaac Sim
      │                                  ▲
      └──────── rt/lowcmd ───────────────┘

Visual output: Xvfb → x11vnc → noVNC → your browser (port 6080)
```

All communication over DDS (CycloneDDS, loopback interface).  
No DimOS, no LCM, no bridge.

---

## Server

| Alias | IP | Port | GPU |
|-------|----|------|-----|
| `x32-techgov-GPU-01` | 176.109.83.84 | 2221 | A100 #0 (primary) |
| `x32-techgov-GPU-02` | 176.109.83.84 | 2222 | A100 #1 |

---

## First-time Setup (once per server)

### 1. Clone with submodules

```bash
ssh x32-techgov-GPU-01
cd /root/skurchev/workspace

git clone --recurse-submodules \
  https://github.com/SergKurchev/WAM_transition_experiment.git wam-stack
```

If you cloned without `--recurse-submodules`:
```bash
cd wam-stack
git submodule update --init --recursive
```

### 2. Build the heavy Isaac Sim base image

Only needed once. Takes **20–40 minutes**. Run inside `tmux`.

```bash
cd /root/skurchev/workspace/wam-stack
tmux new -s base-build

docker build \
  -f docker/isaac-sim/Dockerfile.base \
  -t mws-dimos/isaac-sim-base:unitree-lab-5.1 \
  .
```

Check if it's already built (skip if yes):
```bash
docker images | grep isaac-sim-base
```

### 3. Generate the G1 robot USD (if `g1_29dof_base.usd` is missing)

The large robot geometry file is not in git. Generate it once:

```bash
cd /root/skurchev/workspace/wam-stack
# TODO: run prepare script (requires Isaac Sim env)
# scripts/isaac/prepare_g1_robot_usd.sh
```

> Until this is automated, copy `g1_29dof_base.usd` from the existing server path:
> ```bash
> cp /root/skurchev/workspace/mws-dimos/assets/robots/g1/configuration/g1_29dof_base.usd \
>    /root/skurchev/workspace/wam-stack/assets/robots/g1/configuration/
> ```

---

## Launch

Always work in `tmux` so the session survives SSH disconnects.

```bash
ssh x32-techgov-GPU-01
tmux new -s sim        # or: tmux attach -t sim

cd /root/skurchev/workspace/wam-stack
```

### First launch (build WAM image + start)

```bash
./scripts/start.sh --build \
  --scene /root/skurchev/workspace/assets/office_demo.usdz
```

### Subsequent launches

```bash
./scripts/start.sh \
  --scene /root/skurchev/workspace/assets/office_demo.usdz
```

### Startup sequence

```
gear-sonic ──┐
             ├─► (both healthy) ─► wam starts inference
isaac-sim  ──┘
```

Isaac Sim writes `/tmp/isaac_ready` after warm-up (**~2–5 min**).  
GEAR-SONIC writes `/tmp/gear-sonic-ready` after TRT engine load (**~1–3 min**, cached after first run).  
`wam` container starts only after both are healthy.

---

## Visual Monitoring (noVNC)

Isaac Sim renders to a virtual display inside the container.  
You access it through a browser on your local machine.

### Step 1 — open SSH tunnel (local machine, new terminal)

```bash
# From anywhere on your local machine:
ssh -N -L 6080:localhost:6080 x32-techgov-GPU-01

# Or use the helper script (from CoRL2026/wam-stack/):
./scripts/view.sh
```

Keep this terminal open while you're watching.

### Step 2 — open browser

```
http://localhost:6080
```

You'll see the Isaac Sim viewport with the G1 robot in the office scene.  
The view is live — physics and robot motion update in real time.

### Switch to GPU-02

```bash
./scripts/view.sh x32-techgov-GPU-02
```

---

## Monitoring Logs

```bash
# All containers at once
docker compose -f compose.yml logs -f

# Individual containers
docker compose -f compose.yml logs -f wam
docker compose -f compose.yml logs -f isaac-sim
docker compose -f compose.yml logs -f gear-sonic

# Last 100 lines + follow
docker compose -f compose.yml logs --tail=100 -f wam
```

### Container health

```bash
docker compose -f compose.yml ps
```

Expected output once everything is up:
```
NAME                STATUS
wam-gear-sonic      running (healthy)
wam-isaac-sim       running (healthy)
wam-inference       running
```

---

## Updating WAM Code (hot-reload)

Edit `src/` locally → sync to server → restart only the `wam` container.  
Isaac Sim and GEAR-SONIC keep running.

```bash
# 1. Edit src/ on local machine, then sync:
./scripts/sync.sh x32-techgov-GPU-01:/root/skurchev/workspace

# 2. On server — restart WAM only (~5 seconds):
ssh x32-techgov-GPU-01
cd /root/skurchev/workspace/wam-stack
docker compose -f compose.yml restart wam

# 3. Watch logs:
docker compose -f compose.yml logs -f wam
```

### Switching models

```bash
# Run with UnifoLM
WAM_MODEL=unifolm WAM_CHECKPOINT=/path/to/checkpoint \
  docker compose -f compose.yml restart wam

# Run with EVA
WAM_MODEL=eva WAM_CHECKPOINT=/path/to/checkpoint \
  docker compose -f compose.yml restart wam

# Stub (walks forward, no model needed)
WAM_MODEL=stub docker compose -f compose.yml restart wam
```

---

## Stop / Reset

```bash
# Stop everything
./scripts/stop.sh

# Stop and remove volumes
docker compose -f compose.yml down -v

# Rebuild WAM image (after changing Dockerfile)
./scripts/start.sh --build-only wam
```

---

## Debug: exec into a container

```bash
# WAM container — inspect Python env, test imports
docker compose -f compose.yml exec wam bash

# Inside: test DDS
python -c "import unitree_sdk2py; print('DDS ok')"

# Isaac Sim container — inspect scene, check logs
docker compose -f compose.yml exec isaac-sim bash
```

---

## Pull latest code

```bash
# On server:
ssh x32-techgov-GPU-01
cd /root/skurchev/workspace/wam-stack

git pull
git submodule update --remote modules/gwbc   # update GEAR-SONIC if needed

# Restart WAM to pick up src/ changes:
docker compose -f compose.yml restart wam
```

---

## Project Structure

```
wam-stack/
├── compose.yml                  # 3 services: isaac-sim, gear-sonic, wam
├── docker/
│   ├── isaac-sim/
│   │   ├── Dockerfile.base      # heavy base: Isaac Sim 5.1 + Isaac Lab 2.3.2 (build once)
│   │   ├── Dockerfile           # thin runtime layer (noVNC, pyzmq)
│   │   └── entrypoint.sh        # Xvfb + x11vnc + websockify → port 6080
│   ├── gear-sonic/
│   │   └── Dockerfile           # C++ TensorRT inference binary
│   └── wam/
│       └── Dockerfile           # CUDA 12.2 + Python 3.12 + torch + DDS
├── sim/isaac/                   # Isaac Sim runtime (Python)
│   ├── g1_sim.py                # main loop: InteractiveScene + SimulationContext
│   ├── dds_bridge.py            # rt/lowstate publisher, rt/lowcmd subscriber
│   ├── startup_support.py       # floating-base PD hold during init
│   └── launch_g1.py             # AppLauncher entrypoint
├── scripts/
│   ├── start.sh                 # launch stack
│   ├── stop.sh                  # docker compose down
│   ├── sync.sh                  # rsync to server
│   ├── view.sh                  # SSH tunnel → http://localhost:6080
│   └── isaac/launch_g1.sh       # conda activate + python launch_g1.py
├── assets/robots/g1/            # G1 USD assets (g1_29dof_base.usd not in git — generate)
├── modules/gwbc/                # submodule: GR00T-Kimodo (GEAR-SONIC + robot model data)
└── src/                         # YOUR WAM code (hot-reload)
    ├── main.py                  # control loop
    ├── dds_interface.py         # DDS subscribe/publish
    └── models/
        ├── unifolm.py           # UnifoLM-WMA-0 adapter
        └── eva.py               # EVA adapter
```

---

## DDS Interface Reference

| Topic | Direction | Type | Rate | Content |
|-------|-----------|------|------|---------|
| `rt/lowstate` | Isaac Sim → WAM | `unitree_hg.msg.dds_.LowState_` | 50–200 Hz | 29-DOF joint pos/vel/torque + IMU |
| `rt/lowcmd` | GEAR-SONIC → Isaac Sim | `unitree_hg.msg.dds_.LowCmd_` | 50 Hz | 29-DOF joint targets |
| `rt/run_command/cmd` | WAM → GEAR-SONIC | `std_msgs.msg.dds_.String_` | 10 Hz | JSON `[vx, vy, wz, body_height]` |

Your WAM reads `rt/lowstate`, does inference, publishes `rt/run_command/cmd`.  
GEAR-SONIC converts velocity commands into stable 29-DOF joint targets.
