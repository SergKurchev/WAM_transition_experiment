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

Visual output: Xvfb → x11vnc → noVNC → your browser
```

All communication over CycloneDDS 0.10.2, loopback interface.  
No DimOS, no LCM, no bridge.

---

## Server

| Alias | IP | Port | GPU |
|-------|----|------|-----|
| `x32-techgov-GPU-01` | 176.109.83.84 | 2221 | A100 #0 |
| `x32-techgov-GPU-02` | 176.109.83.84 | 2222 | A100 #1 ← **wam-stack lives here** |

---

## Quick Deploy (daily use)

Run from your local machine, from anywhere in the repo:

```bash
# Sync code + start all containers + show ports
bash scripts/deploy.sh

# Same, but rebuild WAM Docker image first (after Dockerfile changes)
bash scripts/deploy.sh --build

# Reset after robot fell
bash scripts/deploy.sh --reset
```

Then open a **new terminal** and run the tunnel:

```bash
ssh -N -L 6181:localhost:6180 x32-techgov-GPU-02
```

Open browser: **http://localhost:6181**

---

## First-time Setup (once per server)

### 1. Clone with submodules

```bash
ssh x32-techgov-GPU-02
cd /root/skurchev/workspace

git clone --recurse-submodules \
  https://github.com/SergKurchev/WAM_transition_experiment.git wam-stack
```

### 2. Build the WAM Docker image (first time only, ~10 min)

```bash
bash scripts/deploy.sh --build
```

This builds CycloneDDS 0.10.x from source (required — no binary wheels exist for
cyclonedds 0.10.2 + Python 3.10) and installs PyTorch 2.7 + CUDA 12.8.

> **Why 0.10.2?** GEAR-SONIC uses CycloneDDS 0.10.x. Using 11.x in WAM crashes
> GEAR-SONIC with a segfault in `ddsi_xt_type_init_impl` during DDS discovery.

### 3. Server-only binary files (already in place on GPU-02)

These large files are NOT in git. They live at:

```
modules/gwbc/gear_sonic_deploy/
├── policy/release/
│   ├── model_encoder.onnx        (50 MB)
│   └── model_decoder.onnx        (40 MB)
├── planner/target_vel/V2/
│   └── planner_sonic.onnx
└── thirdparty/unitree_sdk2/thirdparty/lib/x86_64/
    ├── libddsc.so / libddsc.so.0
    └── libddscxx.so / libddscxx.so.0
```

If they're missing, copy from mws-dimos on the same server:
```bash
cp -r /root/skurchev/workspace/mws-dimos/modules/gwbc/gear_sonic_deploy/policy \
      /root/skurchev/workspace/wam-stack/modules/gwbc/gear_sonic_deploy/
```

---

## Visual Monitoring (noVNC)

Isaac Sim renders to a virtual display (Xvfb) inside its container.
noVNC serves it over port 6180 on the server.

### Open the tunnel (local machine, new terminal)

```bash
ssh -N -L 6181:localhost:6180 x32-techgov-GPU-02
```

Keep this terminal open. Then open: **http://localhost:6181**

> Port 6180 is often busy on Windows. Use 6181 locally (maps to 6180 on server).

### Verify the tunnel works

```bash
curl -s http://localhost:6181 | head -3
# Expected: <!DOCTYPE html> ...
```

### Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `bind: Permission denied` | Local port 6180 is busy | Use 6181: `ssh -N -L 6181:localhost:6180 ...` |
| `channel: open failed: connect failed` | Stack not running | Run `bash scripts/deploy.sh` first |
| noVNC loads, black screen | Isaac Sim still warming up | Wait 2–5 min |
| Robot is lying down | GEAR-SONIC lost connection briefly | Run `bash scripts/deploy.sh --reset` |

---

## Updating WAM Code (hot-reload)

`src/` is bind-mounted into the container — changes are live after a restart.

```bash
# Edit src/*.py locally, then:
bash scripts/deploy.sh          # syncs and restarts everything

# Or if only WAM changed and stack is already up:
scp -P 2222 src/main.py root@176.109.83.84:/root/skurchev/workspace/wam-stack/src/
ssh -p 2222 root@176.109.83.84 "cd /root/skurchev/workspace/wam-stack && docker compose restart wam"
```

### Switching models

```bash
# Set env in compose.yml or pass on restart:
ssh -p 2222 root@176.109.83.84 "
  cd /root/skurchev/workspace/wam-stack
  WAM_MODEL=unifolm WAM_CHECKPOINT=/path/to/ckpt docker compose restart wam
"
```

---

## Monitoring

```bash
# All containers
ssh -p 2222 root@176.109.83.84 "cd /root/skurchev/workspace/wam-stack && docker compose logs -f"

# Individual
docker compose logs -f wam
docker compose logs -f isaac-sim
docker compose logs -f gear-sonic

# Health
docker compose ps
```

Expected healthy state:
```
NAME               STATUS
wam-gear-sonic     running (healthy)
wam-isaac-sim      running (healthy)
wam-inference      running
```

---

## Project Structure

```
wam-stack/
├── compose.yml                  # 3 services: isaac-sim, gear-sonic, wam
├── docker/
│   ├── isaac-sim/
│   │   ├── Dockerfile           # Isaac Sim 5.1 + Isaac Lab 2.3.2 + noVNC
│   │   └── entrypoint.sh        # Xvfb + x11vnc + websockify → port 6180
│   ├── gear-sonic/
│   │   ├── Dockerfile           # FROM mws-sim-gear-sonic-policy:latest
│   │   └── entrypoint.sh        # fixes x86_64 DDS lib path, enables lo multicast
│   └── wam/
│       └── Dockerfile           # CUDA 12.2 + Python 3.10 + torch + cyclonedds==0.10.2
├── scripts/
│   ├── deploy.sh                # ← MAIN SCRIPT: sync + start + show ports
│   ├── start.sh                 # server-side launch (used by deploy.sh)
│   ├── stop.sh                  # docker compose down
│   ├── sync.sh                  # rsync only (no start)
│   └── view.sh                  # SSH tunnel helper
├── modules/gwbc/                # submodule: GR00T-Kimodo (GEAR-SONIC + robot model data)
└── src/                         # WAM code (bind-mounted, hot-reload)
    ├── main.py                  # control loop: state → model → DDS command (10 Hz)
    ├── dds_interface.py         # rt/lowstate subscriber + rt/run_command/cmd publisher
    └── models/
        ├── unifolm.py           # UnifoLM-WMA-0 adapter (stub — TODO)
        └── eva.py               # EVA adapter (stub — TODO)
```

---

## DDS Interface Reference

| Topic | Direction | Type | Rate | Content |
|-------|-----------|------|------|---------|
| `rt/lowstate` | Isaac Sim → WAM | `unitree_hg.msg.dds_.LowState_` | 50 Hz | 29-DOF joint pos/vel/torque + IMU |
| `rt/lowcmd` | GEAR-SONIC → Isaac Sim | `unitree_hg.msg.dds_.LowCmd_` | 50 Hz | 29-DOF joint targets |
| `rt/run_command/cmd` | WAM → GEAR-SONIC | `std_msgs.msg.dds_.String_` | 10 Hz | JSON `[vx, vy, wz, body_height]` |

WAM reads `rt/lowstate`, runs inference, publishes `rt/run_command/cmd`.  
GEAR-SONIC converts velocity commands into stable 29-DOF joint targets.
