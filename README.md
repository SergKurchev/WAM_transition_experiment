# WAM Transition Experiment — Isaac Sim + GEAR-SONIC + WAM Pipeline

Fine-tune World Action Models (WAMs) on Unitree G1, test transfer to UBTech Walker Tienkung.

**Goal:** Train WAM on (G1, Activity 1+2), evaluate zero-shot transfer on (UBTech, Activity 2)  
**Models:** UnifoLM-WMA-0, EVA  
**Environment:** Isaac Sim 5.1 + GEAR-SONIC WBC + ROS2 DDS

---

## 🔴 CRITICAL ARCHITECTURE (Fixed 23 May 2026)

```
Isaac Sim (physics)
  ├─ rt/lowstate (50D joint state) [10 Hz]
  │
sim-ros2-bridge ← THIS IS CRITICAL
  ├─ Transforms to 994D observation vector
  │   (joint pos/vel history, IMU, token_state, actions)
  │
GEAR-SONIC (WBC policy)
  ├─ Reads 994D observations
  ├─ Generates rt/lowcmd (29-DOF joint targets)
  │
Isaac Sim (applies joint targets, re-publishes state)
  │
  ↑ Loop 10 Hz
  │
WAM (your model)
  ├─ Reads rt/lowstate
  ├─ Publishes rt/run_command/cmd (velocity goals)
  │
GEAR-SONIC
  └─ Converts velocity → stable 29-DOF motion via WBC
```

**Without sim-ros2-bridge:** GEAR-SONIC crashes with `observation dimension (0) ≠ 994`

---

## Prerequisites

### On Server (`x32-techgov-GPU-01`)

```bash
ssh x32-techgov-GPU-01
cd /root/skurchev/workspace

# 1. Clone mws-dimos (REQUIRED — contains Isaac Sim, GEAR-SONIC, ROS2 bridge, WBC)
git clone --branch feat/real-transfer \
  https://github.com/MWS-Physical-AI/mws-dimos.git

# 2. Verify structure
ls -d mws-dimos wam-stack
# Should show: mws-dimos  wam-stack
```

### Images Already Built on GPU-01

These are pre-built from mws-dimos and will be used directly:
- `mws-dimos/sim-isaac:unitree-lab-5.1`
- `mws-sim-gear-sonic-policy:latest`
- `mws-sim-ros2-bridge:latest`

**First-time setup on another server:**
```bash
cd /root/skurchev/workspace/mws-dimos
docker compose -f deploy/sim/ros2/compose.yml build  # ~20 min
```

---

## Quick Start (Daily Use)

### Terminal 1: Start Stack

```bash
ssh x32-techgov-GPU-01
cd /root/skurchev/workspace/wam-stack
docker compose up -d
```

**Verify all 4 containers are healthy:**
```bash
docker compose ps
# Expected:
# wam-isaac-sim      running (healthy)
# wam-gear-sonic     running (healthy)  [after ~30s]
# wam-ros2-bridge    running (healthy)  [after ~15s]
# wam-inference      running
```

### Terminal 2: noVNC Tunnel (Local Machine)

```bash
ssh -N -L 6081:localhost:6080 -p 2221 x32-techgov-GPU-01
```

**Open browser:** `http://localhost:6081`

You should see:
1. Robot balancing/moving in Isaac Sim (after 30–60s warmup)
2. GEAR-SONIC generating smooth motion from velocity goals
3. WAM agent publishing commands

---

## Directory Structure & Dependencies

```
/root/skurchev/workspace/
├── mws-dimos/                    ← EXTERNAL (do NOT modify)
│   ├── deploy/sim/ros2/
│   │   ├── compose.yml           ← Reference (we don't use this directly)
│   │   ├── config.toml           ← DDS bridge config (referenced by our compose)
│   │   └── entrypoints/
│   │       └── launch_g1.sh      ← Isaac Sim entrypoint
│   ├── modules/gwbc/
│   │   ├── gear_sonic_deploy/    ← ONNX models + WBC config
│   │   └── ...
│   └── docker/
│       └── images/
│           ├── ros2-bridge/Dockerfile
│           └── ...
│
└── wam-stack/                    ← YOUR REPO (this one)
    ├── compose.yml               ← MAIN: uses mws-dimos images + builds wam-inference
    ├── docker/
    │   └── wam/Dockerfile        ← Only image built from wam-stack
    ├── src/
    │   ├── main.py               ← WAM control loop (10 Hz)
    │   ├── dds_interface.py      ← DDS subscriber/publisher
    │   └── models/
    │       ├── unifolm.py        ← TODO: implement
    │       └── eva.py            ← TODO: implement
    └── scripts/
        ├── deploy.sh             ← Deploy to server (local machine)
        ├── sync.sh               ← Rsync to server
        └── ...
```

### Key File Roles

| File | From | Purpose |
|------|------|---------|
| `deploy/sim/ros2/config.toml` | mws-dimos | DDS bridge config (defines observation streams) |
| `modules/gwbc/gear_sonic_deploy/` | mws-dimos | GEAR-SONIC ONNX models + motion library |
| `src/main.py` | wam-stack | Your WAM agent (reads lowstate, publishes commands) |
| `docker/wam/Dockerfile` | wam-stack | WAM container (Python 3.10 + torch + cyclonedds 0.10.2) |

---

## Troubleshooting

### "observation dimension (0) ≠ 994" (GEAR-SONIC crash)

**Cause:** ros2-bridge not running or config.toml path wrong

**Fix:**
```bash
docker compose ps | grep ros2-bridge
# Should see: wam-ros2-bridge running (healthy)

docker logs wam-ros2-bridge | tail -20
# Should see: "bridge ready" or observation stream subscriptions
```

If ros2-bridge is not running:
```bash
docker compose up -d ros2-bridge
docker logs wam-ros2-bridge
```

### "Cannot connect to http://localhost:6081"

**Cause:** SSH tunnel died or not started

**Fix:**
```bash
# Local machine — kill old tunnel
killall ssh 2>/dev/null

# Restart tunnel
ssh -N -L 6081:localhost:6080 -p 2221 x32-techgov-GPU-01
```

### Robot lying down / not moving

**Cause:** GEAR-SONIC initializing (normal for 30–60s after startup)

**Check:**
```bash
docker logs wam-gear-sonic | grep -E "Running|ready"
```

**If stuck:**
```bash
docker compose restart wam-gear-sonic
```

### Isaac Sim crashes / container exits

**Cause:** Usually DISPLAY or shader compilation issue

**Check:**
```bash
docker logs wam-isaac-sim | tail -50
```

**Fix:**
```bash
docker compose down
docker compose up -d wam-isaac-sim
docker logs -f wam-isaac-sim  # wait 2–5 min
```

---

## Running the Stack

### Full Restart
```bash
cd /root/skurchev/workspace/wam-stack
docker compose down
docker compose up -d
```

### Restart One Service
```bash
docker compose restart wam  # just WAM agent
docker compose restart wam-gear-sonic
# etc.
```

### View Logs
```bash
docker compose logs -f wam              # WAM agent (10 Hz loop)
docker compose logs -f wam-gear-sonic   # GEAR-SONIC inference + errors
docker compose logs -f wam-isaac-sim    # Isaac Sim physics + rendering
docker compose logs -f wam-ros2-bridge  # DDS bridge (observation transforms)
```

---

## Modifying WAM Code

### Hot-reload (src/ is bind-mounted)

```bash
# Edit locally
vim src/main.py

# On server
cd /root/skurchev/workspace/wam-stack
docker compose restart wam

# Verify changes
docker logs -f wam
```

### Switching Models

```bash
WAM_MODEL=unifolm docker compose restart wam
# OR
WAM_MODEL=eva WAM_CHECKPOINT=/path/to/ckpt docker compose restart wam
```

### Installing Dependencies

If you add packages to `src/`:
```bash
# On server, update WAM container
cd /root/skurchev/workspace/wam-stack
docker compose up -d --build wam
```

---

## DDS Topics Reference

| Topic | Publisher | Subscriber | Rate | Format |
|-------|-----------|------------|------|--------|
| `rt/lowstate` | Isaac Sim | GEAR-SONIC, WAM | 10 Hz | 29-DOF state (joint pos/vel/torque + IMU) |
| `rt/run_command/cmd` | WAM | GEAR-SONIC | 10 Hz | JSON `[vx, vy, wz, body_height]` |
| `rt/lowcmd` | GEAR-SONIC | Isaac Sim | 10 Hz | 29-DOF joint targets |

DDS domain ID: `42` (configured in compose.yml)  
DDS interface: `localhost` (loopback, no external network)

---

## Project Status

- ✅ Isaac Sim + GEAR-SONIC + ROS2 bridge working (verified 23 May)
- ✅ DDS communication (rt/lowstate, rt/run_command/cmd, rt/lowcmd)
- ⏳ UnifoLM-WMA-0 model integration (TODO)
- ⏳ EVA model integration (TODO)
- ⏳ Dataset collection (~400 samples per activity)
- ⏳ Fine-tuning on G1
- ⏳ Transfer evaluation on UBTech

---

## Support / Questions

See `CLAUDE.md` for architecture deep-dive and `QUICKSTART.md` for 5-minute getting started.

For setup issues on a new server, refer to [SETUP.md](SETUP.md).
