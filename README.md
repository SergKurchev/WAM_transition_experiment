# WAM Transition Experiment — Isaac Sim + GEAR-SONIC + WAM Pipeline

Fine-tune World Action Models (WAMs) on Unitree G1, test transfer to UBTech Walker Tienkung.

**Goal:** Train WAM on (G1, Activity 1+2), evaluate zero-shot transfer on (UBTech, Activity 2)  
**Models:** UnifoLM-WMA-0, EVA  
**Environment:** Isaac Sim 5.1 + GEAR-SONIC WBC + ROS2 DDS

**Current Status:** Stable commit: `5615141` (main branch) — Full stack verified working (23 May 2026)

---

## 🚀 Quick Start (5 minutes)

### Prerequisites (on local machine)

- Git access to both: `https://github.com/SergKurchev/wam-stack.git` (this repo) and `https://github.com/MWS-Physical-AI/mws-dimos.git`
- SSH key to `x32-techgov-GPU-01` (port 2221)
- `rsync` installed (Linux/Mac) or `scp` (Windows)

### Workflow

**Terminal 1: Deploy stack**
```bash
# 1. Clone both repos (if not already cloned)
cd ~/workspace  # or your workspace dir
git clone https://github.com/SergKurchev/wam-stack.git
git clone --branch feat/real-transfer https://github.com/MWS-Physical-AI/mws-dimos.git

# 2. Verify structure
ls -d wam-stack mws-dimos
# Output: mws-dimos  wam-stack

# 3. Sync code to server and start stack
cd wam-stack
bash scripts/deploy.sh
# Expected: All 4 containers healthy within 2 minutes
```

**Terminal 2: Visual monitoring (local machine)**
```bash
# After deploy.sh completes, open separate terminal
ssh -N -L 6081:localhost:6080 -p 2221 x32-techgov-GPU-01

# Then open browser: http://localhost:6081
# You should see robot balancing in Isaac Sim
```

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

## Directory Structure & Dependencies

```
~/workspace/
├── mws-dimos/                    ← EXTERNAL (feat/real-transfer branch)
│   ├── deploy/sim/ros2/
│   │   ├── compose.yml           ← Reference (we don't use directly)
│   │   ├── config.toml           ← DDS bridge config (referenced by our compose)
│   │   └── entrypoints/
│   │       └── launch_g1.sh      ← Isaac Sim entrypoint
│   ├── modules/gwbc/
│   │   ├── gear_sonic_deploy/    ← ONNX models + WBC config
│   │   └── ...
│   └── docker/
│       └── images/
│           ├── isaac-sim/Dockerfile
│           ├── ros2-bridge/Dockerfile
│           └── ...
│
└── wam-stack/                    ← THIS REPO (your code)
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
        ├── deploy.sh             ← ← MAIN DEPLOYMENT SCRIPT
        ├── sync.sh               ← Sync code to server (uses rsync)
        └── ...
```

### Key Files

| File | From | Purpose |
|------|------|---------|
| `deploy/sim/ros2/config.toml` | mws-dimos | DDS bridge config (defines observation streams) |
| `modules/gwbc/gear_sonic_deploy/` | mws-dimos | GEAR-SONIC ONNX models + motion library |
| `src/main.py` | wam-stack | Your WAM agent (reads lowstate, publishes commands) |
| `docker/wam/Dockerfile` | wam-stack | WAM container (Python 3.10 + torch + cyclonedds 0.10.2) |

---

## Daily Usage

### Start stack
```bash
cd ~/workspace/wam-stack
bash scripts/deploy.sh [--build] [--reset]

# --build : rebuild wam-inference image (if dependencies changed)
# --reset : stop and restart all containers
```

### Monitor in real-time
```bash
docker compose logs -f wam              # WAM agent
docker compose logs -f wam-gear-sonic   # GEAR-SONIC inference
docker compose logs -f wam-isaac-sim    # Isaac Sim physics
docker compose logs -f wam-ros2-bridge  # DDS bridge
```

### Modify WAM code (hot-reload)
```bash
# Edit locally
vim src/main.py

# Restart on server (src/ is bind-mounted)
docker compose restart wam

# Verify changes
docker logs -f wam
```

### Stop stack
```bash
docker compose down
```

---

## DDS Topics Reference

| Topic | Publisher | Subscriber | Rate | Format |
|-------|-----------|------------|------|--------|
| `rt/lowstate` | Isaac Sim | GEAR-SONIC, WAM | 10 Hz | 29-DOF state (joint pos/vel/torque + IMU) |
| `rt/run_command/cmd` | WAM | GEAR-SONIC | 10 Hz | JSON `[vx, vy, wz, body_height]` |
| `rt/lowcmd` | GEAR-SONIC | Isaac Sim | 10 Hz | 29-DOF joint targets |

**DDS domain ID:** `42` (configured in compose.yml)  
**DDS interface:** `lo` (loopback, no external network)

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

### Cannot connect to http://localhost:6081

**Cause:** SSH tunnel died or not started

**Fix:**
```bash
# Local machine — kill old tunnel
killall ssh 2>/dev/null

# Restart tunnel
ssh -N -L 6081:localhost:6080 -p 2221 x32-techgov-GPU-01
```

### Robot not moving / stuck at startup

**Cause:** GEAR-SONIC initializing (normal for 30–60s after startup)

**Check:**
```bash
docker logs wam-gear-sonic | grep -E "Running|ready"
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

## Dependency Versions (Stable)

| Component | Version |
|-----------|---------|
| Isaac Sim | 5.1 |
| GEAR-SONIC | feat/real-transfer commit |
| CycloneDDS | 0.10.2 |
| PyTorch | 2.7.0 |
| Python | 3.10 |
| CUDA | 12.2 |

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

## Architecture Deep-dive

See `CLAUDE.md` for:
- Why sim-ros2-bridge is critical (50D→994D transformation)
- DDS data flow and timing
- mws-dimos dependency model
- Pre-approved Claude actions

See `SETUP.md` for:
- Complete first-time server setup
- Dependency versions
- Troubleshooting for each component
- Update procedures

---

## Support

- Issues: Check troubleshooting section above
- Questions: See `CLAUDE.md` for architecture details
- Feedback: Report at https://github.com/SergKurchev/wam-stack/issues (if public)
