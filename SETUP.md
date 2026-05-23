# Complete Setup Guide for wam-stack

Follow this guide **once per server** to set up the full WAM pipeline.

---

## Prerequisites

- **Hardware:** NVIDIA GPU with CUDA 12.2+ (tested on A100)
- **OS:** Ubuntu 20.04 or 22.04 with Docker + NVIDIA Docker runtime
- **Network:** SSH access to server, port 2221 (or your config)
- **Disk:** ~100 GB free (Isaac Sim + Docker images)

---

## Step 1: Clone Repositories

```bash
ssh x32-techgov-GPU-01
cd /root/skurchev/workspace

# Clone mws-dimos (REQUIRED—contains Isaac Sim, GEAR-SONIC, WBC, ROS2 bridge)
git clone --branch feat/real-transfer \
  https://github.com/MWS-Physical-AI/mws-dimos.git

# Clone wam-stack (this repo)
git clone https://github.com/SergKurchev/wam-stack.git

# Verify
ls -ld mws-dimos wam-stack
```

**Critical:** Both must be in the same parent directory (`/root/skurchev/workspace/`)

---

## Step 2: Verify Docker & GPU

```bash
docker --version
docker run --rm --gpus all nvidia/cuda:12.2.0-runtime-ubuntu22.04 nvidia-smi
```

---

## Step 3: Build mws-dimos Images (First-time ~20 min)

```bash
cd /root/skurchev/workspace/mws-dimos
docker compose -f deploy/sim/ros2/compose.yml build
```

Builds:
- `mws-dimos/sim-isaac:unitree-lab-5.1` (Isaac Sim)
- `mws-sim-gear-sonic-policy:latest` (GEAR-SONIC)
- `mws-sim-ros2-bridge:latest` (ROS2/DDS bridge)

---

## Step 4: Build WAM Container (First-time ~10 min)

```bash
cd /root/skurchev/workspace/wam-stack
docker compose build wam
```

Compiles CycloneDDS 0.10.2 from source (no binary wheels).

---

## Step 5: Test Full Stack

```bash
cd /root/skurchev/workspace/wam-stack
docker compose up -d
sleep 30
docker compose ps
```

**Expected:** All containers healthy within 30s

---

## Step 6: Visual Monitoring

### Local Machine:
```bash
ssh -N -L 6081:localhost:6080 -p 2221 x32-techgov-GPU-01
# Open: http://localhost:6081
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

## Troubleshooting

### "observation dimension (0) ≠ 994"

ROS2 bridge not running:
```bash
docker compose restart wam-ros2-bridge
docker logs wam-ros2-bridge | tail
```

### Isaac Sim stuck

```bash
docker compose down
docker compose up -d
sleep 120
```

### GPU not detected

```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-runtime-ubuntu22.04 nvidia-smi
```

---

## Updating

**WAM code (safe):**
```bash
git pull origin main
docker compose restart wam
```

**mws-dimos (careful):**
```bash
cd mws-dimos
git fetch origin feat/real-transfer
git diff HEAD origin/feat/real-transfer
git pull origin feat/real-transfer
docker compose -f deploy/sim/ros2/compose.yml build
```

---

See `README.md` for daily usage and `QUICKSTART.md` for 5-minute intro.
