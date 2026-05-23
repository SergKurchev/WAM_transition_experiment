# UnifoLM-WMA-0 Deployment Guide

Integrate and deploy UnifoLM-WMA-0 world action model for G1 arm control.

## Quick Start

### 1. Test Locally (no server needed)

```bash
cd wam-stack
python scripts/test_unifolm.py
```

Expected output:
```
[TEST 1] Initialize UnifoLM in test mode (arm clapping demo)
[PASS] Model initialized successfully
  Device: cpu
  Mode: TEST (arm clapping demo)

[TEST 2] Create mock robot state (G1 29-DOF)
[PASS] Mock state created

[TEST 3] Run 10 seconds of inference (100 steps at 10 Hz)
...
[PASS] All tests passed!
```

### 2. Deploy to Server (with test mode)

```bash
# Terminal 1: Deploy stack with default stub model
cd wam-stack
bash scripts/deploy.sh

# Or: Deploy with UnifoLM in test mode (arm clapping demo)
WAM_MODEL=unifolm bash scripts/deploy.sh

# Or: With --build flag (if dependencies changed)
WAM_MODEL=unifolm bash scripts/deploy.sh --build
```

### 3. Monitor on Server

```bash
# Terminal 2: Watch WAM logs
ssh -p 2221 root@176.109.83.84
docker logs -f wam-inference

# Look for: [UnifoLM demo] step=... (arm clapping output)
```

### 4. Visual Monitoring

```bash
# Terminal 3 (local machine): Port forwarding
ssh -N -L 6081:localhost:6080 -p 2221 x32-techgov-GPU-01

# Then open browser: http://localhost:6081
# You should see G1 robot in Isaac Sim
```

---

## Load Real Model

Once you have a UnifoLM checkpoint, you can load it instead of test mode:

### Option A: From Local Checkpoint

```bash
# Copy checkpoint to server
scp -P 2221 /path/to/unifolm_checkpoint.pt root@176.109.83.84:/root/skurchev/workspace/wam-stack/checkpoints/

# Deploy with checkpoint
WAM_MODEL=unifolm WAM_CHECKPOINT=/workspace/wam/checkpoints/unifolm_checkpoint.pt bash scripts/deploy.sh --build
```

### Option B: From HuggingFace Hub

```bash
# Deploy with HuggingFace model ID
WAM_MODEL=unifolm WAM_CHECKPOINT=org/unifolm-wma-0 bash scripts/deploy.sh --build
```

The model will be auto-downloaded on first run.

---

## Model Architecture

### Input

- **Type:** RobotState dataclass
- **q:** 29-DOF joint positions (radians)
- **dq:** 29-DOF joint velocities (rad/s)
- **tau:** 29-DOF joint torques (N·m)

### Output

- **vx, vy, wz:** Body-frame velocity commands (m/s, m/s, rad/s)
  - vx: forward/back velocity
  - vy: left/right velocity
  - wz: yaw rate (rotation)

### Processing

1. **Input prep:** Concatenate q + dq → 58D observation
2. **Model forward:** 58D → 3D velocity command
3. **Clipping:** Clamp to [-1, 1] for safety

---

## Test Mode: Arm Clapping Demo

When `WAM_CHECKPOINT` is not set, the model runs in **test mode** with heuristic arm clapping:

```python
# Oscillating pattern (0.5 Hz clapping)
phase = step / 10.0  # At 10 Hz control loop
if phase < 0.25:     # Arms opening
    arm_effort = phase / 0.25
elif phase < 0.5:    # Arms closing (clap!)
    arm_effort = 1.0 - (phase - 0.25) / 0.25
# ... continues
```

Output: `[vx=0.0, vy=0.0, wz=0.0]` (stand in place, let GEAR-SONIC handle arm motion)

---

## Troubleshooting

### "WAM_CHECKPOINT must be set for UnifoLM"

**Cause:** Model not in test mode and no checkpoint provided.

**Fix:**
```bash
# Either: Use test mode (no checkpoint)
WAM_MODEL=unifolm bash scripts/deploy.sh

# Or: Provide a checkpoint
WAM_CHECKPOINT=/path/to/checkpoint.pt WAM_MODEL=unifolm bash scripts/deploy.sh
```

### "observation dimension (0) ≠ 994" (GEAR-SONIC error)

**Cause:** sim-ros2-bridge not running or observations not flowing.

**Fix:**
```bash
# Check bridge status
docker compose ps | grep ros2-bridge
docker logs wam-ros2-bridge | tail -20

# Restart if needed
docker compose restart wam-ros2-bridge
```

### Robot not moving / stuck at startup

**Cause:** GEAR-SONIC initializing (normal for 30–60s).

**Check:**
```bash
docker logs wam-gear-sonic | grep -E "Running|ready|initialized"
```

### Slow inference / exceeds control budget

**Check timing:**
```bash
docker logs wam-inference | grep inference_ms

# Should see < 10ms for test mode, < 50ms for real model
```

If exceeding 100ms (10 Hz budget), optimize or use GPU.

---

## Files Modified

| File | Change |
|------|--------|
| `src/models/unifolm.py` | Full implementation (model loading, inference, test mode) |
| `scripts/test_unifolm.py` | New: Local validation script (no server needed) |

---

## Next Steps

1. **[Now]** Test locally: `python scripts/test_unifolm.py`
2. **[Day 1]** Deploy to server with test mode, verify arm clapping in Isaac Sim
3. **[Week 1]** Collect dataset (~400 samples of G1 arm clapping)
4. **[Week 2]** Fine-tune UnifoLM on G1 arm data
5. **[Week 3]** Load fine-tuned checkpoint and test transfer to UBTech

---

## References

- **UnifoLM Repo:** https://github.com/unitreerobotics/unifolm-world-model-action
- **CLAUDE.md:** Architecture details (why sim-ros2-bridge is critical)
- **README.md:** Quick reference
- **SETUP.md:** Troubleshooting & dependency versions
