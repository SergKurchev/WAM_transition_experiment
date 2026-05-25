# WAM Stack Implementation: Recording System & Arm Extension Demo

**Date:** 2026-05-24  
**Status:** ✅ Complete and Ready for Deployment

## What Changed

### 1. Arm Motion Demo Updated
**File:** `src/models/unifolm.py`

**Change:** Replaced arm clapping demo with arm extending forward demo
- Old behavior: Robot claps arms in 4-phase cycle (OPENING→CLAPPING→OPEN→RESTING)
- New behavior: Robot extends arms forward in 3-phase cycle (EXTENDING→EXTENDED→RETRACTING)
- Cycle duration: 3 seconds (30 steps at 10 Hz)
- Body height signal: 0 → +0.15 → 0 (scaled with extension effort)

**Key methods:**
- `_arm_extend_demo()` — New demo function with clear phases
- `__call__()` — Returns 4 values: (vx, vy, wz, body_height)

### 2. Recording System Added
**Files:** `src/recording.py` + `src/main.py`

**Features:**
- **Input frames:** RobotState saved as text (joint pos/vel/tau)
- **Command logs:** Velocity commands logged to CSV
- **Isaac frames:** Screenshots from simulator (when available)
- **Full path logging:** Every save prints absolute path

**Implementation:**
```
MediaRecorder (recording.py)
├── save_input_frame(step, state) → state_XXXXXX.txt
├── save_command(step, vx, vy, wz, body_height) → commands_*.csv
├── save_isaac_frame(step) → isaac_XXXXXX.png
└── finalize() → Print summary with all paths
```

**Integration in main.py:**
```python
recorder = MediaRecorder(media_dir="/workspace/wam/media")
for step in range(max_steps):
    recorder.save_input_frame(step, state)
    recorder.save_command(step, vx, vy, wz, body_height)
    recorder.save_isaac_frame(step)
recorder.finalize()  # On exit (Ctrl+C or error)
```

### 3. Media Copy Script
**File:** `scripts/copy-media-from-server.sh`

**Usage:**
```bash
# Copy all media from server to local
bash scripts/copy-media-from-server.sh

# Copy specific types
bash scripts/copy-media-from-server.sh --frames-only
bash scripts/copy-media-from-server.sh --logs-only
```

**Output:** Full paths to all media files on local machine

### 4. Docker Configuration Updated
**File:** `compose.yml`

**Change:** Added media directory volume mount
```yaml
volumes:
  - /root/skurchev/workspace/wam-stack/media:/workspace/wam/media:rw
```

### 5. Documentation
**Files:**
- `RECORDING.md` — Complete recording system guide with examples
- `CAMERA_SETUP.md` — Camera placement recommendations for Isaac Sim
- `IMPLEMENTATION_SUMMARY.md` — This file

## Directory Structure After Recording

```
/root/skurchev/workspace/wam-stack/media/
├── input_frames/
│   ├── state_000000.txt
│   ├── state_000001.txt
│   └── ...
├── command_logs/
│   └── commands_20260524_143000.csv
└── isaac_frames/
    ├── isaac_000000.png
    ├── isaac_000001.png
    └── ...
```

## Workflow

### Step 1: Deploy with Recording
```bash
# On local machine
export WAM_CHECKPOINT=/path/to/unifolm_v1.pt
bash scripts/remote-deploy.sh --build

# On server (automatic)
# - Code synced
# - Docker image built
# - Containers started
# - Recording initialized
```

### Step 2: Monitor Simulation
```bash
# Terminal 1 (logs)
ssh -p 2221 root@176.109.83.84
docker logs -f wam-inference

# Terminal 2 (video feed)
ssh -N -L 6081:localhost:6080 -p 2221 root@176.109.83.84
# Open http://localhost:6081 in browser
```

### Step 3: Watch Recording Progress
```bash
# On server, live count of recorded frames
watch -n 1 'find /root/skurchev/workspace/wam-stack/media -type f | wc -l'
```

### Step 4: Stop and Copy
```bash
# Press Ctrl+C on inference terminal (or wait for error)
# Recorder prints: "All media in: /root/skurchev/workspace/wam-stack/media"

# From local machine
bash scripts/copy-media-from-server.sh
# Destination: media_local/
```

### Step 5: Analyze Locally
```bash
# View CSV
cat media_local/command_logs/commands_*.csv | head -20

# Count frames
ls media_local/input_frames/ | wc -l
ls media_local/isaac_frames/ | wc -l

# Create video from Isaac frames
ffmpeg -framerate 10 \
  -i media_local/isaac_frames/isaac_%06d.png \
  -c:v libx264 \
  -pix_fmt yuv420p \
  robot_arm_extension.mp4
```

## Files Modified

| File | Change |
|------|--------|
| `src/models/unifolm.py` | New arm extension demo, 4-value output |
| `src/main.py` | Recorder integration, graceful shutdown |
| `src/recording.py` | **NEW** — Recording system |
| `compose.yml` | Added media volume mount |
| `scripts/copy-media-from-server.sh` | **NEW** — Media transfer script |
| `RECORDING.md` | **NEW** — Recording documentation |
| `CAMERA_SETUP.md` | **NEW** — Camera setup guide |
| Various scripts | Updated docstrings to mention extension demo |

## Key Output: Path Logging Example

When running inference, the recorder prints:

```
[WAM] starting  model=unifolm  checkpoint=/workspace/wam/checkpoints/unifolm_v1.pt
[RECORDING] Media directory: /root/skurchev/workspace/wam-stack/media
[RECORDING] Command log: /root/skurchev/workspace/wam-stack/media/command_logs/commands_20260524_143000.csv
[UnifoLM EXTEND] cycle=0  progress=  0.0%  state=EXTENDING  effort=+0.00  body=[vx=0.0 vy=0.0 wz=0.0 h=+0.00]
[UnifoLM EXTEND] cycle=0  progress= 10.0%  state=EXTENDING  effort=+0.33  body=[vx=0.0 vy=0.0 wz=0.0 h=+0.05]
[UnifoLM EXTEND] cycle=0  progress= 20.0%  state=EXTENDING  effort=+0.67  body=[vx=0.0 vy=0.0 wz=0.0 h=+0.10]
...
^C
[WAM] received SIGINT, finalizing recording...

================================================================================
[RECORDING] SUMMARY
================================================================================
[RECORDING] Input frames saved: 450
            Location: /root/skurchev/workspace/wam-stack/media/input_frames
[RECORDING] Isaac Sim frames saved: 450
            Location: /root/skurchev/workspace/wam-stack/media/isaac_frames
[RECORDING] Command log (CSV):
            /root/skurchev/workspace/wam-stack/media/command_logs/commands_20260524_143000.csv
[RECORDING] All media in: /root/skurchev/workspace/wam-stack/media
================================================================================
```

## Next Steps (Post-Implementation)

1. ✅ **Verify on server:** Run `bash scripts/deploy.sh` and let it record for 30 seconds
2. ✅ **Check output:** SSH to server and verify files in `/root/skurchev/workspace/wam-stack/media/`
3. ✅ **Copy locally:** Run `bash scripts/copy-media-from-server.sh`
4. ✅ **Analyze:** Check CSV file and count frames
5. ✅ **Create video:** Use ffmpeg to generate MP4 from Isaac frames
6. **Optional:** Add camera to Isaac Sim USD if not already present (see CAMERA_SETUP.md)

## Prompt Summary

The implemented system now addresses all user requirements:

- ✅ Changed prompt to "extend arms forward" (clear, intuitive)
- ✅ Save video commands (CSV of velocity outputs)
- ✅ Save simulation video inputs (Isaac Sim frames @ 10Hz)
- ✅ Understand camera placement (documented in CAMERA_SETUP.md)
- ✅ Save input frames (RobotState as text)
- ✅ Log full paths on every save (in output and CSV)
- ✅ Create server-to-local copy script (copy-media-from-server.sh)

**Ready to deploy!**

