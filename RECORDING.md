# WAM Inference Recording System

## Overview

The WAM stack automatically records all inference data during simulation runs:
- **Input frames:** Robot state (joint positions, velocities, torques)
- **Command logs:** Velocity commands sent to GEAR-SONIC
- **Isaac Sim frames:** Visual snapshots from the simulator (when available)

All files are saved with timestamped directories and full paths logged for easy access.

## Directory Structure

```
/workspace/wam/media/
├── input_frames/       # RobotState snapshots (state_XXXXXX.txt)
├── command_logs/       # CSV files with velocity commands
└── isaac_frames/       # PNG screenshots from Isaac Sim
```

**Server location:** `/root/skurchev/workspace/wam-stack/media/`

## Recording Files

### Input Frames (`input_frames/`)

Each frame captures the complete robot state at a control step:

**Filename:** `state_XXXXXX.txt` (where XXXXXX = loop step number, 0-padded)

**Contents:**
```
Step: 123
Timestamp: 1716547800.1234

Joint Positions (q) [29]:
  0.1234 +0.5678 -0.1234 ... (29 values)

Joint Velocities (dq) [29]:
  +0.0123 -0.0456 +0.0789 ... (29 values)

Joint Torques (tau) [29]:
  -0.5 +1.2 +0.8 ... (29 values)
```

**Usage:** 
- Inspect individual state snapshots
- Analyze joint motion patterns
- Debug state preprocessing

### Command Logs (`command_logs/`)

CSV file with all velocity commands sent to GEAR-SONIC.

**Filename:** `commands_YYYYMMDD_HHMMSS.csv`

**Columns:** `timestamp, step, vx, vy, wz, body_height`

**Example:**
```
timestamp,step,vx,vy,wz,body_height
2026-05-24T14:30:00.123456,0,0.0000,0.0000,0.0000,0.0000
2026-05-24T14:30:00.223456,1,0.0000,0.0000,0.0000,0.0300
2026-05-24T14:30:00.323456,2,0.0000,0.0000,0.0000,0.0600
```

**Usage:**
- Plot velocity commands over time
- Analyze control policy behavior
- Validate arm extension cycles
- Calculate statistics (min/max/avg commands)

### Isaac Sim Frames (`isaac_frames/`)

PNG screenshots from the Isaac Sim virtual environment.

**Filename:** `isaac_XXXXXX.png`

**Capture:** One frame per control loop iteration (10 Hz = every 0.1 seconds)

**Usage:**
- Create time-lapse videos of simulation
- Verify visual behavior matches commands
- Debug camera placement

## Usage

### 1. Run Inference with Recording

```bash
# On server
cd /root/skurchev/workspace/wam-stack
export WAM_MODEL=unifolm
export WAM_CHECKPOINT=/path/to/checkpoint.pt
bash scripts/deploy.sh

# Control loop runs and records automatically
# Press Ctrl+C to stop (finalizes recording with summary)
```

### 2. Monitor Recording During Run

```bash
# On server, watch media directory
watch -n 1 'find /root/skurchev/workspace/wam-stack/media -type f | wc -l'
```

### 3. Copy Media to Local Machine

```bash
# From local machine
bash scripts/copy-media-from-server.sh

# Destination: media_local/
# Output shows full paths to all files
```

### 4. Analyze Media Locally

```bash
# View command log statistics
python3 -c "
import pandas as pd
df = pd.read_csv('media_local/command_logs/commands_*.csv')
print('Mean commands:')
print(df[['vx', 'vy', 'wz', 'body_height']].mean())
print('\nStd deviation:')
print(df[['vx', 'vy', 'wz', 'body_height']].std())
"

# Count frames
ls media_local/input_frames/ | wc -l
ls media_local/isaac_frames/ | wc -l

# Create video from Isaac frames
ffmpeg -framerate 10 \
  -i media_local/isaac_frames/isaac_%06d.png \
  -c:v libx264 -pix_fmt yuv420p \
  output_simulation.mp4
```

## Output Path Logging

When the inference loop exits (Ctrl+C), the recorder prints a summary:

```
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

Each saved file also logs its full absolute path when created (logged to stdout at interval).

## Environment Variables

**WAM_MEDIA_DIR** - Override default media location (default: `/workspace/wam/media`)

```bash
export WAM_MEDIA_DIR=/custom/path
docker compose up wam
```

## Troubleshooting

### No Isaac frames being saved?
- Check if Isaac Sim is configured to write to `/tmp/isaac_frame.png`
- Confirm media directory is mounted RW: `docker exec wam-inference ls -la /workspace/wam/media/`

### CSV file not updating?
- Verify DDS connection: `docker logs wam-inference | grep "rt/lowstate"`
- Check file permissions: `ls -la media/command_logs/`

### Disk space issues?
- Each input frame is ~1KB
- Each Isaac frame is ~100KB (depending on resolution)
- 1 hour at 10Hz = 36,000 frames = 3.6GB for Isaac frames
- Monitor with: `du -sh /root/skurchev/workspace/wam-stack/media/`

## Integration with Analysis

The recording system is designed to support:

1. **Behavioral Analysis:** Examine robot motion through command logs
2. **Debugging:** Replay individual states and commands
3. **Video Generation:** Create videos from Isaac frames
4. **Model Validation:** Compare recorded vs expected behavior
5. **Transfer Learning:** Use recordings as training data

All files are preserved on the server in `/root/skurchev/workspace/wam-stack/media/` until manually deleted.

