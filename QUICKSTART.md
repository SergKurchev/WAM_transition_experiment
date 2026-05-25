# WAM Stack Quick Start: Deploy & Record

**Goal:** Run arm extension demo and record all inference data with full path logging.

---

## 1. Prepare Local Machine

```bash
cd ~/projects/wam-stack

# Ensure SSH key is ready
ls -la ~/.ssh/id_ed25519

# Test SSH connection
ssh -p 2221 root@176.109.83.84 "echo 'Connected!'"
```

---

## 2. Create or Use Checkpoint

### Option A: Use existing checkpoint
```bash
export WAM_CHECKPOINT=/root/skurchev/workspace/wam-stack/checkpoints/unifolm_v1.pt
```

### Option B: Create new checkpoint
```bash
# On local machine
python scripts/create_model.py

# Copy to server
scp -P 2221 checkpoints/unifolm_v1.pt root@176.109.83.84:/root/skurchev/workspace/wam-stack/checkpoints/
```

---

## 3. Deploy to Server

```bash
# Set checkpoint path
export WAM_CHECKPOINT=/root/skurchev/workspace/wam-stack/checkpoints/unifolm_v1.pt

# Deploy (syncs code, builds image, starts containers)
bash scripts/remote-deploy.sh --build
```

**What happens:**
- Code synced from local → server
- Docker image rebuilt
- Containers started (isaac-sim, gear-sonic, ros2-bridge, wam-inference)
- Recording auto-initialized in `/workspace/wam/media/`

---

## 4. Monitor (3 Terminal Windows)

### Terminal 1: Follow inference logs
```bash
ssh -p 2221 root@176.109.83.84
docker logs -f wam-inference
```

**Look for:**
```
[UnifoLM EXTEND] cycle=0  progress=  0.0%  state=EXTENDING
[UnifoLM EXTEND] cycle=0  progress= 10.0%  state=EXTENDING
...
[RECORDING] Input frames saved: 450
[RECORDING] Command log: /root/skurchev/workspace/wam-stack/media/command_logs/commands_...
```

### Terminal 2: Live frame count
```bash
ssh -p 2221 root@176.109.83.84
watch -n 1 'find /root/skurchev/workspace/wam-stack/media -type f | wc -l'
```

### Terminal 3: Visual feedback (noVNC)
```bash
ssh -N -L 6081:localhost:6080 -p 2221 root@176.109.83.84
# Then open: http://localhost:6081 in browser
```

---

## 5. Stop Recording

```bash
# In Terminal 1 (logs), press Ctrl+C
# Recorder will print:
# ================================================================================
# [RECORDING] SUMMARY
# [RECORDING] Input frames saved: 450
# [RECORDING] Isaac Sim frames saved: 450
# [RECORDING] Command log: /root/skurchev/workspace/wam-stack/media/command_logs/commands_20260524_143000.csv
# ================================================================================
```

---

## 6. Copy Media to Local

```bash
# From local machine
bash scripts/copy-media-from-server.sh

# Output shows destination:
# All media in: /home/user/projects/wam-stack/media_local
```

**Result:** All files copied to `media_local/` with full paths logged

---

## 7. Analyze Locally

### Check recordings
```bash
# Count input frames (RobotState snapshots)
ls media_local/input_frames/ | wc -l

# Count Isaac Sim frames (visual snapshots)
ls media_local/isaac_frames/ | wc -l

# Check command log (CSV)
head -20 media_local/command_logs/commands_*.csv
```

### Create video from Isaac frames
```bash
ffmpeg -framerate 10 \
  -i media_local/isaac_frames/isaac_%06d.png \
  -c:v libx264 -pix_fmt yuv420p \
  robot_arm_extension.mp4

# Play video
open robot_arm_extension.mp4  # macOS
ffplay robot_arm_extension.mp4  # Linux
```

### Analyze commands (Python)
```python
import pandas as pd

df = pd.read_csv('media_local/command_logs/commands_*.csv')

print("Command statistics:")
print(df[['vx', 'vy', 'wz', 'body_height']].describe())

print("\nBody height (arm extension signal):")
print(f"  Min: {df['body_height'].min():.4f}")
print(f"  Max: {df['body_height'].max():.4f}")
print(f"  Mean: {df['body_height'].mean():.4f}")
```

---

## File Locations

### On Server
```
/root/skurchev/workspace/wam-stack/media/
├── input_frames/       # RobotState data (29-DOF joint state)
├── command_logs/       # CSV with velocity commands
└── isaac_frames/       # Visual snapshots from simulator
```

### On Local (after copy)
```
./media_local/
├── input_frames/
├── command_logs/
└── isaac_frames/
```

---

## What Gets Recorded

### Input Frames (state_XXXXXX.txt)
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

### Commands (commands_YYYYMMDD_HHMMSS.csv)
```
timestamp,step,vx,vy,wz,body_height
2026-05-24T14:30:00.123456,0,0.0000,0.0000,0.0000,0.0000
2026-05-24T14:30:00.223456,1,0.0000,0.0000,0.0000,0.0300
2026-05-24T14:30:00.323456,2,0.0000,0.0000,0.0000,0.0600
```

### Isaac Frames (isaac_XXXXXX.png)
PNG screenshot of the simulation at each step (10 Hz = 0.1s intervals)

---

## Troubleshooting

### Issue: Docker build fails
```bash
# Rebuild without cache
bash scripts/remote-deploy.sh --build
```

### Issue: No Isaac frames being saved
```bash
# Check if Isaac Sim is writing to /tmp/isaac_frame.png
docker exec wam-isaac-sim ls -lh /tmp/isaac_frame.png

# If missing, Isaac Sim may not be configured for camera output
# See CAMERA_SETUP.md for instructions
```

### Issue: Command log file is empty
```bash
# Verify DDS connection
docker logs wam-inference | grep "rt/lowstate"

# Should show messages about receiving state
# If not, check that sim-isaac and sim-ros2-bridge are running
docker compose ps
```

### Issue: Disk space full
```bash
# Each Isaac frame is ~100KB
# 1 hour of recording = ~3.6GB

# Check space
du -sh /root/skurchev/workspace/wam-stack/media/

# Clean old recordings if needed
rm -rf /root/skurchev/workspace/wam-stack/media/isaac_frames
```

---

## Key Points

✅ **Automatic recording:** No additional flags needed, everything is logged by default

✅ **Full path logging:** Every save prints absolute path to stdout

✅ **Graceful shutdown:** Press Ctrl+C anytime, recorder finalizes cleanly

✅ **CSV format:** Easy to analyze with pandas/Excel/Python

✅ **Video generation:** Use ffmpeg to create MP4 from frames

✅ **Camera-ready:** Isaac frames captured at 10 Hz (if Isaac Sim has camera configured)

---

## Next: Advanced Analysis

Once you have recorded data:

1. **Visualize motion:** Plot body_height over time to see arm extension cycles
2. **Generate video:** Create MP4 using ffmpeg as shown above
3. **Debug behavior:** Compare recorded frames with expected commands
4. **Model validation:** Use recordings to evaluate model performance

See `RECORDING.md` for detailed analysis examples.

---

## Environment Setup (One-Time)

```bash
# Install ffmpeg (for video generation)
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows
choco install ffmpeg  # or download from https://ffmpeg.org/download.html
```

---

## Need Help?

- **Recording details:** See `RECORDING.md`
- **Camera setup:** See `CAMERA_SETUP.md`
- **Implementation details:** See `IMPLEMENTATION_SUMMARY.md`
- **Project context:** See `CLAUDE.md`

