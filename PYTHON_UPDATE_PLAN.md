# Plan: Update Python to 3.10.18 and Install unifolm_wma

## Executive Summary

**Objective:** Enable real UnifoLM-WMA-0 model inference by installing `unifolm_wma` library, which requires Python 3.10.18 (currently have 3.10.12).

**Current Status:** System runs in graceful fallback mode — returns zero trajectories. Infrastructure is correct, only dependency is missing.

**Result:** Docker image with Python 3.10.18 + unifolm_wma installed. Model inference will activate automatically when checkpoint loads.

**Estimated Time:** 30-45 minutes (includes rebuild, test, verify)

---

## Context & References

### Files to Modify
1. **docker/wam/Dockerfile** — Add Python 3.10.18 installation and unifolm_wma pip install
   - Current path: `wam-stack/docker/wam/Dockerfile`
   - Lines 82-84: Current pragmatic note explains why unifolm_wma is missing

### Key Files (DO NOT MODIFY)
- `src/models/unifolm.py` — Model loading logic ✅ Already updated (27 May 2026)
- `src/main.py` — Control loop and logging ✅ Already updated (27 May 2026)
- `compose.yml` — Docker Compose setup ✅ No changes needed
- `src/recording.py` — Media recording ✅ No changes needed

### Background References
- **unifolm_wma** GitHub: https://github.com/unitreerobotics/unifolm-world-model-action
- **Python 3.10.18** from deadsnakes PPA: `ppa:deadsnakes/ppa` (Ubuntu standard)
- **Current Docker base:** `nvidia/cuda:12.2.0-runtime-ubuntu22.04` (Ubuntu 22.04, has Python 3.10.12)

---

## Step-by-Step Implementation

### Phase 1: Update Dockerfile

**File:** `docker/wam/Dockerfile`

**Location to change:** After line 39 (where Python 3.10 alternatives are set), before line 41 (pip upgrade)

**What to add:**

```dockerfile
# Install Python 3.10.18 from deadsnakes PPA (required by unifolm_wma)
# deadsnakes PPA is the standard way to get non-standard Python versions on Ubuntu
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10.18 python3.10.18-dev && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10.18 1 && \
    update-alternatives --install /usr/bin/python  python  /usr/bin/python3.10.18 1
```

**Why this approach:**
- deadsnakes PPA is the standard Ubuntu way to get Python versions
- Installing python3.10.18-dev ensures pip can compile packages
- update-alternatives ensures python3 and python commands use 3.10.18
- This replaces the previous 3.10.12 as the default

**What to change:** Replace lines 38-39 with a new block that:
1. Adds deadsnakes PPA
2. Updates apt
3. Installs Python 3.10.18 + 3.10.18-dev
4. Sets alternatives (same as before but for .18)

Keep everything else: line 41 (pip upgrade) and all subsequent lines unchanged.

### Phase 2: Add unifolm_wma to pip requirements

**File:** `docker/wam/Dockerfile`

**Location to change:** Line 79 (end of pip install block, after xformers)

**What to add:**

After the existing pip install (line 60-80), add new lines:

```dockerfile
# unifolm_wma requires Python 3.10.18 (installed above)
# This provides the actual model loading, DDIM sampling, and embedders
RUN pip install --no-cache-dir \
    unifolm_wma
```

**Alternative (if unifolm_wma is not on PyPI yet):**

```dockerfile
# If installing from GitHub repo (uncomment if PyPI version unavailable)
# RUN pip install --no-cache-dir \
#     git+https://github.com/unitreerobotics/unifolm-world-model-action.git
```

**Expected behavior:** pip will:
1. Validate Python version is 3.10.18 ✓
2. Download unifolm_wma and dependencies
3. Compile any C extensions (via setuptools)
4. Install to site-packages

### Phase 3: Update Dockerfile comments

**File:** `docker/wam/Dockerfile`

**Location:** Lines 82-84 (current pragmatic note)

**Replace:**
```dockerfile
# Note: unifolm_wma requires exactly Python 3.10.18 (not available in apt)
# Using pragmatic version with transformers.AutoModel instead
# This provides production-ready inference without external library dependency
```

**With:**
```dockerfile
# unifolm_wma is now installed above (Python 3.10.18 from deadsnakes PPA).
# This provides real LatentVisualDiffusion model, DDIM sampling (16 steps),
# text/image embedders (OpenCLIP), and trajectory prediction.
```

---

## Build & Test Procedure

### Build New Image

**Command (on server):**
```bash
cd /root/skurchev/workspace/wam-stack
docker compose build wam --no-cache
```

**Expected output:**
- Step 1-20: Build base CUDA image ✓
- Step 21-25: Install system packages ✓
- Step 26-30: Install Python 3.10.18 from deadsnakes PPA ✓
- Step 31-35: Build CycloneDDS from source ✓
- Step 36-42: Install PyTorch, torch vision ✓
- Step 43-45: Install ML deps (transformers, diffusers, xformers) ✓
- Step 46-47: Install unifolm_wma ← **KEY STEP** ✓

**Build time:** ~10-15 minutes (depending on network, unifolm_wma size)

### Verify Python Version

After build completes, verify Python version in image:

```bash
docker run --rm wam-inference python --version
```

**Expected output:**
```
Python 3.10.18
```

### Start Container and Check Logs

```bash
cd /root/skurchev/workspace/wam-stack
docker compose up wam
```

**Expected logs (within 10 seconds):**

```
[UnifoLM] Initializing with REAL model loading
[UnifoLM] Task: pick and place green cube in white basket
[UnifoLM] Device: cuda
[UnifoLM] DDIM steps: 16
[UnifoLM] Loading model from checkpoint: /workspace/wam/checkpoints/unifolm_wma_dual.ckpt
[UnifoLM] Config loaded
[UnifoLM] Instantiating model from config...
[UnifoLM] Instantiating model...
[UnifoLM] Model moved to cuda
[UnifoLM] Loading weights from checkpoint...
[UnifoLM] Checkpoint loaded successfully
[UnifoLM] Model ready for inference
```

**If successful:** Model inference will activate automatically. Logs will show:
```
[WAM] rt/lowstate received. Control loop starting at 10 Hz.
[UnifoLM] Step 10 - inference starting
[UnifoLM] Step 10: action[0]_norm=X.XXX full_norm=Y.YYY
[WAM] loop=50  action_0[0:3]=[+X.XXX +Y.YYY +Z.ZZZ]  norm=N.NNN  traj_norm=T.TTT
```

(Instead of "Model not loaded, returning zero trajectories")

### Verify Model Output Structure

Check that model outputs correct 14-DOF trajectories:

```bash
docker compose logs wam | grep "action_0\[0:3\]" | head -3
```

**Expected output:**
```
[WAM] loop=50  action_0[0:3]=[+0.123 -0.456 +0.789]  norm=1.234  traj_norm=5.678  state_age=125ms
```

Values should be non-zero (when model is actively predicting), not all zeros.

---

## Success Criteria

✅ **Dockerfile builds without errors**
- No failed pip installs
- No missing dependencies
- Build log shows all steps completed

✅ **Python 3.10.18 is default**
```bash
docker run --rm wam-inference python --version
# Output: Python 3.10.18
```

✅ **unifolm_wma module is importable**
```bash
docker run --rm wam-inference python -c "import unifolm_wma; print('✓ unifolm_wma imported')"
# Output: ✓ unifolm_wma imported
```

✅ **Model loads successfully**
- Log contains: `[UnifoLM] Model ready for inference`
- NOT: `[UnifoLM] Model not loaded, returning zero trajectories`

✅ **Model outputs 14-DOF trajectories**
- Log contains: `action_0[0:3]=[...]` with non-zero values
- Trajectories shape: (16, 14) as expected

✅ **Control loop runs at 10 Hz**
- Logs show continuous `[WAM] loop=X` entries every 5 seconds
- No crashes or exceptions in `docker compose logs wam`

---

## Troubleshooting

### Build fails at "Install Python 3.10.18"

**Symptom:**
```
E: Unable to locate package python3.10.18
```

**Fix:**
- deadsnakes PPA may be temporarily unavailable
- Alternative: Build Python 3.10.18 from source (add to Dockerfile after cmake install):
  ```dockerfile
  RUN wget https://www.python.org/ftp/python/3.10.18/Python-3.10.18.tgz && \
      tar xzf Python-3.10.18.tgz && \
      cd Python-3.10.18 && \
      ./configure --prefix=/usr/local && \
      make && make install && \
      cd .. && rm -rf Python-3.10.18*
  ```

### unifolm_wma pip install fails

**Symptom:**
```
ERROR: Could not find a version that satisfies the requirement unifolm_wma
```

**Fix:**
- unifolm_wma may not be on PyPI yet
- Install from GitHub instead:
  ```dockerfile
  RUN pip install --no-cache-dir \
      git+https://github.com/unitreerobotics/unifolm-world-model-action.git
  ```

### Model loads but outputs are zero

**Symptom:**
```
[WAM] loop=50  action_0[0:3]=[+0.000 +0.000 +0.000]  norm=0.000  traj_norm=0.000
```

**Check:**
1. Is checkpoint at `/workspace/wam/checkpoints/unifolm_wma_dual.ckpt`?
2. Is checkpoint file valid (not corrupted)?
3. Check GPU memory: `docker exec wam-inference nvidia-smi`

**If checkpoint missing:** Download from HuggingFace Hub or copy manually.

### GPU out of memory

**Symptom:**
```
RuntimeError: CUDA out of memory
```

**Fix (reduce model size in unifolm.py):**
```python
# In __call__ method, add:
model.half()  # Convert to float16 precision
# Or reduce ddim_steps from 16 to 8 for faster inference
```

---

## Rollback Plan

If update causes issues:

1. Keep backup of old Dockerfile:
   ```bash
   cp docker/wam/Dockerfile docker/wam/Dockerfile.backup
   ```

2. Revert to old Python:
   ```bash
   git checkout docker/wam/Dockerfile
   docker compose build wam --no-cache
   ```

3. Verify old version still works:
   ```bash
   docker compose up wam
   docker compose logs wam | grep "Model not loaded"  # Should appear (graceful fallback)
   ```

---

## Timeline

| Phase | Task | Time |
|-------|------|------|
| 1 | Edit Dockerfile | 5 min |
| 2 | Build Docker image | 15 min |
| 3 | Start container & verify | 5 min |
| 4 | Check logs for model loading | 5 min |
| 5 | Test trajectory output | 5 min |
| **Total** | | **35 min** |

---

## Next Steps After Success

Once this update is complete:

1. **Monitor inference performance:**
   - Check if action trajectories are reasonable
   - Verify inference time < 1 second per step (at 10 Hz)
   - Monitor GPU memory usage

2. **Fine-tune if needed:**
   - If memory issues: reduce DDIM steps or use float16
   - If trajectories are noisy: tune ACT temporal ensemble weights
   - If too slow: profile with `pytorch-profiler`

3. **Prepare for robot deployment:**
   - Save model checkpoints to persistent storage
   - Add checkpointing/resumption logic to main.py
   - Test with real robot or Isaac Sim scenarios

4. **Update documentation:**
   - Update README with successful inference notes
   - Document GPU memory requirements
   - Add inference timing benchmarks

---

## Agent Notes

**For the implementing agent:**

- This is a **non-destructive update** — only adds Python 3.10.18, doesn't remove 3.10.12
- **No changes to source code** — unifolm.py and main.py already expect this
- **Backward compatible** — if unifolm_wma fails to install, graceful fallback activates
- **Idempotent** — can rebuild image multiple times safely
- **Testable** — success criteria are clear and verifiable via logs

**Critical dependencies check:**
- Ensure cyclonedds==0.10.2 still works with Python 3.10.18 (should be fine, it's version-agnostic)
- Verify torch==2.7.0 has CUDA 12.2 wheels for Python 3.10.18 (confirmed on PyTorch website)
- Confirm transformers version (currently unspecified, installed as latest) is compatible with unifolm_wma

**If stuck:**
- Check Ubuntu 22.04 + deadsnakes PPA compatibility (standard combo)
- Verify unifolm_wma GitHub releases for correct installation method
- Test Python 3.10.18 outside Docker first if needed
