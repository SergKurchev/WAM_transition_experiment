# WAM Stack — Status

_Last updated: 2026-05-21_

---

## Stack is WORKING ✓

All three containers run on GPU-02 (`176.109.83.84:2222`).

| Container | Status | Notes |
|-----------|--------|-------|
| `wam-isaac-sim` | healthy | G1 robot, physics ~35 Hz, publishes `rt/lowstate` |
| `wam-gear-sonic` | healthy | TRT engines loaded, WBC running |
| `wam-inference` | running | DDS initialized, receives `rt/lowstate`, control loop active |
| noVNC port 6080 | listening | tunnel: `ssh -N -L 6081:localhost:6080 x32-techgov-GPU-02` |

---

## How to Start

```bash
# From local machine (anywhere in the repo):
bash scripts/deploy.sh

# With WAM image rebuild (after Dockerfile changes):
bash scripts/deploy.sh --build

# Reset (robot fell over):
bash scripts/deploy.sh --reset
```

Then open tunnel in a new terminal:
```bash
ssh -N -L 6081:localhost:6080 x32-techgov-GPU-02
```
Browser: **http://localhost:6081**

---

## Key Bugs Fixed

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| GEAR-SONIC segfault (exit 139) | cyclonedds 11.x sends XTypes 1.3 packets → crashes GEAR-SONIC's 0.10.x | WAM now uses Python 3.10 + cyclonedds==0.10.2 built from source |
| WAM receives 0 DDS messages | same cyclonedds incompatibility | same fix |
| Isaac Sim `FileNotFoundError /run/mws/` | `/run/mws/` dir not created | `os.makedirs` added to `_open_shm_file` in `dds_bridge.py` |
| `NameError: LowState_` in WAM | type annotation evaluated at import time | `from __future__ import annotations` added to `dds_interface.py` |
| `String_() missing argument 'data'` | API changed between cyclonedds 11.x → 0.10.2 | `String_(data=payload)` instead of `msg = String_(); msg.data = ...` |
| GEAR-SONIC `libddsc.so.0` not found | upstream image has aarch64 libs, not x86_64 | entrypoint.sh sets `LD_LIBRARY_PATH` to x86_64 dir from bind-mount |
| entrypoint.sh `bash\r: not found` | Windows CRLF line endings | `deploy.sh` runs `sed -i 's/\r//'` automatically on every deploy |

---

## DDS Architecture

```
Isaac Sim  ──rt/lowstate──►  GEAR-SONIC (WBC)  ──rt/lowcmd──►  Isaac Sim
                │
                └──────────────────────────────►  WAM (inference)
                                                       │
                                               rt/run_command/cmd
                                                       │
                                                       ▼
                                                  GEAR-SONIC
```

All on `network_mode: host`, loopback `lo`, CycloneDDS domain 0, multicast 239.255.0.1.

---

## What's Not Done Yet

| Item | Notes |
|------|-------|
| UnifoLM-WMA-0 adapter | `src/models/unifolm.py` — stub, raises `NotImplementedError` |
| EVA adapter | `src/models/eva.py` — stub, raises `NotImplementedError` |
| Walker Tienkung USD asset | Not found yet |
| Dataset collection | ~400 samples per activity needed |

---

## Server-only Files (not in git)

These binary files exist only on GPU-02 and must be copied manually if lost:

```
modules/gwbc/gear_sonic_deploy/
├── policy/release/model_encoder.onnx         (50 MB, from mws-dimos)
├── policy/release/model_decoder.onnx         (40 MB, from mws-dimos)
├── planner/target_vel/V2/planner_sonic.onnx  (from mws-dimos)
└── thirdparty/unitree_sdk2/thirdparty/lib/x86_64/
    ├── libddsc.so + libddsc.so.0              (7.5 MB, from mws-dimos)
    └── libddscxx.so + libddscxx.so.0          (3.1 MB, from mws-dimos)
```
