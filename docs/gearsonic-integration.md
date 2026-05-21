# GEAR-SONIC Integration: How mws-dimos Does It and How We Replicate It

## How mws-dimos Wires GEAR-SONIC

### Architecture Overview

```
WAM model (src/main.py)
     │  rt/run_command/cmd  (DDS String_, JSON "[vx,vy,wz,height]")
     ▼
GEAR-SONIC  (--input-type dds  →  DDSInputHandler)
     │  rt/lowcmd  (DDS LowCmd_, 29-DOF joint targets @ 50 Hz)
     ▼
Isaac Sim  (sim/isaac/g1_sim.py  +  dds_bridge.py)
     │  rt/lowstate  (DDS LowState_, 29-DOF state @ 50 Hz)
     ▲──────────────────────────────────────────────────────┘
```

### The DDSInputHandler (patched binary feature)

The prebuilt `g1_deploy_onnx_ref` binary in the MWS docker image includes a custom
`DDSInputHandler` (source: `gear_sonic_deploy/src/.../dds_input_handler.hpp`).
This handler is NOT in the open-source gwbc source tree — it is a MWS-owned extension.

**What DDSInputHandler does:**

1. **Subscribes** to DDS topic `rt/run_command/cmd` (`std_msgs.msg.dds_.String_`) and parses
   JSON `[vx, vy, wz, body_height]` into a `MovementState`:
   ```
   speed             = sqrt(vx² + vy²)
   locomotion_mode   = WALK (2) if speed > 0.05 m/s, else IDLE (0)
   movement_direction = normalize([vx, vy, 0])
   facing_direction  = [1, 0, 0]  (heading tracked: delta_heading += wz * dt)
   height            = body_height if body_height > 0 else -1.0 (policy default)
   ```

2. **Auto-start sequence** — 5 seconds after "Init Done" prints, automatically:
   - Sets `operator_state.start = true`  → GEAR-SONIC enters CONTROL state
   - Sets `planner_state.enabled = true` → locomotion planner activates
   - Sends ZMQ PUSH byte to `localhost:5558` → **we do not listen here** (wam-stack uses rt/lowcmd detection instead)

   **No manual `control.py start-control` or `drop` commands are needed.**

### Startup Timing Diagram

```
t=0        Isaac Sim boots, loads USD, physics warmup (~2 min)
           GEAR-SONIC boots, loads TensorRT engines (~2-3 min first run, <30s after)

t=A        GEAR-SONIC: "Init Done" → writes /tmp/gear-sonic-ready
           Isaac: healthcheck /tmp/isaac_ready already set (burn-in done)
           → Compose: WAM container starts (depends_on both healthy)

t=A+5s     DDSInputHandler auto-trigger:
           1. Enter CONTROL state
           2. Enable planner
           3. ZMQ PUSH → localhost:5558 (wam-stack ignores; wrench released by rt/lowcmd detection)

t=A+5s+ε   Robot standing freely, GEAR-SONIC tracking MovementState
           WAM must already be publishing rt/run_command/cmd by this point
```

**Critical window:** WAM must have DDS initialized and be publishing commands within ~5 seconds
of GEAR-SONIC's "Init Done". If WAM takes longer, the wrench releases while GEAR-SONIC is in
IDLE mode → robot falls.

### ZMQ Control Port (mws-dimos only)

mws-dimos also exposes a ZMQ PUB socket on port 5556 (`command` topic) for manual operator
commands:
- `start-control` — `{start=True, normal=True}` (manual override)
- `stop` — `{stop=True}`
- `reset-policy` — `{reset=True}`

These are used by `scripts/control.py` for debugging/recovery. In normal operation the
DDSInputHandler auto-start replaces all of these.

---

## What Our wam-stack Currently Has

| Component | Status |
|-----------|--------|
| `--input-type dds` in compose.yml | ✓ correct |
| WAM publishes `rt/run_command/cmd` | ✓ correct format |
| DDSInputHandler auto-drops wrench | ✓ GEAR-SONIC handles this |
| WAM starts after both healthchecks | ⚠ timing risk |
| AUTO-DROP in g1_sim.py | ⚠ redundant / possibly premature |
| GEAR-SONIC restart on crash | ✗ no `restart: always` in compose |
| ZMQ `start-control` signal | ✗ not sent (but not needed if DDSInputHandler works) |

### Identified Issues

**Issue 1: Timing window**
WAM depends on `gear-sonic: condition: service_healthy`. GEAR-SONIC writes
`/tmp/gear-sonic-ready` at Init Done. DDSInputHandler fires 5 s later. WAM must initialize
DDS and publish its first `rt/run_command/cmd` in those 5 seconds.

Potential failure path:
- GEAR-SONIC Init Done → gear-sonic-ready written
- Compose starts WAM
- WAM inits DDS, loads model, waits for `rt/lowstate` → may take >5 s
- DDSInputHandler fires, drops wrench, GEAR-SONIC = IDLE (no WAM command yet)
- Robot falls

**Issue 2: (non-issue) AUTO-DROP code**
Earlier sessions added AUTO-DROP logic to `g1_sim.py` based on `rt/lowcmd` presence, but
server git stash restored the original clean code. Current g1_sim.py has no AUTO-DROP.
DDSInputHandler handles the drop correctly — no change needed here.

**Issue 3: GEAR-SONIC container doesn't restart on crash**
When GEAR-SONIC crashes (`lowstate_age_ms > 500` watchdog), the container exits. No
`restart: always` is configured. Everything must be manually restarted.

---

## Implementation Plan

### Step 1: Fix WAM startup timing  ← ROOT CAUSE (confirmed by log timestamps)
**Root cause confirmed by timestamps (2026-05-21):**
```
09:52:00  GEAR-SONIC Init Done → writes gear-sonic-ready
09:52:05  DDSInputHandler drops wrench; robot in IDLE mode (no WAM yet)
09:52:16  GEAR-SONIC crashes (lowstate timeout after robot falls)
09:52:17  WAM starts — 12 seconds too late!
```
Docker container startup overhead (~12s) exceeds the 5-second DDSInputHandler window.

**Fix (applied to compose.yml):** Remove WAM's `depends_on: gear-sonic: condition: service_healthy`.
WAM only needs Isaac Sim healthy (for `rt/lowstate`). WAM starts when Isaac is ready (typically
30-60 seconds before GEAR-SONIC Init Done), so it is publishing WALK commands well before the
wrench drops.

Startup sequence after fix:
```
t=0      Isaac Sim starts + GEAR-SONIC starts (parallel)
t=~120s  Isaac burn-in done → writes isaac_ready
t=~120s  WAM starts (depends on isaac only) → DDS init → publishes [0.15,0,0,0] at 10Hz
t=~180s  GEAR-SONIC Init Done → writes gear-sonic-ready
t=~185s  DDSInputHandler drops wrench; GEAR-SONIC receives WALK from WAM → robot walks!
```

### Step 2: Add GEAR-SONIC restart policy
Add `restart: on-failure` to gear-sonic in compose.yml so it automatically recovers if
something does crash. With the timing fix above, GEAR-SONIC should no longer crash from
lowstate timeout, but `restart: on-failure` is a useful safety net.

### Step 3: Validate end-to-end
1. Deploy with `bash scripts/update.sh`
2. Watch logs: `docker logs wam-gear-sonic --follow`
3. Check for `[DDSInputHandler] Init Done` → 5 s later → `Isaac drop sent`
4. Check WAM logs for WALK commands flowing
5. Watch noVNC to confirm robot walks, not falls

---

## DDS Command Format Reference

```python
# WAM side (dds_interface.py) — already correct
import json
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

payload = json.dumps([vx, vy, wz, body_height])   # "[0.15, 0.0, 0.0, 0.0]"
publisher.Write(String_(data=payload))
```

GEAR-SONIC interprets (from architecture.md):
```
speed    = sqrt(vx² + vy²)         # 0.15 → 0.15 m/s
mode     = WALK if speed > 0.05    # → LocomotionMode::WALK (2)
movement = normalize([vx, vy, 0])  # [1, 0, 0]
facing   = [1, 0, 0]               # default forward
height   = -1.0                    # use policy default
```

## ZMQ Ports Summary

| Port | Direction | Purpose | Owner |
|------|-----------|---------|-------|
| 5556 | PUB → SUB | GEAR-SONIC ZMQ Manager: command/planner/pose (manual ops) | GEAR-SONIC binary |
| 5558 | PUSH → PULL | DDSInputHandler drop signal (sent by binary, **wam-stack does not bind this port**) | GEAR-SONIC binary → /dev/null |
| 6559 | PUSH → PULL | Isaac reset-sim (`scripts/reset_sim.sh` → `g1_sim.py`) | wam-stack |

**Wrench drop mechanism (wam-stack):** Isaac detects the first `rt/lowcmd` from GEAR-SONIC
and calls `support.trigger_drop()` internally — no ZMQ port needed on our side.
GEAR-SONIC tries to push to 5558 (silently fails); we ignore it.
