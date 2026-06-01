#!/usr/bin/env python3
"""Capture left + right wrist camera snapshots + poses — reads CURRENT SHM state.

Run inside wam-inference container (via grab_snapshot.sh):
    python3 /tmp/capture_snapshot.py

The caller already uploaded the desired offset and waited one render cycle,
so whatever is in SHM now reflects the current camera state.

Reads immediately (no waiting):
    /run/mws/camera.rgb                    — head camera (1280×720)
    /run/mws/camera_left_wrist.rgb         — left wrist (1280×720)
    /run/mws/camera_right_wrist.rgb        — right wrist (1280×720)
    /run/mws/camera_left_wrist_pose.json   — pose + offset (Patches 16/18)
    /run/mws/camera_right_wrist_pose.json  — pose + offset (Patch 16)

Saves to /workspace/wam/media/camera_test/:
    snapshot_left_TIMESTAMP.{png,json}   )
    snapshot_right_TIMESTAMP.{png,json}  )  timestamped copies
    snapshot_composite_TIMESTAMP.png     )
    snapshot_left_latest.{png,json}      )
    snapshot_right_latest.{png,json}     )  always overwritten
    snapshot_composite_latest.png        )
"""
import json, os, sys, time
import numpy as np, cv2

SHM_DIR = "/run/mws"
OUT_DIR = "/workspace/wam/media/camera_test"
W, H = 1280, 720

CAMERAS = {
    "head":  f"{SHM_DIR}/camera.rgb",
    "left":  f"{SHM_DIR}/camera_left_wrist.rgb",
    "right": f"{SHM_DIR}/camera_right_wrist.rgb",
}
POSES = {
    "left":  f"{SHM_DIR}/camera_left_wrist_pose.json",
    "right": f"{SHM_DIR}/camera_right_wrist_pose.json",
}

os.makedirs(OUT_DIR, exist_ok=True)


def read_shm(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        raw = f.read(W * H * 3)
    if len(raw) < W * H * 3:
        return None
    return np.frombuffer(raw, dtype=np.uint8).reshape(H, W, 3).copy()


def read_pose(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def print_pose(side, pose):
    if pose is None:
        print(f"  [{side}] pose JSON not found")
        return
    wrist_key = f"{side}_wrist_cam"
    link_key  = f"{side}_wrist_yaw_link"
    cam_w   = pose.get(wrist_key, {}).get("world") or {}
    wrist_w = pose.get(link_key,  {}).get("world") or {}
    rel     = pose.get(wrist_key, {}).get(f"relative_to_{side}_wrist_yaw_link") or {}
    ts      = pose.get("timestamp")
    print(f"  [{side} wrist]  sim_time={ts:.3f}s" if ts else f"  [{side} wrist]")
    if cam_w:
        print(f"    cam world pos  : {[round(v, 4) for v in cam_w.get('position', [])]}")
        print(f"    cam world quat : {[round(v, 4) for v in cam_w.get('quaternion_wxyz', [])]}")
    if wrist_w:
        print(f"    wrist world pos: {[round(v, 4) for v in wrist_w.get('position', [])]}")
    if rel:
        print(f"    offset pos     : {rel.get('position')}")
        print(f"    offset quat    : {rel.get('quaternion_wxyz')}")


# ── read ──────────────────────────────────────────────────────────────────────
frames = {}
for name, path in CAMERAS.items():
    img = read_shm(path)
    if img is None:
        print(f"WARNING: {path} not found or too small — using black frame")
        img = np.zeros((H, W, 3), dtype=np.uint8)
    frames[name] = img
    nz   = np.count_nonzero(img) / img.size * 100
    mean = img.mean(axis=(0, 1)).round(1)
    std  = img.std(axis=(0, 1)).round(1)
    print(f"{name:6s}: nonzero={nz:.1f}%  mean={mean}  std={std}")

print()
poses = {side: read_pose(path) for side, path in POSES.items()}
for side in ("left", "right"):
    print_pose(side, poses[side])

# ── save ──────────────────────────────────────────────────────────────────────
ts = int(time.time())

# Composite: head | left wrist | right wrist
thumb_w, thumb_h = 640, 360
composite = np.hstack([
    cv2.resize(cv2.cvtColor(frames["head"],  cv2.COLOR_RGB2BGR), (thumb_w, thumb_h)),
    cv2.resize(cv2.cvtColor(frames["left"],  cv2.COLOR_RGB2BGR), (thumb_w, thumb_h)),
    cv2.resize(cv2.cvtColor(frames["right"], cv2.COLOR_RGB2BGR), (thumb_w, thumb_h)),
])

null_pose = {"timestamp": None, "world": None, "_note": "pose unavailable"}

saved = []
for stem_ts, stem_lat in [(f"snapshot_left_{ts}", "snapshot_left_latest"),
                           (f"snapshot_right_{ts}", "snapshot_right_latest")]:
    side = "left" if "left" in stem_ts else "right"
    img_bgr = cv2.cvtColor(frames[side], cv2.COLOR_RGB2BGR)
    pose    = poses[side] or null_pose

    cv2.imwrite(f"{OUT_DIR}/{stem_ts}.png", img_bgr)
    cv2.imwrite(f"{OUT_DIR}/{stem_lat}.png", img_bgr)
    with open(f"{OUT_DIR}/{stem_ts}.json",  "w") as f: json.dump(pose, f, indent=2)
    with open(f"{OUT_DIR}/{stem_lat}.json", "w") as f: json.dump(pose, f, indent=2)
    saved.append(f"{stem_ts}.{{png,json}}")

cv2.imwrite(f"{OUT_DIR}/snapshot_composite_{ts}.png", composite)
cv2.imwrite(f"{OUT_DIR}/snapshot_composite_latest.png", composite)
saved.append(f"snapshot_composite_{ts}.png")

print(f"\nSaved:")
for s in saved:
    print(f"  {OUT_DIR}/{s}")
print(f"  {OUT_DIR}/snapshot_*_latest.* (overwritten)")

# ── copy calibrate result if present ─────────────────────────────────────────
CALIB_SHM = f"{SHM_DIR}/cam_calibrate_result.json"
if os.path.exists(CALIB_SHM):
    import shutil
    dst = f"{OUT_DIR}/cam_calibrate_result.json"
    shutil.copy2(CALIB_SHM, dst)
    with open(dst) as f:
        calib = json.load(f)
    print(f"\n[calibrate] Measured offset saved to {dst}")
    print(json.dumps(calib, indent=2))
