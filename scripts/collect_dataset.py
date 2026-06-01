#!/usr/bin/env python3
"""WAM dataset collection for UnifoLM fine-tuning.

Collects N episodes and saves them in WMAData-compatible format:

  dataset/
  ├── g1_pick_place.csv
  ├── videos/g1_pick_place/front_camera/
  │   ├── 0.mp4 ... N.mp4
  └── transitions/g1_pick_place/
      ├── meta_data/stats.safetensors
      └── 0.h5 ... N.h5

Each H5:
  action            (T, 14) float32  — joint targets q[14:28]
  observation.state (T, 14) float32  — joint positions q[14:28]
  attrs: action_type, state_type, robot_type

Usage (inside wam-inference container):
  python /workspace/wam/scripts/collect_dataset.py \\
      --n-samples 10 --model stub --output-dir /workspace/wam/dataset

  python /workspace/wam/scripts/collect_dataset.py \\
      --n-samples 10 --model unifolm \\
      --checkpoint /workspace/wam/checkpoints/unifolm_wma_dual.ckpt \\
      --output-dir /workspace/wam/dataset
"""

import argparse
import os
import sys
import time
import numpy as np
import cv2
import pandas as pd
from pathlib import Path

sys.path.insert(0, "/workspace/wam")
sys.path.insert(0, "/workspace/unitree_sdk2py_src")
sys.path.insert(0, "/workspace/unifolm/src")

import torch
import h5py
from safetensors.torch import save_file

DATASET_NAME = "g1_pick_place"
CAMERA_VIEW  = "front_camera"
CONTROL_HZ   = 10
ARM_START    = 14
ARM_END      = 28
ARM_DOF      = ARM_END - ARM_START   # 14
CAM_W, CAM_H = 1280, 720
CAM_SHM_PATH = "/run/mws/camera.rgb"
CAM_SHM_SIZE = CAM_W * CAM_H * 3

INIT_ARM_Q = np.array([
    0.3,  0.2,  0.0,  1.2,  0.0,  0.0,  0.0,
    0.3, -0.2,  0.0,  1.2,  0.0,  0.0,  0.0,
], dtype=np.float32)

PROMPTS = [
    "pick and place green cube in white basket",
    "grasp the green cube and put it in the box",
    "pick up the green object and place it in the white container",
    "move the green cube to the white storage box",
    "lift the green cube and drop it in the basket",
    "take the green cube and place it in the white box",
    "grasp green cube, place in white basket",
    "pick green cube, put in white container",
    "move green object to white basket",
    "transfer green cube to white box",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def flatten_dict(d, parent_key="", sep="/"):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def save_video_mp4(frames: list, path: str, fps: int = CONTROL_HZ) -> bool:
    """Save list of HxWx3 uint8 RGB frames as H.264 MP4."""
    if not frames:
        return False
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (w, h))
    for frame in frames:
        bgr = np.ascontiguousarray(frame[:, :, ::-1])   # RGB → BGR
        writer.write(bgr)
    writer.release()
    size = os.path.getsize(path)
    return size > 1000


# ── camera ────────────────────────────────────────────────────────────────────

class CameraReader:
    def __init__(self):
        self._shm = None
        if os.path.exists(CAM_SHM_PATH):
            try:
                self._shm = open(CAM_SHM_PATH, "rb")
                print(f"[Camera] SHM opened: {CAM_SHM_PATH}", flush=True)
            except Exception as e:
                print(f"[Camera] SHM warning: {e}", flush=True)

    def read_rgb(self) -> np.ndarray:
        """Return uint8 RGB [H, W, 3] or black frame on failure."""
        if self._shm is not None:
            try:
                self._shm.seek(0)
                raw = self._shm.read(CAM_SHM_SIZE)
                if len(raw) >= CAM_SHM_SIZE:
                    return np.frombuffer(raw, dtype=np.uint8).reshape(CAM_H, CAM_W, 3).copy()
            except Exception:
                pass
        return np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)

    def close(self):
        if self._shm:
            self._shm.close()


# ── stub episode (no DDS / model needed) ──────────────────────────────────────

def collect_stub_episode(ep_id: int, n_steps: int):
    """Synthetic sinusoidal arm motion — for testing without Isaac Sim."""
    np.random.seed(ep_id * 42 + 7)
    t = np.linspace(0, 2 * np.pi, n_steps)
    amp   = np.random.uniform(0.05, 0.18, ARM_DOF).astype(np.float32)
    phase = np.random.uniform(0, np.pi,   ARM_DOF).astype(np.float32)

    states  = np.zeros((n_steps, ARM_DOF), dtype=np.float32)
    actions = np.zeros((n_steps, ARM_DOF), dtype=np.float32)
    frames  = []

    for i in range(n_steps):
        noise    = np.random.normal(0, 0.008, ARM_DOF).astype(np.float32)
        states[i]  = INIT_ARM_Q + amp * np.sin(t[i] + phase) + noise
        actions[i] = INIT_ARM_Q + amp * np.sin(t[i] + phase + 0.1)

        # Simple colored frame: green channel oscillates
        frame = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
        g_val = int(abs(60 + 80 * np.sin(2 * np.pi * i / n_steps))) % 256
        frame[:, :, 1] = g_val           # green
        frame[:, :, 2] = 30              # slight red
        # episode label stripe
        frame[:20, :, :] = np.array([ep_id * 25 % 200, 100, 200], dtype=np.uint8)
        frames.append(frame)

    return states, actions, frames


# ── real episode (DDS + model) ────────────────────────────────────────────────

def collect_real_episode(model, dds, camera: CameraReader, ep_id: int,
                         n_steps: int, prompt: str):
    """Run control loop and record state/action/camera for one episode."""
    states  = np.zeros((n_steps, ARM_DOF), dtype=np.float32)
    actions = np.zeros((n_steps, ARM_DOF), dtype=np.float32)
    frames  = []
    dt = 1.0 / CONTROL_HZ

    print(f"[Ep {ep_id}] Running {n_steps} steps @ {CONTROL_HZ} Hz …", flush=True)

    for step_i in range(n_steps):
        t0 = time.time()

        robot_state = dds.get_state()
        if robot_state is None:
            q = INIT_ARM_Q.copy()
        else:
            q = np.array(robot_state.q, dtype=np.float32)[ARM_START:ARM_END]

        # model inference
        action = q.copy()
        if model is not None and robot_state is not None:
            try:
                action_traj, _, _ = model(robot_state)
                action = action_traj[0].cpu().numpy().astype(np.float32)
            except Exception as e:
                if step_i % 20 == 0:
                    print(f"  [step {step_i}] model error: {e}", flush=True)

        frame = camera.read_rgb()

        states[step_i]  = q
        actions[step_i] = action
        frames.append(frame)

        try:
            dds.publish_lowcmd(action.tolist())
        except Exception:
            pass

        elapsed = time.time() - t0
        if (dt - elapsed) > 0:
            time.sleep(dt - elapsed)

        if step_i % 20 == 0:
            print(f"  step {step_i:3d}/{n_steps}  "
                  f"q_norm={np.linalg.norm(q):.3f}  "
                  f"a_norm={np.linalg.norm(action):.3f}", flush=True)

    return states, actions, frames


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="WAM dataset collection")
    parser.add_argument("--n-samples",  type=int, default=10)
    parser.add_argument("--n-steps",    type=int, default=80,
                        help="Steps per episode at 10 Hz (80 = 8 sec)")
    parser.add_argument("--model",      choices=["stub", "unifolm"], default="stub")
    parser.add_argument("--checkpoint", default="/workspace/wam/checkpoints/unifolm_wma_dual.ckpt")
    parser.add_argument("--output-dir", default="/workspace/wam/dataset")
    parser.add_argument("--prompt",     default=None,
                        help="Override prompt (default: cycle through PROMPTS list)")
    args = parser.parse_args()

    output_dir   = Path(args.output_dir)
    videos_dir   = output_dir / "videos"   / DATASET_NAME / CAMERA_VIEW
    trans_dir    = output_dir / "transitions" / DATASET_NAME
    meta_dir     = trans_dir  / "meta_data"

    for d in (videos_dir, trans_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"[Collect] output  : {output_dir}", flush=True)
    print(f"[Collect] model   : {args.model}", flush=True)
    print(f"[Collect] samples : {args.n_samples}  steps/ep: {args.n_steps}", flush=True)

    # ── init DDS + model ──────────────────────────────────────────────────────
    model  = None
    dds    = None
    camera = CameraReader()

    if args.model == "unifolm":
        from dds_interface import DDSInterface
        dds = DDSInterface(iface=os.environ.get("DDS_IFACE", "lo"))
        dds.init()
        print("[Collect] Waiting 3 s for DDS …", flush=True)
        time.sleep(3.0)

        from models.unifolm import UnifoLMModel
        prompt0 = args.prompt or PROMPTS[0]
        model = UnifoLMModel(checkpoint=args.checkpoint, prompt=prompt0)

    # ── collect episodes ──────────────────────────────────────────────────────
    all_actions = []
    all_states  = []
    csv_rows    = []

    for ep_id in range(args.n_samples):
        prompt = args.prompt or PROMPTS[ep_id % len(PROMPTS)]
        print(f"\n[Ep {ep_id+1}/{args.n_samples}] prompt='{prompt}'", flush=True)

        if args.model == "stub":
            states, actions, frames = collect_stub_episode(ep_id, args.n_steps)
        else:
            if hasattr(model, "prompt"):
                model.prompt = prompt
            states, actions, frames = collect_real_episode(
                model, dds, camera, ep_id, args.n_steps, prompt)

        # ── save H5 ───────────────────────────────────────────────────────────
        h5_path = trans_dir / f"{ep_id}.h5"
        with h5py.File(str(h5_path), "w") as f:
            f.create_dataset("action",            data=actions)
            f.create_dataset("observation.state", data=states)
            f.attrs["action_type"] = "joint position"
            f.attrs["state_type"]  = "joint position"
            f.attrs["robot_type"]  = "Unitree G1"
        print(f"  [H5]    {h5_path}  shape={actions.shape}", flush=True)

        # ── save video ────────────────────────────────────────────────────────
        vid_path = str(videos_dir / f"{ep_id}.mp4")
        ok = save_video_mp4(frames, vid_path, fps=CONTROL_HZ)
        print(f"  [video] {vid_path}  frames={len(frames)}  ok={ok}", flush=True)

        all_actions.append(torch.tensor(actions))
        all_states.append(torch.tensor(states))

        csv_rows.append({
            "videoid":                ep_id,
            "contentUrl":             "x",
            "duration":               "x",
            "data_dir":               f"{DATASET_NAME}/{CAMERA_VIEW}",
            "instruction":            prompt,
            "dynamic_confidence":     "x",
            "dynamic_wording":        "x",
            "dynamic_source_category":"x",
            "embodiment":             "Unitree G1",
            "fps":                    CONTROL_HZ,
        })

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = output_dir / f"{DATASET_NAME}.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"\n[Collect] CSV saved: {csv_path}", flush=True)

    # ── stats.safetensors ─────────────────────────────────────────────────────
    all_act = torch.cat(all_actions, dim=0)   # (N_total, 14)
    all_st  = torch.cat(all_states,  dim=0)   # (N_total, 14)

    stats = {
        "action": {
            "max":  all_act.max(dim=0).values,
            "min":  all_act.min(dim=0).values,
            "mean": all_act.mean(dim=0),
            "std":  all_act.std(dim=0).clamp(min=1e-4),
        },
        "observation.state": {
            "max":  all_st.max(dim=0).values,
            "min":  all_st.min(dim=0).values,
            "mean": all_st.mean(dim=0),
            "std":  all_st.std(dim=0).clamp(min=1e-4),
        },
    }
    flat = flatten_dict(stats)
    stats_path = str(meta_dir / "stats.safetensors")
    save_file(flat, stats_path)
    print(f"[Collect] stats saved: {stats_path}", flush=True)
    print(f"  action min[:4]={stats['action']['min'][:4].tolist()}", flush=True)
    print(f"  action max[:4]={stats['action']['max'][:4].tolist()}", flush=True)

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print(f"[Collect] DONE  {args.n_samples} episodes", flush=True)
    print(f"  CSV:    {csv_path}", flush=True)
    print(f"  Videos: {videos_dir}", flush=True)
    print(f"  H5:     {trans_dir}", flush=True)
    print(f"  Stats:  {stats_path}", flush=True)
    print(f"{'='*60}", flush=True)

    camera.close()


if __name__ == "__main__":
    main()
