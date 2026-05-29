"""UnifoLM-WMA-0 inference with real model loading and DDIM sampling.

Loads actual UnifoLM model from checkpoint with OmegaConf config.
Implements proper image-guided synthesis with text conditioning.
Outputs trajectory predictions and video frames.
"""

import os
import sys
import time
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from collections import deque
from collections import OrderedDict

import numpy as np
import torch
import cv2

from omegaconf import OmegaConf
from einops import rearrange, repeat

from dds_interface import RobotState

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Model input dimensions (from config_model.yaml)
MODEL_INPUT_H = 320       # image height expected by model
MODEL_INPUT_W = 512       # image width expected by model
MODEL_STATE_DIM = 16      # universal state dim (model always uses 16)
MODEL_ACTION_DIM = 16     # universal action dim (model always uses 16)
MODEL_CHANNELS = 4        # latent channels (from config: channels: 4)
MODEL_HORIZON = 16        # temporal length (from config: temporal_length: 16)
MODEL_FPS = 15            # fps for sampling: 30 / frame_stride=2 (from official run_real_eval_server.sh)
                          # default_fs=10 in wma_config is just a fallback; official script uses frame_stride=2 → fps=15
LATENT_H = MODEL_INPUT_H // 8   # = 40
LATENT_W = MODEL_INPUT_W // 8   # = 64
NOISE_SHAPE = [1, MODEL_CHANNELS, MODEL_HORIZON, LATENT_H, LATENT_W]

# Normalization stats: G1 Pack Camera dataset (unitree_g1_pack_camera)
# Bundled in the unifolm repo examples directory, always available in container.
G1_PACK_CAMERA_STATS_PATH = (
    "/workspace/unifolm/examples/world_model_interaction_prompts"
    "/transitions/unitree_g1_pack_camera/meta_data/stats.safetensors"
)

# G1 joint mapping in unitree_hg LowState (35 slots total):
#   [00-05] left leg  (hip pitch/roll/yaw, knee, ankle pitch/roll)
#   [06-11] right leg (same order)
#   [12-13] waist (yaw, roll)
#   [14-20] left arm  (7 DOF)  ← model controls these
#   [21-27] right arm (7 DOF)  ← model controls these
#   [28]    gripper
#   [29-34] unused (zeros)
G1_ARM_JOINT_START = 14  # first arm joint index in LowState / LowCmd
G1_ARM_JOINT_END   = 28  # last arm joint index + 1 → q[14:28] gives 14 DOF

# Robot head camera SHM — D435i mounted on G1, looking at workspace.
# Same perspective as training data (G1_Dex1_MountCameraRedGripper_Dataset).
# Format: raw RGB bytes 1280×720, written by Isaac Sim bridge.
CAMERA_SHM_PATH = "/run/mws/camera.rgb"
CAMERA_SHM_W, CAMERA_SHM_H = 1280, 720
CAMERA_SHM_SIZE = CAMERA_SHM_W * CAMERA_SHM_H * 3  # bytes


def instantiate_from_config(config):
    """Instantiate a module from OmegaConf config or plain dict.

    Mirrors unifolm_wma.utils.utils.instantiate_from_config.
    Handles OmegaConf DictConfig WITHOUT converting to plain dict —
    the nested sub-configs must stay as DictConfig so model code can
    use attribute access (config.params, config.target etc.).
    """
    try:
        from omegaconf import DictConfig
        if isinstance(config, DictConfig):
            # DictConfig supports dict-like .get() and **unpacking.
            # Keep nested values as OmegaConf objects for the model code.
            target = config.get("target", None) or config.get("_target_", None)
            if target is None:
                raise ValueError(f"Missing 'target' in config: {config}")
            params = config.get("params", {})
            if isinstance(target, str):
                module_path, class_name = target.rsplit(".", 1)
                mod = __import__(module_path, fromlist=[class_name])
                cls = getattr(mod, class_name)
            else:
                cls = target
            return cls(**params)
    except ImportError:
        pass

    # Plain-dict path (fallback)
    if not isinstance(config, dict):
        if isinstance(config, str):
            return config
        raise ValueError(f"Cannot instantiate config: {config}")

    target = config.get("target") or config.get("_target_")
    if not target:
        raise ValueError(f"Missing 'target' in config: {config}")

    params = config.get("params", {})

    if isinstance(target, str):
        module_path, class_name = target.rsplit(".", 1)
        mod = __import__(module_path, fromlist=[class_name])
        cls = getattr(mod, class_name)
    else:
        cls = target

    return cls(**params)


def load_model_checkpoint(model: torch.nn.Module, ckpt: str) -> torch.nn.Module:
    """Load model weights from checkpoint file."""
    print(f"[UnifoLM] Loading checkpoint from: {ckpt}", flush=True)

    state_dict = torch.load(ckpt, map_location="cpu")
    if "state_dict" in state_dict:
        raw_sd = state_dict["state_dict"]
    elif "module" in state_dict:
        raw_sd = OrderedDict()
        for key in state_dict['module'].keys():
            raw_sd[key[16:]] = state_dict['module'][key]
    else:
        raw_sd = state_dict

    # Rename legacy keys
    renamed_sd = OrderedDict()
    for k, v in raw_sd.items():
        new_k = k.replace("framestride_embed", "fps_embedding")
        renamed_sd[new_k] = v

    # Load with strict=False: mismatched-shape keys are silently skipped
    result = model.load_state_dict(renamed_sd, strict=False)
    if result.missing_keys:
        print(f"[UnifoLM] Missing keys ({len(result.missing_keys)}): {result.missing_keys[:5]}...", flush=True)
    if result.unexpected_keys:
        print(f"[UnifoLM] Unexpected keys ({len(result.unexpected_keys)}): {result.unexpected_keys[:3]}...", flush=True)

    print(f"[UnifoLM] Checkpoint loaded successfully", flush=True)
    return model


def get_device_from_parameters(module: torch.nn.Module) -> torch.device:
    """Get device from module parameters."""
    return next(iter(module.parameters())).device


def get_latent_z(model: torch.nn.Module, videos: torch.Tensor) -> torch.Tensor:
    """Encode videos into latent space.

    Args:
        model: Model with encode_first_stage method.
        videos: Input videos [B, C, T, H, W] in [-1, 1].

    Returns:
        Latent tensor [B, C, T, h, w] where h=H//8, w=W//8.
    """
    b, c, t, h, w = videos.shape
    x = rearrange(videos, 'b c t h w -> (b t) c h w')
    z = model.encode_first_stage(x)
    z = rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)
    return z


def image_guided_synthesis(
        model: torch.nn.Module,
        prompts: list,
        observation: Dict[str, torch.Tensor],
        noise_shape: list,
        ddim_steps: int = 16,
        ddim_eta: float = 1.0,
        unconditional_guidance_scale: float = 1.0,
        fs: int = MODEL_FPS,
        timestep_spacing: str = 'uniform',
        guidance_rescale: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run inference with DDIM sampling.

    Mirrors scripts/evaluation/real_eval_server.py::image_guided_synthesis.

    Args:
        model: LatentVisualDiffusion model instance.
        prompts: List of text prompts (length = batch_size).
        observation: Dict with keys:
            - 'observation.images.top': [B, T, C, H, W] in [-1, 1]
            - 'observation.state':      [B, T, state_dim]
            - 'action':                 [B, T, action_dim] (zeros placeholder)
        noise_shape: [B, C, T, h, w] latent noise shape.
        ddim_steps: Number of DDIM denoising steps.
        ddim_eta: DDIM eta (1.0 = stochastic, 0.0 = deterministic).
        unconditional_guidance_scale: CFG scale (1.0 = no guidance).
        fs: Frame stride / FPS condition value.
        timestep_spacing: DDIM timestep spacing strategy.
        guidance_rescale: Guidance rescale factor.

    Returns:
        (batch_variants, actions, states):
            batch_variants: Decoded video [B, C, T, H, W].
            actions: Predicted action trajectory [B, horizon, action_dim].
            states: Predicted state trajectory [B, horizon, state_dim].
    """
    from unifolm_wma.models.samplers.ddim import DDIMSampler

    batch_size = noise_shape[0]
    ddim_sampler = DDIMSampler(model)
    fs_tensor = torch.tensor([fs] * batch_size, dtype=torch.long, device=model.device)

    img = observation['observation.images.top']          # [B, T, C, H, W]
    cond_img = img[:, -1, ...]                           # [B, C, H, W] — last obs frame

    # Image cross-attention embeddings (from CLIP image encoder + projector)
    cond_img_emb = model.embedder(cond_img)              # [B, 257, 1280]
    cond_img_emb = model.image_proj_model(cond_img_emb)  # [B, 16, 1024]

    # Build latent conditioning (hybrid mode: concat latent of obs frame)
    cond = {}
    if model.model.conditioning_key == 'hybrid':
        # Encode all observation frames: img [B, T, C, H, W] → [B, C, T, H, W]
        z = get_latent_z(model, img.permute(0, 2, 1, 3, 4))  # [B, 4, T, 40, 64]
        # Use last frame as concat condition, repeated for all future frames
        img_cat_cond = z[:, :, -1:, :, :]                    # [B, 4, 1, 40, 64]
        img_cat_cond = repeat(img_cat_cond,
                              'b c t h w -> b c (repeat t) h w',
                              repeat=noise_shape[2])          # [B, 4, 16, 40, 64]
        cond["c_concat"] = [img_cat_cond]

    # Text conditioning
    cond_ins_emb = model.get_learned_conditioning(prompts)  # [B, 77, 1024]

    # State conditioning
    cond_state = model.state_projector(observation['observation.state'])     # [B, T, 1024]
    cond_state_emb = model.agent_state_pos_emb + cond_state                  # [B, T, 1024]

    # Action conditioning (zeroed out during inference — model predicts actions)
    cond_action = model.action_projector(observation['action'])              # [B, T, 1024]
    cond_action_emb = model.agent_action_pos_emb + cond_action
    cond_action_emb = torch.zeros_like(cond_action_emb)                     # zero out

    # Combined cross-attention context: [state | instruction | image_emb]
    cond["c_crossattn"] = [
        torch.cat([cond_state_emb, cond_ins_emb, cond_img_emb], dim=1)
    ]

    # Action head conditioning: raw images + states for last n_obs_steps_acting
    n_act = model.n_obs_steps_acting
    cond["c_crossattn_action"] = [
        img.permute(0, 2, 1, 3, 4)[:, :, -n_act:],   # [B, C, n_act, H, W]
        observation['observation.state'][:, -n_act:]   # [B, n_act, state_dim]
    ]

    kwargs = {"unconditional_conditioning_img_nonetext": None}

    # DDIM sampling — returns (latent_samples, actions, states, intermediates)
    samples, actions, states, _ = ddim_sampler.sample(
        S=ddim_steps,
        conditioning=cond,
        batch_size=batch_size,
        shape=noise_shape[1:],
        verbose=False,
        unconditional_guidance_scale=unconditional_guidance_scale,
        unconditional_conditioning=None,
        eta=ddim_eta,
        cfg_img=None,
        mask=None,
        x0=None,
        fs=fs_tensor,
        timestep_spacing=timestep_spacing,
        guidance_rescale=guidance_rescale,
        **kwargs
    )

    # Decode latent → pixel space
    batch_images = model.decode_first_stage(samples)

    return batch_images, actions, states


class ACTTemporalEnsembler:
    """Exponential weighted temporal ensemble (from ACT paper)."""

    def __init__(self, temporal_ensemble_coeff: float = 0.01, chunk_size: int = 16, exe_steps: int = 8):
        self.chunk_size = chunk_size
        self.exe_steps = exe_steps
        self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.reset()

    def reset(self):
        self.ensembled_actions = None
        self.ensembled_actions_count = None

    def update(self, actions: torch.Tensor) -> torch.Tensor:
        """Apply exponential weighted temporal ensemble smoothing."""
        self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device=actions.device)

        if self.ensembled_actions is None:
            self.ensembled_actions = actions.clone()
            self.ensembled_actions_count = torch.ones(
                (self.chunk_size, 1), dtype=torch.long, device=self.ensembled_actions.device
            )
        else:
            self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
            self.ensembled_actions += (
                actions[:, : -self.exe_steps] * self.ensemble_weights[self.ensembled_actions_count]
            )
            self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
            self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size)
            self.ensembled_actions = torch.cat([self.ensembled_actions, actions[:, -self.exe_steps :]], dim=1)
            self.ensembled_actions_count = torch.cat(
                [
                    self.ensembled_actions_count,
                    torch.ones((self.exe_steps, 1), dtype=torch.long, device=self.ensembled_actions_count.device),
                ]
            )

        actions_out, self.ensembled_actions, self.ensembled_actions_count = (
            self.ensembled_actions[:, : self.exe_steps],
            self.ensembled_actions[:, self.exe_steps :],
            self.ensembled_actions_count[self.exe_steps :],
        )
        return actions_out


class UnifoLMModel:
    """UnifoLM-WMA-0 using real model loading with DDIM sampling."""

    def __init__(self, checkpoint: str | None, prompt: str | None = None):
        """Initialize with real model loading from config and checkpoint."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.step_count = 0
        self.prompt = prompt or "pick and place green cube in white basket"

        # Config from deployment
        self.ddim_steps = 16
        self.ddim_eta = 1.0
        self.horizon = MODEL_HORIZON          # 16 timesteps
        self.agent_action_dim = 14            # G1 arms: 14 DOF (7 + 7)
        self.model_action_dim = MODEL_ACTION_DIM  # model output: 16-dim
        self.model_state_dim = MODEL_STATE_DIM    # model input: 16-dim
        # n_obs_steps must match config: n_obs_steps_imagen = 2
        self.n_obs_steps = 2

        # Temporal ensemble
        self.temporal_ensembler = ACTTemporalEnsembler(
            temporal_ensemble_coeff=0.01,
            chunk_size=self.horizon,
            exe_steps=8
        )

        # History buffers — keep last n_obs_steps frames/states
        self.obs_image_history = deque(maxlen=self.n_obs_steps)
        self.obs_state_history = deque(maxlen=self.n_obs_steps)
        self.action_history = deque(maxlen=self.horizon)
        self.last_camera_image = None

        # Normalization stats (loaded from G1 pack camera dataset)
        self.norm_stats = self._load_normalization_stats()

        # Robot head camera SHM — preferred over Isaac top-down frame
        self.camera_shm_map = None
        self._init_camera_shm()

        print(f"[UnifoLM] Initializing with REAL model loading", flush=True)
        print(f"[UnifoLM] Task: {self.prompt}", flush=True)
        print(f"[UnifoLM] Device: {self.device}", flush=True)
        print(f"[UnifoLM] DDIM steps: {self.ddim_steps}", flush=True)
        print(f"[UnifoLM] n_obs_steps: {self.n_obs_steps}", flush=True)
        print(f"[UnifoLM] model_action_dim: {self.model_action_dim} (trim to {self.agent_action_dim} for G1)", flush=True)

        if checkpoint is None:
            raise ValueError("WAM_CHECKPOINT must be set (path to .ckpt file)")

        self._load_model_real(checkpoint)

    # ── normalisation ─────────────────────────────────────────────────────────

    def _load_normalization_stats(self) -> Optional[Dict[str, Any]]:
        """Load G1 Pack Camera min/max stats for state + action normalisation.

        Stats are bundled in the unifolm repo examples directory at:
          G1_PACK_CAMERA_STATS_PATH
        Format: safetensors with keys like "observation.state/min", "action/max", …
        After unflatten_dict → {"observation.state": {"min": tensor, "max": tensor}, …}
        """
        try:
            from safetensors.torch import load_file as safetensors_load

            def _unflatten(d, sep="/"):
                out: Dict[str, Any] = {}
                for k, v in d.items():
                    parts = k.split(sep)
                    node = out
                    for part in parts[:-1]:
                        node = node.setdefault(part, {})
                    node[parts[-1]] = v
                return out

            raw = safetensors_load(G1_PACK_CAMERA_STATS_PATH)
            stats = _unflatten(raw)

            state_min = stats['observation.state']['min'].float()  # [16]
            state_max = stats['observation.state']['max'].float()  # [16]
            action_min = stats['action']['min'].float()            # [16]
            action_max = stats['action']['max'].float()            # [16]

            print(f"[UnifoLM] ✓ Loaded G1 pack camera normalization stats", flush=True)
            print(f"[UnifoLM]   state min[:5]={state_min[:5].tolist()}", flush=True)
            print(f"[UnifoLM]   state max[:5]={state_max[:5].tolist()}", flush=True)
            print(f"[UnifoLM]   action min[:5]={action_min[:5].tolist()}", flush=True)
            print(f"[UnifoLM]   action max[:5]={action_max[:5].tolist()}", flush=True)

            return {
                'observation.state': {'min': state_min, 'max': state_max},
                'action':            {'min': action_min, 'max': action_max},
            }
        except Exception as e:
            print(f"[UnifoLM] WARNING: Could not load norm stats ({e})", flush=True)
            print(f"[UnifoLM]   → feeding raw joint angles to model (suboptimal)", flush=True)
            return None

    def _unnormalize_actions(self, actions_norm: torch.Tensor) -> torch.Tensor:
        """Convert model action output from [-1, 1] back to actual joint angles.

        Formula (inverse of min-max normalise):
            joint_angle = (norm + 1) / 2 * (max - min) + min

        Args:
            actions_norm: [horizon, action_dim] in [-1, 1].

        Returns:
            [horizon, action_dim] in original joint-angle space (radians).
        """
        if self.norm_stats is None:
            return actions_norm  # passthrough — no stats available

        action_min = self.norm_stats['action']['min'].to(actions_norm.device)  # [16]
        action_max = self.norm_stats['action']['max'].to(actions_norm.device)  # [16]

        # Shape: [horizon, 16] × [16] — broadcast over horizon dimension
        actions_unnorm = (actions_norm + 1.0) / 2.0 * (action_max - action_min) + action_min
        return actions_unnorm

    # ── camera ────────────────────────────────────────────────────────────────

    def _init_camera_shm(self) -> None:
        """Open robot head camera SHM (/run/mws/camera.rgb).

        The SHM is written by the Isaac Sim bridge and contains raw RGB bytes
        for the D435i camera mounted on G1's head — same perspective as the
        training dataset (G1_Dex1_MountCameraRedGripper).  Falls back to the
        Isaac editor top-down frame (/tmp/isaac_frame.png) when not available.
        """
        try:
            import os as _os
            if not _os.path.exists(CAMERA_SHM_PATH):
                print(f"[UnifoLM] Camera SHM not found at {CAMERA_SHM_PATH} "
                      f"— will use /tmp/isaac_frame.png", flush=True)
                return
            self.camera_shm_map = open(CAMERA_SHM_PATH, "rb")
            print(f"[UnifoLM] ✓ Camera SHM: {CAMERA_SHM_PATH} "
                  f"(D435i head cam, {CAMERA_SHM_W}×{CAMERA_SHM_H} RGB)", flush=True)
        except Exception as e:
            print(f"[UnifoLM] Camera SHM init warning: {e}", flush=True)

    def _read_camera_shm(self) -> Optional[np.ndarray]:
        """Read one frame from the D435i head-camera SHM.

        Returns:
            uint8 BGR array [720, 1280, 3] or None on error.
        """
        if self.camera_shm_map is None:
            return None
        try:
            self.camera_shm_map.seek(0)
            raw = self.camera_shm_map.read(CAMERA_SHM_SIZE)
            if len(raw) < CAMERA_SHM_SIZE:
                return None
            rgb = np.frombuffer(raw, dtype=np.uint8).reshape(CAMERA_SHM_H, CAMERA_SHM_W, 3)
            # Convert RGB → BGR so the rest of the pipeline (cv2) is uniform
            return np.ascontiguousarray(rgb[:, :, ::-1])
        except Exception:
            return None

    def _load_model_real(self, checkpoint: str) -> None:
        """Load model using OmegaConf config and instantiate_from_config."""
        print(f"[UnifoLM] Loading model from checkpoint: {checkpoint}", flush=True)

        try:
            # Load config
            config_path = "/workspace/wam/config_model.yaml"
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config not found: {config_path}")

            config = OmegaConf.load(config_path)
            print(f"[UnifoLM] Config loaded", flush=True)

            # Instantiate model
            print(f"[UnifoLM] Instantiating model from config...", flush=True)
            self.model = instantiate_from_config(config.model)
            self.model = self.model.to(self.device)
            self.model.eval()

            # Load checkpoint
            if os.path.exists(checkpoint):
                self.model = load_model_checkpoint(self.model, checkpoint)
                print(f"[UnifoLM] Model ready for inference", flush=True)
            else:
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

        except Exception as e:
            print(f"[UnifoLM] ERROR loading model: {e}", flush=True)
            import traceback
            traceback.print_exc()
            self.model = None

    def _get_camera_image(self) -> np.ndarray:
        """Get camera image, preferring the robot head camera (D435i SHM).

        Priority:
          1. /run/mws/camera.rgb  — D435i head camera (same perspective as
             training data G1_Dex1_MountCameraRedGripper).  PREFERRED.
          2. /tmp/isaac_frame.png — Isaac Sim editor top-down view (fallback;
             distribution mismatch with training data causes dark video output).
          3. Last cached image.
          4. Black frame (zeros).

        Returns BGR uint8 array.
        """
        try:
            # 1. Robot head camera from SHM (preferred — matches training distribution)
            shm_img = self._read_camera_shm()
            if shm_img is not None:
                self.last_camera_image = shm_img
                return shm_img

            # 2. Isaac Sim editor frame (top-down, fallback)
            isaac_path = Path("/tmp/isaac_frame.png")
            if isaac_path.exists():
                img = cv2.imread(str(isaac_path))
                if img is not None:
                    self.last_camera_image = img
                    return img

            # 3. Last cached frame
            if self.last_camera_image is not None:
                return self.last_camera_image

            # 4. Black frame
            return np.zeros((CAMERA_SHM_H, CAMERA_SHM_W, 3), dtype=np.uint8)

        except Exception as e:
            print(f"[UnifoLM] Camera error: {e}", flush=True)
            return np.zeros((CAMERA_SHM_H, CAMERA_SHM_W, 3), dtype=np.uint8)

    def _prepare_image_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert BGR numpy image to normalized float tensor [C, H, W] in [-1, 1].

        Resizes to MODEL_INPUT_H x MODEL_INPUT_W (320x512) as expected by the model.
        """
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (MODEL_INPUT_W, MODEL_INPUT_H), interpolation=cv2.INTER_LINEAR)
        # Normalize to [-1, 1]: (x/255 - 0.5) * 2 = x/127.5 - 1
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float()
        img_tensor = (img_tensor / 255.0 - 0.5) * 2.0
        return img_tensor  # [3, 320, 512]

    def _prepare_state_tensor(self, q: np.ndarray) -> torch.Tensor:
        """Pad 14-DOF arm state to 16-DOF and min-max normalise to [-1, 1].

        The model was trained with agent_state_dim=16 and min_max normalisation
        using G1 Pack Camera dataset stats.  Dims 14–15 are padded with zeros
        (gripper channels in training data; they map to ≈ -1 after normalisation,
        which the model has seen during training for the padded-robot case).

        Normalisation formula:
            norm = (x - min) / (max - min + 1e-8) * 2 - 1   → [-1, 1]
        """
        state_16 = np.zeros(self.model_state_dim, dtype=np.float32)
        state_16[:len(q)] = q
        state_tensor = torch.from_numpy(state_16).float()  # [16]

        if self.norm_stats is not None:
            state_min = self.norm_stats['observation.state']['min']  # [16]
            state_max = self.norm_stats['observation.state']['max']  # [16]
            state_tensor = (state_tensor - state_min) / (state_max - state_min + 1e-8) * 2.0 - 1.0
            state_tensor = torch.clamp(state_tensor, -1.0, 1.0)

        return state_tensor  # [16]

    def __call__(self, state: RobotState) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Production inference: returns (action_traj, state_traj, video_output).

        action_traj shape: (16, 14) — 16 timesteps × 14 DOF (both arms).
        Send action_traj[0] directly to robot as joint targets.
        """
        try:
            self.step_count += 1

            if self.step_count % 10 == 0:
                print(f"[UnifoLM] Step {self.step_count} - inference starting", flush=True)

            if self.model is None:
                print(f"[UnifoLM] Model not loaded, returning zero trajectories", flush=True)
                return (
                    torch.zeros((self.horizon, self.agent_action_dim)),
                    torch.zeros((self.horizon, self.agent_action_dim)),
                    None,
                )

            # --- Prepare inputs ---
            image = self._get_camera_image()
            # Extract arm joints from LowState: indices 14-27 (left arm 14-20, right arm 21-27)
            q_raw = np.array(state.q, dtype=np.float32)[G1_ARM_JOINT_START:G1_ARM_JOINT_END]

            img_tensor = self._prepare_image_tensor(image)      # [3, 320, 512]
            state_tensor = self._prepare_state_tensor(q_raw)    # [16]

            # --- Update observation history ---
            self.obs_image_history.append(img_tensor)
            self.obs_state_history.append(state_tensor)
            self.action_history.append(torch.zeros(self.agent_action_dim))

            # --- Warmup: wait until we have n_obs_steps frames ---
            if len(self.obs_image_history) < self.n_obs_steps:
                if self.step_count % 5 == 0:
                    print(f"[UnifoLM] Warming up: {len(self.obs_image_history)}/{self.n_obs_steps}", flush=True)
                return (
                    torch.zeros((self.horizon, self.agent_action_dim)),
                    torch.zeros((self.horizon, self.agent_action_dim)),
                    None,
                )

            # --- Run real DDIM inference ---
            with torch.no_grad():
                try:
                    # Stack history into batched tensors
                    # obs images: [B, T, C, H, W] = [1, 2, 3, 320, 512]
                    img_seq = torch.stack(list(self.obs_image_history))   # [T, C, H, W]
                    img_input = img_seq.unsqueeze(0).to(self.device)       # [1, T, C, H, W]

                    # obs states: [B, T, D] = [1, 2, 16]
                    state_seq = torch.stack(list(self.obs_state_history))  # [T, D]
                    state_input = state_seq.unsqueeze(0).to(self.device)   # [1, T, D]

                    # zero action placeholder: [B, horizon, D] = [1, 16, 16]
                    # agent_action_pos_emb has shape [1, horizon, 1024] — one slot per
                    # predicted timestep, NOT per obs step.
                    action_input = torch.zeros(
                        1, self.horizon, self.model_action_dim,
                        device=self.device
                    )

                    observation = {
                        'observation.images.top': img_input,
                        'observation.state': state_input,
                        'action': action_input,
                    }

                    if self.step_count % 10 == 0:
                        print(f"[UnifoLM] Calling DDIM sampler, ddim_steps={self.ddim_steps}", flush=True)

                    # Real DDIM inference
                    video_output, pred_actions, pred_states = image_guided_synthesis(
                        model=self.model,
                        prompts=[self.prompt],
                        observation=observation,
                        noise_shape=NOISE_SHAPE,
                        ddim_steps=self.ddim_steps,
                        ddim_eta=self.ddim_eta,
                        unconditional_guidance_scale=1.0,
                        fs=MODEL_FPS,
                        timestep_spacing='uniform',
                        guidance_rescale=0.0,
                    )

                    # pred_actions: [B, horizon, model_action_dim] = [1, 16, 16]
                    # pred_states:  [B, horizon, model_state_dim]  = [1, 16, 16]
                    if pred_actions is not None and pred_actions.shape[0] == 1:
                        pred_actions = pred_actions.squeeze(0)  # [16, 16]
                    if pred_states is not None and pred_states.shape[0] == 1:
                        pred_states = pred_states.squeeze(0)    # [16, 16]

                    # Unnormalise actions from model space [-1, 1] → actual joint angles (rad)
                    # pred_actions is [16, 16] here (already squeezed above)
                    # Must unnormalise BEFORE trimming so the 16 stats dims align properly.
                    pred_actions_unnorm = self._unnormalize_actions(pred_actions)  # [16, 16] rad

                    # Trim from 16-dim to 14-dim (G1 arm DOF)
                    action_traj = pred_actions_unnorm[:, :self.agent_action_dim].cpu()  # [16, 14]
                    state_traj = pred_states[:, :self.agent_action_dim].cpu()           # [16, 14]

                except Exception as e:
                    print(f"[UnifoLM] Inference failed: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                    # Do NOT fall back to randn — return zeros so we can tell if inference worked
                    action_traj = torch.zeros((self.horizon, self.agent_action_dim))
                    state_traj = torch.zeros((self.horizon, self.agent_action_dim))
                    video_output = None

            # --- Apply temporal ensemble ---
            pred_actions_batched = action_traj.unsqueeze(0)  # [1, 16, 14]
            actions_ensemble = self.temporal_ensembler.update(pred_actions_batched)

            # Log every 10 steps
            if self.step_count % 10 == 0:
                action_0 = actions_ensemble[0, 0].cpu().numpy()
                action_0_norm = float(np.linalg.norm(action_0))
                action_full_norm = float((action_traj ** 2).sum() ** 0.5)
                print(
                    f"[UnifoLM] Step {self.step_count}: "
                    f"action[0]_norm={action_0_norm:.4f} "
                    f"traj_norm={action_full_norm:.4f}",
                    flush=True,
                )

            # Record latest step action in history
            action_0_val = actions_ensemble[0, 0].cpu()
            self.action_history.append(action_0_val.float())

            return action_traj, state_traj, video_output

        except Exception as e:
            print(f"[UnifoLM] FATAL ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return (
                torch.zeros((self.horizon, self.agent_action_dim)),
                torch.zeros((self.horizon, self.agent_action_dim)),
                None,
            )
