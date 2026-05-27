# Changelog

## [Unreleased] — branch `G1_unifolm_with_GS`

### 2026-05-27 — Real DDIM inference, state norm, action unnorm, SHM camera

**Commit:** `a19d73b`

#### Что добавлено / изменено

**`src/models/unifolm.py`** — полная реализация UnifoLM-WMA-0:

- `_load_normalization_stats()` — загрузка G1 pack camera min/max stats из
  `stats.safetensors` (bundled в unifolm repo, всегда доступен в контейнере)
- `_prepare_state_tensor()` — min-max нормализация 14-DOF стейта → [-1, 1]
  перед подачей в модель (ранее подавались сырые радианы — неправильно)
- `_unnormalize_actions()` — обратная min-max из [-1, 1] → реальные углы (рад)
  применяется **до** обрезки 16→14 DOF
- `_init_camera_shm()` / `_read_camera_shm()` — чтение D435i из SHM
  `/run/mws/camera.rgb`; та же перспектива что в training data → фикс domain mismatch
- `_get_camera_image()` — приоритет: SHM > `/tmp/isaac_frame.png` > cached > zeros
- `image_guided_synthesis()` — полный DDIM inference, mirrors `real_eval_server.py`
- `ACTTemporalEnsembler` — экспоненциальный temporal ensemble
- `MODEL_FPS = 15` (исправлено с 10; источник: `run_real_eval_server.sh`)
- Константы: `G1_PACK_CAMERA_STATS_PATH`, `CAMERA_SHM_PATH/W/H/SIZE`

**`src/recording.py`** — `save_model_output()`:

- cv2 всегда импортируется (исправлен `UnboundLocalError`)
- `np.ascontiguousarray()` для отрицательных stride от `[::-1]` flip

**`src/config_model.yaml`** — создан:
- `n_obs_steps_imagen: 2`, `agent_state_dim/action_dim: 16`, `input_dim: 16`
- `decision_making_only: True`, `temporal_length: 16`

**`src/configs/train/meta.json`** — создан:
- Метаданные форм для `MultiImageObsEncoder`

**`src/main.py`** — обновлён:
- Логирование 14-DOF joint targets вместо velocity commands
- Unpacking `(action_traj, state_traj, video_output)` из модели

**`CLAUDE.md`** — создан (актуальные инструкции для агента)

**`README.md`** — полностью переписан (правильная архитектура)

**`UNIFOLM_DEPLOYMENT.md`** — переписан (реальный деплой)

#### Результат

```
[UnifoLM] ✓ Loaded G1 pack camera normalization stats
[UnifoLM] ✓ Camera SHM: /run/mws/camera.rgb (D435i head cam, 1280×720 RGB)
[WAM] loop=2  action_0[0:3]=[-0.420 +0.415 -0.166]  norm=1.185  traj_norm=4.960
[RECORDING] Saved video: media/model_output/output_000002.mp4
```

---

## [0.2.0] — 2026-05-25

### Recording system + DDS working

- `src/recording.py` — MediaRecorder (MP4, PNG, TXT, CSV)
- Shared memory camera `/run/mws/camera.rgb` (D435i, 1280×720)
- Docker volume mount для `/workspace/wam/media`

---

## [0.1.0] — 2026-05-23

### Базовый стек

- Isaac Sim + GEAR-SONIC + ROS2 bridge работает
- DDS коммуникация (rt/lowstate, rt/lowcmd) подтверждена
- `sim-ros2-bridge` критичен для трансформации 50D→994D наблюдений GEAR-SONIC
