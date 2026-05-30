# CoRL 2026 — WAM Stack (CLAUDE.md)

Инструкции для Claude Code агента. Читай прежде чем менять что-либо.

---

## Research Plan (big picture)

**Цель:** Проверить, может ли WAM, дообученный на (G1, Task 1) и (G1, Task 2), без дообучения выполнять (UBTech Walker, Task 2).

**Команда:** Сергей Курчев + Jeffrin Sam + Артём Лыков.

**Матрица экспериментов:**

| Fine-tune | Evaluate |
|-----------|----------|
| (G1, Task 1) | — |
| (G1, Task 2) | — |
| (UBTech, Task 1) | — |
| *(UBTech, Task 2)* | ← **здесь тестируем transfer** |

**Jeffrin's models** (не трогай): NVIDIA Cosmos, LingBot-VA, Dream Zero.

---

## Статус на 28 мая 2026

✅ **ПОЛНЫЙ ПАЙПЛАЙН РАБОТАЕТ. РУКИ ДВИГАЮТСЯ В SUPPORT MODE.**

```
[UnifoLM] ✓ Loaded G1 pack camera normalization stats
[UnifoLM] ✓ Camera SHM: /run/mws/camera.rgb (D435i head cam, 1280×720 RGB)
[WAM] rt/lowstate received. Control loop starting at 10 Hz.
[WAM] loop=2  action_0[0:3]=[-0.238 +0.560 +0.168]  norm=0.935  traj_norm=3.761
[RECORDING] Saved video: /workspace/wam/media/steps/step_000001/model_output.mp4
# Arm joints changing in Isaac Sim (confirmed 28 May):
#   arm_q[14:28] sample1: [0.3215, 0.1123, 0.0244, ...]
#   arm_q[14:28] sample2: [0.5064, 0.028,  0.0974, ...]  ← delta up to 0.18 rad/step
```

**Patch infrastructure (28 May 2026):**
- `scripts/patch_mws_dimos.py` — idempotent patch runner, applied before every `docker compose up`
- `PATCHES.md` — full documentation of external code changes
- Patch 1: g1_sim.py arm-in-support-mode → arms move during startup wrench phase
- Patch 2: unifolm attention.py → removes xformers assert, enables vanilla attention
- Patch 3: g1_sim.py kinematic-robot → `fix_root_link=True` when `G1_KINEMATIC_ROBOT=1`

**Scene system (28 May 2026):**
- `scene_config.yaml` — object positions (cm) + physics config for mws_office.usdz
- `scripts/patch_scene.py` — edits USD inside USDZ in-place (tmpdir), no loose .usd ever left on disk
- Runs automatically in `deploy.sh` step 3; `G1_KINEMATIC_ROBOT=1` default in compose.yml

---

## Архитектура пайплайна

```
Isaac Sim 5.1
  ├── G1 physics (29 DOF)
  ├── D435i sim camera → /run/mws/camera.rgb (SHM, 1280×720 RGB)
  └── rt/lowstate → DDS lo:42 (50 Hz)
          │
          ▼
   WAM Inference (10 Hz)
   main.py → UnifoLMModel.__call__(RobotState)
   │
   ├── 1. Камера: /run/mws/camera.rgb (SHM, ПРИОРИТЕТ)
   │             fallback: /tmp/isaac_frame.png
   │             → resize 320×512 → norm (x/255-0.5)*2 → [-1,1]
   │
   ├── 2. Стейт: state.q[14:28]  (14-DOF G1 arms, slots 14-27)
   │             pad → [16] zeros → min-max norm → [-1,1]
   │             (stats: unitree_g1_pack_camera/meta_data/stats.safetensors)
   │
   ├── 3. image_guided_synthesis()  ← mirrors real_eval_server.py
   │   ├── get_latent_z(obs_frames)             # encode VAE
   │   ├── c_concat = latent_last_frame × 16    # [1,4,16,40,64]
   │   ├── c_crossattn = [state|text|clip_img]  # [1,95,1024]
   │   ├── c_crossattn_action = [imgs, states]  # last 2 steps
   │   └── DDIMSampler.sample(S=16, eta=1.0, fps=15)
   │       └── → (latent [1,4,16,40,64], actions [1,16,16], states [1,16,16])
   │
   ├── 4. decode_first_stage(latent)  → video [1,3,16,320,512]
   ├── 5. _unnormalize_actions(actions)  → [16,16] radians
   ├── 6. trim [:, :14]  → action_traj [16,14]
   └── 7. temporal_ensemble()  → сглаженные действия
          │
          ▼  rt/lowcmd → Isaac Sim (arm_q[14:28], LEG hold, gripper passive)
   Recording:
   ├── model_output/output_*.mp4
   ├── robot_camera/camera_*.png
   ├── input_frames/state_*.txt
   └── command_logs/*.csv
```

---

## Файловая структура

```
wam-stack/
├── CLAUDE.md                    ← этот файл
├── README.md                    ← полная документация
├── PATCHES.md                   ← документация всех патчей внешнего кода
├── UNIFOLM_DEPLOYMENT.md        ← деплой и troubleshooting модели
├── scene_config.yaml            ← позиции объектов сцены (cm) + физика
├── compose.yml                  ← Docker Compose: sim-isaac + wam
├── docker/wam/Dockerfile        ← WAM контейнер
├── scripts/
│   ├── deploy.sh                ← деплой: патчи → compose up → restart sim-isaac
│   ├── patch_mws_dimos.py       ← идемпотентный патч-runner для mws-dimos/unifolm
│   └── patch_scene.py           ← правит USD внутри USDZ (tmpdir), без loose .usd
├── src/                         ← bind-mount → /workspace/wam
│   ├── main.py                  ← control loop 10 Hz
│   ├── dds_interface.py         ← DDS subscriber
│   ├── recording.py             ← MediaRecorder (MP4, PNG, CSV)
│   ├── config_model.yaml        ← OmegaConf конфиг модели
│   ├── configs/train/meta.json  ← формы для MultiImageObsEncoder
│   └── models/
│       ├── unifolm.py           ✅ DONE — real DDIM inference
│       └── eva.py               🔲 TODO — заглушка
├── checkpoints/                 ← bind-mount ro
│   └── unifolm_wma_dual.ckpt    ← ≈16 GB
└── repos/
    └── unifolm/                 ← bind-mount ro → /workspace/unifolm
        └── src/unifolm_wma/     ← library via PYTHONPATH
```

---

## Критические зависимости

### mws-dimos (EXTERNAL, READ-ONLY)

**Репо:** https://github.com/MWS-Physical-AI/mws-dimos  
**Ветка:** `feat/real-transfer` (НЕ main!)  
**На сервере:** `/root/skurchev/workspace/mws-dimos/`

Мы используем его pre-built Docker-образы и assets как bind-mount. **Не копируй, не модифицируй.**

Что нам нужно:
- `mws-dimos/sim-isaac:unitree-lab-5.1` — Isaac Sim + G1 сцена
- `deploy/images/isaac-sim/entrypoint.sh` → монтируется как `/entrypoint.sh`
- `assets/robots/g1/` → монтируется как `/workspace/robot_assets`
- `/root/skurchev/workspace/assets/office_demo.usdz` → сцена (рядом с mws-dimos)

### unifolm (EXTERNAL, READ-ONLY + patch)

**Репо:** https://github.com/unitreerobotics/unifolm-world-model-action  
**На сервере:** `/root/skurchev/workspace/wam-stack/repos/unifolm/`  
**В контейнере:** `/workspace/unifolm/` (PYTHONPATH, НЕ pip install)

**КРИТИЧЕСКИЙ ПАТЧ (только на сервере):**
```
repos/unifolm/src/unifolm_wma/modules/attention.py
```
Удалена строка `assert 1 > 2` (~строка 128 в `forward()`).  
Причина: xformers не работает (собран для PyTorch 2.10+cu128, контейнер — 2.7.0+cu126).  
Патч активирует vanilla PyTorch attention, которое полностью реализовано ниже assert.

**Normalization stats (bundled):**
```
/workspace/unifolm/examples/world_model_interaction_prompts/
  transitions/unitree_g1_pack_camera/meta_data/stats.safetensors
```
Ключи после unflatten: `observation.state/{min,max}`, `action/{min,max}` — тензоры [16].

### G1 checkpoint

**Путь:** `/workspace/wam/checkpoints/unifolm_wma_dual.ckpt`  
Обучен на нескольких датасетах Unitree (включая G1_Dex1_MountCameraRedGripper_Dataset).  
Dual = видео-диффузия + action head одновременно.

---

## Ключевые реализационные детали

### src/models/unifolm.py

**Константы:**
```python
MODEL_INPUT_H, W = 320, 512          # размер кадра для модели
MODEL_STATE_DIM = MODEL_ACTION_DIM = MODEL_HORIZON = 16
MODEL_FPS = 15                        # 30 / frame_stride=2 (из run_real_eval_server.sh)
NOISE_SHAPE = [1, 4, 16, 40, 64]     # [B, C, T, h=320//8, w=512//8]
G1_PACK_CAMERA_STATS_PATH = "/workspace/unifolm/examples/.../stats.safetensors"
CAMERA_SHM_PATH = "/run/mws/camera.rgb"  # 1280×720 RGB, raw bytes
```

**Класс UnifoLMModel:**
- `_load_normalization_stats()` — загружает stats.safetensors через safetensors.torch
- `_unnormalize_actions(norm)` — `(norm+1)/2 * (max-min) + min` → радианы
- `_init_camera_shm()` / `_read_camera_shm()` — D435i из SHM `/run/mws/camera.rgb`
- `_get_camera_image()` — приоритет: SHM > /tmp/isaac_frame.png > cached > zeros
- `_prepare_state_tensor(q)` — pad 14→16, min-max norm → [-1,1]
- `_prepare_image_tensor(img)` — BGR→RGB, resize, `/255*2-1`
- `image_guided_synthesis()` — полный DDIM inference (mirrors real_eval_server.py)

**Порядок unnorm:**
```python
pred_actions = pred_actions.squeeze(0)          # [16, 16]
pred_actions_unnorm = _unnormalize_actions(pred_actions)  # [16, 16] radians
action_traj = pred_actions_unnorm[:, :14]       # [16, 14] G1 arms
```
Unnorm **до** обрезки до 14 DOF — иначе stats dims не совпадут.

### config_model.yaml

Ключевые параметры:
- `n_obs_steps_imagen: 2` — история наблюдений (deque maxlen=2)
- `agent_state_dim: 16`, `agent_action_dim: 16` — универсальная размерность
- `conditioning_key: hybrid` — concat latent + cross-attention
- `decision_making_only: True` — action head активен

### compose.yml

Shared memory volume `sim_bridge_shm` пробрасывается в оба контейнера (`sim-isaac` и `wam`) через `/run/mws`. Именно через него D435i симулированная камера доступна WAM контейнеру.

**Env vars sim-isaac:**
- `G1_KINEMATIC_ROBOT=1` (default) — `fix_root_link=True`: корень таза закреплён в spawn,
  суставы управляются PD-контроллером по командам WAM, коллизии активны.
  Робот никогда не падает, рукой можно взаимодействовать с кубом/коробкой.
- `G1_KINEMATIC_ROBOT=0` — обычная физика + startup support wrench.

**Важно: GEAR-SONIC WBC не используется в нашем пайплайне.**
Мы подключаемся напрямую: Isaac Sim → DDS rt/lowstate → WAM inference → DDS rt/lowcmd.
Аналогично тому, как сделано с UnifoLM (Patch 1 держит руки управляемыми в SUPPORT режиме).

### scene_config.yaml

Объекты сцены `mws_office.usdz`. Единицы — **сантиметры** (metersPerUnit=0.01, Y-up).

```yaml
# Значение из Isaac Sim Properties (в метрах) → умножить на 100 → получишь значение здесь
```

Изменения вступают в силу после `bash scripts/deploy.sh` (или `python3 scripts/patch_scene.py && docker compose restart sim-isaac`).

---

## Сервер

```
Host x32-techgov-GPU-01
  HostName 176.109.83.84
  Port 2221
  User root
  IdentityFile ~/.ssh/id_ed25519
```

**Workspace на сервере:** `/root/skurchev/workspace/`

```bash
# Стандартное подключение
eval $(ssh-agent -s) && ssh-add ~/.ssh/id_ed25519
ssh -p 2221 root@176.109.83.84

# Рестарт WAM (src/ bind-mounted, image пересобирать не нужно)
docker restart wam-inference

# Логи
docker logs -f wam-inference

# Очистка медиа
find /root/skurchev/workspace/wam-stack/media -type f -delete
```

---

## Pre-approved Actions (Claude может делать без дополнительного разрешения)

✅ SSH к GPU-01 для работы с wam-stack  
✅ `docker compose up/down/restart` (без `--volumes`)  
✅ `docker logs`, `docker exec` (readonly)  
✅ Редактировать файлы в `wam-stack/src/`  
✅ Читать файлы из `repos/unifolm/` (read-only)

❌ Требует явного разрешения:
- Обучение / GPU-intensive jobs
- Изменения за пределами `wam-stack/`
- `docker compose down --volumes` или `git reset --hard`
- Изменения в `mws-dimos/`
- Пересборка Docker образа (`docker compose build`)

---

## DO's and DON'Ts

### DO ✓

- Используй DDS для коммуникации — не менять без понимания FastDDS
- Перед запуском на сервере синхронизируй локальные файлы (`scp`)
- Обновляй `README.md` и `CLAUDE.md` при изменении архитектуры
- При проблемах сначала читай `/workspace/unifolm/scripts/evaluation/real_eval_server.py`

### DON'T ✗

- **НЕ** модифицируй mws-dimos — только read-only reference
- **НЕ** делай `pip install unifolm_wma` — используй PYTHONPATH bind-mount
- **НЕ** меняй DDS config без понимания ROS2 FastDDS
- **НЕ** запускай training на GPU-01 без явного разрешения
- **НЕ** добавляй Claude как co-author в коммиты

---

## EVA Model — Plan & Blockers (29 May 2026, ветка EVA_without_GS)

### Что такое EVA

**EVA** = WAN 2.1 14B (Alibaba I2V diffusion, flow-matching) + IDM (ResNet50, inverse dynamics).

Pipeline (trajectory-replay mode, НЕ real-time):
```
1. Snapshot  — 1 кадр камеры (640×480) + текстовый промпт
2. Plan      — WAN генерирует 49 кадров @ 16 FPS (≈3 сек траектории)
               40 шагов flow-matching ODE solver (Euler/DPM-Solver++)
               Реальное время на A100: ориентировочно 3–8 мин (НЕ проверено)
3. Decode    — IDM: sliding window (frame_i, frame_i+2, frame_i+4) → 14D joint action
               Denorm: actions * train_std + train_mean (RoboTwin stats → нужна адаптация)
4. Execute   — replay 49 actions via DDS rt/lowcmd @ 16 Hz (3 сек в симуляторе)
5. Repeat
```

**Без GEAR-SONIC** — пайплайн подключается напрямую как UnifoLM:
Isaac Sim → DDS rt/lowstate → EVA inference → DDS rt/lowcmd.
Patch 1 (arm-in-support-mode) также актуален.

### Поток данных EVA

```
Isaac Sim
  ├── /run/mws/camera.rgb (SHM, 1280×720) → resize → 640×480
  └── rt/lowstate DDS → arm joint positions (для логирования)
          ↓
  EVA контейнер (eva service)
  ├── WAN 2.1 14B I2V → video [49, 640, 480, 3]
  ├── IDM ResNet50: (B, 9, 512, 512) → [49, 14D] actions
  └── DDS publish rt/lowcmd → arm_q[14:28]  (аналогично unifolm.py)
```

Посредник между видео и роботом: **IDM (ResNet50)**.
Референс robot interface: `src/dds_interface.py::publish_lowcmd()` (тот же что у UnifoLM).

### Чекпоинты (подтверждено 29 May 2026)

**EVA:** `RobbinWang123/EVA` — Apache 2.0, публичный
```bash
huggingface-cli download RobbinWang123/EVA --include "IDM_singleview.pt" --local-dir ./checkpoints/eva
huggingface-cli download RobbinWang123/EVA --include "eva_i2v_14B.ckpt"  --local-dir ./checkpoints/eva
```
- `IDM_singleview.pt` — 348 MB (ResNet-based IDM, single-view)
- `eva_i2v_14B.ckpt` — 32.8 GB (WAN 2.1 14B fine-tuned с RL на RoboTwin, НЕ base WAN)

**WAN 2.1 base (если нужен отдельно):** `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers` — Apache 2.0
Полный размер: ~82 GB (включает T5-XXL 11.4 GB + CLIP 4.77 GB + 7 shard-файлов ~65 GB).
Для EVA нужен только `eva_i2v_14B.ckpt` — он уже содержит всё (32.8 GB merged).

### VRAM-бюджет (A100 80 GB)

| Компонент | VRAM без оптимизаций | VRAM с group offload |
|-----------|---------------------|----------------------|
| WAN 14B bf16 веса | ~28 GB | ~13 GB (offload блоками) |
| T5-XXL энкодер | ~11 GB | **0 GB (CPU offload)** |
| CLIP энкодер | ~5 GB | ~5 GB |
| VAE decode | ~4–6 GB | ~4–6 GB |
| IDM ResNet | < 1 GB | < 1 GB |
| Isaac Sim | ~6–8 GB | ~6–8 GB |
| **Итого** | **~55–59 GB** | **~29–33 GB ✅** |

### Рекомендуемая стратегия памяти

**Приоритет 1 — Group Offloading (официальный путь diffusers, без потери качества):**
```python
from diffusers.hooks.group_offloading import apply_group_offloading
# T5 на CPU (экономит 11 GB):
apply_group_offloading(text_encoder, onload_device=cuda, offload_device=cpu,
                       offload_type="block_level", num_blocks_per_group=4)
# Transformer offload по блокам (fits 14B в ~13 GB):
transformer.enable_group_offload(onload_device=cuda, offload_device=cpu,
                                 offload_type="leaf_level", use_stream=True)
```
Итог: ~30 GB VRAM. Замедление ~15–20%, качество = full precision.

**Приоритет 2 — GGUF Q4_K_M (если нужно ещё меньше):**
- `city96/Wan2.1-I2V-14B-480P-gguf` — трансформер 11.3 GB (Apache 2.0)
- Работает через ComfyUI-GGUF nodes, не нативный diffusers
- Community: хорошее качество для Q4_K_M

**НЕ использовать:** bitsandbytes int4 на I2V — known issue #11006 (сломанный video output).

### Блокеры (финальный статус)

| # | Блокер | Статус |
|---|--------|--------|
| ~~1~~ | ~~IDM checkpoint недоступен~~ | ✅ Решён — `RobbinWang123/EVA`, Apache 2.0 |
| ~~2~~ | ~~WAN checkpoint и лицензия~~ | ✅ Решён — `eva_i2v_14B.ckpt` 32.8 GB, Apache 2.0 |
| ~~5~~ | ~~GEAR-SONIC watchdog~~ | ✅ Неактуально — GEAR-SONIC не используется |
| **3** | **RoboTwin→G1 IDM normalization**: circular dependency | ⚠️ Bootstrap через G1 URDF joint limits |
| **4** | **xformers в Dockerfile**: нужен для OOM-safe inference | ⚠️ Нужен Dockerfile |

### Ссылки в коде EVA

| Нужно | Файл | Путь |
|-------|------|------|
| Inference entry | `main.py` + `experiment=exp_inference` | `repos/eva/main.py` |
| IDM forward | `idm/idm.py::IDMResNetPlus.forward()` | input `(B,9,512,512)` → `[14]` |
| IDM конфиг | `configurations/model/idm_resnet_plus.yaml` | `num_frames:3, frame_stride:2` |
| Denormalization | `idm/idm.py` | `train_mean`, `train_std` |
| Kinematic limits | `flow_grpo/idm_reward.py` | max vel 2.66–3.32 rad/s |
| WAN sampling | `algorithms/wan_i2v*.py::sample()` | flow-matching ODE, не DDIM |

---

## LingBot-VA — Plan & Integration (30 May 2026, ветка LingBot-VA_without_GS)

### Что такое LingBot-VA

**LingBot-VA** = Causal World Model (Wan2.2-5B видео backbone + action stream 350M).
Dual-stream Mixture-of-Transformers, autoregressive, flow-matching (Euler solver).
Общий размер: **5.3B параметров**.

Paper: https://arxiv.org/abs/2601.21998 | GitHub: https://github.com/robbyant/lingbot-va
Checkpoints: https://huggingface.co/robbyant/ | License: Apache 2.0

### Архитектура и выход

```
RGB кадр(ы) + текстовый промпт (T5)
        ↓
Video stream (Wan2.2-5B) — 3 шага Euler до s=0.6
        ↓ предсказанные будущие латентные кадры
Action stream (350M) — 10 шагов Euler до s=1.0
        ↓
30D action chunk (K=4 video frames × τ=4 = 16 robot actions)
```

**Структура 30D action вектора:**
```
[0:7]   left arm EEF pose (XYZ + quaternion)
[7:14]  left arm joint angles  ← нам нужно это (7 DOF)
[14]    left gripper
[15:22] right arm EEF pose
[22:29] right arm joint angles ← нам нужно это (7 DOF)
[29]    right gripper
```
Для G1 извлекаем `action[[7:14, 22:29]]` → 14D joint targets, напрямую в rt/lowcmd.

### Скорость (КЛЮЧЕВОЕ ПРЕИМУЩЕСТВО)

| Параметр | Значение |
|----------|---------|
| Размер модели | 5.3B (vs EVA 14B, UnifoLM ~16B) |
| VRAM с offload | **~24 GB** |
| Video inference | 3 шага Euler |
| Action inference | 10 шагов Euler |
| Robot execution | **50 Hz** |
| Режим | **Async**: пока робот выполняет chunk_t, модель считает chunk_{t+1} |
| Эффективно | Real-time closed-loop при 50 Hz |

Async pipeline (Algorithm 2 в paper):
```
t=0: inference(obs_0) → action_chunk_0        # ~0.5-1 сек
t=1: robot выполняет action_chunk_0 @ 50 Hz   # параллельно
     inference(obs_FDM_1) → action_chunk_1    # FDM-grounded prediction
t=2: robot выполняет action_chunk_1           # и т.д.
```

### Поток данных и посредники

```
Isaac Sim
  ├── /run/mws/camera.rgb (SHM, 1280×720) → resize → нужный размер
  └── rt/lowstate DDS → arm joint positions
          ↓
  WAM control loop (src/main.py)
    ├── _read_camera_shm() → frame
    ├── WebSocket client → LingBot-VA server:29056
    │     отправляет: {obs: [frame], state: action, reset: bool, prompt: str}
    │     получает:   {action: [30D numpy array]}
    ├── extract: action[[7:14, 22:29]] → 14D joint targets
    └── DDS publish rt/lowcmd → arm_q[14:28]
          ↓
  LingBot-VA Server (отдельный контейнер lingbot-server)
    wan_va/wan_va_server.py --port 29056
```

**Посредник:** WebSocket (порт 29056, localhost). Никакого DDS на стороне модели.

### Референс в коде

| Что | Где |
|-----|-----|
| Сервер (entry point) | `repos/lingbot-va/wan_va/wan_va_server.py` |
| Пример клиента | `repos/lingbot-va/evaluation/robotwin/eval_polict_client_openpi` |
| Async algorithm | Paper Algorithm 2, Section 3.3 |
| Config (порт, модель) | `repos/lingbot-va/configs/robotwin.yaml` |
| **КРИТИЧНО**: `attn_mode` | `<checkpoint>/transformer/config.json` → поменять `"flex"` → `"torch"` |

### Чекпоинты

| Checkpoint | HF repo | Размер | Обучен на |
|-----------|---------|--------|-----------|
| `lingbot-va-base` | `robbyant/lingbot-va-base` | 24.4 GB | Общий pretraining (~16K ч данных) |
| `lingbot-va-posttrain-robotwin` | `robbyant/lingbot-va-posttrain-robotwin` | ~24 GB | RoboTwin 2.0 (50 задач, bimanual) |
| `lingbot-va-posttrain-libero-long` | `robbyant/lingbot-va-posttrain-libero-long` | ~24 GB | LIBERO-Long |

**Для G1 специальных весов нет.** Путь адаптации:
- Старт с `lingbot-va-base`
- Post-training на G1 данных (paper: достаточно **50 демонстраций**, 3K шагов, LR 1e-5)
- Это достижимо — можно собрать в Isaac Sim через teleop или через UnifoLM

### Контейнер

Новый `docker/lingbot/Dockerfile`:
```
Base: nvidia/cuda:12.6.0-runtime-ubuntu22.04
Python: 3.10.16
PyTorch: 2.9.0+cu126
Ключевые зависимости:
  - diffusers==0.36.0
  - transformers==4.55.2
  - websockets
  - flash-attn --no-build-isolation   (обязателен для attn_mode="flashattn")
  - einops, msgpack, opencv-python
НЕ нужен CycloneDDS — LingBot-VA не использует DDS
```

Второй сервис `wam` (существующий) получает только клиентский код WebSocket.

### Сравнение моделей

| | UnifoLM | EVA | **LingBot-VA** |
|---|---|---|---|
| Параметры | ~16B | 14B | **5.3B** |
| Скорость | 10 Hz sync | 2-5 мин/траектория | **50 Hz async** |
| VRAM | ~16 GB | ~50 GB | **~24 GB** |
| Интерфейс к роботу | DDS | DDS (добавить) | **WebSocket** |
| G1 веса | ✅ есть | ❌ нет | ❌ нет (50 демо) |
| Качество (RoboTwin) | — | — | **92.9% SOTA** |

### Блокеры

| # | Блокер | Статус |
|---|--------|--------|
| **1** | **G1 post-training данные**: нужно 50 демонстраций | ⚠️ Собрать в Isaac Sim |
| **2** | **flash-attn в Dockerfile**: `--no-build-isolation`, долгая сборка | ⚠️ Нужен Dockerfile |
| **3** | **WebSocket bridge**: клиент в wam/main.py вместо прямого вызова модели | ⚠️ Нужна реализация |

---

## Ego-VCP — Plan & Integration (30 May 2026, ветка Ego-VCP_without_GS)

### Что такое Ego-VCP

**Ego-VCP** = Ego-**V**ision World Model for humanoid **C**ontact **P**lanning.
Paper: arXiv:2510.11682 (ICRA 2026) | GitHub: https://github.com/HybridRobotics/Ego-VCP
Авторы: UC Berkeley, U Michigan, CUHK | License: MIT

**Тип:** Латентная world model (RSSM / DreamerV2-стиль) + Stochastic MPC планировщик (CEM, 2000 сэмплов).
**НЕ** diffusion, **НЕ** VLA, **НЕ** autoregressive LLM.

### ⚠️ КРИТИЧЕСКИЙ НЮАНС: задачи Ego-VCP ≠ наши задачи

Ego-VCP обучен на **контактное планирование при локомоции**:
- `g1_wall` — опора на стену после толчка
- `g1_ball` — блокирование летящих объектов руками
- `g1_tunnel` — проход через низкие арки (нагнуться)

Это **НЕ** манипуляция. Нет pick-and-place, нет захвата куба.
Выход модели — 10D EEF targets для всего тела (не 14D arm joints).
Для задачи "взять куб → положить в корзину" модель нужно переобучать с нуля.

### Архитектура и выход

```
Proprioception (9 + 3N dim)  +  Depth image [48×64] (НЕ RGB)
        ↓
ObsEncoder → z_t [32]  +  GRU h_t [256]
        ↓
SMPC: 2000 случайных последовательностей × horizon=4
  → roll out через DynamicEncoder + QvalueDecoder
  → выбрать top-64 по Q-value
        ↓
10D high-level action: [left_EEF_xyz (3), right_EEF_xyz (3), body_RPY (3), height (1)]
        ↓
Frozen locomotion policy (policy.pt / policy.onnx) — RSL-RL export
        ↓
29D joint position targets → Isaac Sim
```

**Наблюдение:** Только **depth** (48×64), никакого RGB, никакого текста, никакого состояния рук.

### G1-специфичность — ✅ ВЕСЬ СТЕК ПОД G1

Единственный робот во всём репозитории — Unitree G1:
- `ego_vcp/assets/unitree/g1/g1.usd` — USD модель
- `ego_vcp/envs/g1/g1_*.py` — все среды
- `logs/g1_flat/pre-trained/exported/policy.pt` — предобученный низкоуровневый контроллер

### Предобученные веса — ✅ В РЕПО

| Чекпоинт | Путь в репо | Задачи |
|----------|------------|--------|
| All-tasks | `wm_logs/all/world_model.pt` | wall + ball + tunnel |
| Ball only | `wm_logs/ball/world_model.pt` | g1_ball |
| Tunnel only | `wm_logs/tunnel/world_model.pt` | g1_tunnel |
| Wall only | `wm_logs/wall/world_model.pt` | g1_wall |
| Low-level ctrl | `logs/g1_flat/pre-trained/exported/policy.pt` | G1 29-DOF локомоция |

Все под MIT license. Датасет: `Hang917/EgoVCP_Dataset` на HuggingFace (20.5 GB, `.npz`).

### Скорость

**25 Hz real-time** — подтверждено на физическом G1, GPU **RTX 2060** (consumer-grade!).

SMPC: 2000 сэмплов × horizon=4 — в реал-тайм благодаря pre-allocated tensor buffers и in-place ops. Латентная GRU модель очень лёгкая (32-dim z, 256-dim h).

### Контейнер — ПРОБЛЕМА

**Нет Dockerfile.** Зависимость: **Isaac Lab 2.1.0 → Isaac Sim 4.5.0**.

Наш стек использует **Isaac Sim 5.1** (image: `mws-dimos/sim-isaac:unitree-lab-5.1`).
Ego-VCP требует Isaac Sim **4.5.0** — другая мажорная версия.

Варианты:
1. **Изолированный контейнер** с Isaac Sim 4.5.0 только для Ego-VCP (тяжело, ~20 GB образ)
2. **Без Isaac Lab**: вытащить только `world_model.pt` + GRU inference код → запустить standalone, без Isaac Lab API
3. **Переписать на нашу Isaac Sim 5.1** — риск несовместимостей API

Рекомендация: вариант 2 — standalone инференс без Isaac Lab.

### Поток данных и референс

```
Isaac Sim 5.1 (наш)
  ├── rt/lowstate DDS → arm joint positions + base velocity + IMU
  └── (нет depth SHM пока) → нужно добавить depth публикацию через SHM
          ↓
  wam-bridge (новый компонент)
    ├── Конвертировать rt/lowstate → ego-vcp obs vector (проприоцепция)
    ├── Depth image → из sim camera (нужно настроить в mws_office.usdz)
    ├── Запустить SMPC inference (models/ego_vcp.py)
    │     Референс: repos/ego-vcp/ego_vcp/scripts/play_wm.py
    ├── 10D EEF → policy.pt → 29D joint targets (frozen loco controller)
    └── DDS publish rt/lowcmd → joint_q[0:29]
```

**Референс кода:**
| Что | Файл |
|-----|------|
| Inference loop | `repos/ego-vcp/ego_vcp/scripts/play_wm.py` |
| World model класс | `repos/ego-vcp/rsl_rl/rsl_rl/modules/ego_world_model.py` |
| SMPC planner | `repos/ego-vcp/ego_vcp/scripts/smpc_controller.py` |
| Obs конфигурация | `repos/ego-vcp/ego_vcp/envs/g1/g1_ball_config.py` |
| Low-level ctrl | `repos/ego-vcp/logs/g1_flat/pre-trained/exported/policy.pt` |

### Блокеры

| # | Блокер | Сложность |
|---|--------|-----------|
| **1** | **Задачи не те**: обучен на контакт при локомоции, не манипуляция | ❌ Фундаментальное |
| **2** | **Isaac Lab 4.5 vs 5.1**: либо standalone inference, либо новый контейнер | ⚠️ Высокая |
| **3** | **Depth SHM**: нужно добавить публикацию depth из Isaac Sim 5.1 | ⚠️ Средняя |
| **4** | **Obs format bridge**: rt/lowstate → ego-vcp proprioception vector | ⚠️ Средняя |
| **5** | **Нет обучения на манипуляцию**: нужен новый датасет и переобучение | ❌ Фундаментальное |

---

## TrajBooster / PPTmodel4UnitreeG1 — Plan (30 May 2026, ветка TrajBooster_PPTmodel4UnitreeG1_without_GS)

### Что это

**TrajBooster** = Trajectory-Centric Learning для whole-body humanoid manipulation.
Paper: arXiv:2509.11839 (ICRA 2026) | GitHub: https://github.com/OpenHelix-Team/OpenTrajBooster
Project page: https://jiachengliu3.github.io/TrajBooster/

**PPTmodel4UnitreeG1** = Post-Pre-Trained checkpoint из TrajBooster, готовый к fine-tune на G1.
HuggingFace: `l2aggle/PPTmodel4UnitreeG1` (~6 GB, Apache 2.0)

### Архитектура (три уровня)

```
RGB (head + wrist) + language prompt + 6D EEF wrist poses
        ↓
① VLA: GR00T N1.5 (3B) + Diffusion Transformer (flow-matching, 4 шага)
        ↓ 16-step action chunk @ 20 Hz
② Manager Policy (heuristic DAgger)
        ↓ base velocity (vx, vy, vyaw) + torso height
③ Worker Policy "Homie" (goal-conditioned RL)
        ↓ 12-DOF lower body joint PD targets

Upper body IK (Pinocchio) → arm joint angles
Dex3 hand → 7-DOF hand targets
```

Весь стек → **joint position targets** для всего G1 (29 DOF).

### ✅ Ключевые преимущества

| Преимущество | Деталь |
|---|---|
| **G1-специфичные веса** | `l2aggle/PPTmodel4UnitreeG1`, Apache 2.0, **~6 GB** |
| **Тот же DDS** | Unitree SDK2 = CycloneDDS — **никакого bridge не нужно** |
| **Малый чекпоинт** | 6 GB vs EVA 32 GB, vs WAN 82 GB |
| **Быстро fine-tune** | Только **10 минут** реальных демонстраций |
| **Задачи = манипуляция** | Whole-body: рука + базовое движение |

### Скорость

| Параметр | Значение |
|---|---|
| Control loop | **20 Hz** |
| Action chunk | 16 шагов |
| Denoising steps | 4 (flow-matching) |
| Latency | ~47.88 мс (H100) |
| GPU requirement | A100 класс для inference |

### G1-специфичные веса — ✅ ЕСТЬ

```bash
# Скачать PPT checkpoint:
huggingface-cli download l2aggle/PPTmodel4UnitreeG1 --local-dir ./checkpoints/trajbooster

# Загрузить:
from transformers import AutoModel
model = AutoModel.from_pretrained("l2aggle/PPTmodel4UnitreeG1")
```

Этот чекпоинт: GR00T N1.5 после post-pre-training на **35 ч ретаргетированных Agibot→G1 данных**.
Для task-specific fine-tune нужно всего **10 минут** реального teleop.

Лицензии:
- PPT checkpoint: **Apache 2.0** ✅
- Retargeting model: CC BY-NC-SA 4.0 (non-commercial, research OK) ✅

### Контейнер

**Нет Dockerfile в репо.** Нужно писать самим.

```
Base:    nvidia/cuda:12.6.0-runtime-ubuntu22.04
Python:  3.10
PyTorch: latest stable (GR00T N1.5 via HuggingFace Transformers)
Deps:    transformers, safetensors, opencv-python-headless,
         pinocchio (IK для upper body → joint angles)
         cyclonedds==0.10.2  ← тот же что в UnifoLM!
```

Один сервис `trajbooster` в compose.yml.

### Поток данных и интерфейс

```
Isaac Sim
  ├── /run/mws/camera.rgb (SHM, 1280×720)        ← head cam
  ├── [нужно добавить] wrist camera SHM           ← ⚠️ пока нет
  └── rt/lowstate DDS → joint positions + IMU
          ↓
  WAM inference (src/models/trajbooster.py)
    ├── RGB head + [wrist] → VLA input
    ├── language prompt → T5/CLIP encode
    ├── rt/lowstate → 6D EEF pose (FK через Pinocchio)
    ├── GR00T N1.5 inference → 16-step action chunk
    ├── Manager Policy → base commands
    ├── Worker Policy → lower body joints
    └── DDS publish rt/lowcmd → joint_q[0:29]
          ↓
  Isaac Sim (all 29 DOF)
```

**Посредник**: НИКАКОГО дополнительного — чистый DDS, как в UnifoLM.

### Референс в коде

| Что | Где |
|-----|-----|
| **Inference entry** | `g1_deploy/scripts/G1_inference.py` |
| **Deploy framework** | `g1_deploy/HomieDeploy/` |
| **VLA forward pass** | `gr00t` module (HuggingFace GR00T API) |
| Manager + Worker | `g1_deploy/HomieDeploy/homie*.py` |
| Fine-tuning | `vla_fine_tune/` директория в репо |
| Dataset (retarget) | `l2aggle/Agibot2UnitreeG1Retarget` (35 ч, ~30 GB) |

### Блокеры

| # | Блокер | Статус |
|---|--------|--------|
| **1** | **Wrist camera**: в нашей сцене только head cam | ⚠️ Добавить в Isaac Sim или адаптировать VLA |
| **2** | **Pinocchio IK**: нужна библиотека + G1 URDF в контейнере | ⚠️ Dockerfile |
| **3** | **Manager + Worker pipeline**: три модели одновременно | ⚠️ Реализация |
| **4** | **10 мин teleop**: нужны G1 демонстрации для fine-tune | ⚠️ Собрать в Isaac Sim |

---

## Сравнение всех рассмотренных моделей

| | UnifoLM | LingBot-VA | EVA | Ego-VCP | **TrajBooster** |
|---|---|---|---|---|---|
| Задачи | Манипуляция ✅ | Манипуляция ✅ | Манипуляция ✅ | Контакт/локомоция ⚠️ | Whole-body ✅ |
| G1 веса | ✅ Есть | ❌ Нет | ❌ Нет | ✅ Есть (но не манипуляция) | **✅ Apache 2.0** |
| Checkpoint | ~16 GB | ~24 GB | ~33 GB | в репо (мал) | **~6 GB** |
| Скорость | 10 Hz | 50 Hz async | ~5 мин/traj | 25 Hz | **20 Hz** |
| VRAM inference | ~16 GB | ~24 GB | ~50 GB | < 2 GB | ~A100 class |
| Интерфейс | DDS ✅ | WebSocket | DDS (добавить) | Isaac Lab only | **DDS ✅** |
| Статус | ✅ Работает | Plan only | Plan only | Wrong tasks | **Лучший кандидат** |

---

## FastWAM — Plan (30 May 2026, ветка fastwam_without_GS)

### Что такое FastWAM

**FastWAM** = "Fast-WAM: Do World Action Models Need Test-time Future Imagination?"
arXiv: 2603.16666 | GitHub: локально в `repos/fastwam`
Checkpoints: `yuanty/fastwam` на HuggingFace | License: уточнить

**Backbone**: Wan2.2-TI2V-5B (видео DiT) + ActionDiT (350M) — unified via **MoT** (Mixture of Transformers).
Архитектура аналогична LingBot-VA, но вопрос бенчмарка: нужно ли генерировать видео на инференсе?

### Является ли WAM? — Частично

Главный тезис paper: **WAM не обязан генерировать видео в test time**.

| Метод | Режим | Видео |
|-------|-------|-------|
| `infer_action()` | **Основной** — только actions, без видео | ❌ нет |
| `infer_joint()` | Опциональный — actions + видео | ✅ есть |

Архитектурно — **да, WAM**: видео и action стримы обучены совместно, loss комбинированный.
В деплойменте — **нет**: используется только `infer_action()`, видео не генерируется.

Для нашего проекта: **можно включить `infer_joint()`** и получить видео траектории для датасета.

### Архитектура и выход

```
RGB images (3 камеры для RoboTwin) + proprioception (14D) + text (T5)
        ↓
VAE encode → image latents [1, 48, 1, h//8, w//8]
        ↓
MoT: Video DiT (Wan2.2, 30 layers, hidden=3072) ←cross-attn→ ActionDiT (30 layers, hidden=1024)
        ↓
Flow-matching scheduler (WanContinuousFlowMatchScheduler), 20 шагов
        ↓
action latents → denorm → action_traj [T, 14]   (RoboTwin: 14D ✅ = G1 arm DOF)
(+ опционально: video latents → VAE decode → video frames)
```

### Скорость

Timing-код есть (`infer_s`, `sim_s` в `deploy_policy.py`), но **конкретных цифр в paper нет**.
Ориентир по аналогии с LingBot-VA (аналогичный backbone Wan2.x, 20 шагов flow-matching):
- Ожидаемо **~0.5–2 сек** на A100 для action-only режима
- Для joint (action+video): значительно дольше

### G1-специфичных весов нет

Обучен на LIBERO (Mujoco sim) и RoboTwin (sim). Реальных G1 данных нет.
Выход RoboTwin: **14D joint actions** → прямое совпадение с G1 arm slots 14–27.

Путь к G1: post-train на RoboTwin checkpoint (он уже в 14D пространстве) + G1-специфичные демо.

### Чекпоинты

```bash
huggingface-cli download yuanty/fastwam \
  libero_uncond_2cam224.pt \
  robotwin_uncond_3cam_384.pt \
  --local-dir ./checkpoints/fastwam
```

| Файл | Задачи |
|------|--------|
| `robotwin_uncond_3cam_384.pt` | RoboTwin bimanual (14D) ← использовать для G1 |
| `libero_uncond_2cam224.pt` | LIBERO (7D) |

Базовая модель: `Wan-AI/Wan2.2-TI2V-5B` (нужна отдельно).

### Контейнер — важный нюанс: CUDA 12.8

FastWAM требует **PyTorch 2.7.1+cu128** (CUDA 12.8). Текущий WAM контейнер: cu126.
Проверить: поддерживает ли GPU-01 CUDA 12.8 (`nvidia-smi` должен показать >= 12.8).

```
Base:    nvidia/cuda:12.8.0-runtime-ubuntu22.04
Python:  3.10
PyTorch: 2.7.1+cu128
Deps:    einops, hydra-core, transformers==4.49.0, accelerate, deepspeed
         cyclonedds==0.10.2  ← добавить (нет в оригинале)
```

**Нет Dockerfile** в репо — пишем сами.

### Поток данных

```
Isaac Sim
  ├── /run/mws/camera.rgb (SHM head, 1280×720) → resize → 240×320
  ├── [⚠️ нужен] left wrist SHM  → 240×320
  ├── [⚠️ нужен] right wrist SHM → 240×320
  └── rt/lowstate DDS → arm joints [14:28] → proprio [14D]
          ↓
  WAM inference (src/models/fastwam.py)
    ├── Stack 3 cameras → [384, 320] concat
    ├── fastwam.infer_action(prompt, image, proprio)
    │   Референс: repos/fastwam/experiments/robotwin/fastwam_policy/deploy_policy.py
    │   Model class: repos/fastwam/fastwam/fastwam.py → FastWAM.infer_action()
    ├── → action_traj [T, 14] → denorm
    └── DDS publish rt/lowcmd → arm_q[14:28]
```

**Посредник**: никакого — прямой вызов Python library, как UnifoLM.

### Референс в коде

| Что | Файл |
|-----|------|
| **Inference pipeline** | `repos/fastwam/experiments/robotwin/fastwam_policy/deploy_policy.py` |
| **Model class** | `repos/fastwam/fastwam/fastwam.py` → `FastWAM.infer_action()` / `infer_joint()` |
| **MoT architecture** | `repos/fastwam/fastwam/mot.py` |
| **ActionDiT** | `repos/fastwam/fastwam/action_dit.py` |
| **Config** | `repos/fastwam/configs/model/fastwam.yaml` |
| **Dataset stats** | `checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json` |

### Блокеры

| # | Блокер | Сложность |
|---|--------|-----------|
| **1** | **CUDA 12.8**: проверить совместимость с GPU-01 | ⚠️ Проверить |
| **2** | **Wan2.2-TI2V-5B**: доступность модели на HF | ⚠️ Проверить |
| **3** | **2 wrist cameras**: нет в текущей сцене | ⚠️ Добавить |
| **4** | **DDS bridge**: нет в репо, добавить cyclonedds | ⚠️ Нужна реализация |
| **5** | **G1 fine-tune**: нет G1 весов, нужны демо | ⚠️ Собрать |

---

## Следующие шаги

### Краткосрочно (TrajBooster — приоритет)
1. **Скачать `l2aggle/PPTmodel4UnitreeG1`** на сервер (~6 GB)
2. **`docker/trajbooster/Dockerfile`** — transformers, pinocchio, cyclonedds
3. **`src/models/trajbooster.py`** — GR00T inference + Manager/Worker pipeline
4. **Wrist camera**: добавить вторую камеру в Isaac Sim или адаптировать под one-camera
5. **10 мин teleop демо** → fine-tune на pick-and-place задачу

### Среднесрочно
1. **Собрать датасет** (~400 сэмплов/активность) для UnifoLM fine-tune
2. **Протестировать transfer** UnifoLM → UBTech Walker Tienkung

### Долгосрочно
1. Деплой на реальный G1 (Isaac Sim → реальное железо)
2. Мультиробот: UBTech Walker USD + DOF mapping
3. Сравнение UnifoLM vs TrajBooster на transfer task
