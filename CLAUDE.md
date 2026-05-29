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
- `G1_KINEMATIC_ROBOT=0` — обычная физика + startup support wrench (для тестирования GEAR-SONIC).

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

## Следующие шаги

### Краткосрочно
1. ✅ ~~**Publish actions via DDS**~~ — `dds.publish_lowcmd()` работает, руки движутся (28 May 2026)
2. **Собрать датасет** — ~400 сэмплов на активность в Isaac Sim (СЛЕДУЮЩИЙ ШАГ)

### Среднесрочно
1. **Fine-tune** UnifoLM на собранных данных G1
2. **Протестировать transfer** на UBTech Walker Tienkung
3. **Реализовать EVA** (`src/models/eva.py`)

### Долгосрочно
1. Деплой на реальный G1 (Isaac Sim → реальное железо)
2. Мультиробот: добавить UBTech Walker USD + DOF mapping
3. Сравнение UnifoLM vs EVA на transfer task
