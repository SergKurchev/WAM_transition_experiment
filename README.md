# WAM Stack — UnifoLM-WMA-0 на G1 в Isaac Sim

**Цель:** дообучить World Action Model (WAM) на действиях G1, проверить zero-shot transfer на UBTech Walker Tienkung.

**Статус на 27 мая 2026:** ✅ **Полный пайплайн работает.** Real DDIM inference, нормализация состояния, видео генерируется, действия в правильном пространстве радиан.

---

## Архитектура (полный пайплайн)

```
┌──────────────────────────────────────────────────────────────────┐
│  Isaac Sim 5.1  (wam-isaac-sim контейнер)                        │
│  ├─ Физика G1 (29 DOF)                                           │
│  ├─ Сцена: office_demo.usdz                                      │
│  ├─ D435i камера → /run/mws/camera.rgb  (SHM, 1280×720 RGB)      │
│  └─ rt/lowstate → DDS (50 Hz, 29-DOF joint pos/vel/tau)          │
└────────────────────────┬─────────────────────────────────────────┘
                         │ DDS domain 42, iface lo
┌────────────────────────▼─────────────────────────────────────────┐
│  WAM Inference (wam-inference контейнер)  10 Hz                  │
│                                                                  │
│  main.py ──► UnifoLMModel.__call__(RobotState)                   │
│               │                                                  │
│   1. Камера   ├─ /run/mws/camera.rgb  (SHM, приоритет)           │
│               │   └─ fallback: /tmp/isaac_frame.png              │
│               │   resize → 320×512, norm (x/255-0.5)*2           │
│               │                                                  │
│   2. Стейт    ├─ state.q[:14] (14-DOF G1 arms)                   │
│               │   pad → [16] с нулями                            │
│               │   min-max норм. → [-1,1] (G1 pack camera stats)  │
│               │                                                  │
│   3. DDIM     └─ image_guided_synthesis()  (16 шагов)            │
│       ├─ c_concat:  latent obs frame [1,4,16,40,64]              │
│       ├─ c_crossattn: [state_emb | text_emb | img_emb]           │
│       │              [1, 2+77+16, 1024]                          │
│       └─ c_crossattn_action: [imgs_BCTHU, states_BTD]            │
│                                                                  │
│  DDIMSampler.sample(S=16) → (latent, actions[1,16,16], states)   │
│       │                                                          │
│  decode_first_stage() → video [B,C,T,H,W]                        │
│  unnormalize_actions() → joint angles [16,14] rad                │
│  temporal_ensemble()   → сглаженные действия                     │
│                                                                  │
│  Recording:                                                      │
│   ├─ model_output/output_*.mp4  (генерируемое видео)             │
│   ├─ robot_camera/camera_*.png  (D435i кадры)                    │
│   ├─ input_frames/state_*.txt   (состояние робота)               │
│   └─ command_logs/*.csv         (история команд)                 │
└──────────────────────────────────────────────────────────────────┘
```

---

## Критические зависимости

### 1. `mws-dimos` (внешний репо, READ-ONLY)

**Репо:** https://github.com/MWS-Physical-AI/mws-dimos  
**Ветка:** `feat/real-transfer` — **не main!**  
**Расположение на сервере:** `/root/skurchev/workspace/mws-dimos/`

Предоставляет **готовые Docker-образы** и **assets** — мы их не собираем, только используем:

| Что | Откуда | Зачем |
|-----|--------|-------|
| `mws-dimos/sim-isaac:unitree-lab-5.1` | pre-built image | Isaac Sim + G1 сцена |
| `mws-sim-gear-sonic-policy:latest` | pre-built image | GEAR-SONIC WBC (29-DOF баланс) |
| `mws-sim-ros2-bridge:latest` | pre-built image | DDS bridge 50D→994D |
| `deploy/images/isaac-sim/entrypoint.sh` | bind-mount | Isaac Sim entrypoint |
| `assets/robots/g1/` | bind-mount | USD assets G1 |
| `modules/gwbc/` | bind-mount | GEAR-SONIC модели |

> ⚠️ **НЕЛЬЗЯ** копировать или модифицировать файлы mws-dimos. Только bind-mount как read-only.

### 2. `unifolm` (inference library, READ-ONLY)

**Репо:** https://github.com/unitreerobotics/unifolm-world-model-action  
**Расположение на сервере:** `/root/skurchev/workspace/wam-stack/repos/unifolm/`  
**В контейнере:** `/workspace/unifolm/` (bind-mount, `PYTHONPATH=/workspace/unifolm/src`)

> ⚠️ **НЕ pip-install.** Загружается через PYTHONPATH bind-mount. Причина: нужна конкретная версия с патчем (см. ниже).

**Критический патч (только на сервере):**  
`repos/unifolm/src/unifolm_wma/modules/attention.py` — удалена строка `assert 1 > 2` (~строка 128 внутри `forward()`).  
Причина: xformers собран для PyTorch 2.10+cu128, контейнер использует 2.7.0+cu126. Патч переключает на vanilla PyTorch attention, который работает корректно.

**Что берём из unifolm:**
- `unifolm_wma.models.ddpms.LatentVisualDiffusion` — основная модель
- `unifolm_wma.models.samplers.ddim.DDIMSampler` — DDIM inference
- `examples/world_model_interaction_prompts/transitions/unitree_g1_pack_camera/meta_data/stats.safetensors` — stats нормализации

### 3. Чекпойнт модели

**Файл:** `checkpoints/unifolm_wma_dual.ckpt` (≈16 GB)  
**Путь в контейнере:** `/workspace/wam/checkpoints/unifolm_wma_dual.ckpt`  
**Env var:** `WAM_CHECKPOINT`

Чекпойнт обучен на нескольких датасетах Unitree, включая `G1_Dex1_MountCameraRedGripper_Dataset`. Dual = одновременно видео-генерация + action head.

### 4. G1 Pack Camera Stats

**Файл:** `/workspace/unifolm/examples/.../unitree_g1_pack_camera/meta_data/stats.safetensors`  
**Назначение:** min-max нормализация стейта и обратная нормализация действий

```
observation.state/min[:5] = [-1.6514, -0.1329, -1.3105, -1.0467, -1.9663]
observation.state/max[:5] = [ 0.9437,  1.6497,  0.9905,  1.3529,  1.4207]
action/min[:5]            = [-1.6897, -0.3179, -1.3181, -1.0472, -1.9722]
action/max[:5]            = [ 0.9541,  1.6616,  0.9965,  1.3657,  1.4339]
```

---

## Структура репозитория

```
wam-stack/
├── compose.yml                  # Docker Compose — 2 сервиса: sim-isaac, wam
├── docker/
│   └── wam/Dockerfile           # WAM контейнер: CUDA 12.2 + Python 3.10 + torch 2.7
├── src/                         # Bind-mount → /workspace/wam в контейнере
│   ├── main.py                  # Control loop 10 Hz: DDS → model → record
│   ├── dds_interface.py         # DDS subscriber (rt/lowstate), RobotState dataclass
│   ├── recording.py             # MediaRecorder: MP4/PNG/TXT/CSV
│   ├── config_model.yaml        # OmegaConf конфиг модели (LatentVisualDiffusion)
│   ├── configs/train/meta.json  # Метаданные форм для MultiImageObsEncoder
│   └── models/
│       ├── unifolm.py           # ✅ UnifoLM-WMA-0 (DDIM inference)
│       └── eva.py               # 🔲 EVA (заглушка)
├── checkpoints/                 # Bind-mount → /workspace/wam/checkpoints (ro)
│   └── unifolm_wma_dual.ckpt    # ≈16 GB чекпойнт
├── repos/
│   └── unifolm/                 # Bind-mount → /workspace/unifolm (ro)
│       └── src/unifolm_wma/     # unifolm_wma library (PYTHONPATH)
└── media/                       # Bind-mount → /workspace/wam/media (rw)
    ├── model_output/            # output_*.mp4 — генерируемое видео
    ├── robot_camera/            # camera_*.png — D435i кадры
    ├── input_frames/            # state_*.txt — стейт робота
    └── command_logs/            # commands_*.csv
```

---

## Быстрый старт

### Требования

На сервере `x32-techgov-GPU-01` (`176.109.83.84:2221`):
```
/root/skurchev/workspace/
├── wam-stack/          ← этот репо
├── mws-dimos/          ← feat/real-transfer (READ-ONLY)
└── assets/
    └── office_demo.usdz
```

### Запуск

```bash
# Подключиться к серверу
ssh -p 2221 root@176.109.83.84

# Запустить стек
cd /root/skurchev/workspace/wam-stack
docker compose up -d

# Следить за логами
docker logs -f wam-inference

# Ожидаемый вывод после загрузки (~90 сек):
# [UnifoLM] ✓ Loaded G1 pack camera normalization stats
# [UnifoLM] ✓ Camera SHM: /run/mws/camera.rgb
# [WAM] rt/lowstate received. Control loop starting at 10 Hz.
# [WAM] loop=2  action_0[0:3]=[-0.42 +0.42 -0.17]  norm=1.18  traj_norm=4.96
```

### Мониторинг (визуальный)

```bash
# На локальной машине — SSH туннель к noVNC
ssh -N -L 6081:localhost:6080 -p 2221 root@176.109.83.84
# Открыть: http://localhost:6081
```

### Перезапуск после изменений кода

`src/` bind-mounted, поэтому достаточно:
```bash
docker restart wam-inference
```

### Остановка

```bash
docker compose down
```

---

## Модель: UnifoLM-WMA-0

### Что это

**LatentVisualDiffusion** — диффузионная видео-модель с встроенным action head (ConditionalUnet1D). Одновременно:
- генерирует следующие 16 кадров (предсказание мира)
- предсказывает траекторию из 16 шагов × 14 DOF (управление роботом)

### Inference pipeline

```python
# 1. Получить кадр с D435i (SHM /run/mws/camera.rgb)
img: [B, T=2, C=3, H=320, W=512]  # 2 obs steps, нормализовано [-1,1]

# 2. Нормализовать стейт
state: [B, T=2, D=16]  # 14 DOF G1 + 2 нуля, min-max → [-1,1]

# 3. Кондиционирование
c_concat       = latent(last_obs_frame)      # [1, 4, 16, 40, 64]
c_crossattn    = [state_emb | text | clip]   # [1, 95, 1024]
c_crossattn_action = [imgs, states]          # последние 2 шага

# 4. DDIM sampling (16 шагов, eta=1.0, fps=15)
samples, actions, states = DDIMSampler.sample(S=16, ...)

# 5. Decode + unnormalize
video  = decode_first_stage(samples)         # [1, 3, 16, 320, 512]
joints = unnormalize(actions[:,:14])         # [16, 14] radians
```

### Параметры модели

| Параметр | Значение |
|----------|---------|
| DDIM шагов | 16 |
| FPS condition | 15 (= 30 / frame_stride=2) |
| Horizon | 16 timesteps |
| State dim | 16 (G1: 14 DOF + 2 zeros) |
| Action dim | 16 → обрезается до 14 |
| Image size | 320×512 → latent 40×64 |
| Latent channels | 4 |

### Выход модели

| Тип | Форма | Описание |
|-----|-------|----------|
| `action_traj` | `[16, 14]` float32, радианы | 16-шаговая траектория, 14 DOF руки |
| `state_traj` | `[16, 14]` float32 | предсказанные стейты мира |
| `video_output` | `[1, 3, 16, 320, 512]` float32 | генерируемое видео [-1, 1] |

> Сейчас `action_traj[0]` логируется, но **не отправляется** роботу (в `main.py` нет DDS publish команды). Это следующий шаг.

---

## Конфигурация: переменные окружения

| Переменная | По умолчанию | Описание |
|------------|-------------|----------|
| `WAM_MODEL` | `unifolm` | Модель: `unifolm` / `eva` / `stub` |
| `WAM_CHECKPOINT` | `/workspace/wam/checkpoints/unifolm_wma_dual.ckpt` | Путь к чекпойнту |
| `WAM_PROMPT` | `pick and place green cube in white basket` | Текстовый промпт |
| `WAM_MEDIA_DIR` | `/workspace/wam/media` | Директория для записи |
| `DDS_IFACE` | `lo` | Сетевой интерфейс DDS |

---

## DDS топики

| Топик | Кто публикует | Кто читает | Частота | Формат |
|-------|--------------|-----------|---------|--------|
| `rt/lowstate` | Isaac Sim | WAM, GEAR-SONIC | 50 Hz | 29-DOF state |
| `rt/lowcmd` | GEAR-SONIC | Isaac Sim | 10 Hz | 29-DOF joint targets |

DDS domain ID: `42`, интерфейс: `lo` (loopback).

---

## Медиа-данные

```
media/
├── model_output/output_*.mp4    # Генерируемое видео (16 кадров, 10 FPS)
├── robot_camera/camera_*.png    # D435i кадры (1280×720 RGB)
├── input_frames/state_*.txt     # Стейт робота (q, dq, tau текстом)
└── command_logs/commands_*.csv  # Лог команд (timestamp, step, ...)
```

Автопрунинг: каждый час оставляет первые и последние 10 файлов в каждой папке.

Очистка вручную:
```bash
find /root/skurchev/workspace/wam-stack/media -type f -delete
```

---

## Версии компонентов

| Компонент | Версия |
|-----------|--------|
| Isaac Sim | 5.1 |
| GEAR-SONIC | feat/real-transfer |
| CycloneDDS | 0.10.2 |
| PyTorch | 2.7.0+cu126 |
| Python | 3.10.12 |
| CUDA | 12.2 |
| unifolm_wma | git HEAD (с патчем attention.py) |

---

## Статус

| Компонент | Статус |
|-----------|--------|
| Isaac Sim + G1 физика | ✅ работает |
| DDS коммуникация (rt/lowstate) | ✅ работает |
| UnifoLM-WMA-0 загрузка чекпойнта | ✅ работает |
| DDIM inference (16 шагов) | ✅ работает |
| State normalization | ✅ работает (G1 pack camera stats) |
| Action unnormalization | ✅ работает (радианы) |
| D435i SHM камера | ✅ работает |
| MP4 видео генерация | ✅ работает |
| DDS publish действий → GEAR-SONIC | 🔲 не реализовано |
| Сбор датасета (~400 sample/activity) | 🔲 предстоит |
| Fine-tuning на G1 | 🔲 предстоит |
| Перенос на UBTech Walker Tienkung | 🔲 предстоит |

---

## Troubleshooting

### Модель не загружается

```bash
docker logs wam-inference | grep -E "ERROR|FATAL|Traceback"
# Проверить checkpoint
docker exec wam-inference ls -lh /workspace/wam/checkpoints/
```

### Видео чёрное / мутное

Вероятные причины:
1. Используется `/tmp/isaac_frame.png` вместо SHM камеры → проверить: `[UnifoLM] ✓ Camera SHM` в логах
2. Domain mismatch (модель обучена на реальных данных, стек кормит симуляцию) → улучшается после fine-tuning

### GEAR-SONIC crash "observation dimension ≠ 994"

```bash
# sim-ros2-bridge не запущен или не healthy
docker compose ps | grep ros2-bridge
docker logs wam-ros2-bridge | tail -20
```

### Нет rt/lowstate

```bash
docker logs wam-isaac-sim | tail -30
# Isaac Sim стартует 2-5 минут, это нормально
```

---

## Авторы

Sergey Kurchev (@SergKurchev) — WAM fine-tuning pipeline  
Jeffrin Sam — NVIDIA Cosmos, LingBot-VA, Dream Zero  
Artem Lykov — mws-dimos infrastructure
