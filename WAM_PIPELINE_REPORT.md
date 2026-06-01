# UnifoLM WAM Fine-Tuning Pipeline — Промежуточный отчёт

**Дата:** 1 июня 2026  
**Автор:** Сергей Курчев  
**Проект:** CoRL 2026 — кросс-роботный transfer через World Action Model  
**Команда:** Сергей Курчев · Jeffrin Sam · Артём Лыков

---

## Кратко

UnifoLM-WMA-0 запущен на Unitree G1 в Isaac Sim. Инференс стабилен, робот производит
осмысленные движения руками. Полный цикл fine-tuning pipeline —
сбор датасета → тренировка → сохранение чекпоинта — реализован и проверен end-to-end
на тестовом датасете. Тот же pipeline адаптирован для UBTech Walker Tienkung.

---

## 1. Архитектура модели

**UnifoLM-WMA-0** — это latent video diffusion модель с диффузионной action head.
Backbone генерирует предсказанное будущее видео; action head предсказывает траекторию
суставов робота, обусловленную этим латентным предсказанием.

```
Кадр с камеры (1280×720 RGB)
    ↓  resize → [3, 320, 512]
DINOSigLIP (визуальный энкодер)
    ↓  [B, 257, 1280]
Resampler (image_proj_model)
    ↓  image tokens [B, 16, 1024]
                         ┌── CLIP text encoder ── токены промпта [B, 77, 1024]
                         │
WMAModel (temporal U-Net, 16-шаговый DDIM) ──────────────────────────────────────────
    │
    ├──▶  Предсказанное видео  [1, 3, 16, 320, 512]   (16 будущих кадров)
    │
    └──▶  ConditionalUnet1D (action head)
               ↓
          action_traj  [16, 14]   → action_traj[0] отправляется в суставы @ 10 Hz
```

### Компоненты модели

| Компонент | Роль | Параметры |
|-----------|------|-----------|
| VAE (`AutoencoderKL`) | Кодирует видеокадры → латент `[B,4,T,40,64]` | ~84 M |
| CLIP text (`FrozenOpenCLIPEmbedder`) | Кондиционирование по тексту | ~354 M |
| DINOSigLIP (`FrozenOpenCLIPImageEmbedderV2`) | Эмбеддинг визуального наблюдения | ~630 M |
| Resampler (`image_proj_model`) | Визуальные фичи → cross-attention токены | ~26 M |
| WMAModel (U-Net) | Временной диффузионный backbone | ~3 060 M |
| ConditionalUnet1D | Action diffusion head | ~25 M |
| SATokenProjector + проекторы | Токенизация state/action, pos. embeddings | ~1 M |
| **Итого** | | **~4 180 M** |

---

## 2. Unitree G1 — Результаты инференса

### Конфигурация

- Isaac Sim публикует `rt/lowstate` со скоростью 50 Hz через DDS (CycloneDDS 0.10.2)
- Control loop WAM работает на **10 Hz**: читает стейт → запускает DDIM → публикует `rt/lowcmd`
- Выход модели: `action_traj [16, 14]` — горизонт 16 шагов, 14 DOF (обе руки)
- На робота отправляется только `action_traj[0]` — первый шаг траектории
- ACT-style temporal ensemble (коэффициент 0.01) сглаживает команды во времени
- Нормализация: min/max из G1 pack-camera датасета; обратное преобразование применяется
  перед отправкой joint targets

### Кадры с камеры (D435i head camera)

> **[ВСТАВИТЬ: step_000000/camera_input.png]** — начальная сцена, кубик на столе

> **[ВСТАВИТЬ: step_000190/camera_input.png]** — та же сцена примерно через 19 секунд

Вид не меняется между шагами — head камера робота статична в текущей конфигурации
симулятора. Кубик остаётся на столе на протяжении всего эпизода.

### Предсказанное видео модели

Каждый вызов модели генерирует 16-кадровое предсказание будущего вместе с траекторией
действий. Ниже показаны: само предсказание и сравнение side-by-side с входным кадром.

> **[ВСТАВИТЬ: step_000001/model_output.mp4]** — предсказанное будущее (16 кадров, 10 fps)

> **[ВСТАВИТЬ: step_000001/model_output_cmp.mp4]** — side-by-side: камера | предсказание

**Наблюдение:** Сгенерированное видео визуально практически не меняется от шага к шагу —
модель продуцирует одно и то же «воображаемое» будущее вне зависимости от небольших
изменений стейта. Это ожидаемое поведение для pretrained модели, которая не проходила
fine-tuning на нашей конкретной сцене. Предсказание видео является prior-ом, а не
реальным rollout-ом.

### Поведение робота

Руки движутся целенаправленно к зелёному кубику. Движение плавное и билатерально
скоординированное, соответствующее намерению pick-and-place. Однако **задача не
выполняется до конца**: руки не достигают объекта и не захватывают его.

Причина — дистрибуционный сдвиг: чекпоинт обучался на реальном G1 с конкретной
компоновкой рабочего пространства. В нашей Isaac Sim сцене положение кубика, освещение
и перспектива камеры отличаются. Для устранения этого зазора необходим fine-tuning на
данных конкретной сцены.

---

## 3. UBTech Walker Tienkung — Статус

Inference pipeline адаптирован для Walker Tienkung. Маппинг DOF, DDS-схема и конфигурация
робота интегрированы в `dds_interface.py`. Control loop, загрузчик модели и система
записи используются совместно с реализацией для G1.

---

## 4. Формат датасета

Для валидации полного pipeline был собран тестовый датасет из 10 эпизодов. Формат
соответствует спецификации `WMAData`, которую использует тренировочный код UnifoLM.

### Структура директорий

```
dataset/
├── g1_pick_place.csv
├── videos/
│   └── g1_pick_place/
│       └── front_camera/
│           ├── 0.mp4  …  9.mp4      # H.264, 10 fps, 1280×720, 80 кадров
└── transitions/
    └── g1_pick_place/
        ├── meta_data/
        │   └── stats.safetensors    # пер-DOF min/max/mean/std по всем эпизодам
        ├── 0.h5  …  9.h5
```

### H5 файл (один эпизод)

```python
action             shape=(80, 14)  float32   # joint targets  q[14:28], радианы
observation.state  shape=(80, 14)  float32   # joint readings q[14:28], радианы
attrs:
    action_type = "joint position"
    state_type  = "joint position"
    robot_type  = "Unitree G1"
```

80 шагов × 10 Hz = 8 секунд на эпизод. 14 DOF — обе руки (по 7 суставов);
ноги и торс не включены.

### Ключи stats.safetensors

```
action/min, action/max, action/mean, action/std             → каждый shape (14,)
observation.state/min, …/max, …/mean, …/std                 → каждый shape (14,)
```

---

## 5. Fine-Tuning

### Стратегия заморозки

Обучаются только компоненты, связанные с action head. Большой видео-backbone заморожен.

| | Компоненты | Параметры |
|--|-----------|-----------|
| **Заморожено** | VAE, CLIP text, DINOSigLIP, U-Net backbone | 4 128.5 M |
| **Обучается** | ConditionalUnet1D, Resampler, state/action проекторы, pos. embeddings | **50.9 M** |

Обоснование: 10–400 сэмплов недостаточно для осмысленного обновления 4B диффузионного
backbone без catastrophic forgetting. Action head (50 M) адаптирует поведение под
конкретного робота, сохраняя при этом визуальное понимание backbone.

### GPU и временны́е затраты

| Параметр | Значение |
|----------|---------|
| GPU | NVIDIA A100 80 GB PCIe |
| Batch size | **1** (при batch size 2 — OOM, не хватает ~3.9 GB) |
| Занятый VRAM | ~42–50 GB (замороженный backbone держится в памяти) |
| Время одного шага | ~8–10 сек |
| **10 сэмплов, 10 эпох** (100 шагов) | **~13–15 минут** |
| Прогноз: 400 сэмплов, 60 эпох | ~8–10 часов |

### Кривая лосса (10-сэмпловый тестовый прогон)

> **[ВСТАВИТЬ: reports/loss_curve.png]**

| Эпоха | Mean loss | Action loss (mean) |
|-------|-----------|--------------------|
| 1 | 1.51 | 0.89 (warmup, lr растёт 0 → 1e-4) |
| 2 | 1.54 | 1.11 |
| 3 | 0.83 | 0.68 |

Лосс нестабилен на синтетическом тестовом датасете (синусоидальные движения, нет
реального визуального контента). На реальных данных из симулятора action loss ожидается
ниже и стабильнее с первой эпохи.

---

## 6. Как запустить

### Запуск стека (инференс)

```bash
# На сервере x32-techgov-GPU-01 (порт 2221):
cd /root/skurchev/workspace/wam-stack
bash scripts/deploy.sh           # sync кода + сборка wam образа + docker compose up

# Тоннель для визуального мониторинга (локально):
ssh -N -L 6081:localhost:6080 -p 2221 x32-techgov-GPU-01
# Открыть: http://localhost:6081  (Isaac Sim через noVNC)
```

### Сбор датасета

```bash
# Stub-режим — без Isaac Sim, синтетические данные:
docker exec wam-inference python /workspace/wam/scripts/collect_dataset.py \
    --n-samples 10 --model stub --output-dir /workspace/wam/dataset

# Реальный режим — читает DDS стейт + камеру, запускает UnifoLM:
docker exec wam-inference python /workspace/wam/scripts/collect_dataset.py \
    --n-samples 400 --model unifolm \
    --checkpoint /workspace/wam/checkpoints/unifolm_wma_dual.ckpt \
    --output-dir /workspace/wam/dataset
```

### Fine-tuning

```bash
# Dry-run (проверка загрузки модели и данных за 2 шага):
docker exec -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    wam-inference python /workspace/wam/scripts/finetune.py \
    --dry-run --batch-size 1 \
    --dataset /workspace/wam/dataset \
    --checkpoint /workspace/wam/checkpoints/unifolm_wma_dual.ckpt \
    --output-dir /workspace/wam/media/finetune_checkpoints \
    --report-dir  /workspace/wam/media/reports

# Полная тренировка:
docker exec -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    wam-inference python /workspace/wam/scripts/finetune.py \
    --epochs 10 --batch-size 1 \
    --dataset /workspace/wam/dataset \
    --checkpoint /workspace/wam/checkpoints/unifolm_wma_dual.ckpt \
    --output-dir /workspace/wam/media/finetune_checkpoints \
    --report-dir  /workspace/wam/media/reports
```

---

## 7. Структура репозитория

```
wam-stack/
├── src/
│   ├── main.py                  # control loop 10 Hz
│   ├── dds_interface.py         # подписка/публикация DDS
│   ├── recording.py             # сохранение кадров, CSV, MP4
│   └── models/
│       ├── unifolm.py           # DDIM инференс + ACT ensemble
│       └── eva.py               # (заглушка)
├── scripts/
│   ├── collect_dataset.py       # сбор датасета (stub + real)
│   └── finetune.py              # single-GPU fine-tuning
├── docker/wam/Dockerfile        # CUDA 12.2 + Python 3.10 + h5py + pandas
├── compose.yml                  # контейнеры Isaac Sim + WAM
├── DATASET_FORMAT.md            # полная спецификация формата данных
└── WAM_PIPELINE_REPORT.md       # этот файл
```

**Сервер:** `x32-techgov-GPU-01` · `176.109.83.84:2221`  
**Рабочая директория:** `/root/skurchev/workspace/wam-stack/`  
**Чекпоинт:** `/root/skurchev/workspace/wam-stack/checkpoints/unifolm_wma_dual.ckpt` (16 GB)
