# UnifoLM-WMA-0: Deployment & Operations Guide

Руководство по запуску реального DDIM inference UnifoLM-WMA-0 на G1.

**Статус:** ✅ Работает на GPU-01 с 27 мая 2026.

---

## Предварительные условия

На сервере должны быть:

```
/root/skurchev/workspace/
├── wam-stack/
│   ├── checkpoints/unifolm_wma_dual.ckpt   ← ≈16 GB чекпойнт
│   └── repos/unifolm/                       ← unifolm repo с патчем
└── mws-dimos/                               ← feat/real-transfer ветка
```

Проверить:
```bash
ls -lh /root/skurchev/workspace/wam-stack/checkpoints/unifolm_wma_dual.ckpt
# Должно быть ≈16G

python3 -c "import sys; sys.path.insert(0, '/root/skurchev/workspace/wam-stack/repos/unifolm/src'); \
  from unifolm_wma.models.ddpms import LatentVisualDiffusion; print('OK')"
# Должно вывести: OK
```

---

## Запуск

```bash
cd /root/skurchev/workspace/wam-stack
docker compose up -d
```

Полный старт занимает **~90 секунд** (Isaac Sim grpc инициализация + загрузка модели).

### Прогресс загрузки (что искать в логах)

```bash
docker logs wam-inference -f
```

| Что видим | Что происходит |
|-----------|----------------|
| `[UnifoLM] ✓ Loaded G1 pack camera normalization stats` | stats загружены |
| `[UnifoLM] ✓ Camera SHM: /run/mws/camera.rgb` | D435i камера доступна |
| `AE working on z of shape (1, 4, 32, 32)` | AutoEncoder инициализируется |
| `[UnifoLM] Checkpoint loaded successfully` | чекпойнт загружен |
| `[WAM] rt/lowstate received` | DDS соединение установлено |
| `[WAM] loop=2  action_0[0:3]=...` | DDIM inference работает |

---

## Что делает модель

Каждую итерацию (10 Hz) модель:

1. **Читает** 2 последних кадра D435i (1280×720 RGB из SHM)
2. **Нормализует** стейт робота (14 DOF → 16, min-max → [-1, 1])
3. **Запускает DDIM** (16 шагов, ~2-4 сек на GPU)
4. **Декодирует** латентное видео → пиксели
5. **Денормализует** действия → радианы
6. **Записывает** MP4, PNG, TXT

Выходной лог:
```
[WAM] loop=5  action_0[0:3]=[-0.42 +0.42 -0.17]  norm=1.18  traj_norm=4.96  state_age=16ms  q0=-0.275
```

- `action_0[0:3]` — первые 3 DOF (shoulder pitch/roll/yaw) в **радианах**
- `norm` — L2-норма первого шага (типично 0.8–1.5 рад)
- `traj_norm` — L2-норма всей 16-шаговой траектории (типично 4–6)

---

## Нормализация

### Стейт (вход в модель)

```python
# Формула: (x - min) / (max - min + 1e-8) * 2 - 1
# Источник: unitree_g1_pack_camera/meta_data/stats.safetensors

state_min = [-1.6514, -0.1329, -1.3105, -1.0467, -1.9663, -1.4105,
             -1.6218, -1.4388, -1.7361, -0.7223, -1.0379, -0.5346,
             -1.0349, -0.6417, -0.0307,  0.0230]

state_max = [ 0.9437,  1.6497,  0.9905,  1.3529,  1.4207,  1.6086,
              1.1852,  0.6558,  0.1548,  1.4194,  1.3507,  1.3449,
              1.6028,  1.6194,  5.4722,  5.4958]
```

Dims 0-13 — 14 DOF G1 arms. Dims 14-15 — gripper (нули, т.к. G1 без gripper в этой конфигурации).

### Действия (выход из модели)

```python
# Формула обратная: (norm + 1) / 2 * (max - min) + min

action_min = [-1.6897, -0.3179, -1.3181, -1.0472, -1.9722, -1.4171,
              -1.6144, -1.4506, -1.7541, -0.7332, -1.0460, -0.5455,
              -1.0137, -0.6167,  0.0000,  0.0000]

action_max = [ 0.9541,  1.6616,  0.9965,  1.3657,  1.4339,  1.6144,
               1.1964,  0.6619,  0.1525,  1.4254,  1.3613,  1.3555,
               1.6144,  1.6144,  5.4000,  5.4000]
```

---

## Камера

Приоритет источников:

```
1. /run/mws/camera.rgb          ← D435i head cam (PREFERRED)
   Формат: raw RGB bytes, 1280×720×3, unsigned int8
   Перспектива: совпадает с G1_Dex1_MountCameraRedGripper training data

2. /tmp/isaac_frame.png         ← Isaac Sim editor view (fallback)
   Проблема: top-down view, domain mismatch с training data
   → видео выход модели тёмное/мутное

3. Последний закешированный кадр
4. Чёрный кадр (zeros)
```

Проверить какой источник используется:
```bash
docker logs wam-inference | grep "Camera SHM"
# Если видишь ✓ — SHM доступен (нормально)
# Если видишь "not found" — используется fallback
```

---

## Медиа-файлы

### Расположение

```
/root/skurchev/workspace/wam-stack/media/
├── model_output/output_XXXXXX.mp4   # Генерируемое видео (16 кадров, 10 FPS)
├── robot_camera/camera_XXXXXX.png   # D435i кадры (1280×720)
├── input_frames/state_XXXXXX.txt    # Стейт (q, dq, tau текстом)
└── command_logs/commands_*.csv      # Лог шагов
```

### Скачать на локальную машину

```bash
# На локальной машине
scp -r -P 2221 root@176.109.83.84:/root/skurchev/workspace/wam-stack/media/model_output/ ./
scp -r -P 2221 root@176.109.83.84:/root/skurchev/workspace/wam-stack/media/robot_camera/ ./
```

### Очистка

```bash
find /root/skurchev/workspace/wam-stack/media -type f -delete
```

---

## Конфигурация модели

Файл: `src/config_model.yaml`

Критические параметры (не менять без понимания):

```yaml
n_obs_steps_imagen: 2       # размер истории (deque maxlen)
n_obs_steps_acting: 2       # история для action head
agent_state_dim: 16         # должно совпадать с чекпойнтом
agent_action_dim: 16        # должно совпадать с чекпойнтом
decision_making_only: True  # action head активен
temporal_length: 16         # горизонт (DDIM T)
default_fs: 10              # fallback FPS (мы используем 15 в inference)
```

DDIM параметры (в `unifolm.py`):
```python
ddim_steps = 16   # шагов семплирования
ddim_eta   = 1.0  # стохастичность (1.0 = DDIM-stochastic)
MODEL_FPS  = 15   # FPS conditioning (30 / frame_stride=2)
```

---

## Изменение промпта

```bash
# На сервере или через env var в compose.yml
WAM_PROMPT="robot arm picks up red block" docker compose restart wam
```

Или в `compose.yml`:
```yaml
environment:
  - WAM_PROMPT=robot arm picks up red block
```

---

## Troubleshooting

### "Missing keys" при загрузке чекпойнта

Норма — модель загружается с `strict=False`. Несколько missing/unexpected ключей не критично.  
Проблема есть если inference возвращает нули (`norm=0.000 traj_norm=0.000`).

### `assert 1 > 2` ошибка

```
AssertionError: >>> ERROR: should setup xformers
```

**Причина:** Патч attention.py не применён.  
**Фикс на сервере:**
```bash
sed -i '/assert 1 > 2/d' /root/skurchev/workspace/wam-stack/repos/unifolm/src/unifolm_wma/modules/attention.py
docker restart wam-inference
```

### Inference слишком медленный

DDIM 16 шагов на GPU занимает ~2-4 сек. Control loop 10 Hz, то есть одна итерация inference занимает больше 1 control step — это нормально, inference запускается асинхронно.

Если inference вообще не запускается (только zeros в логах), проверить:
```bash
docker logs wam-inference | grep -E "DDIM|Inference failed|FATAL"
```

### Видео чёрное

1. Проверить источник камеры (SHM vs fallback)
2. Проверить нормализацию — если stats не загрузились, стейт подаётся ненормализованным
3. Domain mismatch с training data — уменьшится после fine-tuning на Isaac Sim данных

### Нет SHM камеры

```bash
# На сервере: проверить что файл есть
ls -la /run/mws/camera.rgb
# Если нет — Isaac Sim или ros2-bridge не писал в SHM

# Проверить isaac-sim контейнер
docker logs wam-isaac-sim | tail -30
```

---

## Следующий шаг: публикация действий

Сейчас `action_traj` только логируется. Чтобы отправить роботу, нужно добавить в `main.py`:

```python
# После получения action_traj из модели
action_0 = action_traj[0].numpy()  # [14] DOF, radians
dds.publish_lowcmd(action_0)       # нужно реализовать в dds_interface.py
```

⚠️ Это требует понимания как GEAR-SONIC принимает команды по `rt/lowcmd`.
