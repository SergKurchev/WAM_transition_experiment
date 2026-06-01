# WAM Dataset Format

## Dataset discovery (verified from wma_data.py + real .h5 examples 2026-06-01)

### На что опирается WMAData при загрузке

```python
WMAData(
    meta_path   = "dataset/g1_pick_place.csv",       # путь к CSV
    data_dir    = "dataset/",                         # корень датасета
    dataset_name = "g1_pick_place",                  # имя датасета
    transition_dir = "dataset/transitions/",          # корень transitions
)
```

---

## Структура директорий

```
dataset/
├── g1_pick_place.csv                              ← список эпизодов
├── videos/
│   └── g1_pick_place/
│       └── front_camera/
│           ├── 0.mp4                              ← видео эпизода 0
│           ├── 1.mp4
│           └── ...
└── transitions/
    └── g1_pick_place/
        ├── meta_data/
        │   └── stats.safetensors                  ← статистика по всему датасету
        ├── 0.h5                                   ← траектория эпизода 0
        ├── 1.h5
        └── ...
```

---

## Формат CSV (g1_pick_place.csv)

**Обязательные колонки:**

| Колонка | Тип | Пример | Описание |
|---------|-----|--------|----------|
| `videoid` | int | `0` | номер эпизода — совпадает с именем .h5 и .mp4 |
| `contentUrl` | str | `x` | заглушка, не используется |
| `duration` | str | `x` | заглушка, не используется |
| `data_dir` | str | `g1_pick_place/front_camera` | имя_датасета/имя_камеры → WMAData строит путь к видео и H5 |
| `instruction` | str | `"pick and place green cube"` | текстовый промпт для модели |
| `dynamic_confidence` | str | `x` | заглушка |
| `dynamic_wording` | str | `x` | заглушка |
| `dynamic_source_category` | str | `x` | заглушка |
| `embodiment` | str | `Unitree G1` | имя робота |
| `fps` | int | `10` | FPS видео (наш control loop = 10 Hz) |

**Как WMAData строит пути из CSV:**
```python
# Видео:
videos/<data_dir>/<videoid>.mp4
# = videos/g1_pick_place/front_camera/0.mp4

# H5 (data_dir.name != dataset_name → берёт parent):
transitions/<data_dir.parent>/<videoid>.h5
# = transitions/g1_pick_place/0.h5
```

---

## Формат H5 файла (один эпизод)

**Подтверждено из реального G1 примера:**
`examples/.../unitree_g1_pack_camera/0.h5`

```python
h5py.File("0.h5", "w") as f:
    f.create_dataset("action",            data=actions)           # shape (T, 14) float32
    f.create_dataset("observation.state", data=states)            # shape (T, 14) float32
    f.attrs["action_type"] = "joint position"
    f.attrs["state_type"]  = "joint position"
    f.attrs["robot_type"]  = "Unitree G1"
```

| Ключ | Shape | dtype | Описание |
|------|-------|-------|----------|
| `action` | `(T, 14)` | float32 | joint targets q[14:28] обеих рук в радианах |
| `observation.state` | `(T, 14)` | float32 | реальные joint positions q[14:28] |

- **T** = длина эпизода в шагах (при 10 Hz: ~50-200 шагов = 5-20 секунд)
- **14 DOF** = 7 суставов левой руки + 7 правой
- **Порядок суставов (из main.py INIT_ARM_Q):**
  ```
  left  [0:7]:  shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw
  right [7:14]: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw
  ```
- `action` и `observation.state` совпадают по индексам с `lowstate.q[14:28]` из DDS

**Примечание:** max_action_dim=16 и max_state_dim=16 в конфиге — WMAData автоматически zero-pad до 16 через `_map_to_uni_action()`. Сохранять в H5 нужно 14-мерные векторы (без padding).

---

## Формат stats.safetensors

Сохраняется через `safetensors.torch.save_file(flattened_dict, path)`.
Ключи — **flattened** через "/" (см. `flatten_dict` в prepare_training_data.py).

```python
{
    "action/max":              Tensor shape (14,) float32   # per-dim максимум по всем T×episodes
    "action/min":              Tensor shape (14,) float32
    "action/mean":             Tensor shape (14,) float32
    "action/std":              Tensor shape (14,) float32
    "observation.state/max":   Tensor shape (14,) float32
    "observation.state/min":   Tensor shape (14,) float32
    "observation.state/mean":  Tensor shape (14,) float32
    "observation.state/std":   Tensor shape (14,) float32
}
```

**Как считать:**
```python
all_actions = torch.cat([episode_actions, ...], dim=0)  # shape (N_total, 14)
stats["action"]["max"]  = all_actions.max(dim=0).values
stats["action"]["min"]  = all_actions.min(dim=0).values
stats["action"]["mean"] = all_actions.mean(dim=0)
stats["action"]["std"]  = all_actions.std(dim=0)
# flatten и save_file(flatten_dict(stats), path)
```

---

## Формат видео (.mp4)

- Кодек: **H.264** (decord не читает AV1 без конвертации)
- Разрешение: любое ≥530×300 (WMAData загружает с resize до 530×300 при `load_raw_resolution=False`)
  - В нашем конфиге `load_raw_resolution=True` и `resolution=[320, 512]` с `spatial_transform=resize_center_crop`
  - Рекомендуем сохранять 1280×720 (нативное D435i), WMAData сам сделает resize
- FPS: 10 (наш control loop)
- **Минимум кадров:** `frame_stride*(video_length-1)+1 = 2*15+1 = 31` кадров
  - При 10 Hz это ≥ 3.1 секунды на эпизод
- Содержимое: RGB кадры с D435i камеры из Isaac Sim

---

## Пример одного сэмпла в псевдокоде

```python
episode_id = 0
T = 80  # 8 секунд @ 10 Hz

# Сбор данных в control loop:
states  = np.zeros((T, 14), dtype=np.float32)  # lowstate.q[14:28]
actions = np.zeros((T, 14), dtype=np.float32)  # action_traj[0] из модели

for t in range(T):
    state = dds.get_lowstate().q[14:28]          # 14-dim
    action_traj, _, frame = model(state)          # (16, 14), _, (1,3,16,320,512)
    states[t]  = state
    actions[t] = action_traj[0]                  # первый шаг траектории
    frames.append(frame)

# Сохранение:
with h5py.File(f"transitions/g1_pick_place/{episode_id}.h5", "w") as f:
    f.create_dataset("action",            data=actions)
    f.create_dataset("observation.state", data=states)
    f.attrs["action_type"] = "joint position"
    f.attrs["state_type"]  = "joint position"
    f.attrs["robot_type"]  = "Unitree G1"

save_video(frames, f"videos/g1_pick_place/front_camera/{episode_id}.mp4", fps=10)
```

---

## Конфиг WMAData для нашего датасета

```yaml
data:
  params:
    train:
      params:
        data_dir: "/workspace/wam/dataset"
        video_length: 16
        frame_stride: 2
        load_raw_resolution: True
        resolution: [320, 512]
        spatial_transform: resize_center_crop
        normalization_mode: "min_max"
        individual_normalization: True
        n_obs_steps: 2
        max_action_dim: 16       # 14 actual + 2 zero-pad
        max_state_dim: 16
    dataset_and_weights:
      g1_pick_place: 1.0
```

---

## Известные подводные камни

1. **H5 записывать в float32** — torch сохраняет float32 по умолчанию, numpy тоже.
2. **Видео кодек AV1 не поддерживается decord** — использовать libx264 при сохранении через OpenCV.
3. **T в H5 должен быть > `max(next_state_indices)`** где `next_state_indices = frame_indices + frame_stride`. При T=80 и frame_stride=2 запас большой.
4. **stats считать по всему датасету** (concat всех эпизодов) — иначе нормализация будет неверной.
5. **data_dir в CSV** должен быть `g1_pick_place/front_camera` (с именем камеры), иначе путь к H5 сломается.
