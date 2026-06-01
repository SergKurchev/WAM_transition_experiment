# Отчёт: тестирование камер Isaac Sim (G1 + wrist cameras)

**Дата:** 30 мая 2026  
**Цель:** Запустить простой Докер с Isaac Sim, снять кадры со всех трёх камер (head + left wrist + right wrist), собрать всё, что подаётся в модель, и проверить корректность установки камер.

---

## 1. Конфигурация камер

Робот Unitree G1 в сцене `mws_office.usdz`. Три камеры:

| Камера | SHM путь | Разрешение | Назначение |
|--------|----------|------------|------------|
| Head (D435i) | `/run/mws/camera.rgb` | 1280×720 | Уже работала — голова робота |
| Left wrist | `/run/mws/camera_left_wrist.rgb` | 1280×720 | Новая — левое запястье |
| Right wrist | `/run/mws/camera_right_wrist.rgb` | 1280×720 | Новая — правое запястье |

Wrist-камеры — часть пайплайна для моделей TrajBooster / LingBot-VA / FastWAM, которым нужны 3 вида (head + 2 wrist).

---

## 2. Первые результаты: wrist-камеры серые

После запуска Isaac Sim и захвата кадров wrist-камеры показали **однородный серый RGB = [128.9, 128.9, 128.9]** — это default clear color Isaac Sim, когда камера ничего не рендерит.

![Первый захват — все три камеры](media/camera_test/composite_all_cameras.png)

*Слева направо: HEAD (норм), LEFT WRIST (серый), RIGHT WRIST (серый).*

Статистика первого захвата:

```
head:        nonzero=99.8%  mean=[191.7, 192.8, 189.2]  std≠0  ← реальная сцена
left_wrist:  nonzero=100.0% mean=[128.9, 128.9, 128.9]  std=0  ← пустота
right_wrist: nonzero=100.0% mean=[128.9, 128.9, 128.9]  std=0  ← пустота
```

---

## 3. Диагностика

### Head-камера работала сразу

![Head camera](media/camera_test/head.png)

Сцена видна: монитор, зелёный куб, стол, рука робота. Head-камера определена **прямо в G1 USD** как дочерний прим `d435_optical_frame → d435_link → torso_link` — Isaac Sim знает о ней с начала и правильно обновляет позу.

### Почему wrist-камеры серые — корень проблемы

**Isaac Sim 5.x использует Fabric** для хранения физических трансформ. При рендере Fabric обновляет `xformOp` только для **корневого тела** артикуляции (pelvis). Остальные звенья (руки, запястья) рендерятся через внутренний кинематический кэш артикуляции, **минуя** записть `xformOp` в USD.

Камеры, созданные динамически через `CameraCfg` (в отличие от head, которая pre-existed в USD), прикрепляются к USD-прим как дочерние узлы. Но поскольку `xformOp` родительского звена (`left_wrist_yaw_link`) **не обновляется** Fabric-ом в USD стейдже, камера остаётся замёрзшей в T-позе мировых координат.

```
G1 артикуляция (Fabric, быстро)
  ├── pelvis → xformOp обновляется ✓
  ├── torso_link → head camera следует (pre-existing USD prim) ✓
  └── left_wrist_yaw_link → xformOp НЕ обновляется в USD ✗
        └── camera_left_wrist → висит в T-позе, смотрит в пустоту ✗
```

---

## 4. Решение: Patches 14 и 15

### Patch 14 — явная синхронизация поз перед рендером

Добавляем явный вызов `set_world_poses()` перед каждым `sim.render()`. Берём актуальную физическую позу запястья из `robot.data.body_pos_w` / `body_quat_w` и записываем её в camera prim.

```python
# Индексы звеньев:
# [g1_sim] wrist body indices: left=28  right=29
_lp = robot.data.body_pos_w[0:1, _left_wrist_body_idx]
_lq = robot.data.body_quat_w[0:1, _left_wrist_body_idx]
left_wrist_cam.set_world_poses(_lp, _lq)
right_wrist_cam.set_world_poses(_rp, _rq)
```

После Patch 14 камеры начали **видеть реальную геометрию** — тело робота:

![После Patch 14](media/camera_test/composite_v3.png)

*Слева направо: HEAD, LEFT WRIST (тело/рука), RIGHT WRIST (плечо/корпус).*

```
left_wrist:  mean=[160.6, 160.6, 160.6]  std=30   ← реальная геометрия ✓
right_wrist: mean=[136.7, 136.7, 136.7]  std=45.9 ← реальная геометрия ✓
```

### Patch 15 — применяем OffsetCfg

Patch 14 ставил камеру точно в **центр запястного сустава** с ориентацией тела. `CameraCfg.OffsetCfg(pos=(0.06, 0, 0), rot=(0.7071, 0, -0.7071, 0))` при этом игнорировался.

Patch 15 применяет offset явно через `quat_apply` и `quat_mul`:

```python
from isaaclab.utils.math import quat_apply, quat_mul

_off_p = torch.tensor([[0.06, 0., 0.]], device=robot.device)   # +6 см по +X
_off_q = torch.tensor([[0.7071, 0., -0.7071, 0.]], device=robot.device)  # -90° вокруг Y

# cam_pos  = wrist_pos  + rotate(wrist_quat, offset_pos)
# cam_quat = wrist_quat * offset_quat   →  camera смотрит по +X запястья
left_wrist_cam.set_world_poses(
    _lp + quat_apply(_lq, _off_p),
    quat_mul(_lq, _off_q),
)
right_wrist_cam.set_world_poses(
    _rp + quat_apply(_rq, _off_p),
    quat_mul(_rq, _off_q),
)
```

**Почему −90° вокруг Y:** камера смотрит по своей оси −Z. Поворот −90°Y отображает −Z → +X в системе запястья. +X запястья G1 направлен к пальцам. Итого: камера смотрит от запястья **к пальцам и рабочему пространству** перед ними.

---

## 5. Финальные результаты

![Финальный захват (Patch 15)](media/camera_test/composite_v4.png)

*Слева направо: HEAD, LEFT WRIST (пол + рука), RIGHT WRIST (предплечье + рука).*

### Head camera

![Head camera final](media/camera_test/head_v3.png)

Корректный вид сцены: монитор, зелёный куб, рука робота внизу.

### Left wrist camera (финал)

![Left wrist final](media/camera_test/left_wrist_v4.png)

Камера смотрит вдоль левой руки к пальцам и вниз. Видны:
- Пол рабочей зоны (левая половина кадра)
- Пальцы/кисть руки (правая часть)
- Зелёный куб в верхнем левом углу

### Right wrist camera (финал)

![Right wrist final](media/camera_test/right_wrist_v4.png)

Камера смотрит вдоль правой руки. Видны предплечье, сочленения, рабочее пространство за рукой.

### Финальная статистика

```
head:        nonzero=99.8%  mean=[184.5, 185.5, 181.6]  std=[76.8, 76.8, 79.2]  ← пёстрая сцена ✓
left_wrist:  nonzero=100.0% mean=[151.3, 151.4, 151.3]  std=[18.8, 18.6, 18.9]  ← рука + пол ✓
right_wrist: nonzero=100.0% mean=[139.0, 139.0, 139.0]  std=[34.6, 34.6, 34.6]  ← рука ✓
```

---

## 6. Сводка применённых патчей

| Патч | Файл | Что делает |
|------|------|-----------|
| Patch 8–12 | `dds_bridge.py` | SHM-файлы для wrist камер, методы `configure_wrist_cameras_export()` и `publish_wrist_cameras()` |
| Patch 4–7 | `g1_sim.py` | `CameraCfg` для wrist камер в `G1SceneCfg`, acquire + configure + publish |
| **Patch 14a** | `g1_sim.py` | Находим body indices `left=28, right=29` |
| **Patch 14b** | `g1_sim.py` | `set_world_poses(wrist_pos, wrist_quat)` перед каждым `sim.render()` |
| **Patch 15** | `g1_sim.py` | Применяем OffsetCfg: `cam_pos = wrist_pos + rotate(wrist_quat, 0.06m·X)`, `cam_quat = wrist_quat * (-90°Y)` |

Все патчи идемпотентны (sentinel-based) и живут в `scripts/patch_mws_dimos.py`.

---

## 7. Что подаётся в модель

Для UnifoLM (текущая основная модель) используется только **head camera**:

```
/run/mws/camera.rgb  →  resize 320×512  →  norm (x/255)*2−1 → [-1, 1]
```

Для будущих моделей (LingBot-VA, TrajBooster, FastWAM) — все три камеры:

```
/run/mws/camera.rgb              →  head    (scene context)
/run/mws/camera_left_wrist.rgb   →  left    (left hand + workspace)
/run/mws/camera_right_wrist.rgb  →  right   (right hand + workspace)
```

SHM обновляется синхронно с `sim.render()` через `omni.replicator` annotators (RGB CPU readback).

---

## 8. Оставшиеся замечания

1. **Вид в rest-позе**: в стандартной позе покоя руки у туловища — wrist камеры смотрят вниз и немного в сторону тела. При вытянутой к столу руке поле зрения сместится на стол/объекты. Это нормальное поведение для wrist-mounted camera.

2. **Текстуры сцены**: в логах есть предупреждения о missing textures для `mws_office.usdz` (RubikCube, Mug и др.) — не блокирует работу, но объекты выглядят без текстур (plain grey/white).

3. **Rotation fine-tuning**: для более агрессивного взгляда вниз на стол можно добавить дополнительный наклон (например, −15° вокруг Z в offset quaternion), но это имеет смысл проверять с реальными демонстрациями.
