# PATCHES.md — External Code Patches

Этот файл документирует **все изменения во внешнем коде** (`mws-dimos`, `unifolm`),
которые необходимы для работы WAM-стека без GEAR-SONIC.

Патчи применяются скриптом `scripts/patch_mws_dimos.py`, который запускается
автоматически в `scripts/deploy.sh` перед `docker compose up`.

> **Принцип**: внешние репозитории (`mws-dimos`, `unifolm`) остаются нетронутыми
> в git-смысле. Патч-скрипт модифицирует файлы на хосте, создавая `.wam-bak.*`
> резервные копии. Это позволяет:
> - Обновить внешние репо (`git pull`) и переприменить патчи заново.
> - Явно видеть все изменения в одном месте.
> - Легко откатить: `.wam-bak.*` файлы сохраняют оригинал.

---

## Patch 1 — Arm commands in SUPPORT mode

**Файл:** `mws-dimos/sim/isaac/g1_sim.py`
**Функция:** `run()`, основной loop (≈ строка 680)
**Сентинель:** `# ── WAM-patch: arm-in-support-mode`

### Проблема

`g1_sim.py` разработан под GEAR-SONIC WBC. При старте симуляции Isaac Sim
удерживает робота внешней wrench-силой (`IsaacStartupSupport`), пока GEAR-SONIC
не инициализируется и не даёт команду `drop`. В это время (`support.released = False`)
g1_sim.py **игнорирует** `rt/lowcmd` и ставит все суставы в `_default_q`:

```python
# ОРИГИНАЛЬНЫЙ КОД (g1_sim.py ~строка 680):
if support.released:
    q_target = q_isaac      # ← применяет lowcmd только после дропа
    dq_target = dq_isaac
else:
    q_target = _default_q   # ← WAM не может двигать руками!
    dq_target = _zero_dq
```

Это сделано намеренно (комментарий в коде): "forwarding policy output to joints
causes arms/legs to thrash against the pinned body". Для GEAR-SONIC это правильно —
у WBC балансировочные предположения несовместимы с телом на wrench.

**Для WAM это не так**: WAM управляет только руками (14 DOF), не пытаясь
балансировать. Wrench держит корпус — физически это безопасно.

### Решение

В SUPPORT режиме: ноги/пояс держать в `_default_q`, **руки** — из `rt/lowcmd`.

```python
# ПОСЛЕ ПАТЧА:
if support.released:
    q_target = q_isaac
    dq_target = dq_isaac
else:
    # ── WAM-patch: arm-in-support-mode ───────────────────────────────
    # Keep legs/waist at default; apply arm targets from rt/lowcmd.
    _arm_mujoco_idx = torch.arange(14, 28, dtype=torch.long, device=q_hw.device)
    _arm_isaac_idx = mujoco_to_isaac[_arm_mujoco_idx]
    q_target = _default_q.clone()
    q_target[0, _arm_isaac_idx] = q_hw[_arm_mujoco_idx]
    dq_target = _zero_dq
```

### Joint mapping

```
Hardware/MuJoCo ordering (rt/lowcmd motor_cmd indices):
  DOF  0-13 → legs + waist    → остаются в _default_q (устойчивая стойка)
  DOF 14-20 → left arm (7 DOF) → берутся из rt/lowcmd
  DOF 21-27 → right arm (7 DOF) → берутся из rt/lowcmd
  DOF 28    → gripper           → passive (zeros)

Isaac Lab ordering (G1_MUJOCO_TO_ISAACLAB_DOF):
  MuJoCo 14 → IsaacLab 10
  MuJoCo 15 → IsaacLab 16
  MuJoCo 16 → IsaacLab 23
  MuJoCo 17 → IsaacLab 5
  MuJoCo 18 → IsaacLab 11
  MuJoCo 19 → IsaacLab 17
  MuJoCo 20 → IsaacLab 24
  MuJoCo 21 → IsaacLab 18
  MuJoCo 22 → IsaacLab 25
  MuJoCo 23 → IsaacLab 19
  MuJoCo 24 → IsaacLab 26
  MuJoCo 25 → IsaacLab 20
  MuJoCo 26 → IsaacLab 27
  MuJoCo 27 → IsaacLab 21
```

### Почему это безопасно

- Wrench удерживает корень (`root_z ≈ 0.777 m`).
- Ноги/пояс командуются `_default_q` (те же значения что были в SUPPORT режиме).
- Руки получают joint targets от модели UnifoLM (≈ 1 рад L2-норма).
- Нет риска падения или нестабильности — wrench компенсирует гравитацию.
- Симуляция продолжает работать на RTF ≈ 0.08 (достаточно для записи датасета).

### Как сбросить

```bash
# Посмотреть резервную копию:
ls /root/skurchev/workspace/mws-dimos/sim/isaac/g1_sim.py.wam-bak.*

# Восстановить оригинал:
cp /root/skurchev/workspace/mws-dimos/sim/isaac/g1_sim.py.wam-bak.<timestamp> \
   /root/skurchev/workspace/mws-dimos/sim/isaac/g1_sim.py
```

---

## Patch 3 — Kinematic robot root (G1_KINEMATIC_ROBOT)

**Файл:** `mws-dimos/sim/isaac/g1_sim.py`
**Функция:** `_make_robot_cfg()` (настройка ArticulationRootPropertiesCfg)
**Сентинель:** `# WAM-patch: kinematic-robot`
**Управление:** env var `G1_KINEMATIC_ROBOT=1` в контейнере `sim-isaac`

### Проблема

По умолчанию G1 симулируется с полной физикой + startup support wrench.
WAM отправляет команды рукам, робот их выполняет, но физика продолжает
действовать на корень таза. Когда wrench снимается (или не успевает
компенсировать), робот падает. Для сбора датасета это критично —
нам важны сцены с конкретными движениями рук, а не падения.

### Решение

Isaac Lab `ArticulationRootPropertiesCfg.fix_root_link = True` закрепляет
корень таза в точке спауна (кинематический root). Суставы по-прежнему
управляются PD-контроллером по командам WAM. Коллизии активны —
робот может взаимодействовать с кубом и коробкой.

```python
# ПОСЛЕ ПАТЧА (g1_sim.py):
if os.environ.get("G1_KINEMATIC_ROBOT", "0") == "1":  # WAM-patch: kinematic-robot
    cfg.spawn.articulation_props.fix_root_link = True
    # Robot pelvis anchored to spawn; joints driven by WAM targets.
    # Collision geometry active → can interact with physics objects.
```

### Почему это безопасно

- `fix_root_link` — официальный Isaac Lab параметр, не хак.
- Физика объектов сцены (куб, коробка) работает полностью.
- Коллизии робота активны — захват/столкновение работает корректно.
- Суставы управляются теми же PD-targets что и без патча.
- Откат: `G1_KINEMATIC_ROBOT=0` (или убрать из compose.yml).

### Как включить/выключить

```bash
# Включить (по умолчанию в compose.yml):
G1_KINEMATIC_ROBOT=1 docker compose restart sim-isaac

# Выключить (обычная физика):
G1_KINEMATIC_ROBOT=0 docker compose restart sim-isaac
```

---

## Patch 2 — Remove xformers assertion (unifolm attention.py)

**Файл:** `repos/unifolm/src/unifolm_wma/modules/attention.py`
**Функция:** `forward()` (≈ строка 128)
**Сентинель:** `# ── WAM-patch: xformers-assert-removed`

### Проблема

unifolm_wma собирался с xformers для PyTorch 2.10+cu128.
Контейнер WAM использует PyTorch 2.7+cu126 — несовместимая версия.
При загрузке модели xformers бросает исключение, а в `attention.py`
стоял guard:

```python
assert 1 > 2, ">>> ERROR: should setup xformers"
```

Эта строка блокирует fallback на обычный PyTorch attention,
хотя он реализован ниже и полностью функционален.

### Решение

Удалить/закомментировать assert. Vanilla PyTorch attention (Flash-Attention
или SDPA) работает корректно на cu126 и даёт те же результаты.

### Почему это безопасно

- Vanilla attention математически идентичен xformers для inference.
- Незначительно медленнее (≈10-15%), но DDIM на GPU всё равно ограничен
  пропускной способностью диффузионной модели, а не attention.
- Модель производит корректные ненулевые действия (норма ≈ 1.0–1.5 рад).

---

## Применение патчей

```bash
# На сервере, из директории wam-stack/:
python3 scripts/patch_mws_dimos.py

# Dry run (без записи, только показать что изменится):
python3 scripts/patch_mws_dimos.py --dry-run --verbose

# После патча — перезапустить Isaac Sim (чтобы загрузил новый g1_sim.py):
docker compose restart sim-isaac
# Или полный рестарт стека:
docker compose down && docker compose up -d
```

`scripts/deploy.sh` запускает патчи автоматически на шаге 2.

---

## История патчей

| Дата | Патч | Автор |
|------|------|-------|
| 2026-05-29 | Patch 3: g1_sim.py kinematic-robot (G1_KINEMATIC_ROBOT) | Сергей |
| 2026-05-27 | Patch 2: unifolm attention.py assert | Сергей |
| 2026-05-27 | Patch 1: g1_sim.py arm-in-support-mode | Сергей |
