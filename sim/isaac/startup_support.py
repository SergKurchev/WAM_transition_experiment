"""Floating-base PD spring-damper startup support for the slim Isaac G1 runtime.

Analogous to ``IsaacSupport`` in ``unitree_sim_isaaclab/startup_support.py``,
ported for the bare ``InteractiveScene`` + ``SimulationContext`` loop in
``g1_sim.py``.  No ``ManagerBasedRLEnv``, no ``robot_dds`` param.

Usage (inside g1_sim.run())::

    support = IsaacStartupSupport(scene, sim, device=robot.device)
    # ... enter main loop ...
    for _ in range(DECIMATION):
        if not support.released:
            support.apply()
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(dt=SIM_DT)
    support.check_and_release()

Release: call ``support.trigger_drop()`` from the main loop when GEAR-SONIC
sends its first ``rt/lowcmd``.  The wrench is then linearly ramped to zero
over RELEASE_RAMP_STEPS sub-steps (1.0 s at sim_dt=0.005 s) to avoid PhysX
stiff-contact buckling on landing.

Three hard-won fixes carried over from the reference implementation:
1. Sign correction — restoring torque uses ``-KP_ANG * rotvec`` (not +).
2. Pre-flush ordering — ``apply()`` must run BEFORE ``scene.write_data_to_sim()``.
3. Persistent-setter override — on release, ``robot.has_external_wrench = True``
   forces the next ``write_data_to_sim()`` to flush zeros to PhysX, clearing the
   persistent force that would otherwise keep the robot suspended.
"""
from __future__ import annotations

import builtins as _builtins
import threading
from typing import Any, List

import torch


def print(*args: Any, **kwargs: Any) -> None:  # noqa: A001 — intentional module-scoped shadow
    """Module-scoped ``print`` that always flushes so loop-rate diagnostics are live."""
    kwargs.setdefault("flush", True)
    _builtins.print(*args, **kwargs)


class IsaacStartupSupport:
    """Floating-base PD spring-damper support for the slim Isaac G1 startup.

    Args:
        scene: The ``InteractiveScene`` instance (must contain ``"robot"`` articulation).
        sim: The ``SimulationContext`` instance (used only for gravity-suppression fallback).
        device: Torch device string (e.g. ``"cuda:0"``).
    """

    KP_POS: float = 10_000.0
    KD_POS: float = 1_000.0
    KP_ANG: float = 1_000.0
    KD_ANG: float = 10.0

    # Raise the support target Z by this much above spawn so feet hang free.
    # See reference implementation for full rationale; 0.05 m gives ~1.6 cm sag
    # above spawn_z so the release drop is small.
    VERTICAL_LIFT_OFFSET: float = 0.05

    # Safety floor: honour drop only after this many physics sub-steps.
    RELEASE_MIN_STEPS: int = 200
    RELEASE_MIN_ROOT_Z: float = 0.70

    # Ramp-down: 200 sub-steps × 0.005 s = 1.0 s gradual descent.
    RELEASE_RAMP_STEPS: int = 200

    def __init__(
        self,
        scene: Any,
        sim: Any,
        device: str = "cuda:0",
    ) -> None:
        self._scene = scene
        self._sim = sim
        self._device = device
        self._released: bool = False
        self._step: int = 0
        self._ramping: bool = False
        self._ramp_step: int = 0

        self._drop_event = threading.Event()

        robot = scene.articulations["robot"]
        root_state = robot.data.root_state_w[0].clone()
        self._spawn_pos = root_state[0:3].clone()
        self._support_target_pos = self._spawn_pos.clone()
        self._support_target_pos[2] += self.VERTICAL_LIFT_OFFSET
        self._spawn_quat = root_state[3:7].clone()
        self._spawn_quat_inv = torch.stack([
            self._spawn_quat[0], -self._spawn_quat[1],
            -self._spawn_quat[2], -self._spawn_quat[3],
        ])

        self._body_ids: List[int] = [0]
        _attach_usd_name: str = "pelvis"
        for candidate in ("pelvis", "torso_link"):
            try:
                ids, names = robot.find_bodies(candidate)
                if ids:
                    self._body_ids = list(ids[:1])
                    _attach_usd_name = names[0]
                    print(f"[IsaacStartupSupport] attach body: '{names[0]}' (id={ids[0]})")
                    break
            except Exception:
                continue
        else:
            print("[IsaacStartupSupport] WARNING: body name resolution failed; using body id=0")

        self._use_wrench: bool = True
        self._use_gravity_suppression: bool = False
        self._force_buf = torch.zeros(1, 1, 3, device=device, dtype=torch.float32)
        self._torque_buf = torch.zeros(1, 1, 3, device=device, dtype=torch.float32)

        _wrench_ok = False
        try:
            robot.set_external_force_and_torque(
                torch.zeros(1, 1, 3, device=device),
                torch.zeros(1, 1, 3, device=device),
                body_ids=self._body_ids[:1],
                env_ids=[0],
                is_global=True,
            )
            _wrench_ok = True
            print(
                f"[IsaacStartupSupport] mode: articulation wrench  "
                f"body='{_attach_usd_name}'  body_id={self._body_ids[0]}"
            )
        except Exception as e:
            print(
                f"[IsaacStartupSupport] WARNING: set_external_force_and_torque probe "
                f"failed ({e}); falling back to gravity suppression"
            )

        if not _wrench_ok:
            self._use_wrench = False
            self._use_gravity_suppression = True
            self._gravity_restore_mag: float = 9.81
            try:
                from pxr import UsdPhysics
                _stage = sim.stage
                _found = False
                for _prim in _stage.Traverse():
                    if _prim.IsA(UsdPhysics.Scene):
                        _usd_scene = UsdPhysics.Scene(_prim)
                        _existing = _usd_scene.GetGravityMagnitudeAttr().Get()
                        if _existing is not None and _existing > 0.0:
                            self._gravity_restore_mag = float(_existing)
                        _usd_scene.GetGravityMagnitudeAttr().Set(0.0)
                        _found = True
                        break
                if not _found:
                    raise RuntimeError("no UsdPhysics.Scene prim found in stage")
                print(
                    f"[IsaacStartupSupport] mode: gravity suppression  "
                    f"(will restore {self._gravity_restore_mag:.3f} m/s²)"
                )
            except Exception as e:
                raise RuntimeError(
                    f"[IsaacStartupSupport] gravity suppression fallback failed ({e})"
                ) from e

        assert self._use_wrench != self._use_gravity_suppression

        spawn_pos_str = [f"{v:.3f}" for v in self._spawn_pos.tolist()]
        print(
            f"[IsaacStartupSupport] initialised — spawn_pos={spawn_pos_str}  "
            f"support_target_z={self._support_target_pos[2].item():.3f}  "
            f"release=trigger_drop() on first rt/lowcmd  "
            f"min_steps={self.RELEASE_MIN_STEPS}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def released(self) -> bool:
        return self._released

    @property
    def support_scale(self) -> float:
        """Fraction of support wrench currently applied: 1.0 before drop,
        linearly ramping to 0.0 across the release ramp window, 0.0 once
        fully released.  Callers (e.g., joint-target blending in the main
        loop) can use ``1.0 - support_scale`` as the policy-authority
        fraction so policy authority ramps up while the wrench ramps down.
        """
        if self._released:
            return 0.0
        if self._ramping:
            return max(0.0, 1.0 - self._ramp_step / max(1, self.RELEASE_RAMP_STEPS))
        return 1.0

    def apply(self) -> None:
        """Apply support for one physics sub-step.

        Must run BEFORE ``scene.write_data_to_sim()`` in the decimation loop.
        No-op once released.
        """
        if self._released:
            return

        self._step += 1
        data = self._scene.articulations["robot"].data

        if self._use_wrench:
            if self._ramping:
                scale = max(0.0, 1.0 - self._ramp_step / max(1, self.RELEASE_RAMP_STEPS))
                self._apply_wrench(data, scale=scale)
                self._ramp_step += 1
                if self._ramp_step >= self.RELEASE_RAMP_STEPS:
                    self._release()
                    return
            else:
                self._apply_wrench(data)
        # gravity-suppression: nothing to do per sub-step

        if self._step % 50 == 0:
            root_z = data.root_pos_w[0][2].item()
            _mode = "wrench" if self._use_wrench else "grav-suppress"
            if self._ramping:
                _scale = max(0.0, 1.0 - self._ramp_step / max(1, self.RELEASE_RAMP_STEPS))
                _state = f"RAMP {self._ramp_step}/{self.RELEASE_RAMP_STEPS} scale={_scale:.2f}"
            else:
                _state = "armed" if self._drop_event.is_set() else "waiting"
            print(
                f"[IsaacStartupSupport] SUPPORT mode={_mode} step={self._step} "
                f"root_z={root_z:.4f} drop_signal={_state}"
            )

    def trigger_drop(self) -> None:
        """Signal that the wrench should be released.

        Call once from the main loop when GEAR-SONIC sends its first rt/lowcmd.
        Idempotent — safe to call on every loop iteration once the condition is met.
        """
        if not self._drop_event.is_set():
            print("[IsaacStartupSupport] drop triggered by first rt/lowcmd from GEAR-SONIC")
            self._drop_event.set()

    def check_and_release(self) -> None:
        """Check drop signal once per control step (outside decimation loop). No-op once released."""
        if self._released:
            return
        self._check_release()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
        w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
        return torch.stack([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ])

    def _apply_wrench(self, data: Any, scale: float = 1.0) -> None:
        pos = data.root_pos_w[0]
        root_state = data.root_state_w[0]
        quat = root_state[3:7]
        lin_vel = root_state[7:10]
        ang_vel = root_state[10:13]

        delta_pos = self._support_target_pos - pos
        force = (self.KP_POS * delta_pos + self.KD_POS * (-lin_vel)) * scale

        q_err = self._quat_mul(quat, self._spawn_quat_inv)
        q_err = q_err / q_err.norm().clamp(min=1e-8)
        if q_err[0] < 0.0:
            q_err = -q_err
        rotvec = 2.0 * q_err[1:4]
        torque = (-self.KP_ANG * rotvec - self.KD_ANG * ang_vel) * scale

        self._force_buf[0, 0].copy_(force)
        self._torque_buf[0, 0].copy_(torque)

        robot = self._scene.articulations["robot"]
        robot.set_external_force_and_torque(
            self._force_buf,
            self._torque_buf,
            body_ids=self._body_ids,
            env_ids=[0],
            is_global=True,
        )

    def _check_release(self) -> None:
        if not self._drop_event.is_set():
            return
        if self._step < self.RELEASE_MIN_STEPS:
            return
        if self._ramping:
            return
        root_z = self._scene.articulations["robot"].data.root_pos_w[0][2].item()
        if root_z < self.RELEASE_MIN_ROOT_Z:
            print(
                f"[IsaacStartupSupport] WARNING: drop signal with "
                f"root_z={root_z:.3f} < {self.RELEASE_MIN_ROOT_Z:.3f} m — releasing anyway"
            )
        if self._use_wrench and self.RELEASE_RAMP_STEPS > 0:
            self._ramping = True
            self._ramp_step = 0
            print(
                f"[IsaacStartupSupport] RAMP_START step={self._step} root_z={root_z:.4f} "
                f"ramp_steps={self.RELEASE_RAMP_STEPS} "
                f"({self.RELEASE_RAMP_STEPS * 0.005:.2f} s at sim_dt=0.005 s)"
            )
        else:
            self._release()

    def rearm(self) -> None:
        """Reset support state for the next episode.

        Caller must have already teleported the robot to its spawn pose (so that
        the freshly-read root state reflects the new spawn position).
        The next trigger_drop() call will re-release the wrench.
        """
        self._drop_event.clear()
        self._released = False
        self._ramping = False
        self._ramp_step = 0
        self._step = 0

        robot = self._scene.articulations["robot"]
        root_state = robot.data.root_state_w[0].clone()
        self._spawn_pos = root_state[0:3].clone()
        self._support_target_pos = self._spawn_pos.clone()
        self._support_target_pos[2] += self.VERTICAL_LIFT_OFFSET
        self._spawn_quat = root_state[3:7].clone()
        self._spawn_quat_inv = torch.stack([
            self._spawn_quat[0], -self._spawn_quat[1],
            -self._spawn_quat[2], -self._spawn_quat[3],
        ])

        spawn_pos_str = [f"{v:.3f}" for v in self._spawn_pos.tolist()]
        print(
            f"[IsaacStartupSupport] rearm — spawn_pos={spawn_pos_str}  "
            f"support_target_z={self._support_target_pos[2].item():.3f}"
        )

    def _release(self) -> None:
        if self._use_wrench:
            robot = self._scene.articulations["robot"]
            robot.set_external_force_and_torque(
                torch.zeros(1, 1, 3, device=self._device),
                torch.zeros(1, 1, 3, device=self._device),
                body_ids=self._body_ids,
                env_ids=[0],
                is_global=True,
            )
            # Fix #3: override False so write_data_to_sim flushes zeros to PhysX,
            # clearing the persistent force from the tensor API's SET semantics.
            robot.has_external_wrench = True
        elif self._use_gravity_suppression:
            try:
                from pxr import UsdPhysics
                _stage = self._sim.stage
                for _prim in _stage.Traverse():
                    if _prim.IsA(UsdPhysics.Scene):
                        UsdPhysics.Scene(_prim).GetGravityMagnitudeAttr().Set(
                            self._gravity_restore_mag
                        )
                        print(
                            f"[IsaacStartupSupport] gravity restored: "
                            f"{self._gravity_restore_mag:.3f} m/s²"
                        )
                        break
            except Exception as e:
                print(f"[IsaacStartupSupport] WARNING: gravity restore failed ({e})")

        _mode = "wrench" if self._use_wrench else "grav-suppress"
        self._released = True
        root_z = self._scene.articulations["robot"].data.root_pos_w[0][2].item()
        print(
            f"[IsaacStartupSupport] SUPPORT_RELEASED mode={_mode} step={self._step} "
            f"root_z={root_z:.4f}"
        )
