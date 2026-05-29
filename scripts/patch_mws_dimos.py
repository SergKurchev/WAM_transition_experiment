#!/usr/bin/env python3
"""Apply WAM-specific patches to external mws-dimos code.

Usage
-----
    python3 scripts/patch_mws_dimos.py [--dry-run] [--verbose]

Options
-------
--dry-run   Print what would change without writing anything.
--verbose   Show full diff for each patch.

Why a script instead of a git patch?
-------------------------------------
mws-dimos is maintained by the MWS team and must stay on
``feat/real-transfer`` without local commits.  A patch script is:
  - Idempotent (safe to run multiple times; skips already-applied patches).
  - Self-documenting (each PATCH entry explains the rationale).
  - Integrated into deploy.sh so it runs automatically before every
    ``docker compose up``.

All patches are documented in detail in ``wam-stack/PATCHES.md``.

Adding a new patch
------------------
Append an entry to the PATCHES list below.  Required keys:
  file        Absolute path on the server.
  description Short one-line description.
  sentinel    A string that appears only in the patched version (for idempotency).
  old         The exact substring to replace (use raw strings r"..." for backslashes).
  new         The replacement substring.
"""
from __future__ import annotations

import argparse
import difflib
import shutil
import sys
import time
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Patch definitions
# Each entry is applied in order.  All are idempotent.
# ─────────────────────────────────────────────────────────────────────────────

PATCHES: list[dict] = [

    # ── Patch 1: arm commands in SUPPORT mode ────────────────────────────────
    {
        "file": "/root/skurchev/workspace/mws-dimos/sim/isaac/g1_sim.py",
        "description": (
            "Allow arm joint commands from rt/lowcmd in Isaac SUPPORT mode "
            "(WAM without GEAR-SONIC WBC)"
        ),
        # Sentinel: unique string present ONLY after patch is applied.
        "sentinel": "# ── WAM-patch: arm-in-support-mode",
        # ── old ──────────────────────────────────────────────────────────────
        # g1_sim.py normally ignores rt/lowcmd while the startup-support wrench
        # is active, holding all joints at the default standing pose.  This is
        # fine for GEAR-SONIC (which needs to ramp up after release) but blocks
        # WAM from actuating the arms at all.
        "old": """\
                else:
                    q_target = _default_q
                    dq_target = _zero_dq""",
        # ── new ──────────────────────────────────────────────────────────────
        # In SUPPORT mode the external wrench holds the robot root; the joints
        # are still fully actuated by the Isaac Lab articulation PD controller.
        # WAM patch: keep leg/waist joints at _default_q (safe standing pose)
        # but apply arm targets from rt/lowcmd.
        #
        # Joint-space remapping:
        #   Hardware/MuJoCo ordering (used in rt/lowcmd):
        #     DOF  0-13 → legs + waist
        #     DOF 14-27 → left arm (14-20) + right arm (21-27)
        #     DOF 28    → gripper
        #   Isaac Lab ordering (used by set_joint_position_target):
        #     mujoco_to_isaac[14:28] gives the corresponding IL indices.
        "new": """\
                else:
                    # ── WAM-patch: arm-in-support-mode ───────────────────────
                    # Applied by wam-stack/scripts/patch_mws_dimos.py
                    # Documented in wam-stack/PATCHES.md (Patch 1).
                    # Keep legs/waist at default; apply arm targets from lowcmd.
                    _arm_mujoco_idx = torch.arange(14, 28, dtype=torch.long,
                                                   device=q_hw.device)
                    _arm_isaac_idx = mujoco_to_isaac[_arm_mujoco_idx]
                    q_target = _default_q.clone()
                    q_target[0, _arm_isaac_idx] = q_hw[_arm_mujoco_idx]
                    dq_target = _zero_dq
                    # ─────────────────────────────────────────────────────────""",
    },

    # ── Patch 3: kinematic robot root — follows WAM commands exactly, no physics fall ──
    # Controlled by env var G1_KINEMATIC_ROBOT=1 in compose.yml (sim-isaac).
    # fix_root_link=True anchors the robot pelvis to its spawn position.
    # The arm/leg joints are still driven by the Isaac Lab PD controller
    # (i.e. WAM joint targets work normally), but gravity/forces cannot
    # push the base — robot never falls regardless of WBC or wrench state.
    # Collision geometry remains active so the robot can interact with
    # physics objects (cube, storage box).
    {
        "file": "/root/skurchev/workspace/mws-dimos/sim/isaac/g1_sim.py",
        "description": (
            "Kinematic robot root: fix_root_link when G1_KINEMATIC_ROBOT=1 "
            "(robot follows WAM commands exactly, never falls)"
        ),
        "sentinel": "# WAM-patch: kinematic-robot",
        "old": (
            "    cfg.spawn.articulation_props.solver_position_iteration_count = 2\n"
            "    cfg.spawn.articulation_props.solver_velocity_iteration_count = 0"
        ),
        "new": (
            "    cfg.spawn.articulation_props.solver_position_iteration_count = 2\n"
            "    cfg.spawn.articulation_props.solver_velocity_iteration_count = 0\n"
            "    if os.environ.get(\"G1_KINEMATIC_ROBOT\", \"0\") == \"1\":  # WAM-patch: kinematic-robot\n"
            "        cfg.spawn.articulation_props.fix_root_link = True\n"
            "        # Robot pelvis anchored to spawn; joints driven by WAM targets.\n"
            "        # Collision geometry active → can interact with physics objects.\n"
            "        # Applied by wam-stack/scripts/patch_mws_dimos.py (Patch 3)."
        ),
    },

    # ── Patch 2: unifolm attention.py — remove xformers assertion ────────────
    {
        "file": (
            "/root/skurchev/workspace/wam-stack/repos/unifolm"
            "/src/unifolm_wma/modules/attention.py"
        ),
        "description": (
            "Remove xformers assertion that blocks vanilla attention "
            "(container has PyTorch 2.7/cu126, xformers requires cu128)"
        ),
        "sentinel": "# ── WAM-patch: xformers-assert-removed",
        "old": 'assert 1 > 2, ">>> ERROR: should setup xformers"',
        "new": (
            "# ── WAM-patch: xformers-assert-removed ─────────────────────────\n"
            "                # Applied by wam-stack/scripts/patch_mws_dimos.py\n"
            "                # Documented in wam-stack/PATCHES.md (Patch 2).\n"
            "                # xformers built for cu128; container uses cu126.\n"
            "                # Vanilla PyTorch attention (below) is fully functional.\n"
            "                # ─────────────────────────────────────────────────"
        ),
        # attention.py was manually patched before this script existed —
        # the assert was deleted directly without leaving a sentinel.
        # old_missing_ok lets the script treat that case as already-done.
        "old_missing_ok": True,
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Core logic
# ─────────────────────────────────────────────────────────────────────────────

GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
BLUE   = "\033[34m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def apply_patch(patch: dict, *, dry_run: bool = False, verbose: bool = False) -> str:
    """Apply a single patch entry.

    Returns
    -------
    "skipped"   — patch already applied (sentinel found), or manually applied
                  with old_missing_ok=True (target already absent).
    "applied"   — patch applied successfully.
    "error"     — old string not found (file may have changed upstream).

    Optional patch keys
    -------------------
    old_missing_ok : bool (default False)
        If True and the target ``old`` string is not found (and sentinel is
        absent too), treat the file as already manually patched instead of
        reporting an error.  Use when the change was applied by hand before
        the patch script existed.
    """
    path = Path(patch["file"])
    description = patch["description"]
    sentinel = patch["sentinel"]
    old = patch["old"]
    new = patch["new"]
    old_missing_ok = patch.get("old_missing_ok", False)

    print(f"\n{'─'*70}")
    print(f"{BOLD}Patch:{RESET} {description}")
    print(f"File:  {path}")

    if not path.exists():
        print(f"{RED}  ✗ File not found — skipping.{RESET}")
        return "error"

    content = path.read_text(encoding="utf-8")

    # Idempotency check
    if sentinel in content:
        print(f"{GREEN}  ✓ Already applied (sentinel found) — skipping.{RESET}")
        return "skipped"

    if old not in content:
        if old_missing_ok:
            print(f"{YELLOW}  ○ Target string absent and sentinel missing — file was likely{RESET}")
            print(f"{YELLOW}    patched manually before this script existed. Treating as OK.{RESET}")
            return "skipped"
        print(f"{RED}  ✗ Target string not found — file may have changed upstream.{RESET}")
        print(f"    Looking for:\n    {old!r}")
        return "error"

    new_content = content.replace(old, new, 1)

    if verbose:
        diff = difflib.unified_diff(
            content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"a/{path.name}",
            tofile=f"b/{path.name}",
            n=3,
        )
        print("".join(list(diff)[:60]))

    if dry_run:
        print(f"{YELLOW}  ○ DRY RUN — would apply patch (not writing).{RESET}")
        return "applied"

    # Backup
    backup = path.with_suffix(path.suffix + f".wam-bak.{int(time.time())}")
    shutil.copy2(path, backup)
    print(f"  Backup: {backup.name}")

    path.write_text(new_content, encoding="utf-8")
    print(f"{GREEN}  ✓ Patch applied.{RESET}")
    return "applied"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply WAM patches to mws-dimos and unifolm external repos."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would change without writing.")
    parser.add_argument("--verbose", action="store_true",
                        help="Show unified diff for each patch.")
    args = parser.parse_args()

    print(f"\n{BOLD}{'═'*70}")
    print(f"WAM patch script — mws-dimos & unifolm external code")
    print(f"{'═'*70}{RESET}")
    if args.dry_run:
        print(f"{YELLOW}DRY RUN mode — no files will be written.{RESET}")

    results: dict[str, int] = {"applied": 0, "skipped": 0, "error": 0}

    for patch in PATCHES:
        result = apply_patch(patch, dry_run=args.dry_run, verbose=args.verbose)
        results[result] += 1

    print(f"\n{'═'*70}")
    print(
        f"{BOLD}Summary:{RESET}  "
        f"{GREEN}applied={results['applied']}{RESET}  "
        f"{YELLOW}skipped(already done)={results['skipped']}{RESET}  "
        f"{RED}errors={results['error']}{RESET}"
    )

    if results["error"] > 0:
        print(f"\n{RED}Some patches failed — check output above.{RESET}")
        return 1

    print(f"\n{GREEN}All patches OK.{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
