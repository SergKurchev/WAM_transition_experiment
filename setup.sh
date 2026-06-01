══ 4/4  Running deploy.sh on server ══
Enter passphrase for key '/home/sergkurchev/.ssh/id_ed25519':
[INFO] Verifying directory structure...
[INFO] ✓ Directory structure correct
[INFO] ✓ mws-dimos found at /root/jdoe/workspace/mws-dimos
[INFO] ✓ mws-dimos on feat/real-transfer
[INFO] Applying WAM patches to external code...

══════════════════════════════════════════════════════════════════════
WAM patch script — mws-dimos & unifolm external code
══════════════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────────────
Patch: Allow arm joint commands from rt/lowcmd in Isaac SUPPORT mode (WAM without GEAR-SONIC WBC)
File:  /root/jdoe/workspace/mws-dimos/sim/isaac/g1_sim.py
  ✗ File not found — skipping.

──────────────────────────────────────────────────────────────────────
Patch: Kinematic robot root: fix_root_link when G1_KINEMATIC_ROBOT=1 (robot follows WAM commands exactly, never falls)
File:  /root/jdoe/workspace/mws-dimos/sim/isaac/g1_sim.py
  ✗ File not found — skipping.

──────────────────────────────────────────────────────────────────────
Patch: Remove xformers assertion that blocks vanilla attention (container has PyTorch 2.7/cu126, xformers requires cu128)
File:  /root/jdoe/workspace/wam-stack/repos/unifolm/src/unifolm_wma/modules/attention.py
  Backup: attention.py.wam-bak.1780320629
  ✓ Patch applied.

══════════════════════════════════════════════════════════════════════
Summary:  applied=1  skipped(already done)=0  errors=2

Some patches failed — check output above.
[ERROR] Patch script failed — check output above. Aborting deploy.

C:\Users\NeverGonnaGiveYouUp\OneDrive\Рабочий стол\study_materials\Skoltech\!INDUSTRIAL_IMMERSION_MWS\test\