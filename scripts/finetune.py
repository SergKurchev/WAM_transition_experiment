#!/usr/bin/env python3
"""UnifoLM-WMA-0 fine-tuning on WAM collected dataset.

Strategy:
  - FREEZE: VAE (first_stage), CLIP text (cond_stage), DINOSigLIP (embedder)
  - TRAIN:  action head (ConditionalUnet1D), image projection (Resampler),
            state/action projectors, SATokenProjector
  - LOSS:   action prediction MSE via diffusion policy (decision_making_only=True)
  - GPU:    single GPU, no DDP

Usage:
  python /workspace/wam/scripts/finetune.py --dry-run
  python /workspace/wam/scripts/finetune.py --epochs 10 --dataset /workspace/wam/dataset
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

# single-GPU env for PL
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("RANK",       "0")
os.environ.setdefault("WORLD_SIZE", "1")

sys.path.insert(0, "/workspace/wam")
sys.path.insert(0, "/workspace/unifolm/src")

import torch
import numpy as np
from omegaconf import OmegaConf

DATASET_NAME = "g1_pick_place"
BASE_CONFIG  = "/workspace/unifolm/configs/train/config.yaml"
DEFAULT_CKPT = "/workspace/wam/checkpoints/unifolm_wma_dual.ckpt"


# ── config ────────────────────────────────────────────────────────────────────

def build_config(args) -> OmegaConf:
    cfg = OmegaConf.load(BASE_CONFIG)

    # Point to our dataset
    cfg.data.params.train.params.data_dir    = args.dataset
    cfg.data.params.batch_size               = args.batch_size
    cfg.data.params.num_workers              = 2
    cfg.data.params.dataset_and_weights      = {DATASET_NAME: 1.0}

    # Fine-tune settings
    cfg.model.pretrained_checkpoint           = args.checkpoint
    cfg.model.params.decision_making_only     = True
    cfg.model.params.num_epochs               = args.epochs
    cfg.model.params.lr_warmup_steps          = 50
    cfg.model.base_learning_rate              = args.lr
    cfg.model.scale_lr                        = False

    return cfg


# ── freeze backbone ───────────────────────────────────────────────────────────

FREEZE_PREFIXES = [
    "first_stage_model",     # VAE encoder/decoder
    "cond_stage_model",      # CLIP text encoder (already frozen by config)
    "embedder",              # DINOSigLIP visual encoder
    "model.diffusion_model", # main temporal U-Net backbone
]

TRAIN_PREFIXES = [
    "image_proj_model",      # Resampler: visual → cross-attn tokens
    "model.unet_head",       # ConditionalUnet1D: action diffusion head
    "state_projector",       # state → 1024-dim
    "action_projector",      # action → 1024-dim
    "stem_process",          # SATokenProjector
    "agent_state_pos_emb",   # learned positional embedding
    "agent_action_pos_emb",
]


def freeze_backbone(model) -> list:
    """Freeze all, then unfreeze action head. Return trainable params list."""
    for p in model.parameters():
        p.requires_grad = False

    trainable = []
    for name, param in model.named_parameters():
        if any(name.startswith(pfx) for pfx in TRAIN_PREFIXES):
            param.requires_grad = True
            trainable.append(param)

    frozen_M    = sum(p.numel() for p in model.parameters() if not p.requires_grad) / 1e6
    trainable_M = sum(p.numel() for p in trainable) / 1e6

    print(f"[Freeze] frozen    : {frozen_M:.1f} M params", flush=True)
    print(f"[Freeze] trainable : {trainable_M:.1f} M params", flush=True)
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"[Freeze] trainable layers ({len(trainable_names)}):", flush=True)
    for n in trainable_names[:12]:
        print(f"  {n}", flush=True)
    if len(trainable_names) > 12:
        print(f"  ... (+{len(trainable_names)-12} more)", flush=True)

    return trainable


# ── data ──────────────────────────────────────────────────────────────────────

def build_dataloader(cfg, args):
    from unifolm_wma.utils.utils import instantiate_from_config
    from unifolm_wma.utils.data  import DataModuleFromConfig

    data_module: DataModuleFromConfig = instantiate_from_config(cfg.data)
    data_module.setup()

    for name, ds in data_module.train_datasets.items():
        print(f"[Data] {name}: {len(ds)} samples", flush=True)

    loader = data_module._train_dataloader()
    return loader


# ── cosine LR schedule ────────────────────────────────────────────────────────

def cosine_lr(step: int, total_steps: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    return base_lr * 0.5 * (1.0 + np.cos(np.pi * progress))


# ── training loop ─────────────────────────────────────────────────────────────

def train(model, loader, trainable_params, args, report_dir: Path):
    device = next(model.parameters()).device

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        betas=(0.95, 0.999),
        eps=1e-8,
        weight_decay=1e-6,
    )

    total_steps   = args.epochs * len(loader)
    warmup_steps  = min(50, total_steps // 10)
    ckpt_dir      = Path(args.output_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    losses_rows = []
    best_loss   = float("inf")
    global_step = 0

    print(f"\n[Train] epochs={args.epochs}  steps/epoch={len(loader)}  "
          f"total={total_steps}  warmup={warmup_steps}", flush=True)
    print(f"[Train] batch_size={args.batch_size}  lr={args.lr}", flush=True)

    model.train()
    t_start = time.time()

    for epoch in range(args.epochs):
        epoch_losses = []

        for batch_idx, batch in enumerate(loader):
            if args.dry_run and batch_idx >= 2:
                break

            # LR schedule
            lr_now = cosine_lr(global_step, total_steps, warmup_steps, args.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

            # Move tensors to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # Forward
            optimizer.zero_grad()
            try:
                loss, loss_dict = model.shared_step(batch, random_uncond=False)
            except Exception as e:
                print(f"  [step {global_step}] forward error: {e}", flush=True)
                import traceback; traceback.print_exc()
                global_step += 1
                continue

            if not torch.isfinite(loss):
                print(f"  [step {global_step}] non-finite loss={loss.item():.4f}, skip", flush=True)
                global_step += 1
                continue

            # Backward
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            loss_val  = float(loss.item())
            act_loss  = float(loss_dict.get("train/loss_action", loss_dict.get("loss_action", 0.0)))
            vid_loss  = float(loss_dict.get("train/loss",        loss_dict.get("loss",        loss_val)))
            epoch_losses.append(loss_val)

            row = {
                "epoch": epoch + 1,
                "step":  global_step,
                "loss":  round(loss_val, 6),
                "action_loss": round(act_loss, 6),
                "video_loss":  round(vid_loss, 6),
                "lr":    round(lr_now, 8),
                "grad_norm": round(float(grad_norm), 4),
            }
            losses_rows.append(row)

            eta = (time.time() - t_start) / max(1, global_step) * max(0, total_steps - global_step)
            print(
                f"[Train] ep={epoch+1}/{args.epochs}  step={batch_idx}/{len(loader)}  "
                f"loss={loss_val:.4f}  act={act_loss:.4f}  "
                f"lr={lr_now:.2e}  gn={float(grad_norm):.3f}  "
                f"ETA={eta/60:.1f}min",
                flush=True,
            )
            global_step += 1

        if not epoch_losses:
            continue

        epoch_mean = float(np.mean(epoch_losses))
        print(f"\n[Train] ── Epoch {epoch+1} mean loss: {epoch_mean:.4f} ──\n", flush=True)

        # Save ONLY trainable params (~200 MB, not full 16 GB model)
        trainable_sd = {k: v for k, v in model.state_dict().items()
                        if any(k.startswith(pfx) for pfx in TRAIN_PREFIXES)}
        ckpt_path = ckpt_dir / f"epoch_{epoch+1:04d}.ckpt"
        torch.save({"trainable_state_dict": trainable_sd,
                    "train_prefixes": TRAIN_PREFIXES,
                    "epoch": epoch + 1,
                    "loss":  epoch_mean}, str(ckpt_path))

        if epoch_mean < best_loss:
            best_loss = epoch_mean
            best_path = ckpt_dir / "best.ckpt"
            torch.save({"trainable_state_dict": trainable_sd,
                        "train_prefixes": TRAIN_PREFIXES,
                        "epoch": epoch + 1,
                        "loss":  best_loss}, str(best_path))
            print(f"[Train] ✓ Best checkpoint: {best_path}  loss={best_loss:.4f}", flush=True)

    return losses_rows, best_loss


# ── save losses ───────────────────────────────────────────────────────────────

def save_losses(rows: list, report_dir: Path):
    report_dir.mkdir(parents=True, exist_ok=True)

    csv_path = report_dir / "losses.csv"
    if rows:
        with open(str(csv_path), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"[Report] Losses CSV: {csv_path}", flush=True)

    # Loss curve
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps  = [r["step"]  for r in rows]
        losses = [r["loss"]  for r in rows]
        act_ls = [r["action_loss"] for r in rows]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(steps, losses, label="total loss", color="steelblue")
        ax1.set_xlabel("step"); ax1.set_ylabel("loss"); ax1.set_title("Total Loss")
        ax1.grid(True, alpha=0.3); ax1.legend()

        ax2.plot(steps, act_ls, label="action loss", color="darkorange")
        ax2.set_xlabel("step"); ax2.set_ylabel("loss"); ax2.set_title("Action Loss")
        ax2.grid(True, alpha=0.3); ax2.legend()

        plt.tight_layout()
        img_path = report_dir / "loss_curve.png"
        plt.savefig(str(img_path), dpi=100)
        plt.close()
        print(f"[Report] Loss curve: {img_path}", flush=True)
    except Exception as e:
        print(f"[Report] matplotlib not available: {e}", flush=True)
        # Text table fallback
        print("\n[Report] Loss table (every 5th step):", flush=True)
        print(f"{'epoch':>5} {'step':>6} {'loss':>10} {'act_loss':>10} {'lr':>12}", flush=True)
        for r in rows[::max(1, len(rows)//20)]:
            print(f"{r['epoch']:>5} {r['step']:>6} {r['loss']:>10.4f} "
                  f"{r['action_loss']:>10.4f} {r['lr']:>12.2e}", flush=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="UnifoLM fine-tuning (single GPU)")
    parser.add_argument("--dry-run",    action="store_true",
                        help="1 epoch, 2 steps — checks that everything loads")
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch-size", type=int,   default=2)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--dataset",    default="/workspace/wam/dataset")
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--output-dir", default="/workspace/wam/checkpoints/finetune")
    parser.add_argument("--report-dir", default="/workspace/wam/reports")
    args = parser.parse_args()

    if args.dry_run:
        args.epochs = 1
        print("[Finetune] ── DRY RUN: 1 epoch, max 2 steps ──", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Finetune] device={device}", flush=True)

    report_dir = Path(args.report_dir)

    # ── load config + model ───────────────────────────────────────────────────
    print(f"\n[Finetune] Building config …", flush=True)
    cfg = build_config(args)

    print(f"[Finetune] Instantiating model …", flush=True)
    from unifolm_wma.utils.utils import instantiate_from_config
    from unifolm_wma.utils.train  import load_checkpoints

    model = instantiate_from_config(cfg.model)
    model = load_checkpoints(model, cfg.model)

    # re-register ZTSNR schedule after checkpoint load
    if getattr(model, "rescale_betas_zero_snr", False):
        model.register_schedule(
            given_betas=model.given_betas,
            beta_schedule=model.beta_schedule,
            timesteps=model.timesteps,
            linear_start=model.linear_start,
            linear_end=model.linear_end,
            cosine_s=model.cosine_s,
        )

    model = model.to(device)
    model.learning_rate = args.lr

    # ── freeze backbone ───────────────────────────────────────────────────────
    print(f"\n[Finetune] Freezing backbone …", flush=True)
    trainable_params = freeze_backbone(model)

    if not trainable_params:
        print("[Finetune] ERROR: no trainable params found — check TRAIN_PREFIXES", flush=True)
        sys.exit(1)

    # ── data ──────────────────────────────────────────────────────────────────
    print(f"\n[Finetune] Loading dataset from {args.dataset} …", flush=True)
    loader = build_dataloader(cfg, args)
    print(f"[Finetune] Loader: {len(loader)} batches/epoch", flush=True)

    # ── train ─────────────────────────────────────────────────────────────────
    print(f"\n[Finetune] Starting training …\n", flush=True)
    t0 = time.time()
    losses_rows, best_loss = train(model, loader, trainable_params, args, report_dir)
    elapsed = time.time() - t0

    # ── save reports ──────────────────────────────────────────────────────────
    save_losses(losses_rows, report_dir)

    # ── final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}", flush=True)
    print(f"[Finetune] DONE", flush=True)
    print(f"  best loss    : {best_loss:.4f}", flush=True)
    print(f"  total time   : {elapsed/60:.1f} min", flush=True)
    print(f"  checkpoints  : {args.output_dir}", flush=True)
    print(f"  reports      : {args.report_dir}", flush=True)
    if losses_rows:
        first = losses_rows[0]["loss"]
        last  = losses_rows[-1]["loss"]
        delta = first - last
        print(f"  loss ep1→last: {first:.4f} → {last:.4f}  (Δ={delta:+.4f})", flush=True)
    print(f"{'='*65}", flush=True)


if __name__ == "__main__":
    main()
