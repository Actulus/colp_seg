#!/usr/bin/env python3
"""
train.py — trains ColposcopyUNet on the unified manifest.

LR schedule: linear warmup -> cosine decay, built ONCE with LambdaLR over a
step-based schedule and never mutated afterward. Each param group (encoder
vs decoder/heads) keeps its own base LR ratio for the entire run via the
lambda function itself (not via editing param_groups later), so there is no
"scheduler forgets my manual unfreeze" bug like the previous pipeline had.

Usage:
    python src/train.py --config configs/config.yaml
"""

import argparse
import json
import math
import time
from pathlib import Path

import pandas as pd
import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import ColposcopyDataset
from model import ColposcopyUNet, build_param_groups
from losses import SegmentationLoss, ClassificationLoss, dice_score


def compute_pos_weights(manifest_path: str):
    """pos_weight = n_negative / n_positive, computed from the TRAIN split
    only (never val/test, to avoid any leakage into calibration)."""
    df = pd.read_csv(manifest_path)
    train = df[df["split"] == "train"]

    cls_pos = train["label"].sum()
    cls_neg = len(train) - cls_pos
    cls_pos_weight = float(cls_neg / max(cls_pos, 1))

    masked = train[train["has_mask"] == True]  # noqa: E712
    # pixel-level positive ratio, sampled cheaply from a subset of masks
    import cv2
    import numpy as np

    pos_px, tot_px = 0, 0
    for p in masked["parsed_mask_path"].sample(min(60, len(masked)), random_state=0):
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(
                f"Could not read mask file: {p!r}. This usually means "
                f"prepare_data.py hasn't been (successfully) run in this "
                f"environment yet, or data/manifest.csv is stale (e.g. left "
                f"over from a different machine/session). Re-run "
                f"prepare_data.py and confirm it prints 'Wrote manifest' "
                f"with no errors before training."
            )
        pos_px += (m > 127).sum()
        tot_px += m.size
    seg_pos_weight = float((tot_px - pos_px) / max(pos_px, 1)) if tot_px else 1.0
    # clip — extreme pixel imbalance weights destabilize BCE more than they help
    seg_pos_weight = float(min(seg_pos_weight, 15.0))

    return cls_pos_weight, seg_pos_weight


def make_lr_lambda(warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.01):
    def fn(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(progress, 1.0)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine

    return fn


def run_epoch(
    model,
    loader,
    seg_crit,
    cls_crit,
    optimizer,
    scheduler,
    device,
    train: bool,
    seg_loss_weight: float,
    cls_loss_weight: float,
    use_cls_head: bool,
    scaler=None,
):
    model.train() if train else model.eval()
    total_loss, total_seg_loss, total_cls_loss = 0.0, 0.0, 0.0
    all_dice = []
    n_batches = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            has_mask = batch["has_mask"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            amp_ctx = torch.autocast(
                device_type=device.type, enabled=scaler is not None
            )
            with amp_ctx:
                out = model(image)
                seg_loss = seg_crit(out["seg_logits"], mask, has_mask)
                loss = seg_loss_weight * seg_loss
                if use_cls_head:
                    cls_loss = cls_crit(out["cls_logits"], label)
                    loss = loss + cls_loss_weight * cls_loss
                else:
                    cls_loss = torch.tensor(0.0)

            if train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                scheduler.step()

            with torch.no_grad():
                if has_mask.any():
                    probs = torch.sigmoid(out["seg_logits"][has_mask])
                    d = dice_score(probs, mask[has_mask], threshold=0.5)
                    all_dice.append(d)

            total_loss += loss.item()
            total_seg_loss += seg_loss.item()
            total_cls_loss += float(cls_loss.item()) if use_cls_head else 0.0
            n_batches += 1

    mean_dice = torch.cat(all_dice).mean().item() if all_dice else float("nan")
    return {
        "loss": total_loss / max(n_batches, 1),
        "seg_loss": total_seg_loss / max(n_batches, 1),
        "cls_loss": total_cls_loss / max(n_batches, 1),
        "dice@0.5": mean_dice,
    }


def build_train_sampler(dataset, high_grade_oversample: float = 2.0):
    """Grade-balanced sampler for the train DataLoader.

    Directly targets what the diagnostic script found: real high-grade
    lesions (has_mask=True, label=1) get found only ~26% of the time,
    identical before and after adding Atlas's high-grade-skewed
    CLASSIFICATION data -- because that data never touched the
    segmentation objective. This does: masked_high rows are up-weighted
    to be sampled `high_grade_oversample`x as often as masked_low rows
    (equalized first, then multiplied). has_mask=False rows (Atlas aux,
    iodine/green) are left at their natural frequency -- this targets
    the segmentation recall problem specifically, not classification
    balance (already handled separately via cls_pos_weight).
    """
    df = dataset.df
    has_mask = df["has_mask"].values
    label = df["label"].values
    weights = np.ones(len(df), dtype=np.float64)

    masked_high = has_mask & (label == 1)
    masked_low = has_mask & (label == 0)
    n_high, n_low = masked_high.sum(), masked_low.sum()

    if n_high > 0 and n_low > 0:
        weights[masked_high] = (n_low / n_high) * high_grade_oversample
        weights[masked_low] = 1.0
    elif n_high > 0:
        weights[masked_high] = high_grade_oversample

    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(df),
        replacement=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/config.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    manifest = cfg["data"]["manifest_path"]
    image_size = cfg["data"]["image_size"]

    train_ds = ColposcopyDataset(manifest, "train", image_size, seg_only=False)
    val_ds = ColposcopyDataset(manifest, "val", image_size, seg_only=False)
    print(f"train={len(train_ds)} val={len(val_ds)}")

    train_sampler = build_train_sampler(
        train_ds,
        high_grade_oversample=cfg["training"].get("high_grade_oversample", 2.0),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        sampler=train_sampler,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
    )

    cls_pos_weight, seg_pos_weight = compute_pos_weights(manifest)
    print(
        f"Computed pos_weights from TRAIN split: cls={cls_pos_weight:.2f} seg={seg_pos_weight:.2f}"
    )

    model = ColposcopyUNet(
        encoder_name=cfg["model"]["encoder"],
        pretrained=cfg["model"]["pretrained"],
        use_cls_head=cfg["model"]["use_cls_head"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    param_groups = build_param_groups(
        model,
        base_lr=cfg["training"]["learning_rate"],
        encoder_lr_mult=cfg["training"]["encoder_lr_mult"],
    )
    optimizer = torch.optim.AdamW(
        param_groups, weight_decay=cfg["training"]["weight_decay"]
    )

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * cfg["training"]["epochs"]
    warmup_steps = steps_per_epoch * cfg["training"]["warmup_epochs"]
    lr_lambda = make_lr_lambda(warmup_steps, total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    seg_crit = SegmentationLoss(
        bce_weight=cfg["training"]["seg_bce_weight"],
        dice_weight=cfg["training"]["seg_dice_weight"],
        use_tversky=cfg["training"]["use_tversky"],
        tversky_alpha=cfg["training"]["tversky_alpha"],
        tversky_beta=cfg["training"]["tversky_beta"],
        pos_weight=seg_pos_weight,
    ).to(device)
    cls_crit = ClassificationLoss(pos_weight=cls_pos_weight).to(device)

    scaler = (
        torch.cuda.amp.GradScaler()
        if (device.type == "cuda" and cfg["training"]["mixed_precision"])
        else None
    )

    best_val_dice = -1.0
    epochs_no_improve = 0
    history = []

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        t0 = time.time()
        train_metrics = run_epoch(
            model,
            train_loader,
            seg_crit,
            cls_crit,
            optimizer,
            scheduler,
            device,
            train=True,
            seg_loss_weight=cfg["training"]["seg_loss_weight"],
            cls_loss_weight=cfg["training"]["cls_loss_weight"],
            use_cls_head=cfg["model"]["use_cls_head"],
            scaler=scaler,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            seg_crit,
            cls_crit,
            optimizer,
            scheduler,
            device,
            train=False,
            seg_loss_weight=cfg["training"]["seg_loss_weight"],
            cls_loss_weight=cfg["training"]["cls_loss_weight"],
            use_cls_head=cfg["model"]["use_cls_head"],
            scaler=None,
        )
        dt = time.time() - t0

        row = {
            "epoch": epoch,
            "time_s": round(dt, 1),
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "lr_encoder": optimizer.param_groups[0]["lr"],
            "lr_decoder": optimizer.param_groups[1]["lr"],
        }
        history.append(row)
        print(
            f"[{epoch:03d}/{cfg['training']['epochs']}] "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"train_dice={train_metrics['dice@0.5']:.3f} val_dice={val_metrics['dice@0.5']:.3f} "
            f"lr_enc={row['lr_encoder']:.2e} ({dt:.1f}s)"
        )

        if val_metrics["dice@0.5"] > best_val_dice:
            best_val_dice = val_metrics["dice@0.5"]
            epochs_no_improve = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_dice": best_val_dice,
                    "config": cfg,
                },
                out_dir / "best_model.pt",
            )
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= cfg["training"]["early_stopping_patience"]:
            print(
                f"Early stopping at epoch {epoch} (no val_dice improvement for "
                f"{cfg['training']['early_stopping_patience']} epochs). "
                f"Best val_dice={best_val_dice:.4f}"
            )
            break

    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    (out_dir / "final_summary.json").write_text(
        json.dumps(
            {
                "best_val_dice": best_val_dice,
                "cls_pos_weight": cls_pos_weight,
                "seg_pos_weight": seg_pos_weight,
            },
            indent=2,
        )
    )
    print(
        f"Done. Best val_dice={best_val_dice:.4f}. Checkpoint: {out_dir / 'best_model.pt'}"
    )


if __name__ == "__main__":
    main()
