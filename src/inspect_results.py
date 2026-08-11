#!/usr/bin/env python3
"""
inspect_results.py — the "did this actually work" script.

Numbers alone (Dice/IoU) can look fine while the model is doing something
silly (e.g. always predicting a blob in the same spot). This produces two
things worth looking at directly:

  1. outputs/training_curves.png  — train/val loss and Dice per epoch.
     Look for: val_dice actually climbing (not flat/near-zero), and
     train/val loss diverging a lot late (overfitting) vs. tracking
     together (fine, or underfitting if both stay high).
  2. outputs/qualitative_predictions.png — a grid of (image | ground truth
     | prediction) for a handful of TEST cases, with per-sample Dice
     printed under each row. This is the fastest way to catch a model
     that's technically hitting a Dice score by, say, only ever predicting
     empty masks (which scores well against the 85 genuinely-empty cases
     but is useless).

Usage:
    python src/inspect_results.py --config configs/config.yaml --n_samples 8
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from dataset import ColposcopyDataset
from model import ColposcopyUNet


def plot_training_curves(history_csv: Path, out_path: Path):
    df = pd.read_csv(history_csv)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(df["epoch"], df["train_loss"], label="train")
    axes[0].plot(df["epoch"], df["val_loss"], label="val")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(df["epoch"], df["train_dice@0.5"], label="train")
    axes[1].plot(df["epoch"], df["val_dice@0.5"], label="val")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("Dice @ 0.5")
    axes[1].set_title("Segmentation Dice"); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[1].set_ylim(0, 1)

    best_epoch = df.loc[df["val_dice@0.5"].idxmax(), "epoch"]
    best_dice = df["val_dice@0.5"].max()
    fig.suptitle(f"Best val Dice = {best_dice:.3f} at epoch {int(best_epoch)}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"Wrote {out_path}  (best val_dice={best_dice:.3f} @ epoch {int(best_epoch)})")


def denormalize(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = (img_tensor * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def dice_np(pred, gt, eps=1e-6):
    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum()
    return (2 * inter + eps) / (union + eps) if union > 0 else 1.0


@torch.no_grad()
def plot_qualitative(model, ds, device, threshold, n_samples, out_path, seed=0):
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(ds), size=min(n_samples, len(ds)), replace=False)

    fig, axes = plt.subplots(len(idxs), 3, figsize=(9, 3 * len(idxs)))
    if len(idxs) == 1:
        axes = axes[None, :]

    model.eval()
    for row, idx in enumerate(idxs):
        sample = ds[idx]
        image = sample["image"].unsqueeze(0).to(device)
        gt = sample["mask"][0].numpy()

        out = model(image)
        prob = torch.sigmoid(out["seg_logits"])[0, 0].cpu().numpy()
        pred = (prob >= threshold).astype(np.float32)
        d = dice_np(pred, gt)

        img_np = denormalize(sample["image"])

        axes[row, 0].imshow(img_np); axes[row, 0].set_title(f"case {sample['case_id']}"); axes[row, 0].axis("off")

        gt_overlay = img_np.copy()
        gt_overlay[gt > 0.5] = [0, 1, 0]
        axes[row, 1].imshow(0.6 * img_np + 0.4 * gt_overlay); axes[row, 1].set_title("ground truth"); axes[row, 1].axis("off")

        pred_overlay = img_np.copy()
        pred_overlay[pred > 0.5] = [1, 0, 0]
        axes[row, 2].imshow(0.6 * img_np + 0.4 * pred_overlay)
        axes[row, 2].set_title(f"prediction (Dice={d:.2f})"); axes[row, 2].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/config.yaml")
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(cfg["output"]["dir"])
    ckpt_path = args.checkpoint or str(out_dir / "best_model.pt")

    history_csv = out_dir / "training_history.csv"
    if history_csv.exists():
        plot_training_curves(history_csv, out_dir / "training_curves.png")
    else:
        print(f"No {history_csv} found yet — run train.py first to see curves.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = ColposcopyUNet(
        encoder_name=cfg["model"]["encoder"], pretrained=False,
        use_cls_head=cfg["model"]["use_cls_head"], dropout=cfg["model"]["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch')}, val_dice={ckpt.get('val_dice'):.4f}")

    test_results_path = out_dir / "test_results.json"
    threshold = 0.5
    if test_results_path.exists():
        import json
        threshold = json.loads(test_results_path.read_text())["calibrated_threshold"]
        print(f"Using calibrated threshold from evaluate.py: {threshold:.2f}")
    else:
        print("No test_results.json found — using default threshold 0.5. "
              "Run evaluate.py first to get a calibrated threshold.")

    ds = ColposcopyDataset(cfg["data"]["manifest_path"], args.split,
                            cfg["data"]["image_size"], seg_only=True)
    plot_qualitative(model, ds, device, threshold, args.n_samples,
                      out_dir / f"qualitative_predictions_{args.split}.png")


if __name__ == "__main__":
    main()
