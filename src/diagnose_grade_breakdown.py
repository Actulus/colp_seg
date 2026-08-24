#!/usr/bin/env python3
"""
diagnose_grade_breakdown.py — the near-zero high-grade Dice from k-fold
could mean two very different things:

  (a) the model finds nothing on high-grade images that DO have a real
      lesion mask (a recall/miss problem -> fix: oversample high-grade
      cases, recall-weighted loss)
  (b) the model over-predicts on high-grade images that have an EMPTY
      mask (a precision/false-positive problem -> fix: different lever
      entirely, possibly unrelated to grade at all)

Aggregate Dice can't tell these apart because both drag the average down
the same way. This script splits every fold's test set into four groups —
{low, high} grade x {has a real lesion mask, empty mask} — and reports
Dice and a simple hit/miss rate for each, then aggregates across folds.

Reads directly from what train_kfold.py already produced:
  - data/manifest_fold{i}.csv       (per-fold splits, written by train_kfold.py)
  - outputs_kfold/fold{i}/best_model.pt
  - outputs_kfold/kfold_results.csv (for each fold's calibrated_threshold)
No retraining, no GPU time beyond a forward pass over each fold's test set.

Usage:
    python src/diagnose_grade_breakdown.py --config configs/config.yaml
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
import time
from torch.utils.data import DataLoader

from dataset import ColposcopyDataset
from model import ColposcopyUNet
from evaluate import collect_predictions, dice_iou


def retry_on_io_error(fn, *args, max_attempts=4, base_delay=2.0, **kwargs):
    """Colab's Drive mount is a network FUSE filesystem and occasionally
    drops mid-read (ConnectionAbortedError, OSError) under repeated I/O --
    not a code bug, just Drive being Drive. Retries with backoff rather
    than losing an otherwise-successful multi-fold run to one flaky read."""
    for attempt in range(1, max_attempts + 1):
        try:
            return fn(*args, **kwargs)
        except (ConnectionAbortedError, ConnectionResetError, OSError) as e:
            if attempt == max_attempts:
                raise
            wait = base_delay * (2 ** (attempt - 1))
            print(
                f"    I/O error ({e!r}), retrying in {wait:.0f}s "
                f"(attempt {attempt}/{max_attempts})..."
            )
            time.sleep(wait)


def classify_sample(prob: np.ndarray, gt: np.ndarray, threshold: float):
    gt_empty = gt.sum() == 0
    pred_bin = (prob >= threshold).astype(np.float32)
    pred_empty = pred_bin.sum() == 0
    dice, _ = dice_iou(pred_bin, gt)

    if gt_empty:
        # "hit" = correctly predicted nothing; "miss" = false positive on a clean image
        hit = pred_empty
    else:
        # "hit" = found at least some real overlap; "miss" = missed it entirely
        hit = not pred_empty and (pred_bin * gt).sum() > 0

    return {
        "gt_empty": bool(gt_empty),
        "pred_empty": bool(pred_empty),
        "dice": float(dice),
        "hit": bool(hit),
    }


def run_fold_diagnostics(
    cfg: dict,
    fold_i: int,
    manifest_path: Path,
    checkpoint_path: Path,
    threshold: float,
    device: torch.device,
) -> pd.DataFrame:
    model = ColposcopyUNet(
        encoder_name=cfg["model"]["encoder"],
        pretrained=False,
        use_cls_head=cfg["model"]["use_cls_head"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    test_ds = ColposcopyDataset(
        str(manifest_path), "test", cfg["data"]["image_size"], seg_only=True
    )
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=2)
    probs, gts, case_ids = collect_predictions(model, test_loader, device)

    manifest_df = pd.read_csv(manifest_path)
    case_to_score = (
        manifest_df.drop_duplicates("case_id")
        .set_index("case_id")["swede_score"]
        .to_dict()
    )

    rows = []
    for prob, gt, cid in zip(probs, gts, case_ids):
        result = classify_sample(prob, gt, threshold)
        score = case_to_score.get(cid, -1)
        grade = "high" if score >= cfg["data"]["swede_threshold"] else "low"
        rows.append({"fold": fold_i, "case_id": cid, "grade": grade, **result})

    return pd.DataFrame(rows)


def summarize(all_rows: pd.DataFrame) -> pd.DataFrame:
    all_rows["gt_status"] = np.where(all_rows["gt_empty"], "empty_mask", "has_lesion")

    # per-fold group means first, then mean+-std ACROSS folds (not pooled
    # over all samples) -- pooling would let folds with more test samples
    # dominate, same reasoning as the main k-fold summary.
    per_fold = (
        all_rows.groupby(["fold", "grade", "gt_status"])
        .agg(n=("dice", "size"), dice_mean=("dice", "mean"), hit_rate=("hit", "mean"))
        .reset_index()
    )

    summary = (
        per_fold.groupby(["grade", "gt_status"])
        .agg(
            n_folds=("fold", "nunique"),
            total_n=("n", "sum"),
            dice_mean=("dice_mean", "mean"),
            dice_std=("dice_mean", "std"),
            hit_rate_mean=("hit_rate", "mean"),
            hit_rate_std=("hit_rate", "std"),
        )
        .reset_index()
    )
    return summary, per_fold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/config.yaml")
    ap.add_argument(
        "--kfold_dir",
        type=str,
        default=None,
        help="defaults to <output.dir's parent>/outputs_kfold",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    kfold_dir = (
        Path(args.kfold_dir)
        if args.kfold_dir
        else Path(cfg["output"]["dir"]).parent / "outputs_kfold"
    )

    results_csv = kfold_dir / "kfold_results.csv"
    if not results_csv.exists():
        raise FileNotFoundError(
            f"{results_csv} not found. Run train_kfold.py first -- this script "
            f"only inspects an existing k-fold run, it doesn't train anything."
        )
    fold_thresholds = (
        pd.read_csv(results_csv).set_index("fold")["calibrated_threshold"].to_dict()
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    manifest_dir = Path(cfg["data"]["manifest_path"]).parent
    all_rows = []
    for fold_i, threshold in sorted(fold_thresholds.items()):
        manifest_path = manifest_dir / f"manifest_fold{fold_i}.csv"
        checkpoint_path = kfold_dir / f"fold{fold_i}" / "best_model.pt"
        if not manifest_path.exists() or not checkpoint_path.exists():
            print(
                f"  fold {fold_i}: missing {manifest_path} or {checkpoint_path}, skipping"
            )
            continue
        print(f"  fold {fold_i}: evaluating at threshold={threshold:.2f} ...")
        df = retry_on_io_error(
            run_fold_diagnostics,
            cfg,
            fold_i,
            manifest_path,
            checkpoint_path,
            threshold,
            device,
        )
        all_rows.append(df)

    if not all_rows:
        raise RuntimeError(
            "No folds could be evaluated -- check kfold_dir and manifest_fold*.csv paths."
        )

    all_rows = pd.concat(all_rows, ignore_index=True)
    summary, per_fold = summarize(all_rows)

    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print("\n=== Per-fold breakdown ===")
    print(per_fold.to_string(index=False))

    print("\n=== Aggregated across folds (mean +/- std) ===")
    print(summary.to_string(index=False))

    print("\nReading this table:")
    print(
        "  grade=high, gt_status=has_lesion, hit_rate  -> fraction of REAL high-grade"
    )
    print(
        "    lesions the model found at least partially. Low here = missing real lesions."
    )
    print("  grade=high, gt_status=empty_mask, hit_rate  -> fraction of lesion-free")
    print(
        "    high-grade images correctly left blank. Low here = false positives, not misses."
    )

    out_dir = kfold_dir
    all_rows.to_csv(out_dir / "diagnostic_per_sample.csv", index=False)
    summary.to_csv(out_dir / "diagnostic_summary.csv", index=False)
    per_fold.to_csv(out_dir / "diagnostic_per_fold.csv", index=False)
    print(
        f"\nWrote diagnostic_summary.csv, diagnostic_per_fold.csv, "
        f"diagnostic_per_sample.csv to {out_dir}"
    )


if __name__ == "__main__":
    main()
