#!/usr/bin/env python3
"""
train_kfold.py — K-fold cross-validation, because with 100 cases a single
70/15/15 split gives you a test Dice with enormous sampling noise (you've
seen this directly: 0.348 / 0.349 / 0.338 across three runs of essentially
the same setup, and per-grade numbers swinging even more). This trains K
independent models, each with a different 1/K of cases held out as test,
and reports mean +/- std across folds — turning "is this run better?" into
an answerable question instead of a guess from one noisy number.

Deliberately reuses the already-tested building blocks from train.py and
evaluate.py (run_epoch, compute_pos_weights, collect_predictions,
sweep_threshold, hausdorff_95, bootstrap_ci) rather than re-implementing
training/eval logic a second time — less code, and no chance of the two
paths silently drifting apart.

How folds are built: cases are split into K stratified groups (by the
case-level label) with sklearn's StratifiedKFold. For fold i, that group is
TEST; the remaining cases are further split train/val (stratified) so
threshold calibration and early stopping still never touch the fold's test
cases. Each fold gets a temporary manifest CSV (data/manifest_fold{i}.csv)
with a fold-specific `split` column — this reuses ColposcopyDataset
completely unmodified, it just reads a different file.

Usage:
    python src/train_kfold.py --config configs/config.yaml --n_folds 5

Runtime: roughly K times a single training run. At ~40s/epoch with early
stopping around epoch 20-30 (per your logs), budget ~30-45 min/fold on a
Colab T4, so ~2.5-4 hours for 5 folds. Each fold checkpoints independently
to outputs_kfold/fold{i}/, so a Colab disconnect only costs you the folds
that hadn't finished yet — completed folds' results are saved as they go
in kfold_results.csv, re-run and it'll happily continue if you add
resume logic (not included by default: reruns retrain every fold, since
these runs are short enough that resume complexity isn't worth it here).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader

from dataset import ColposcopyDataset
from model import ColposcopyUNet, build_param_groups
from losses import SegmentationLoss, ClassificationLoss
from train import compute_pos_weights, make_lr_lambda, run_epoch
from evaluate import collect_predictions, sweep_threshold, dice_iou, hausdorff_95, bootstrap_ci


def build_fold_manifest(full_df: pd.DataFrame, test_cases: set, val_cases: set,
                         out_path: Path) -> None:
    df = full_df.copy()

    def assign(case_id):
        if case_id in test_cases:
            return "test"
        if case_id in val_cases:
            return "val"
        return "train"

    df["split"] = df["case_id"].apply(assign)
    df.to_csv(out_path, index=False)


def train_one_fold(cfg: dict, manifest_path: str, device: torch.device,
                    fold_out_dir: Path) -> dict:
    """Trains one model on one fold's train/val split. Returns the best
    checkpoint's state and val_dice — mirrors train.py's main() loop
    closely enough to trust, but returns rather than writing final files,
    since the caller (run_kfold) handles per-fold bookkeeping."""
    fold_out_dir.mkdir(parents=True, exist_ok=True)
    image_size = cfg["data"]["image_size"]

    train_ds = ColposcopyDataset(manifest_path, "train", image_size, seg_only=False)
    val_ds = ColposcopyDataset(manifest_path, "val", image_size, seg_only=False)

    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"],
                               shuffle=True, num_workers=cfg["training"]["num_workers"],
                               pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"],
                             shuffle=False, num_workers=cfg["training"]["num_workers"],
                             pin_memory=True)

    cls_pos_weight, seg_pos_weight = compute_pos_weights(manifest_path)

    model = ColposcopyUNet(
        encoder_name=cfg["model"]["encoder"], pretrained=cfg["model"]["pretrained"],
        use_cls_head=cfg["model"]["use_cls_head"], dropout=cfg["model"]["dropout"],
    ).to(device)

    param_groups = build_param_groups(model, base_lr=cfg["training"]["learning_rate"],
                                       encoder_lr_mult=cfg["training"]["encoder_lr_mult"])
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg["training"]["weight_decay"])

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * cfg["training"]["epochs"]
    warmup_steps = steps_per_epoch * cfg["training"]["warmup_epochs"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=make_lr_lambda(warmup_steps, total_steps))

    seg_crit = SegmentationLoss(
        bce_weight=cfg["training"]["seg_bce_weight"], dice_weight=cfg["training"]["seg_dice_weight"],
        use_tversky=cfg["training"]["use_tversky"], tversky_alpha=cfg["training"]["tversky_alpha"],
        tversky_beta=cfg["training"]["tversky_beta"], pos_weight=seg_pos_weight,
    ).to(device)
    cls_crit = ClassificationLoss(pos_weight=cls_pos_weight).to(device)

    scaler = torch.cuda.amp.GradScaler() if (device.type == "cuda" and cfg["training"]["mixed_precision"]) else None

    best_val_dice, epochs_no_improve, best_state = -1.0, 0, None
    history = []

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        t0 = time.time()
        train_m = run_epoch(model, train_loader, seg_crit, cls_crit, optimizer, scheduler,
                             device, train=True, seg_loss_weight=cfg["training"]["seg_loss_weight"],
                             cls_loss_weight=cfg["training"]["cls_loss_weight"],
                             use_cls_head=cfg["model"]["use_cls_head"], scaler=scaler)
        val_m = run_epoch(model, val_loader, seg_crit, cls_crit, optimizer, scheduler,
                           device, train=False, seg_loss_weight=cfg["training"]["seg_loss_weight"],
                           cls_loss_weight=cfg["training"]["cls_loss_weight"],
                           use_cls_head=cfg["model"]["use_cls_head"], scaler=None)
        dt = time.time() - t0
        history.append({"epoch": epoch, "time_s": round(dt, 1),
                         **{f"train_{k}": v for k, v in train_m.items()},
                         **{f"val_{k}": v for k, v in val_m.items()}})
        print(f"    [{epoch:03d}/{cfg['training']['epochs']}] "
              f"train_dice={train_m['dice@0.5']:.3f} val_dice={val_m['dice@0.5']:.3f} ({dt:.1f}s)")

        if val_m["dice@0.5"] > best_val_dice:
            best_val_dice = val_m["dice@0.5"]
            epochs_no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= cfg["training"]["early_stopping_patience"]:
            print(f"    early stopping at epoch {epoch}, best_val_dice={best_val_dice:.4f}")
            break

    pd.DataFrame(history).to_csv(fold_out_dir / "training_history.csv", index=False)
    torch.save({"model_state": best_state, "val_dice": best_val_dice}, fold_out_dir / "best_model.pt")
    return {"model_state": best_state, "best_val_dice": best_val_dice}


def evaluate_fold(cfg: dict, manifest_path: str, model_state: dict, device: torch.device) -> dict:
    """Threshold-calibrate on this fold's val, report on this fold's test.
    Mirrors evaluate.py's main() logic."""
    image_size = cfg["data"]["image_size"]
    model = ColposcopyUNet(
        encoder_name=cfg["model"]["encoder"], pretrained=False,
        use_cls_head=cfg["model"]["use_cls_head"], dropout=cfg["model"]["dropout"],
    ).to(device)
    model.load_state_dict(model_state)

    val_ds = ColposcopyDataset(manifest_path, "val", image_size, seg_only=True)
    test_ds = ColposcopyDataset(manifest_path, "test", image_size, seg_only=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=2)

    val_probs, val_gts, _ = collect_predictions(model, val_loader, device)
    best_thr, best_val_dice, _ = sweep_threshold(val_probs, val_gts)

    test_probs, test_gts, test_cases = collect_predictions(model, test_loader, device)

    manifest_df = pd.read_csv(manifest_path)
    case_to_score = manifest_df.drop_duplicates("case_id").set_index("case_id")["swede_score"].to_dict()

    dices, ious, hd95s, grades = [], [], [], []
    for p, g, cid in zip(test_probs, test_gts, test_cases):
        pred_bin = (p >= best_thr).astype(np.float32)
        d, iou = dice_iou(pred_bin, g)
        hd = hausdorff_95(pred_bin, g)
        dices.append(d); ious.append(iou)
        if hd is not None:
            hd95s.append(hd)
        score = case_to_score.get(cid, -1)
        grades.append("high" if score >= cfg["data"]["swede_threshold"] else "low")

    grades_arr, dices_arr = np.array(grades), np.array(dices)
    per_grade = {g: float(dices_arr[grades_arr == g].mean()) for g in set(grades)
                 if (grades_arr == g).sum() > 0}

    return {
        "calibrated_threshold": best_thr,
        "n_test_samples": len(dices),
        "test_dice_mean": float(np.mean(dices)),
        "test_iou_mean": float(np.mean(ious)),
        "test_hd95_mean_px": float(np.mean(hd95s)) if hd95s else None,
        "test_hd95_n_valid": len(hd95s),
        "dice_by_grade": per_grade,
    }


def run_kfold(cfg: dict, n_folds: int, seed: int = 42):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    manifest_path = cfg["data"]["manifest_path"]
    full_df = pd.read_csv(manifest_path)
    cases = full_df[["case_id", "label"]].drop_duplicates().reset_index(drop=True)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_indices = list(skf.split(cases["case_id"], cases["label"]))

    kfold_out_dir = Path(cfg["output"]["dir"]).parent / "outputs_kfold"
    kfold_out_dir.mkdir(parents=True, exist_ok=True)

    fold_results = []
    for fold_i, (trainval_idx, test_idx) in enumerate(fold_indices):
        print(f"\n=== Fold {fold_i + 1}/{n_folds} ===")
        test_cases = set(cases.iloc[test_idx]["case_id"])
        trainval_cases = cases.iloc[trainval_idx]

        # further split remaining cases into train/val, stratified, seeded per-fold
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed + fold_i)
        train_i, val_i = next(sss.split(trainval_cases["case_id"], trainval_cases["label"]))
        val_cases = set(trainval_cases.iloc[val_i]["case_id"])

        fold_manifest_path = Path(manifest_path).parent / f"manifest_fold{fold_i}.csv"
        build_fold_manifest(full_df, test_cases, val_cases, fold_manifest_path)

        n_train = len(cases) - len(test_cases) - len(val_cases)
        print(f"  train={n_train} cases, val={len(val_cases)} cases, test={len(test_cases)} cases")

        fold_dir = kfold_out_dir / f"fold{fold_i}"
        train_result = train_one_fold(cfg, str(fold_manifest_path), device, fold_dir)
        eval_result = evaluate_fold(cfg, str(fold_manifest_path), train_result["model_state"], device)

        row = {"fold": fold_i, "best_val_dice": train_result["best_val_dice"], **eval_result}
        row["dice_by_grade_low"] = row["dice_by_grade"].get("low")
        row["dice_by_grade_high"] = row["dice_by_grade"].get("high")
        del row["dice_by_grade"]
        fold_results.append(row)
        print(f"  fold {fold_i} test_dice={row['test_dice_mean']:.4f} "
              f"(low={row['dice_by_grade_low']}, high={row['dice_by_grade_high']})")

        (fold_dir / "fold_results.json").write_text(json.dumps(row, indent=2))
        pd.DataFrame(fold_results).to_csv(kfold_out_dir / "kfold_results.csv", index=False)

    results_df = pd.DataFrame(fold_results)
    summary = {
        "n_folds": n_folds,
        "test_dice_mean": float(results_df["test_dice_mean"].mean()),
        "test_dice_std": float(results_df["test_dice_mean"].std()),
        "test_iou_mean": float(results_df["test_iou_mean"].mean()),
        "test_iou_std": float(results_df["test_iou_mean"].std()),
        "dice_by_grade_low_mean": float(results_df["dice_by_grade_low"].mean()),
        "dice_by_grade_low_std": float(results_df["dice_by_grade_low"].std()),
        "dice_by_grade_high_mean": float(results_df["dice_by_grade_high"].mean()),
        "dice_by_grade_high_std": float(results_df["dice_by_grade_high"].std()),
        "per_fold": fold_results,
    }
    (kfold_out_dir / "kfold_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== K-fold summary ===")
    print(f"test_dice: {summary['test_dice_mean']:.3f} +/- {summary['test_dice_std']:.3f}")
    print(f"dice_by_grade low:  {summary['dice_by_grade_low_mean']:.3f} +/- {summary['dice_by_grade_low_std']:.3f}")
    print(f"dice_by_grade high: {summary['dice_by_grade_high_mean']:.3f} +/- {summary['dice_by_grade_high_std']:.3f}")
    print(f"\nWrote {kfold_out_dir / 'kfold_summary.json'} and kfold_results.csv")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/config.yaml")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    run_kfold(cfg, args.n_folds, args.seed)


if __name__ == "__main__":
    main()