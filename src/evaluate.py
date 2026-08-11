#!/usr/bin/env python3
"""
evaluate.py — calibrate the decision threshold on VAL, then report final
Dice / IoU / boundary distance on TEST (never touched during calibration),
broken out by swede-score grade, with bootstrap confidence intervals.

Usage:
    python src/evaluate.py --config configs/config.yaml --checkpoint outputs/best_model.pt
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.ndimage import distance_transform_edt
from torch.utils.data import DataLoader

from dataset import ColposcopyDataset
from model import ColposcopyUNet


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Returns lists of (prob_map, gt_mask, case_id, swede_score) for has_mask rows only."""
    model.eval()
    probs_all, gts_all, cases_all = [], [], []
    for batch in loader:
        image = batch["image"].to(device)
        mask = batch["mask"]
        has_mask = batch["has_mask"]
        out = model(image)
        probs = torch.sigmoid(out["seg_logits"]).cpu()
        for i in range(len(has_mask)):
            if has_mask[i]:
                probs_all.append(probs[i, 0].numpy())
                gts_all.append(mask[i, 0].numpy())
                cases_all.append(int(batch["case_id"][i]))
    return probs_all, gts_all, cases_all


def dice_iou(pred: np.ndarray, gt: np.ndarray, eps=1e-6):
    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum()
    dice = (2 * inter + eps) / (union + eps) if union > 0 else 1.0
    iou_union = pred.sum() + gt.sum() - inter
    iou = (inter + eps) / (iou_union + eps) if iou_union > 0 else 1.0
    return dice, iou


def hausdorff_95(pred: np.ndarray, gt: np.ndarray):
    """Symmetric 95th-percentile Hausdorff distance in pixels. Returns None
    if either mask is empty (undefined boundary distance)."""
    if pred.sum() == 0 or gt.sum() == 0:
        return None
    pred_dist = distance_transform_edt(1 - pred)
    gt_dist = distance_transform_edt(1 - gt)
    d_pred_to_gt = pred_dist[gt.astype(bool)]
    d_gt_to_pred = gt_dist[pred.astype(bool)]
    all_d = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    return float(np.percentile(all_d, 95))


def sweep_threshold(probs_list, gts_list, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.1, 0.91, 0.05)
    best_thr, best_dice = 0.5, -1
    results = []
    for thr in thresholds:
        dices = [dice_iou((p >= thr).astype(np.float32), g)[0] for p, g in zip(probs_list, gts_list)]
        mean_dice = float(np.mean(dices))
        results.append((float(thr), mean_dice))
        if mean_dice > best_dice:
            best_dice, best_thr = mean_dice, float(thr)
    return best_thr, best_dice, results


def bootstrap_ci(values, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    values = np.array(values)
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/config.yaml")
    ap.add_argument("--checkpoint", type=str, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    ckpt_path = args.checkpoint or str(Path(cfg["output"]["dir"]) / "best_model.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)

    model = ColposcopyUNet(
        encoder_name=cfg["model"]["encoder"], pretrained=False,
        use_cls_head=cfg["model"]["use_cls_head"], dropout=cfg["model"]["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    manifest = cfg["data"]["manifest_path"]
    image_size = cfg["data"]["image_size"]

    val_ds = ColposcopyDataset(manifest, "val", image_size, seg_only=True)
    test_ds = ColposcopyDataset(manifest, "test", image_size, seg_only=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=2)

    print("Calibrating threshold on VAL...")
    val_probs, val_gts, _ = collect_predictions(model, val_loader, device)
    best_thr, best_val_dice, sweep = sweep_threshold(val_probs, val_gts)
    print(f"  best threshold = {best_thr:.2f}  (val dice = {best_val_dice:.4f})")
    if best_thr <= 0.15:
        print("  WARNING: calibrated threshold is very low — this usually means "
              "predicted probabilities are poorly separated between lesion and "
              "background. Worth checking training curves / trying more epochs "
              "before trusting this threshold.")

    print("Evaluating on TEST at that threshold...")
    test_probs, test_gts, test_cases = collect_predictions(model, test_loader, device)

    import pandas as pd
    manifest_df = pd.read_csv(manifest)
    case_to_score = manifest_df.drop_duplicates("case_id").set_index("case_id")["swede_score"].to_dict()

    dices, ious, hd95s, grades = [], [], [], []
    for p, g, cid in zip(test_probs, test_gts, test_cases):
        pred_bin = (p >= best_thr).astype(np.float32)
        d, iou = dice_iou(pred_bin, g)
        hd = hausdorff_95(pred_bin, g)
        dices.append(d)
        ious.append(iou)
        if hd is not None:
            hd95s.append(hd)
        score = case_to_score.get(cid, -1)
        grades.append("high" if score >= cfg["data"]["swede_threshold"] else "low")

    dice_lo, dice_hi = bootstrap_ci(dices)
    iou_lo, iou_hi = bootstrap_ci(ious)

    grades_arr = np.array(grades)
    dices_arr = np.array(dices)
    per_grade = {g: float(dices_arr[grades_arr == g].mean()) for g in set(grades)
                 if (grades_arr == g).sum() > 0}

    results = {
        "checkpoint": ckpt_path,
        "calibrated_threshold": best_thr,
        "n_test_samples": len(dices),
        "test_dice_mean": float(np.mean(dices)),
        "test_dice_95ci": [dice_lo, dice_hi],
        "test_iou_mean": float(np.mean(ious)),
        "test_iou_95ci": [iou_lo, iou_hi],
        "test_hd95_mean_px": float(np.mean(hd95s)) if hd95s else None,
        "test_hd95_n_valid": len(hd95s),
        "dice_by_grade": per_grade,
        "threshold_sweep": sweep,
    }

    out_path = Path(cfg["output"]["dir"]) / "test_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
