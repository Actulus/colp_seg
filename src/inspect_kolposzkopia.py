#!/usr/bin/env python3
"""
inspect_kolposzkopia.py — runs the trained model on the Kolposzkopia folder
(21 cases, 65 images, NO masks, NOT used for training or in any metric) and
produces prediction-overlay grids for a human to eyeball.

This is deliberately NOT a benchmark. There's no ground truth here, so
there's no Dice/IoU to report -- what this answers is a softer but still
useful question: "on images that look like my actual target domain
(different camera/screenshot pipeline than AnnoCerv), does the model
produce something a clinician would recognize as reasonable, or does it
fall apart completely?" Free-text histopathology diagnosis (from
Kolposzkopia's own CSV, where available) is shown as context in each
panel's title, purely for your own reading -- it was never used in
training or thresholding.

Usage:
    python src/inspect_kolposzkopia.py --config configs/config.yaml \
        --kolposzkopia_dir atlas_raw/Kolposzkopia --threshold 0.3
"""
import argparse
import csv
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import ColposcopyUNet


def load_diagnosis_map(kolposzkopia_dir: Path) -> dict:
    """Reads 'Kolposzkopia - Sheet1.csv' (case_id,diagnosis), if present.
    Missing/absent file just means blank titles -- not fatal.

    Keys are normalized to uppercase: the CSV and the folder names disagree
    on capitalization for a few cases (CSV has 'X24'/'Vi02'/'Vi07', folders
    are 'x24'/'VI02'/'VI07') -- confirmed by actually checking the real
    files, not assumed. Case-sensitive lookup would silently show a blank
    diagnosis for exactly those 3 cases."""
    csv_path = kolposzkopia_dir / "Kolposzkopia - Sheet1.csv"
    mapping = {}
    if not csv_path.exists():
        return mapping
    with open(csv_path, newline="", encoding="utf-8", errors="ignore") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                mapping[row[0].strip().upper()] = row[1].strip()
    return mapping


def collect_case_images(kolposzkopia_dir: Path) -> list:
    """Returns [(case_id_str, image_path), ...] for every image in every
    case subfolder (excludes the CSV and any .DS_Store-type junk)."""
    items = []
    for case_dir in sorted(p for p in kolposzkopia_dir.iterdir() if p.is_dir()):
        for img_path in sorted(case_dir.glob("*.png")) + sorted(case_dir.glob("*.jpg")):
            items.append((case_dir.name, img_path))
    return items


def build_transform(image_size: int) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def load_and_crop(img_path: Path, crop_ratio: float) -> np.ndarray:
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cv2 could not read {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    nh, nw = int(h * crop_ratio), int(w * crop_ratio)
    top, left = (h - nh) // 2, (w - nw) // 2
    return img[top:top + nh, left:left + nw]


def denormalize(img_tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = (img_tensor * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


@torch.no_grad()
def run_and_plot(model, device, items, diagnosis_map, transform, crop_ratio,
                  threshold, out_dir: Path, per_page: int = 6):
    model.eval()
    n_pages = (len(items) + per_page - 1) // per_page

    for page in range(n_pages):
        chunk = items[page * per_page:(page + 1) * per_page]
        fig, axes = plt.subplots(len(chunk), 2, figsize=(7, 3.2 * len(chunk)))
        if len(chunk) == 1:
            axes = axes[None, :]

        for row, (case_id, img_path) in enumerate(chunk):
            raw = load_and_crop(img_path, crop_ratio)
            transformed = transform(image=raw)["image"].unsqueeze(0).to(device)
            prob = torch.sigmoid(model(transformed)["seg_logits"])[0, 0].cpu().numpy()
            pred = (prob >= threshold).astype(np.float32)

            img_np = denormalize(transformed[0].cpu())
            overlay = img_np.copy()
            overlay[pred > 0.5] = [1, 0, 0]
            blended = (0.6 * img_np + 0.4 * overlay)

            diag = diagnosis_map.get(case_id.upper(), "")
            title = f"{case_id}" + (f" — {diag}" if diag else "")
            pred_frac = float(pred.mean())

            axes[row, 0].imshow(img_np)
            axes[row, 0].set_title(title, fontsize=9)
            axes[row, 0].axis("off")

            axes[row, 1].imshow(blended)
            axes[row, 1].set_title(f"prediction ({pred_frac:.1%} of image)", fontsize=9)
            axes[row, 1].axis("off")

        fig.tight_layout()
        out_path = out_dir / f"kolposzkopia_predictions_page{page + 1}.png"
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/config.yaml")
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--kolposzkopia_dir", type=str, required=True,
                     help="e.g. atlas_raw/Kolposzkopia")
    ap.add_argument("--per_page", type=int, default=6)
    ap.add_argument("--threshold", type=float, default=None,
                     help="override the calibrated threshold from test_results.json "
                          "-- useful for Kolposzkopia specifically, since that "
                          "threshold was calibrated on a very different, "
                          "class-imbalanced eval set and may be far too "
                          "conservative for an exploratory out-of-domain check.")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    ckpt_path = args.checkpoint or str(Path(cfg["output"]["dir"]) / "best_model.pt")

    kolposzkopia_dir = Path(args.kolposzkopia_dir)
    items = collect_case_images(kolposzkopia_dir)
    if not items:
        raise FileNotFoundError(f"No images found under {kolposzkopia_dir} -- "
                                 f"check the path points at the 'Kolposzkopia' "
                                 f"folder itself, not its parent repo.")
    print(f"Found {len(items)} images across "
          f"{len({c for c, _ in items})} Kolposzkopia cases.")

    diagnosis_map = load_diagnosis_map(kolposzkopia_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = ColposcopyUNet(
        encoder_name=cfg["model"]["encoder"], pretrained=False,
        use_cls_head=cfg["model"]["use_cls_head"], dropout=cfg["model"]["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    if args.threshold is not None:
        threshold = args.threshold
        print(f"Using manually-specified threshold: {threshold:.2f}")
    else:
        threshold = 0.5
        test_results_path = Path(cfg["output"]["dir"]) / "test_results.json"
        if test_results_path.exists():
            import json
            threshold = json.loads(test_results_path.read_text())["calibrated_threshold"]
            print(f"Using calibrated threshold from evaluate.py: {threshold:.2f}")
        else:
            print("No test_results.json found -- using default threshold 0.5.")

    transform = build_transform(cfg["data"]["image_size"])
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Running inference (no ground truth -- qualitative only) ...")
    run_and_plot(model, device, items, diagnosis_map, transform,
                 cfg["data"].get("crop_ratio", 0.80),
                 threshold, out_dir, per_page=args.per_page)

    print("\nDone. Look through the pages for: does predicted area roughly "
          "track visible acetowhite/vascular regions? Cases with severe "
          "diagnoses (CIN3, carcinoma) predicting near-zero area is the "
          "same recall problem already seen in k-fold -- expected, not new "
          "news. What's worth flagging is the OPPOSITE: images that look "
          "structurally different from AnnoCerv (blurry, oddly cropped, "
          "visible UI elements) producing wildly implausible predictions "
          "(e.g. >50% of the image) would point to a domain-shift problem "
          "specifically, distinct from the recall problem.")


if __name__ == "__main__":
    main()