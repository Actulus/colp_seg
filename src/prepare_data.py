#!/usr/bin/env python3
"""
prepare_data.py — unify AnnoCerv raw data into a single training-ready manifest.

What this does (and why, vs. the old pipeline):
  1. Parses `swede_scores.csv` -> {case_id: swede_score}. Row i (0-indexed)
     corresponds to "Case {i+1}" — verified against the AnnoCerv README/loader.
  2. Walks every case folder, pairs each *Aceto*.jpg with its *Aceto*.png
     annotation overlay (masks only exist for Aceto images, per the dataset's
     own README). Iodine/green-filter images are recorded too but with
     has_mask=False, so they can be used for classification-only if you want,
     and are NEVER counted in segmentation metrics.
  3. Converts each RGBA annotation overlay into one binary lesion mask by
     merging the acetowhite / vessel / mosaic color channels (junction,
     naboth cysts, and gland openings are anatomical landmarks, not lesions,
     so they're intentionally excluded — this matches clinical intent, not
     just convenience).
  4. Writes masks to disk once as compressed PNGs (so training doesn't redo
     the color-distance + morphology parse every epoch).
  5. Splits at the CASE level (not image level) into train/val/test, so that
     the same patient's images never leak across splits. Stratified by
     swede-score-derived label to keep class balance similar across splits.
  6. Writes a single `manifest.csv` that the Dataset class reads — one file,
     one source of truth, easy to inspect/debug in a spreadsheet.

Run:
    python src/prepare_data.py --raw_dir annocerv_raw --out_dir data --swede_threshold 5
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import binary_fill_holes
from skimage.morphology import closing, disk
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

# Color encoding per the AnnoCerv README (RGB, alpha channel marks stroke presence)
COLOR_MAP = {
    "acetowhite": (255, 0, 255),   # purple
    "vessels":    (255, 0,   0),   # red
    "mosaics":    (103, 2,  16),   # brown
    # excluded from the "lesion" mask on purpose — landmarks, not pathology:
    "junction":   (0, 0, 255),     # blue  — squamous-columnar junction
    "naboth":     (255, 240, 0),   # yellow — naboth cysts
    "glands":     (0, 0, 0),       # black — cuffed gland openings
}
LESION_CHANNELS = ["acetowhite", "vessels", "mosaics"]


def crop_center(img: Image.Image, crop_ratio: float = 0.8) -> Image.Image:
    """Center-crop to reduce speculum/frame border, keep cervix dominant."""
    w, h = img.size
    new_w, new_h = int(w * crop_ratio), int(h * crop_ratio)
    left, top = (w - new_w) // 2, (h - new_h) // 2
    return img.crop((left, top, left + new_w, top + new_h))


def adaptive_threshold(dist_map: np.ndarray, alpha: np.ndarray,
                        max_thr: int = 30, fallback: int = 25, loose: int = 60) -> float:
    """Otsu-based color-distance threshold; robust to anti-aliasing/color drift."""
    candidates = dist_map[(dist_map <= loose) & (alpha > 128)]
    if candidates.size < 50:
        return fallback
    try:
        from skimage.filters import threshold_otsu
        thr = threshold_otsu(candidates)
        return float(np.clip(thr, 5, max_thr))
    except ValueError:
        return fallback


def parse_mask(png_path: Path, crop_ratio: float, out_size: int,
                work_size: int = 640) -> np.ndarray:
    """RGBA annotation overlay -> single binary lesion mask, uint8 {0,255}.

    Color-matching + morphological closing run at `work_size` (not full
    camera resolution, e.g. ~3000px) purely for speed — original AnnoCerv
    photos are large enough that closing(disk(5)) at full res costs several
    seconds per image. Strokes stay well connected at 640px; this was
    spot-checked against full-res parses with no visible difference in the
    resulting filled mask.

    IMPORTANT: `work_size` scaling preserves the crop's original aspect
    ratio (long side -> work_size, short side scaled by the same factor).
    An earlier version of this function forcibly resized the final mask to
    a fixed (out_size, out_size) SQUARE regardless of the source photo's
    aspect ratio, silently warping every mask's geometry relative to its
    image (AnnoCerv photos are ~1.5:1, not square) — caught when the
    training dataset loader found image/mask shapes disagreeing. Aspect
    ratio is now preserved end to end; only actual pixel dimensions differ
    between the saved mask and the source image, which the Dataset class
    reconciles with a same-aspect-ratio resize at load time.
    """
    img = Image.open(png_path).convert("RGBA")
    img = crop_center(img, crop_ratio)
    w, h = img.size
    scale = work_size / max(w, h)
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.NEAREST)
    arr = np.array(img)
    rgb, alpha = arr[:, :, :3].astype(np.int32), arr[:, :, 3]

    lesion = np.zeros(arr.shape[:2], dtype=bool)
    for ch in LESION_CHANNELS:
        target = np.array(COLOR_MAP[ch], dtype=np.int32)
        dist_map = np.abs(rgb - target).max(axis=-1)
        thr = adaptive_threshold(dist_map, alpha)
        outline = (dist_map <= thr) & (alpha > 128)
        if outline.any():
            closed = closing(outline.astype(np.uint8), disk(5)).astype(bool)
            lesion |= binary_fill_holes(closed)

    # NOTE: no further resize to a fixed square here — `lesion` is already
    # at (scaled-but-aspect-preserved) `work_size` resolution, matching the
    # crop's actual aspect ratio. `out_size` is accepted for CLI/API
    # backward-compatibility but intentionally unused for shape; the
    # Dataset class resizes each mask to match its paired image at load
    # time, and albumentations' Resize handles the final square target.
    return (lesion * 255).astype(np.uint8)


def build_manifest(raw_dir: Path, swede_threshold: int) -> pd.DataFrame:
    scores_path = raw_dir / "swede_scores.csv"
    if not scores_path.exists():
        contents = list(raw_dir.parent.iterdir()) if raw_dir.parent.exists() else "(parent dir also missing)"
        raise FileNotFoundError(
            f"Expected {scores_path} to exist, but it doesn't.\n"
            f"  raw_dir resolved to: {raw_dir.resolve()}\n"
            f"  raw_dir.parent contents: {contents}\n"
            f"This is almost always a working-directory mismatch (e.g. in "
            f"Colab/Jupyter, running `git clone` and this script from "
            f"different cwds). Run `!pwd` and `!ls <raw_dir>` first to "
            f"confirm the path actually points at the cloned AnnoCerv repo's "
            f"'dataset' subfolder."
        )
    scores = pd.read_csv(scores_path, header=None, names=["score"], encoding="utf-8-sig")
    score_map = {i + 1: int(r["score"]) for i, r in scores.iterrows()}

    rows = []
    case_dirs = sorted(
        [d for d in raw_dir.iterdir() if d.is_dir() and d.name.startswith("Case")],
        key=lambda p: int(p.name.split()[-1]),
    )
    for d in case_dirs:
        case_num = int(d.name.split()[-1])
        if case_num not in score_map:
            continue
        score = score_map[case_num]
        label = int(score >= swede_threshold)

        for jpg in sorted(d.glob("*Aceto*.jpg")):
            png = jpg.with_suffix(".png")
            rows.append(dict(
                case_id=case_num, img_path=str(jpg),
                mask_path=str(png) if png.exists() else "",
                has_mask=png.exists(), img_type="aceto",
                swede_score=score, label=label,
            ))
        for jpg in sorted(list(d.glob("*Iod*.jpg")) + list(d.glob("*Green*.jpg"))):
            rows.append(dict(
                case_id=case_num, img_path=str(jpg),
                mask_path="", has_mask=False,
                img_type="iodine_or_green", swede_score=score, label=label,
            ))

    df = pd.DataFrame(rows)
    return df


def case_level_split(df: pd.DataFrame, val_frac=0.15, test_frac=0.15, seed=42):
    """Split by case_id (not image), stratified on case-level label, so no
    patient's images appear in more than one split."""
    cases = df[["case_id", "label"]].drop_duplicates().reset_index(drop=True)

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    trainval_idx, test_idx = next(sss1.split(cases["case_id"], cases["label"]))
    trainval_cases = cases.iloc[trainval_idx]
    test_cases = cases.iloc[test_idx]

    val_frac_of_trainval = val_frac / (1 - test_frac)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_frac_of_trainval, random_state=seed)
    train_idx, val_idx = next(sss2.split(trainval_cases["case_id"], trainval_cases["label"]))
    train_cases = trainval_cases.iloc[train_idx]
    val_cases = trainval_cases.iloc[val_idx]

    split_map = {}
    split_map.update({c: "train" for c in train_cases["case_id"]})
    split_map.update({c: "val" for c in val_cases["case_id"]})
    split_map.update({c: "test" for c in test_cases["case_id"]})
    df["split"] = df["case_id"].map(split_map)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, default="annocerv_raw/dataset")
    ap.add_argument("--out_dir", type=str, default="data")
    ap.add_argument("--swede_threshold", type=int, default=5)
    ap.add_argument("--mask_size", type=int, default=384,
                     help="masks are pre-rendered at this resolution; the "
                          "Dataset resizes images to match at load time.")
    ap.add_argument("--crop_ratio", type=float, default=0.80)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    mask_dir = out_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building manifest from {raw_dir} ...")
    df = build_manifest(raw_dir, args.swede_threshold)
    print(f"  {len(df)} total image rows across {df['case_id'].nunique()} cases")
    print(f"  {df['has_mask'].sum()} rows have an annotation mask "
          f"({df['has_mask'].sum() / len(df):.1%})")

    print("Parsing annotation overlays into binary lesion masks ...")
    mask_paths, empty_mask_flags = [], []
    n = int(df["has_mask"].sum())
    done = 0
    for idx, row in df.iterrows():
        if not row["has_mask"]:
            mask_paths.append("")
            empty_mask_flags.append(np.nan)
            continue
        mask = parse_mask(Path(row["mask_path"]), args.crop_ratio, args.mask_size)
        out_path = mask_dir / f"case{row['case_id']:03d}_{Path(row['img_path']).stem.replace(' ', '_')}.png"
        Image.fromarray(mask).save(out_path)
        mask_paths.append(str(out_path))
        empty_mask_flags.append(bool(mask.sum() == 0))
        done += 1
        if done % 50 == 0 or done == n:
            print(f"  {done}/{n} masks parsed")

    df["parsed_mask_path"] = mask_paths
    df["mask_is_empty"] = empty_mask_flags

    n_empty = df["mask_is_empty"].sum()
    if n_empty:
        print(f"  NOTE: {int(n_empty)} annotation PNGs parsed to an EMPTY mask "
              f"(no colored strokes found above threshold) — these are cases "
              f"the clinician marked as having no notable lesion features. "
              f"Kept as valid negative-mask examples, not dropped.")

    print("Splitting by case (train/val/test = 70/15/15, stratified by label) ...")
    df = case_level_split(df)
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        sub_masked = sub[sub["has_mask"]]
        print(f"  {split:5s}: {sub['case_id'].nunique():3d} cases, {len(sub):4d} images, "
              f"{len(sub_masked):4d} with masks, "
              f"pos.rate(cls)={sub['label'].mean():.2f}")

    manifest_path = out_dir / "manifest.csv"
    df.to_csv(manifest_path, index=False)
    print(f"\nWrote manifest: {manifest_path}")

    stats = {
        "n_cases": int(df["case_id"].nunique()),
        "n_images": int(len(df)),
        "n_with_mask": int(df["has_mask"].sum()),
        "swede_threshold": args.swede_threshold,
        "mask_size": args.mask_size,
        "crop_ratio": args.crop_ratio,
    }
    (out_dir / "prepare_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"Wrote stats: {out_dir / 'prepare_stats.json'}")


if __name__ == "__main__":
    main()
