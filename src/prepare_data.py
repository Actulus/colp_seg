#!/usr/bin/env python3
"""
prepare_data.py — unify AnnoCerv (real pixel masks) + Atlas high/low-grade
images (classification-only, no masks) into a single training-ready manifest.

Two data sources, two very different roles:

  AnnoCerv (--raw_dir): 100 cases, real clinician-drawn lesion outlines.
    This is the ONLY source of segmentation supervision. Case-level
    stratified train/val/test split happens here.

  Atlas high/low-grade (--atlas_dir, optional): ~62 cases scraped from the
    IARC Colposcopy Atlas, well-documented (metadata.txt per case) but with
    NO pixel masks — only a high/low-grade folder label. These rows are
    added with has_mask=False, so the existing has_mask-aware loss
    (SegmentationLoss in losses.py) automatically excludes them from
    segmentation training and they only ever contribute to the auxiliary
    classification head. They are ALWAYS assigned split="train" and are
    deliberately excluded from the case pool used for train/val/test
    splitting and from train_kfold.py's k-fold rotation — they have no
    masks, so putting them in val/test would just be wasted rows there,
    and forcing them into train consistently means every fold gets the
    same classification-data boost rather than a lottery of which fold
    happens to draw them.

  (Kolposzkopia, the third folder in that same Atlas repo, is NOT loaded
  here at all — it has no masks AND no reliable numeric grade label, and
  is small/inconsistent enough that it's being used as a pure qualitative
  holdout instead. See src/inspect_kolposzkopia.py.)

Run:
    python src/prepare_data.py --raw_dir annocerv_raw/dataset \
        --atlas_dir atlas_raw --out_dir data --swede_threshold 5
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import binary_fill_holes
from skimage.morphology import closing, disk
from sklearn.model_selection import StratifiedShuffleSplit

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

ATLAS_CASE_ID_OFFSET = 100_000  # keeps Atlas case_ids disjoint from AnnoCerv's 1-100


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
    camera resolution, e.g. ~3000px) purely for speed. `work_size` scaling
    preserves the crop's original aspect ratio (long side -> work_size,
    short side scaled by the same factor) -- the mask is saved at that
    (smaller, aspect-correct) resolution; the Dataset class resizes each
    mask to match its paired image at load time, using the same aspect
    ratio, so nothing gets stretched incorrectly.
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

    # No further resize to a fixed square here -- `lesion` is already at
    # (scaled-but-aspect-preserved) work_size resolution.
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
                swede_score=score, label=label, source="annocerv",
            ))
        for jpg in sorted(list(d.glob("*Iod*.jpg")) + list(d.glob("*Green*.jpg"))):
            rows.append(dict(
                case_id=case_num, img_path=str(jpg),
                mask_path="", has_mask=False,
                img_type="iodine_or_green", swede_score=score, label=label,
                source="annocerv",
            ))

    return pd.DataFrame(rows)


def _parse_atlas_swede_score(metadata_text: str):
    m = re.search(r"Swede Score:\s*(-?\d+)", metadata_text)
    return int(m.group(1)) if m else -1  # -1 = sentinel "unknown", never used for
    # anything since these rows always have has_mask=False and their label
    # comes from the folder (high/low), not from a threshold on this score.


def load_atlas_classification_data(atlas_dir: Path) -> pd.DataFrame:
    """Scrapes images_high_grade/ and images_low_grade/ into classification-
    only rows (has_mask=False, split hardcoded to 'train' downstream).
    Every image in a case folder becomes one row (speculum/saline/acetic
    acid/iodine alike) -- more rows of real signal for the classification
    head, and unlike segmentation there's no mask-geometry reason to be
    picky about which stage each image shows."""
    rows = []
    case_counter = ATLAS_CASE_ID_OFFSET
    grade_dirs = [("images_high_grade", 1), ("images_low_grade", 0)]

    for folder_name, label in grade_dirs:
        grade_dir = atlas_dir / folder_name
        if not grade_dir.exists():
            print(f"  WARNING: {grade_dir} not found, skipping {folder_name}")
            continue
        case_dirs = sorted([d for d in grade_dir.iterdir() if d.is_dir()])
        for case_dir in case_dirs:
            meta_path = case_dir / "metadata.txt"
            swede_score = -1
            if meta_path.exists():
                swede_score = _parse_atlas_swede_score(meta_path.read_text(errors="ignore"))

            images = sorted([p for p in case_dir.iterdir()
                              if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
            if not images:
                continue

            case_counter += 1
            for img_path in images:
                rows.append(dict(
                    case_id=case_counter, img_path=str(img_path),
                    mask_path="", has_mask=False, img_type="atlas",
                    swede_score=swede_score, label=label, source="atlas_aux",
                ))

    df = pd.DataFrame(rows)
    n_cases = df["case_id"].nunique() if len(df) else 0
    print(f"  Atlas aux data: {n_cases} cases, {len(df)} images "
          f"({(df['label'] == 1).sum() if len(df) else 0} high-grade rows, "
          f"{(df['label'] == 0).sum() if len(df) else 0} low-grade rows)")
    return df


def case_level_split(df: pd.DataFrame, val_frac=0.15, test_frac=0.15, seed=42) -> pd.DataFrame:
    """Split by case_id (not image), stratified on case-level label, so no
    patient's images appear in more than one split. Only operates on
    source=='annocerv' rows -- Atlas aux rows (if present) have no masks
    and are force-assigned to 'train' via the fillna fallback below, never
    entering the case pool that gets split into val/test."""
    annocerv_df = df[df["source"] == "annocerv"]
    cases = annocerv_df[["case_id", "label"]].drop_duplicates().reset_index(drop=True)

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
    # any case_id not in split_map (i.e. Atlas aux cases) falls back to "train"
    df["split"] = df["case_id"].map(split_map).fillna("train")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, default="annocerv_raw/dataset")
    ap.add_argument("--atlas_dir", type=str, default=None,
                     help="path to cloned colposcopy-digital-atlas-dataset repo root "
                          "(the dir containing images_high_grade/, images_low_grade/). "
                          "Optional -- omit to train on AnnoCerv alone.")
    ap.add_argument("--out_dir", type=str, default="data")
    ap.add_argument("--swede_threshold", type=int, default=5)
    ap.add_argument("--mask_size", type=int, default=384,
                     help="kept for CLI backward-compatibility; no longer forces a "
                          "fixed square (see parse_mask docstring).")
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

    if args.atlas_dir:
        print(f"\nLoading Atlas classification-only data from {args.atlas_dir} ...")
        atlas_df = load_atlas_classification_data(Path(args.atlas_dir))
        if len(atlas_df):
            atlas_df["parsed_mask_path"] = ""
            atlas_df["mask_is_empty"] = np.nan
            df = pd.concat([df, atlas_df], ignore_index=True)

    print("\nSplitting by case (train/val/test = 70/15/15 of AnnoCerv cases; "
          "Atlas aux rows always -> train) ...")
    df = case_level_split(df)
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        sub_masked = sub[sub["has_mask"]]
        n_annocerv_cases = sub[sub["source"] == "annocerv"]["case_id"].nunique()
        n_atlas_cases = sub[sub["source"] == "atlas_aux"]["case_id"].nunique() if "source" in sub else 0
        print(f"  {split:5s}: {n_annocerv_cases:3d} AnnoCerv cases"
              + (f" + {n_atlas_cases} Atlas aux cases" if n_atlas_cases else "")
              + f", {len(sub):4d} images, {len(sub_masked):4d} with masks, "
              f"pos.rate(cls)={sub['label'].mean():.2f}")

    manifest_path = out_dir / "manifest.csv"
    df.to_csv(manifest_path, index=False)
    print(f"\nWrote manifest: {manifest_path}")

    stats = {
        "n_annocerv_cases": int(df[df["source"] == "annocerv"]["case_id"].nunique()),
        "n_atlas_aux_cases": int(df[df["source"] == "atlas_aux"]["case_id"].nunique()) if "source" in df else 0,
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