"""
dataset.py — reads data/manifest.csv (produced by prepare_data.py) and serves
(image, mask, has_mask, label) tuples.

Design notes:
  - Segmentation loss should only ever be computed on rows where has_mask=True.
    `has_mask` is returned alongside the mask so the training loop can mask
    out the loss contribution of unannotated rows exactly, instead of
    silently training on empty/zero masks pretending to be "no lesion".
  - Classification label is available for every row (derived from the case's
    swede score) so the auxiliary classification head can use the full image
    pool even though only ~56% of rows have a segmentation mask.
  - Augmentation is applied identically to image and mask (same random
    params) to keep them pixel-aligned. Color-only augmentation (CLAHE,
    color jitter) is applied to the image alone.
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_transforms(image_size: int, split: str) -> A.Compose:
    if split == "train":
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),          # cervix has no canonical "up"
            A.Rotate(limit=25, border_mode=cv2.BORDER_REFLECT, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15,
                                  val_shift_limit=8, p=0.3),
            A.GaussNoise(std_range=(0.02, 0.08), p=0.2),
            A.ElasticTransform(alpha=40, sigma=6, p=0.15),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])


class ColposcopyDataset(Dataset):
    def __init__(self, manifest_path: str, split: str, image_size: int = 384,
                 crop_ratio: float = 0.80, seg_only: bool = False):
        """
        Args:
            seg_only: if True, drop rows without a mask entirely (use for a
                pure-segmentation dataloader). If False, keep all rows so the
                classification head can also train on unmasked images.
        """
        df = pd.read_csv(manifest_path)
        df = df[df["split"] == split].reset_index(drop=True)
        if seg_only:
            df = df[df["has_mask"] == True].reset_index(drop=True)  # noqa: E712
        self.df = df
        self.crop_ratio = crop_ratio
        self.transform = build_transforms(image_size, split)

    def __len__(self):
        return len(self.df)

    def _load_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        nh, nw = int(h * self.crop_ratio), int(w * self.crop_ratio)
        top, left = (h - nh) // 2, (w - nw) // 2
        return img[top:top + nh, left:left + nw]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = self._load_image(row["img_path"])

        has_mask = bool(row["has_mask"])
        if has_mask:
            mask = cv2.imread(row["parsed_mask_path"], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(
                    f"Could not read mask file: {row['parsed_mask_path']!r} "
                    f"(case {row['case_id']}). data/manifest.csv references "
                    f"this path but the file isn't there — re-run "
                    f"prepare_data.py in THIS environment (mask files aren't "
                    f"portable between machines/sessions, only manifest.csv "
                    f"and the raw images are)."
                )
            if mask.shape[:2] != image.shape[:2]:
                # prepare_data.py saves masks at a fixed square resolution
                # (independent of each photo's native aspect ratio) for
                # storage simplicity. Resizing here to match the freshly
                # cropped image's actual shape — rather than assuming they
                # already match — keeps this correct even if crop_ratio or
                # source photo resolution ever changes without re-running
                # prepare_data.py, and avoids a silent aspect-ratio mismatch
                # feeding into the shared augmentation pipeline below.
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.float32)
        else:
            # placeholder zero mask — has_mask=False tells the loss to ignore it
            mask = np.zeros(image.shape[:2], dtype=np.float32)

        out = self.transform(image=image, mask=mask)
        image_t = out["image"]                       # (3, H, W) float
        mask_t = out["mask"].float().unsqueeze(0)     # (1, H, W) float, {0,1}

        return {
            "image": image_t,
            "mask": mask_t,
            "has_mask": torch.tensor(has_mask, dtype=torch.bool),
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "case_id": int(row["case_id"]),
            "img_path": row["img_path"],
        }


if __name__ == "__main__":
    # quick manual smoke test: `python src/dataset.py`
    ds = ColposcopyDataset("data/manifest.csv", split="train", image_size=384)
    print(f"train dataset size: {len(ds)}")
    sample = ds[0]
    print({k: (v.shape if hasattr(v, "shape") else v) for k, v in sample.items()})
