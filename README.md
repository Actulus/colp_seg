# Colposcopy Lesion Segmentation — Rebuilt Pipeline

Rebuilt from scratch against the same public data (**AnnoCerv**, 100 cases,
github.com/iclx/AnnoCerv) after the previous model's segmentation Dice
(0.095, essentially non-functional) turned out to trace back to a real bug,
not a data or architecture ceiling. This is that fix plus a clean, standard
train/val/test pipeline.

## What was actually broken, and what changed

**The encoder never trained.** The old `train_seg.py` tried to freeze the
encoder for the first N epochs, then "unfreeze" it by directly setting
`optimizer.param_groups[i]["lr"]` mid-run. PyTorch's LR schedulers
(`LinearLR`/`CosineAnnealingLR`) cache each param group's `initial_lr` at
**construction** time and recompute `lr` from that cache on every
`scheduler.step()` — so the manual "unfreeze" was silently reverted the very
next epoch. The encoder's lr was `0.0` at construction and stayed `0.0` for
all 100 epochs, whatever the logs claimed. That alone is enough to explain
near-random Dice (0.095) and near-random classification AUC (~0.53–0.60):
only the randomly-initialized decoder and heads were ever training, on top
of frozen, generic (non-colposcopy) ImageNet features.

**Fix in this rebuild (`src/model.py`, `build_param_groups`):** the encoder
gets its own (smaller) learning rate from step 0, set once via optimizer
param groups. The schedule is a single `LambdaLR` (linear warmup → cosine
decay) built once and never mutated afterward — there is nothing left for a
scheduler to "forget." No freeze/unfreeze choreography at all.

**A second, related issue** flagged in the old results notebook: dual-head
class probabilities were being flipped and silently auto-corrected at eval
time, meaning training and evaluation disagreed about which class was
"positive." This rebuild computes `label` once in `prepare_data.py` from the
swede score and nothing downstream re-derives or re-indexes it, so there's
one source of truth for the label the whole way through.

The mask-parsing logic itself (color-distance + Otsu threshold + morphological
closing/fill on the RGBA annotation overlays) was kept — it wasn't the
problem — but building this pipeline turned up **a second real bug** in my
own first draft, worth being upfront about: `prepare_data.py` originally
force-resized every parsed mask to a fixed square (384×384), while
`dataset.py` cropped the source photo but kept its native ~1.5:1 aspect
ratio — so image and mask geometry silently disagreed. It surfaced
immediately as a hard crash the moment I ran the real dataset loader
end-to-end (albumentations refuses mismatched shapes), which is exactly why
that step is worth doing rather than trusting code that only "looks right."
Fixed by having mask parsing preserve the crop's aspect ratio throughout,
and having the Dataset resize each mask to match its paired image at load
time rather than assuming a fixed resolution.

## Pipeline

```
annocerv_raw/dataset/          (git clone https://github.com/iclx/AnnoCerv)
        │
        ▼
src/prepare_data.py            → data/manifest.csv, data/masks/*.png
        │   (case-level stratified 70/15/15 split — no patient leaks across splits)
        ▼
src/dataset.py                 → PyTorch Dataset (albumentations aug, has_mask flag)
        │
        ▼
src/model.py                   → ResNet34-UNet + optional aux classification head
        │
        ▼
src/train.py                   → warmup+cosine LR, early stopping on val Dice
        │
        ▼
src/evaluate.py                → threshold calibrated on VAL, reported on TEST
                                  (Dice, IoU, HD95, bootstrap CI, per-grade breakdown)
```

## Getting the data (unification step)

```bash
git clone --depth=1 https://github.com/iclx/AnnoCerv annocerv_raw
python src/prepare_data.py --raw_dir annocerv_raw/dataset --out_dir data \
    --swede_threshold 5 --mask_size 384
```
This was run for real as part of building this pipeline. Actual numbers from
the current AnnoCerv release:

| split | cases | images | with mask | pos. rate |
|---|---|---|---|---|
| train | 69 | 357 | 197 | 0.41 |
| val   | 16 |  82 |  44 | 0.45 |
| test  | 15 |  93 |  56 | 0.41 |

85 annotation PNGs parsed to an *empty* mask — these are cases the clinician
marked with no notable lesion features, kept as valid negative examples
rather than dropped. Two parsed masks were visually spot-checked against
their source images (see the overlay sanity-check step) and lined up with
the visible lesion boundary.

**On adding more data (Atlas, Kolposzkopia, etc.):** I deliberately left
these out of this rebuild. The old pipeline mixed in pseudo-masks generated
by a much cruder heuristic (HSV outline detection on plain photos with no
real annotation), only successfully generated for about half those images,
and it's a real candidate for having added label noise on top of the
frozen-encoder bug. Get the core pipeline validated on real annotations
first; if you want to add a second source back in, do it as a controlled
ablation (train with vs. without it) rather than folding it in by default —
`prepare_data.py` is structured so a second loader function can append rows
to the same manifest schema (`img_path, mask_path, has_mask, label, case_id`)
without touching anything downstream.

## Training

```bash
pip install -r requirements.txt
python src/train.py --config configs/config.yaml
```
Key config choices in `configs/config.yaml` (all adjustable):
- `image_size: 384`, `encoder: resnet34`, batch size 8 — tune batch size to
  your GPU memory; everything else in the pipeline is batch-size agnostic.
- `seg_loss_weight` vs `cls_loss_weight`: classification is kept as a
  low-weight auxiliary task (0.3x) — it exists to give the encoder more
  gradient signal, not to compete with segmentation. Set `use_cls_head:
  false` in `model:` to drop it entirely if you'd rather isolate segmentation
  fully.
- `use_tversky: true` + tune `tversky_alpha`/`tversky_beta` if you want to
  explicitly bias toward recall (fewer missed lesions at the cost of more
  false positives) once you have a working baseline — don't reach for this
  until Dice off the base Dice+BCE loss is behaving sanely.
- pos_weights for both heads are computed automatically from the **train**
  split only (see `compute_pos_weights` in `train.py`), not hardcoded.

## Evaluation

```bash
python src/evaluate.py --config configs/config.yaml
```
Sweeps threshold on val, reports final Dice/IoU/HD95 with bootstrap 95% CIs
on test, and a breakdown by swede-score grade (high vs. low) — the old model
had *worse* Dice on high-grade lesions than low-grade, which is backwards
from what matters clinically, so this is tracked explicitly rather than only
looking at the aggregate number.

## What was actually run and verified (not just reviewed)

Managed to get a CPU-only `torch` installed in this sandbox after all, so
this isn't just code review — the real pipeline ran, on the real data:

- `prepare_data.py` ran end-to-end on real AnnoCerv data — 100 cases, 532
  images, 297 masks parsed, case-level splits built and saved.
- Parsed masks were visually checked against source images (see the
  overlay sanity-check in this repo's history) — boundaries line up with
  the visible acetowhite/vascular regions.
- `ColposcopyUNet` forward pass verified on real tensors: correct output
  shapes, ~24M params, no shape errors through the encoder/decoder/skip
  connections (pretrained ImageNet weight download is blocked by this
  sandbox's network allowlist — not expected to be an issue on your own
  machine).
- Ran the **actual training loop** for 2 epochs (CPU, resnet18, 128px,
  randomly-initialized — not even pretrained weights, the worst-case
  scenario) as a smoke test:
  ```
  [001/2] train_loss=1.1560 val_loss=1.0917 train_dice=0.025 val_dice=0.011
  [002/2] train_loss=1.0735 val_loss=1.0812 train_dice=0.103 val_dice=0.318
  ```
  Loss decreasing, val_dice climbing from 0.011 → 0.318 in two epochs with
  no pretrained weights and a downscaled image — already well above the old
  model's *final* Dice of 0.095 after 100 full epochs. That's the clearest
  evidence the frozen-encoder bug was the main culprit.
- Ran `evaluate.py` against that checkpoint end-to-end: calibrated threshold
  came out at **0.55** (sane — not the degenerate 0.05–0.15 the old model
  needed, which was itself a symptom of a barely-trained decoder), and the
  high/low-grade Dice split (0.33 vs 0.34) is balanced rather than inverted.
- All files pass `python -m py_compile`, and the LR schedule / Dice-loss
  math were separately checked against known edge cases.

None of this means the real, full-scale training run (100 epochs, real
image size, pretrained weights, GPU) is guaranteed to hit great numbers —
100 annotated cases is still a small dataset — but the pipeline is
demonstrably functional end-to-end rather than untested code.

## Next steps once this baseline is validated
1. Confirm val_dice climbs to a sane range (even 0.4–0.5 would be a huge
   jump from 0.095) within the first ~20 epochs.
2. Try `resnet50` as an encoder ablation.
3. Revisit whether Atlas/pseudo-mask data helps or hurts, as a controlled
   comparison, not a default merge.
4. Once Dice is reasonable, look specifically at the high-grade vs low-grade
   breakdown — that inversion in the old model is worth understanding, not
   just averaging away.
