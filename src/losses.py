"""
losses.py — Dice + BCE segmentation loss (masked by has_mask), plus a
Tversky variant for asymmetric FP/FN control, and a straightforward BCE
classification loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dice_loss(probs: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """probs, targets: (B, 1, H, W). Returns per-sample loss (B,)."""
    dims = (1, 2, 3)
    intersection = (probs * targets).sum(dim=dims)
    union = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice


def tversky_loss(probs: torch.Tensor, targets: torch.Tensor,
                  alpha: float = 0.5, beta: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    """alpha weights false positives, beta weights false negatives.
    alpha < beta biases toward recall (fewer missed lesions) — useful since
    missing a high-grade lesion is clinically worse than a false alarm."""
    dims = (1, 2, 3)
    tp = (probs * targets).sum(dim=dims)
    fp = (probs * (1 - targets)).sum(dim=dims)
    fn = ((1 - probs) * targets).sum(dim=dims)
    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return 1 - tversky


class SegmentationLoss(nn.Module):
    """Combined BCE + Dice/Tversky, masked so that rows with has_mask=False
    never contribute — this is applied explicitly rather than relying on a
    'mask_quality' weight, so there is no way for pseudo/unannotated rows to
    leak gradient into the segmentation head."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5,
                 use_tversky: bool = False, tversky_alpha: float = 0.4,
                 tversky_beta: float = 0.6, pos_weight: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.use_tversky = use_tversky
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.register_buffer("pos_weight", torch.tensor(pos_weight))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, has_mask: torch.Tensor):
        if has_mask.sum() == 0:
            return logits.sum() * 0.0  # no annotated rows in this batch; zero, still differentiable

        logits_m = logits[has_mask]
        targets_m = targets[has_mask]

        bce = F.binary_cross_entropy_with_logits(
            logits_m, targets_m, pos_weight=self.pos_weight, reduction="mean")

        probs_m = torch.sigmoid(logits_m)
        if self.use_tversky:
            region_loss = tversky_loss(probs_m, targets_m, self.tversky_alpha, self.tversky_beta).mean()
        else:
            region_loss = soft_dice_loss(probs_m, targets_m).mean()

        return self.bce_weight * bce + self.dice_weight * region_loss


class ClassificationLoss(nn.Module):
    def __init__(self, pos_weight: float = 1.0):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor(pos_weight))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        return F.binary_cross_entropy_with_logits(
            logits.squeeze(-1), targets, pos_weight=self.pos_weight, reduction="mean")


@torch.no_grad()
def dice_score(probs: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5,
               eps: float = 1e-6) -> torch.Tensor:
    """Hard Dice at a given threshold, per-sample. probs/targets: (B,1,H,W)."""
    preds = (probs >= threshold).float()
    dims = (1, 2, 3)
    intersection = (preds * targets).sum(dim=dims)
    union = preds.sum(dim=dims) + targets.sum(dim=dims)
    # Convention: both empty -> dice 1.0 (correct negative prediction, not undefined)
    dice = torch.where(union > 0, (2 * intersection + eps) / (union + eps),
                        torch.ones_like(union))
    return dice
