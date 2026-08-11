"""
model.py — ResNet-encoder U-Net for lesion segmentation, with a small
optional auxiliary classification head off the bottleneck.

Deliberately simpler than the previous dual-head design:
  - No manual freeze/unfreeze of the encoder mid-training. The old pipeline
    tried to freeze the encoder for N epochs then unfreeze it by mutating
    optimizer.param_groups[i]["lr"] directly — but PyTorch LR schedulers
    cache each group's `initial_lr` at construction time and recompute lr
    from that cache every `scheduler.step()`, silently reverting the manual
    change. Net effect: the encoder was frozen for the ENTIRE run, not just
    the first N epochs, which is very likely why both heads under-performed.
    This version sidesteps the whole bug class: encoder and head/decoder
    get different LRs from step 0 via optimizer param groups (set once, no
    manual override later), so there's nothing for the scheduler to revert.
  - Segmentation is the primary output. Classification is a cheap auxiliary
    task (extra gradient signal from the swede score) — it is NOT required
    for the segmentation loss to work, and can be disabled via `use_cls_head`.
"""
import torch
import torch.nn as nn
import torchvision.models as tvm


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                                           align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResNetEncoder(nn.Module):
    """Wraps a torchvision ResNet, exposing 5 feature maps for U-Net skips."""

    def __init__(self, name: str = "resnet34", pretrained: bool = True):
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained else None
        backbone = getattr(tvm, name)(weights=weights)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # /2,  64ch
        self.pool = backbone.maxpool                                            # /4
        self.layer1 = backbone.layer1   # /4,  64/256ch
        self.layer2 = backbone.layer2   # /8,  128/512ch
        self.layer3 = backbone.layer3   # /16, 256/1024ch
        self.layer4 = backbone.layer4   # /32, 512/2048ch

        ch = {"resnet18": [64, 64, 128, 256, 512], "resnet34": [64, 64, 128, 256, 512],
              "resnet50": [64, 256, 512, 1024, 2048]}[name]
        self.out_channels = ch  # [stem, layer1, layer2, layer3, layer4]

    def forward(self, x):
        x0 = self.stem(x)          # /2
        x1 = self.layer1(self.pool(x0))   # /4
        x2 = self.layer2(x1)              # /8
        x3 = self.layer3(x2)              # /16
        x4 = self.layer4(x3)              # /32
        return [x0, x1, x2, x3, x4]


class ColposcopyUNet(nn.Module):
    def __init__(self, encoder_name: str = "resnet34", pretrained: bool = True,
                 num_seg_classes: int = 1, use_cls_head: bool = True,
                 num_cls_classes: int = 1, dropout: float = 0.2):
        super().__init__()
        self.encoder = ResNetEncoder(encoder_name, pretrained)
        c0, c1, c2, c3, c4 = self.encoder.out_channels

        self.up4 = UpBlock(c4, c3, 256)
        self.up3 = UpBlock(256, c2, 128)
        self.up2 = UpBlock(128, c1, 64)
        self.up1 = UpBlock(64, c0, 32)
        self.final_up = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)  # back to /1
        self.seg_head = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(32, num_seg_classes, kernel_size=1),
        )

        self.use_cls_head = use_cls_head
        if use_cls_head:
            self.cls_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(c4, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(128, num_cls_classes),
            )

    def forward(self, x):
        x0, x1, x2, x3, x4 = self.encoder(x)

        d = self.up4(x4, x3)
        d = self.up3(d, x2)
        d = self.up2(d, x1)
        d = self.up1(d, x0)
        d = self.final_up(d)
        if d.shape[-2:] != x.shape[-2:]:
            d = nn.functional.interpolate(d, size=x.shape[-2:], mode="bilinear",
                                           align_corners=False)
        seg_logits = self.seg_head(d)

        cls_logits = self.cls_head(x4) if self.use_cls_head else None
        return {"seg_logits": seg_logits, "cls_logits": cls_logits}


def build_param_groups(model: ColposcopyUNet, base_lr: float, encoder_lr_mult: float = 0.1):
    """Differential LR from the start — no mid-training freeze/unfreeze
    mutation of optimizer.param_groups, see module docstring for why."""
    encoder_params = list(model.encoder.parameters())
    other_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    return [
        {"params": encoder_params, "lr": base_lr * encoder_lr_mult, "name": "encoder"},
        {"params": other_params, "lr": base_lr, "name": "decoder_heads"},
    ]


if __name__ == "__main__":
    m = ColposcopyUNet()
    x = torch.randn(2, 3, 384, 384)
    out = m(x)
    print("seg_logits:", out["seg_logits"].shape)
    print("cls_logits:", out["cls_logits"].shape)
