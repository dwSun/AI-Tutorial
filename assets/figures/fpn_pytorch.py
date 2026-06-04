"""
Feature Pyramid Network (FPN) Implementation in PyTorch
========================================================
Reference: Lin et al. "Feature Pyramid Networks for Object Detection" CVPR 2017

This is a minimal, clean, runnable implementation of FPN.
Tested with ResNet-style backbone outputs (C2-C5).

Usage:
    from fpn_pytorch import FPN, FPNWithBackbone
    model = FPNWithBackbone(num_classes=80)  # COCO 80 classes
    outputs = model(images)  # List of 4 feature maps at different scales
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    """
    Feature Pyramid Network (FPN).

    Takes multi-scale backbone features (C2, C3, C4, C5) at strides
    {4, 8, 16, 32} and produces pyramid features (P2, P3, P4, P5)
    at the same strides, with 256 channels each.
    """

    def __init__(self, in_channels_list=(256, 512, 1024, 2048), out_channels=256):
        super().__init__()
        self.out_channels = out_channels

        # 1x1 lateral convs: reduce channel count of each C_i to out_channels
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels_list
        ])

        # 3x3 output convs: smooth the merged features
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels_list
        ])

    def forward(self, features):
        """
        Args:
            features: list of 4 tensors [C2, C3, C4, C5] from backbone
                      shapes: [(B,256,H/4,W/4), (B,512,H/8,W/8),
                               (B,1024,H/16,W/16), (B,2048,H/32,W/32)]
        Returns:
            list of 4 tensors [P2, P3, P4, P5]
            shapes: [(B,256,H/4,W/4), (B,256,H/8,W/8),
                     (B,256,H/16,W/16), (B,256,H/32,W/32)]
        """
        # Step 1: 1x1 lateral convs (channel alignment)
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]

        # Step 2: top-down pathway (from deepest to shallowest)
        # Start with the deepest (P5 = C5 after 1x1)
        outs = [None] * len(laterals)
        outs[-1] = laterals[-1]  # P5 = lateral C5

        for i in range(len(laterals) - 2, -1, -1):
            # 2× upsample the deeper feature
            upsampled = F.interpolate(outs[i + 1], scale_factor=2.0, mode='nearest')
            # Element-wise add with the lateral
            outs[i] = laterals[i] + upsampled

        # Step 3: 3x3 smoothing convs
        outs = [conv(out) for conv, out in zip(self.output_convs, outs)]
        return outs


class FPNWithBackbone(nn.Module):
    """
    Complete FPN with a simple ResNet-style backbone (for demonstration).
    In production, use torchvision.models.resnet50(weights=...) and slice outputs.
    """

    def __init__(self, num_classes=80, in_channels_list=(256, 512, 1024, 2048)):
        super().__init__()
        self.backbone = self._make_backbone()
        self.fpn = FPN(in_channels_list=in_channels_list, out_channels=256)
        # Per-level detection heads (simplified)
        self.heads = nn.ModuleList([
            self._make_head(num_classes) for _ in range(4)
        ])

    def _make_backbone(self):
        """Simple ResNet-like backbone that returns [C2, C3, C4, C5]."""
        # In practice: torchvision.models.resnet50(... features_only=True)
        return nn.Identity()  # placeholder; user provides features externally

    def _make_head(self, num_classes):
        return nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes + 4, 1)  # classes + box (x,y,w,h)
        )

    def forward(self, features):
        pyramid = self.fpn(features)
        outputs = [head(p) for head, p in zip(self.heads, pyramid)]
        return outputs


def demo_fpn():
    """Quick smoke test: feed random multi-scale features through FPN."""
    print("=" * 60)
    print("FPN Demo: feeding 4-scale random features")
    print("=" * 60)

    fpn = FPN(in_channels_list=(256, 512, 1024, 2048), out_channels=256)
    fpn.eval()

    B = 2
    H, W = 224, 224
    c2 = torch.randn(B, 256,  H // 4,  W // 4)
    c3 = torch.randn(B, 512,  H // 8,  W // 8)
    c4 = torch.randn(B, 1024, H // 16, W // 16)
    c5 = torch.randn(B, 2048, H // 32, W // 32)

    with torch.no_grad():
        pyramid = fpn([c2, c3, c4, c5])

    print(f"Input feature shapes:")
    print(f"  C2: {tuple(c2.shape)}  (stride 1/4)")
    print(f"  C3: {tuple(c3.shape)}  (stride 1/8)")
    print(f"  C4: {tuple(c4.shape)}  (stride 1/16)")
    print(f"  C5: {tuple(c5.shape)}  (stride 1/32)")
    print(f"\nOutput pyramid shapes (all 256 channels):")
    for i, p in enumerate(pyramid):
        print(f"  P{i+2}: {tuple(p.shape)}  (stride 1/{4*(i+1)})")
    print(f"\nTotal FPN parameters: {sum(p.numel() for p in fpn.parameters()):,}")


if __name__ == "__main__":
    demo_fpn()
