"""
FPN Multi-Scale Feature Visualization
====================================
Visualizes the multi-scale feature maps (P2-P5) of a Feature Pyramid Network
on a synthetic input image with objects at different scales.

Output: assets/figures/4-4-fpn-multiscale-features.png
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.ndimage import zoom


def make_fake_feature(h, w, n_obj=3, seed=42):
    """Generate a fake feature map with random Gaussian blobs."""
    rng = np.random.RandomState(seed)
    feat = np.zeros((h, w))
    for _ in range(n_obj):
        cy = rng.randint(2, h - 2)
        cx = rng.randint(2, w - 2)
        sigma = rng.uniform(1.5, 3.5)
        y, x = np.ogrid[:h, :w]
        blob = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
        feat += blob
    feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
    return feat


def make_input_image(H=224, W=224):
    """Make a synthetic RGB image with multi-scale objects."""
    img = np.ones((H, W, 3)) * 0.95  # background

    # Big object (top-left): elephant-like
    cy, cx = 60, 70
    yy, xx = np.ogrid[:H, :W]
    big = ((xx - cx) / 35) ** 2 + ((yy - cy) / 25) ** 2 < 1
    img[big] = [0.7, 0.5, 0.4]  # brown

    # Medium object (center): car-like rectangle
    img[90:120, 100:150] = [0.3, 0.3, 0.7]  # blue

    # Small object (bottom-right): bird-like dot
    img[180:195, 195:210] = [0.9, 0.7, 0.2]  # yellow

    # Add some noise
    rng = np.random.RandomState(0)
    img += rng.normal(0, 0.02, img.shape)
    img = np.clip(img, 0, 1)
    return img


fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle('FPN Multi-Scale Feature Pyramid (P2 ~ P5)', fontsize=15, fontweight='bold', y=1.02)

H, W = 224, 224
img = make_input_image(H, W)
axes[0, 0].imshow(img)
axes[0, 0].set_title(f'Input Image\n{H}×{W}×3', fontsize=11, fontweight='bold')
axes[0, 0].axis('off')

scales = [4, 8, 16, 32]
for i, s in enumerate(scales):
    h, w = H // s, W // s
    feat = make_fake_feature(h, w, n_obj=3, seed=42 + i)
    feat_disp = zoom(feat, (s, s), order=1)
    axes[0, i+1].imshow(feat_disp, cmap='hot', interpolation='bilinear', vmin=0, vmax=1)
    axes[0, i+1].set_title(f'P{i+2}: stride 1/{s}\nSize: {h}×{w}', fontsize=11, fontweight='bold')
    axes[0, i+1].axis('off')

axes[0, 4].axis('off')

# Bottom row: ground truth response (where objects of each scale are strongest)
# P2: small objects, P3: medium, P4: large, P5: extra large
gt_per_scale = [
    (5, 7, 60, 65, 195, 200),    # P2: small bird at bottom-right (radius ~5)
    (10, 25, 100, 100, 120, 150), # P3: medium car
    (20, 35, 60, 60, 90, 90),     # P4: large elephant
    (40, 50, 60, 60, 90, 90),     # P5: extra large context
]

# Just show a clean alignment visualization
for i, s in enumerate(scales):
    h, w = H // s, W // s
    feat = np.zeros((h, w))
    cy_gt, rx_gt, gt_y, gt_x, _, _ = gt_per_scale[i]
    # Object center in feature map
    fy = gt_y // s
    fx = gt_x // s
    fr = max(1, rx_gt // s)
    yy, xx = np.ogrid[:h, :w]
    blob = np.exp(-((xx - fx) ** 2 + (yy - fy) ** 2) / (2 * fr ** 2))
    feat = blob
    feat_disp = zoom(feat, (s, s), order=1)
    axes[1, i+1].imshow(feat_disp, cmap='viridis', interpolation='bilinear')
    axes[1, i+1].axis('off')

# Show the input in the same position for alignment
axes[1, 0].imshow(img)
axes[1, 0].set_title('Aligned Response', fontsize=11, fontweight='bold')
axes[1, 0].axis('off')

# Legend / explanation
fig.text(0.5, -0.02,
         'Top: feature maps from FPN (P2 highest resolution ~ small objects, P5 lowest ~ large objects).   '
         'Bottom: aligned response — each scale is most sensitive to objects of corresponding size.',
         ha='center', fontsize=10, style='italic', color='#444')

plt.tight_layout()
plt.savefig('assets/figures/4-4-fpn-multiscale-features.png', dpi=120, bbox_inches='tight', facecolor='white')
print('  ✓ Saved: assets/figures/4-4-fpn-multiscale-features.png')
