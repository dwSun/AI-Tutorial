"""
11-2-segmentation-metrics.py
Visualize IoU (Jaccard Index) and Dice coefficient for semantic segmentation.
Also visualize how segmentation maps look at different IoU levels.

Output: assets/figures/11-2-segmentation-metrics.png
"""
import os
os.makedirs("assets/figures", exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt


def draw_seg(ax, title, iou, dice):
    """Draw two colored blobs (ground truth and prediction) with given IoU."""
    h, w = 100, 100
    yy, xx = np.mgrid[0:h, 0:w]

    # GT: circle
    gt_mask = ((xx - 35) ** 2 + (yy - 50) ** 2) < 22 ** 2
    # Pred: circle offset to control IoU
    center_x = 35 + (1 - iou) * 30
    pred_mask = ((xx - center_x) ** 2 + (yy - 50) ** 2) < 22 ** 2

    img = np.ones((h, w, 3))
    # Background light gray
    img[:] = 0.95
    # GT only: green
    img[gt_mask & ~pred_mask] = [0.4, 0.85, 0.4]
    # Pred only: red
    img[pred_mask & ~gt_mask] = [0.95, 0.4, 0.4]
    # Intersection: yellow
    img[gt_mask & pred_mask] = [1.0, 0.95, 0.3]

    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.text(50, 90, f"IoU = {iou:.2f}\nDice = {dice:.2f}",
            ha="center", va="top", fontsize=14, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, ec="black"))


fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# Different IoU levels
iou_levels = [0.95, 0.75, 0.50, 0.25]
dice_levels = [2 * x / (1 + x) for x in iou_levels]
labels = [
    "(a) 优秀: IoU=0.95\nGT 和 Pred 几乎重合",
    "(b) 良好: IoU=0.75\n主要区域对齐",
    "(c) 勉强: IoU=0.50\n刚刚及格 (COCO mAP@0.5)",
    "(d) 差: IoU=0.25\n模型基本失败"
]

for ax, iou, dice, label in zip(axes, iou_levels, dice_levels, labels):
    draw_seg(ax, label, iou, dice)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=[0.4, 0.85, 0.4], label='GT only (漏检)'),
    Patch(facecolor=[0.95, 0.4, 0.4], label='Pred only (误检)'),
    Patch(facecolor=[1.0, 0.95, 0.3], label='Intersection (正确)'),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=10,
           bbox_to_anchor=(0.5, -0.02))

plt.suptitle("语义分割评估: IoU (Jaccard Index) 与 Dice 系数可视化",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("assets/figures/11-2-segmentation-metrics.png", dpi=120, bbox_inches="tight")
print("Saved: assets/figures/11-2-segmentation-metrics.png")
