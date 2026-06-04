"""
11-1-detection-metrics.py
Visualize the standard mAP (mean Average Precision) curve and IoU computation
for object detection. Critical for understanding detection evaluation.

Output: assets/figures/11-1-detection-metrics.png
"""
import os
os.makedirs("assets/figures", exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt


# IoU computation - draw two boxes
def draw_iou(ax, title):
    # Ground truth box
    gt = [1, 2, 4, 5]  # x1, y1, x2, y2
    # Predicted box
    pred = [2, 1, 5, 4]
    ax.add_patch(plt.Rectangle((gt[0], gt[1]), gt[2] - gt[0], gt[3] - gt[1],
                                 fill=False, edgecolor="green", linewidth=2.5,
                                 label="Ground Truth"))
    ax.add_patch(plt.Rectangle((pred[0], pred[1]), pred[2] - pred[0], pred[3] - pred[1],
                                 fill=False, edgecolor="red", linewidth=2.5,
                                 label="Prediction"))

    # Intersection
    ix1, iy1 = max(gt[0], pred[0]), max(gt[1], pred[1])
    ix2, iy2 = min(gt[2], pred[2]), min(gt[3], pred[3])
    if ix1 < ix2 and iy1 < iy2:
        ax.add_patch(plt.Rectangle((ix1, iy1), ix2 - ix1, iy2 - iy1,
                                     fill=True, facecolor="yellow", alpha=0.4,
                                     label="Intersection"))
    ax.text(1.5, 6, f"GT area: 6", fontsize=10)
    ax.text(2.5, 0.3, f"Pred area: 6", fontsize=10)
    ax.text(2.5, 4.5, f"Inter: 4", fontsize=10, color="darkorange", fontweight="bold")
    ax.text(2.5, 0.5, f"Union: 8", fontsize=10, color="darkgreen", fontweight="bold")
    ax.text(2.5, 3.8, r"$\text{IoU} = \frac{4}{8} = 0.5$", fontsize=14,
            color="red", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", ec="red"))
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 7)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")


# mAP curve
def plot_map_curve(ax):
    """Plot a typical precision-recall curve for an object detector."""
    recall = np.linspace(0, 1, 100)
    # Realistic-looking PR curve
    precision = 1 - 0.7 * recall ** 1.5 - 0.1 * np.sin(recall * 20)
    precision = np.clip(precision, 0, 1)

    # Compute AP via trapezoidal integration
    ap = np.trapezoid(precision, recall)
    ax.plot(recall, precision, "b-", linewidth=2, label=f"PR Curve (AP={ap:.3f})")
    ax.fill_between(recall, precision, alpha=0.2, color="blue")
    ax.set_xlabel("Recall (查全率)", fontsize=11)
    ax.set_ylabel("Precision (查准率)", fontsize=11)
    ax.set_title("(b) PR 曲线与 mAP\nmAP = PR 曲线下面积", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    # Annotate AP
    ax.annotate(f"AP = {ap:.3f}", xy=(0.5, precision[50]), xytext=(0.3, 0.8),
                fontsize=12, color="red", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="red"))


fig, axes = plt.subplots(1, 2, figsize=(16, 6))
draw_iou(axes[0], "(a) IoU (交并比) 计算")
plot_map_curve(axes[1])

plt.suptitle("目标检测两大核心指标: IoU + mAP",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("assets/figures/11-1-detection-metrics.png", dpi=120, bbox_inches="tight")
print("Saved: assets/figures/11-1-detection-metrics.png")
