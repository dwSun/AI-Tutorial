"""
3-5-loss-landscape.py
Draw the 4 key features of a deep learning loss landscape:
  - Local minimum
  - Saddle point
  - Narrow valley (ravine)
  - Plateau
Plus a global minimum for reference.

Output: assets/figures/3-5-loss-landscape.png
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import os

os.makedirs("assets/figures", exist_ok=True)

fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))


def draw_loss(ax, title, feature_fn, special_pts=None, contour_levels=15):
    """Draw a 2D loss landscape with given feature function."""
    x = np.linspace(-3, 3, 200)
    y = np.linspace(-2, 2, 200)
    X, Y = np.meshgrid(x, y)
    Z = feature_fn(X, Y)
    Z = np.clip(Z, 0, 8)

    cs = ax.contourf(X, Y, Z, levels=contour_levels, cmap="viridis", alpha=0.6)
    ax.contour(X, Y, Z, levels=contour_levels, colors="white", linewidths=0.4, alpha=0.5)

    # Mark special points
    if special_pts:
        for (px, py, color, label) in special_pts:
            ax.scatter(px, py, c=color, s=200, zorder=5, edgecolors="white", linewidths=2)
            ax.annotate(label, (px, py), xytext=(px + 0.2, py + 0.3),
                        fontsize=10, color=color, fontweight="bold")

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("$w_1$", fontsize=11)
    ax.set_ylabel("$w_2$", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])


# 1. Local minimum
def f_local_min(X, Y):
    return (X ** 2 + Y ** 2) * 0.6 + (np.exp(-((X - 1.5) ** 2 + Y ** 2) * 1.5) * 3
                                       + np.exp(-((X + 1.5) ** 2 + (Y - 0.8) ** 2) * 1.5) * 1.5)


draw_loss(axes[0], "Local Minima\n(梯度=0, 但非全局最优)",
          f_local_min,
          [(-1.5, 0.8, "red", "local min"),
           (1.5, 0, "red", "local min"),
           (0, 0, "lime", "saddle")])

# 2. Saddle point
def f_saddle(X, Y):
    return (X ** 2 - Y ** 2) * 0.3 + 2 + 0.2 * (X ** 2 + Y ** 2)


draw_loss(axes[1], "Saddle Point\n(一个方向↑ 另一个方向↓)",
          f_saddle,
          [(0, 0, "red", "saddle")])

# 3. Narrow valley (ravine)
def f_valley(X, Y):
    return 0.05 * X ** 2 + 2.5 * Y ** 2 + 1


draw_loss(axes[2], "Narrow Valley\n(SGD 横向震荡, 纵向进展慢)",
          f_valley,
          [(-2.5, 0, "cyan", "start"),
           (0, 0, "red", "min")])

# 4. Plateau
def f_plateau(X, Y):
    return np.minimum(2.5 + np.exp(-((Y) ** 2) * 0.5) * 2, 0.5 * X ** 2 + 0.2 * Y ** 2 + 0.5)


draw_loss(axes[3], "Plateau\n(梯度极小, 训练看似停滞)",
          f_plateau,
          [(-2, 0.2, "cyan", "start"),
           (2, 0, "red", "min")])

plt.suptitle("Loss Landscape 的 4 种典型地形 — 为什么优化器比 GD 更重要",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("assets/figures/3-5-loss-landscape.png", dpi=120, bbox_inches="tight")
print("Saved: assets/figures/3-5-loss-landscape.png")
