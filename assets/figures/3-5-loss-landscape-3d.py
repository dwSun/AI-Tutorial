"""
3-5-loss-landscape-3d.py
Draw the 4 key features of a deep learning loss landscape as 3D surface plots:
  - Local minimum (multiple wells)
  - Saddle point
  - Narrow valley (ravine)
  - Plateau
With appropriate 3D viewing angles.

Output: assets/figures/3-5-loss-landscape-3d.png
"""
import os
os.makedirs("assets/figures", exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)


fig = plt.figure(figsize=(20, 5.5))


def draw_3d_landscape(ax, title, feature_fn, elev=30, azim=45,
                     feature_pts=None, cmap="viridis"):
    """Draw a 3D surface of a 2D loss landscape."""
    x = np.linspace(-3, 3, 200)
    y = np.linspace(-2, 2, 200)
    X, Y = np.meshgrid(x, y)
    Z = feature_fn(X, Y)
    Z = np.clip(Z, 0, 8)

    surf = ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.85,
                            linewidth=0, antialiased=True, rstride=2, cstride=2)
    ax.contour(X, Y, Z, levels=12, colors="black", linewidths=0.4, alpha=0.4,
               offset=Z.min() - 0.5)

    if feature_pts:
        for (px, py, color, label) in feature_pts:
            pz = feature_fn(px, py)
            ax.scatter([px], [py], [pz + 0.15], c=color, s=120, zorder=10,
                       edgecolors="white", linewidths=2)
            ax.text(px, py, pz + 0.7, label, fontsize=9, color=color,
                    fontweight="bold", ha="center")

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("$w_1$", fontsize=10, labelpad=2)
    ax.set_ylabel("$w_2$", fontsize=10, labelpad=2)
    ax.set_zlabel("Loss", fontsize=10, labelpad=2)
    ax.tick_params(axis="both", labelsize=7)
    ax.tick_params(axis="z", labelsize=7)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1, 0.65, 0.5))


# 1. Local minimum (multiple wells)
def f_local_min(X, Y):
    return ((X ** 2 + Y ** 2) * 0.6
            + np.exp(-((X - 1.5) ** 2 + Y ** 2) * 1.5) * 3
            + np.exp(-((X + 1.5) ** 2 + (Y - 0.8) ** 2) * 1.5) * 1.5)


ax1 = fig.add_subplot(1, 4, 1, projection="3d")
draw_3d_landscape(ax1, "(a) Local Minima\n(多个局部低谷, 易困住)",
                  f_local_min, elev=25, azim=-60, cmap="coolwarm",
                  feature_pts=[(-1.5, 0.8, "red", "local"),
                              (1.5, 0, "red", "local"),
                              (0, 0, "lime", "saddle")])

# 2. Saddle point
def f_saddle(X, Y):
    return (X ** 2 - Y ** 2) * 0.3 + 2 + 0.2 * (X ** 2 + Y ** 2)


ax2 = fig.add_subplot(1, 4, 2, projection="3d")
draw_3d_landscape(ax2, "(b) Saddle Point\n(一方向↑另一方向↓)",
                  f_saddle, elev=20, azim=-30, cmap="plasma",
                  feature_pts=[(0, 0, "red", "saddle")])

# 3. Narrow valley (ravine)
def f_valley(X, Y):
    return 0.05 * X ** 2 + 2.5 * Y ** 2 + 1


ax3 = fig.add_subplot(1, 4, 3, projection="3d")
draw_3d_landscape(ax3, "(c) Narrow Valley\n(横向窄, 纵向慢, SGD 噩梦)",
                  f_valley, elev=30, azim=-50, cmap="viridis",
                  feature_pts=[(0, 0, "red", "min")])

# 4. Plateau
def f_plateau(X, Y):
    return np.minimum(2.5 + np.exp(-((Y) ** 2) * 0.5) * 2,
                      0.5 * X ** 2 + 0.2 * Y ** 2 + 0.5)


ax4 = fig.add_subplot(1, 4, 4, projection="3d")
draw_3d_landscape(ax4, "(d) Plateau\n(平台区, 梯度极小)",
                  f_plateau, elev=40, azim=-70, cmap="cividis",
                  feature_pts=[(0, 0, "red", "min")])

plt.suptitle("Loss Landscape 4 种典型地形的 3D 表面 (matplotlib 真实渲染)",
             fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("assets/figures/3-5-loss-landscape-3d.png", dpi=120, bbox_inches="tight")
print("Saved: assets/figures/3-5-loss-landscape-3d.png")
