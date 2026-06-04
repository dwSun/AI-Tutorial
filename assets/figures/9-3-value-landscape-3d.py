"""
9-3-value-landscape-3d.py
Draw a 3D value function landscape for a simple Gridworld.
Each cell has a height = V(s), making it visually clear which cells are
"good" (tall green bars) and which are "bad" (short red bars).

Output: assets/figures/9-3-value-landscape-3d.png
"""
import os
os.makedirs("assets/figures", exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa


# 4x4 gridworld value function (from value iteration)
V = np.array([
    [0.81, 0.90, 0.96, 1.0],
    [0.76, 0.86, 0.78, 0.49],
    [0.66, 0.69, 0.48, -0.27],
    [0.59, 0.41, -0.20, -1.0]
])

# Terminal states
term_pos = [(0, 3, "+1"), (3, 3, "-1")]

fig = plt.figure(figsize=(14, 10))

# Left: 3D bar chart of V(s)
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
xpos, ypos = np.meshgrid(range(4), range(4))
xpos = xpos.flatten()
ypos = ypos.flatten()
zpos = np.zeros_like(xpos, dtype=float)

# Color bars by value
colors = []
for v in V.flatten():
    if v > 0.7:
        colors.append("#4CAF50")  # green for high value
    elif v > 0:
        colors.append("#FFC107")  # yellow for positive
    elif v > -0.5:
        colors.append("#FF9800")  # orange for negative
    else:
        colors.append("#F44336")  # red for very negative

dx = dy = 0.6
dz = V.flatten()
ax1.bar3d(xpos - dx/2, ypos - dy/2, zpos, dx, dy, dz,
          color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)

# Annotate values
for i in range(4):
    for j in range(4):
        v = V[i, j]
        ax1.text(j, i, v + (0.05 if v > 0 else -0.15),
                f"{v:+.2f}", ha="center", va="bottom" if v > 0 else "top",
                fontsize=10, fontweight="bold",
                color="white" if abs(v) > 0.5 else "black")

ax1.set_xlabel("Column")
ax1.set_ylabel("Row")
ax1.set_zlabel("V(s)")
ax1.set_title("(a) Value Function 3D 柱状图\n(高度=价值, 颜色=正负)", fontsize=12, fontweight="bold")
ax1.view_init(elev=30, azim=-50)
ax1.set_box_aspect((1, 1, 0.6))

# Right: 3D value surface with optimal policy
ax2 = fig.add_subplot(1, 2, 2, projection="3d")
X, Y = np.meshgrid(range(4), range(4))
surf = ax2.plot_surface(X, Y, V, cmap="RdYlGn", alpha=0.75,
                        linewidth=0.5, edgecolor="black",
                        vmin=-1, vmax=1)

# Add optimal policy arrows on the surface
policy = np.array([
    ['→', '→', '→', 'T'],
    ['↑', '↑', '↑', '↓'],
    ['↑', '↑', '↑', '↓'],
    ['↑', '→', '↑', 'T']
])
for i in range(4):
    for j in range(4):
        action = policy[i, j]
        if action == 'T':
            ax2.scatter([j], [i], [V[i, j] + 0.05], c="black", s=200, marker='*')
        else:
            # Draw an arrow on the surface
            arrow = {'↑': (0, -0.4), '↓': (0, 0.4), '←': (-0.4, 0), '→': (0.4, 0)}[action]
            ax2.quiver(j, i, V[i, j] + 0.02,
                       arrow[0], arrow[1], 0,
                       color="navy", arrow_length_ratio=0.3, linewidth=2)

ax2.set_xlabel("Column")
ax2.set_ylabel("Row")
ax2.set_zlabel("V(s)")
ax2.set_title("(b) Value Surface + 最优策略\n(箭头 = 最佳动作)", fontsize=12, fontweight="bold")
ax2.view_init(elev=35, azim=-45)
ax2.set_box_aspect((1, 1, 0.6))
fig.colorbar(surf, ax=ax2, fraction=0.04, pad=0.04, label="V(s)")

plt.suptitle("Gridworld 价值函数 3D 可视化: 高价值=绿山, 低价值=红谷",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("assets/figures/9-3-value-landscape-3d.png", dpi=120, bbox_inches="tight")
print("Saved: assets/figures/9-3-value-landscape-3d.png")
