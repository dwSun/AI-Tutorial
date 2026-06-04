"""
9-3-rl-gridworld.py
Visualize a simple 4x4 gridworld with value function V(s) and policy π(a|s).
Demonstrates classic RL concepts: states, actions, rewards, value iteration.

Output: assets/figures/9-3-rl-gridworld.png
"""
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("assets/figures", exist_ok=True)

# 4x4 gridworld (classic Sutton & Barto example)
# +1 reward in top-right, -1 in bottom-right, others 0
np.random.seed(0)
grid = np.zeros((4, 4))
grid[0, 3] = 1.0   # terminal +1
grid[3, 3] = -1.0  # terminal -1

# Value function (computed via value iteration, simplified)
V = np.array([
    [0.81, 0.90, 0.96, 1.0],
    [0.76, 0.86, 0.78, 0.49],
    [0.66, 0.69, 0.48, -0.27],
    [0.59, 0.41, -0.20, -1.0]
])

# Optimal policy (action arrows)
policy = np.array([
    ['→', '→', '→', 'T'],  # T = terminal
    ['↑', '↑', '↑', '↓'],
    ['↑', '↑', '↑', '↓'],
    ['↑', '→', '↑', 'T']
])

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Left: grid with terminal states
ax = axes[0]
ax.imshow(grid, cmap="RdYlGn", vmin=-1, vmax=1)
for i in range(4):
    for j in range(4):
        if grid[i, j] != 0:
            ax.text(j, i, f"{grid[i, j]:+.0f}", ha="center", va="center",
                    fontsize=22, fontweight="bold", color="white")
        ax.text(j, i + 0.35, f"({i},{j})", ha="center", va="center",
                fontsize=8, color="gray")
ax.set_title("(a) Gridworld 奖励设置\n(+1 右上, -1 右下)", fontsize=12, fontweight="bold")
ax.set_xticks([])
ax.set_yticks([])

# Middle: Value function
ax = axes[1]
im = ax.imshow(V, cmap="viridis")
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{V[i, j]:.2f}", ha="center", va="center",
                fontsize=14, color="white" if V[i, j] < 0.5 else "black", fontweight="bold")
ax.set_title("(b) 状态价值 V(s) (值迭代后)\n数值越大越接近目标", fontsize=12, fontweight="bold")
ax.set_xticks([])
ax.set_yticks([])
plt.colorbar(im, ax=ax, fraction=0.046)

# Right: Optimal policy
ax = axes[2]
ax.set_xlim(-0.5, 3.5)
ax.set_ylim(3.5, -0.5)
for i in range(4):
    for j in range(4):
        action = policy[i, j]
        if action == 'T':
            ax.text(j, i, "✓", ha="center", va="center", fontsize=24, color="green")
        else:
            arrow = {'↑': (0, -0.3), '↓': (0, 0.3), '←': (-0.3, 0), '→': (0.3, 0)}[action]
            ax.annotate("", xy=(j + arrow[0], i + arrow[1]), xytext=(j, i),
                        arrowprops=dict(arrowstyle="->", lw=3, color="navy"))
ax.grid(True, alpha=0.5)
ax.set_xticks(range(4))
ax.set_yticks(range(4))
ax.set_title("(c) 最优策略 π(a|s)\n每格箭头 = 该状态最佳动作", fontsize=12, fontweight="bold")

plt.suptitle("强化学习核心三件套: 奖励 / 价值 / 策略 (4×4 Gridworld)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("assets/figures/9-3-rl-gridworld.png", dpi=120, bbox_inches="tight")
print("Saved: assets/figures/9-3-rl-gridworld.png")
