"""
3-5-momentum-weights.py
Visualize the exponential weighted average of past gradients in Momentum.
With beta=0.9, the current step's gradient has weight 1, the previous step has 0.9, etc.
The "effective memory" is approximately 1/(1-beta) = 10 steps.

Output: assets/figures/3-5-momentum-weights.png
"""
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("assets/figures", exist_ok=True)

beta = 0.9
n_steps = 20
weights = np.array([beta ** i for i in range(n_steps)])

# Cumulative weight (sum of past weights relative to total)
cum_weights = np.cumsum(weights) / np.sum(weights)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Left: Weight per step (exponential decay)
ax = axes[0]
ax.bar(range(n_steps), weights, color="steelblue", alpha=0.8, edgecolor="navy")
ax.set_xlabel("距离当前步数 $\\tau$", fontsize=12)
ax.set_ylabel("梯度权重 $\\beta^\\tau$", fontsize=12)
ax.set_title(f"Momentum 的指数衰减权重 (β={beta})", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")
ax.set_xticks(range(0, n_steps, 2))
# Annotate sum
ax.text(n_steps / 2, max(weights) * 0.9,
        f"总权重: $\\sum \\beta^\\tau = \\frac{{1}}{{1-\\beta}} = {1/(1-beta):.1f}$",
        ha="center", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

# Right: Cumulative weight
ax = axes[1]
ax.plot(range(n_steps), cum_weights, "ro-", markersize=8, linewidth=2)
ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="50% 累计权重")
ax.axvline(x=0, color="green", linestyle="--", alpha=0.5)
ax.fill_between(range(n_steps), 0, cum_weights, alpha=0.15, color="orange")
ax.set_xlabel("距离当前步数 $\\tau$", fontsize=12)
ax.set_ylabel("累计权重占比", fontsize=12)
ax.set_title("过去多少步的梯度占 50%?", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.set_xticks(range(0, n_steps, 2))
ax.set_ylim(0, 1.05)
ax.legend(loc="lower right")
# Mark half-life
half_life = np.log(0.5) / np.log(beta)
ax.annotate(f"半衰期 ≈ {half_life:.1f} 步",
            xy=(half_life, 0.5), xytext=(half_life + 3, 0.6),
            fontsize=11, ha="left",
            arrowprops=dict(arrowstyle="->", color="red"))

plt.suptitle("Momentum 指数加权求和 — '有效记忆' ≈ 1/(1-β) = 10 步",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("assets/figures/3-5-momentum-weights.png", dpi=120, bbox_inches="tight")
print("Saved: assets/figures/3-5-momentum-weights.png")
