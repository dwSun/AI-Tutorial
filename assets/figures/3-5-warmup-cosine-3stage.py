"""
3-5-warmup-cosine-3stage.py
Visualize the modern 3-stage training schedule:
  Stage 1: Warmup (0-5% of training)
  Stage 2: Stable training at lr_max
  Stage 3: Cosine decay to lr_min

Output: assets/figures/3-5-warmup-cosine-3stage.png
"""
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("assets/figures", exist_ok=True)

# Parameters
total_steps = 1000
warmup_ratio = 0.05
warmup_steps = int(total_steps * warmup_ratio)
cosine_steps = total_steps - warmup_steps

lr_max = 1e-4
lr_min = 1e-6
lr_warmup_start = 1e-7  # Start from very small


def warmup_cosine_schedule(step, warmup_steps, total_steps, lr_max, lr_min, lr_start):
    if step < warmup_steps:
        # Linear warmup
        return lr_start + (lr_max - lr_start) * (step / warmup_steps)
    else:
        # Cosine decay
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * progress))


steps = np.arange(0, total_steps)
lrs = np.array([warmup_cosine_schedule(s, warmup_steps, total_steps, lr_max, lr_min, lr_warmup_start) for s in steps])

fig, ax = plt.subplots(figsize=(14, 6.5))

# Draw the schedule
ax.plot(steps, lrs, "b-", linewidth=2.5)

# Shade the 3 phases
ax.axvspan(0, warmup_steps, alpha=0.15, color="green", label="Stage 1")
ax.axvspan(warmup_steps, total_steps, alpha=0.06, color="blue")

# Phase dividers
ax.axvline(x=warmup_steps, color="green", linestyle="--", alpha=0.6)
ax.text(warmup_steps / 2, 1.4e-4, "Stage 1\nWarmup (0-5%)", ha="center", va="center", fontsize=11,
        color="darkgreen", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.6))

# Stage 2 / Stage 3 boundary at ~85% (cosine starts kicking in)
stage2_to_3 = int(total_steps * 0.85)
ax.axvline(x=stage2_to_3, color="red", linestyle="--", alpha=0.5)
ax.text((warmup_steps + stage2_to_3) / 2, 1.4e-4, "Stage 2\n稳定训练 (5-85%)", ha="center", va="center",
        fontsize=11, color="darkblue", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.6))
ax.text((stage2_to_3 + total_steps) / 2, 1.4e-4, "Stage 3\nCosine 收尾 (85-100%)", ha="center", va="center",
        fontsize=11, color="darkred", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.6))

# Annotate key points
ax.annotate("lr_max\n(1e-4)", xy=(total_steps / 2, lr_max), xytext=(total_steps / 2 + 50, lr_max * 1.3),
            fontsize=10, ha="left",
            arrowprops=dict(arrowstyle="->", color="red"))
ax.annotate("lr_min\n(1e-6)", xy=(total_steps, lr_min), xytext=(total_steps - 100, lr_min * 3),
            fontsize=10, ha="right",
            arrowprops=dict(arrowstyle="->", color="red"))
ax.annotate("lr_warmup_start\n(1e-7)", xy=(0, lr_warmup_start), xytext=(20, lr_warmup_start * 8),
            fontsize=10, ha="left",
            arrowprops=dict(arrowstyle="->", color="red"))

# Use log y-scale to see the warmup clearly
ax.set_yscale("log")
ax.set_xlabel("Training Step", fontsize=12)
ax.set_ylabel("Learning Rate (log scale)", fontsize=12)
ax.set_title("现代训练流程: Warmup + 稳定训练 + Cosine 收尾 (3 阶段)",
             fontsize=14, fontweight="bold")
ax.grid(True, alpha=0.3, which="both")
ax.set_ylim(1e-8, 5e-4)
ax.set_xlim(-30, total_steps + 30)
ax.legend(loc="lower left", fontsize=11)

plt.tight_layout()
plt.savefig("assets/figures/3-5-warmup-cosine-3stage.png", dpi=120, bbox_inches="tight")
print("Saved: assets/figures/3-5-warmup-cosine-3stage.png")
