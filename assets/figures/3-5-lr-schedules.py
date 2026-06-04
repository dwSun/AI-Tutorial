"""
3-5-lr-schedules.py
Visualize 5 common learning rate schedules:
  - Step Decay
  - Exponential Decay
  - Cosine Decay
  - Warmup + Cosine
  - OneCycleLR

Output: assets/figures/3-5-lr-schedules.png
"""
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("assets/figures", exist_ok=True)

epochs = np.arange(0, 200)
lr_max = 0.1
lr_min = 0.001
warmup_epochs = 10

# 1. Step Decay
def step_lr(epoch, lr_max=lr_max, step_size=40, gamma=0.5):
    return lr_max * (gamma ** (epoch // step_size))


step_lrs = np.array([step_lr(e) for e in epochs])

# 2. Exponential Decay
def exp_lr(epoch, lr_max=lr_max, gamma=0.98):
    return lr_max * (gamma ** epoch)


exp_lrs = np.array([exp_lr(e) for e in epochs])

# 3. Cosine Decay
def cosine_lr(epoch, lr_max=lr_max, lr_min=lr_min, T=epochs[-1]):
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * epoch / T))


cos_lrs = np.array([cosine_lr(e) for e in epochs])

# 4. Warmup + Cosine
def warmup_cosine(epoch, lr_max=lr_max, lr_min=lr_min, warmup=10, T=epochs[-1]):
    if epoch < warmup:
        return lr_max * epoch / warmup
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * (epoch - warmup) / (T - warmup)))


wc_lrs = np.array([warmup_cosine(e) for e in epochs])

# 5. OneCycleLR
def onecycle(epoch, lr_max=lr_max, lr_min=lr_min, pct_start=0.3, T=epochs[-1]):
    pct = epoch / T
    if pct < pct_start:
        return lr_min + (lr_max - lr_min) * (pct / pct_start)
    else:
        return lr_max - (lr_max - lr_min / 100) * ((pct - pct_start) / (1 - pct_start))


oc_lrs = np.array([onecycle(e) for e in epochs])

# Plot
fig, ax = plt.subplots(figsize=(14, 6.5))

ax.plot(epochs, step_lrs, "C0-", linewidth=2, label="Step Decay (阶梯)", alpha=0.85)
ax.plot(epochs, exp_lrs, "C1-", linewidth=2, label="Exponential (指数平滑)", alpha=0.85)
ax.plot(epochs, cos_lrs, "C2-", linewidth=2.5, label="Cosine (余弦退火)", alpha=0.9)
ax.plot(epochs, wc_lrs, "C3-", linewidth=2.5, label="Warmup + Cosine (现代主流)", alpha=0.9)
ax.plot(epochs, oc_lrs, "C4-", linewidth=2.5, label="OneCycleLR (单周期)", alpha=0.9)

# Annotations
ax.axvline(x=warmup_epochs, color="gray", linestyle=":", alpha=0.6, label="Warmup 结束")
ax.annotate("Warmup\n(0 → max)", xy=(warmup_epochs / 2, 0.06), fontsize=9,
            ha="center", color="darkgreen",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))

# Phase labels
ax.axvspan(0, warmup_epochs, alpha=0.08, color="green")
ax.axvspan(warmup_epochs, 60, alpha=0.04, color="blue")
ax.axvspan(60, 200, alpha=0.08, color="red")

ax.text(5, 0.115, "Warmup", ha="center", fontsize=10, color="green", fontweight="bold")
ax.text(40, 0.115, "稳定训练", ha="center", fontsize=10, color="blue", fontweight="bold")
ax.text(150, 0.115, "Cosine 收尾", ha="center", fontsize=10, color="red", fontweight="bold")

ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Learning Rate", fontsize=12)
ax.set_title("5 大常见学习率策略对比", fontsize=14, fontweight="bold")
ax.legend(loc="upper right", fontsize=10, ncol=2)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 0.13)

plt.tight_layout()
plt.savefig("assets/figures/3-5-lr-schedules.png", dpi=120, bbox_inches="tight")
print("Saved: assets/figures/3-5-lr-schedules.png")
