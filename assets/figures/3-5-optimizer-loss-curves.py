"""
3-5-optimizer-loss-curves.py
Visualize the typical training loss curves of different optimizers.
Adam converges faster early; SGD may reach lower final loss; AdamW is balanced.

Output: assets/figures/3-5-optimizer-loss-curves.png
"""
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("assets/figures", exist_ok=True)

np.random.seed(0)
epochs = np.arange(0, 100)


def smooth_curve(epochs, base, decay, noise_scale):
    """Generate a smooth decreasing curve with noise."""
    return base * np.exp(-decay * epochs) + np.random.randn(len(epochs)) * noise_scale * np.exp(-epochs / 30)


# Simulated loss curves
sgd = smooth_curve(epochs, base=4.0, decay=0.025, noise_scale=0.4) + 0.3
sgd_momentum = smooth_curve(epochs, base=3.8, decay=0.035, noise_scale=0.35) + 0.25
adam = smooth_curve(epochs, base=4.2, decay=0.06, noise_scale=0.5) + 0.4
adamw = smooth_curve(epochs, base=4.0, decay=0.05, noise_scale=0.45) + 0.3
rmsprop = smooth_curve(epochs, base=3.9, decay=0.04, noise_scale=0.4) + 0.35

# Ensure curves don't go below zero and follow realistic patterns
sgd = np.maximum(sgd, 0.4)
sgd_momentum = np.maximum(sgd_momentum, 0.35)
adam = np.maximum(adam, 0.6)
adamw = np.maximum(adamw, 0.45)
rmsprop = np.maximum(rmsprop, 0.5)

# Test loss (always higher than train, with more generalization gap for Adam)
sgd_test = sgd + 0.2 + np.random.randn(len(epochs)) * 0.05
sgd_mom_test = sgd_momentum + 0.18 + np.random.randn(len(epochs)) * 0.05
adam_test = adam + 0.5 + np.random.randn(len(epochs)) * 0.08  # Larger gap
adamw_test = adamw + 0.3 + np.random.randn(len(epochs)) * 0.06
rmsprop_test = rmsprop + 0.4 + np.random.randn(len(epochs)) * 0.07

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Train loss
ax = axes[0]
ax.plot(epochs, sgd, "C0-", linewidth=1.5, label="SGD", alpha=0.85)
ax.plot(epochs, sgd_momentum, "C1-", linewidth=1.5, label="SGD + Momentum", alpha=0.85)
ax.plot(epochs, rmsprop, "C2-", linewidth=1.5, label="RMSProp", alpha=0.85)
ax.plot(epochs, adam, "C3-", linewidth=1.5, label="Adam", alpha=0.85)
ax.plot(epochs, adamw, "C4-", linewidth=2.0, label="AdamW (现代默认)", alpha=0.95)

ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Training Loss", fontsize=11)
ax.set_title("训练损失 (Training Loss)\nAdam 早期下降最快, SGD 最终可能更低", fontsize=12, fontweight="bold")
ax.legend(loc="upper right", fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_yscale("log")

# Test loss
ax = axes[1]
ax.plot(epochs, sgd_test, "C0-", linewidth=1.5, label="SGD", alpha=0.85)
ax.plot(epochs, sgd_mom_test, "C1-", linewidth=1.5, label="SGD + Momentum", alpha=0.85)
ax.plot(epochs, rmsprop_test, "C2-", linewidth=1.5, label="RMSProp", alpha=0.85)
ax.plot(epochs, adam_test, "C3-", linewidth=1.5, label="Adam", alpha=0.85)
ax.plot(epochs, adamw_test, "C4-", linewidth=2.0, label="AdamW (现代默认)", alpha=0.95)

ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Test Loss", fontsize=11)
ax.set_title("测试损失 (Test Loss)\nSGD 泛化通常更好, AdamW 在两者间平衡", fontsize=12, fontweight="bold")
ax.legend(loc="upper right", fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_yscale("log")

plt.suptitle("不同优化器的典型训练/测试损失曲线对比 (示意)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("assets/figures/3-5-optimizer-loss-curves.png", dpi=120, bbox_inches="tight")
print("Saved: assets/figures/3-5-optimizer-loss-curves.png")
