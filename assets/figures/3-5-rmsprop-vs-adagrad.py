"""
3-5-rmsprop-vs-adagrad.py
Side-by-side visualization of AdaGrad's cumulative sum vs RMSProp's exponential
moving average. Both compute sqrt(G_t) for the per-parameter learning rate, but
RMSProp's exponential moving average is bounded, while AdaGrad's cumulative sum
grows unboundedly.

Output: assets/figures/3-5-rmsprop-vs-adagrad.png
"""
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("assets/figures", exist_ok=True)

np.random.seed(42)
n_steps = 300

# Realistic gradient sequence: occasional large gradients, mostly small
grads = np.concatenate([
    np.random.randn(50) * 0.3 + 0.5,
    np.random.randn(50) * 0.8 + 1.0,  # spike
    np.random.randn(200) * 0.4 + 0.5,
])

# AdaGrad: G_t = G_{t-1} + g_t^2
G = np.zeros(n_steps)
G[0] = grads[0] ** 2
for t in range(1, n_steps):
    G[t] = G[t - 1] + grads[t] ** 2

# RMSProp: v_t = rho * v_{t-1} + (1 - rho) * g_t^2
rho = 0.9
v = np.zeros(n_steps)
v[0] = grads[0] ** 2
for t in range(1, n_steps):
    v[t] = rho * v[t - 1] + (1 - rho) * grads[t] ** 2

# Effective per-parameter learning rate
lr_max = 0.01
lr_adagrad = lr_max / (np.sqrt(G) + 1e-8)
lr_rmsprop = lr_max / (np.sqrt(v) + 1e-8)

fig, axes = plt.subplots(2, 2, figsize=(15, 9))

# Top left: gradients
ax = axes[0, 0]
ax.plot(grads, "k-", linewidth=0.8, alpha=0.7)
ax.set_xlabel("Step", fontsize=11)
ax.set_ylabel("Gradient $g_t$", fontsize=11)
ax.set_title("(a) 原始梯度序列 (含一次大波峰)", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.axvspan(50, 100, alpha=0.15, color="red", label="梯度波峰区间")

# Top right: G_t vs v_t
ax = axes[0, 1]
ax.plot(G, "r-", linewidth=1.5, label="$G_t$ (AdaGrad 累积)")
ax.plot(v, "b-", linewidth=1.5, label="$v_t$ (RMSProp 滑动平均)")
ax.set_xlabel("Step", fontsize=11)
ax.set_ylabel("梯度平方累积量", fontsize=11)
ax.set_title("(b) AdaGrad 单调递增 vs RMSProp 收敛稳态", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Bottom left: effective learning rate (log scale)
ax = axes[1, 0]
ax.semilogy(lr_adagrad, "r-", linewidth=1.5, label="AdaGrad")
ax.semilogy(lr_rmsprop, "b-", linewidth=1.5, label="RMSProp")
ax.set_xlabel("Step", fontsize=11)
ax.set_ylabel("Per-param LR (log scale)", fontsize=11)
ax.set_title("(c) 有效学习率随时间变化", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which="both")

# Bottom right: gradient spike response
ax = axes[1, 1]
# Zoom in on the spike region
spike_start = 40
spike_end = 150
ax.plot(range(spike_start, spike_end), lr_adagrad[spike_start:spike_end], "r-", linewidth=2,
        label="AdaGrad: 单调下降, 不再回升")
ax.plot(range(spike_start, spike_end), lr_rmsprop[spike_start:spike_end], "b-", linewidth=2,
        label="RMSProp: 滑动平均会回升")
ax.axvspan(50, 100, alpha=0.15, color="red", label="梯度波峰")
ax.set_xlabel("Step", fontsize=11)
ax.set_ylabel("Per-param LR (log scale)", fontsize=11)
ax.set_title("(d) 大梯度波峰后: AdaGrad 永久降低, RMSProp 恢复", fontsize=12, fontweight="bold")
ax.legend(fontsize=10, loc="lower left")
ax.grid(True, alpha=0.3, which="both")
ax.semilogy()

plt.suptitle("AdaGrad (累积) vs RMSProp (滑动平均) — 为什么 RMSProp 不会停摆",
             fontsize=14, fontweight="bold", y=1.00)
plt.tight_layout()
plt.savefig("assets/figures/3-5-rmsprop-vs-adagrad.png", dpi=120, bbox_inches="tight")
print("Saved: assets/figures/3-5-rmsprop-vs-adagrad.png")
