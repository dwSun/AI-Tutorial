"""
3-5-adagrad-decay.py
Visualize AdaGrad's critical flaw: learning rate monotonically decreases to zero.

Output: assets/figures/3-5-adagrad-decay.png
"""
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("assets/figures", exist_ok=True)

# Simulate AdaGrad vs RMSProp
np.random.seed(0)
n_steps = 200

# Simulated gradients (constant magnitude, like a typical parameter)
grads = np.random.randn(n_steps) * 0.5 + 1.0

# AdaGrad: G_t = G_{t-1} + g_t^2
G_adagrad = np.cumsum(grads ** 2)
lr_adagrad = 0.01 / (np.sqrt(G_adagrad) + 1e-8)

# RMSProp: v_t = rho * v_{t-1} + (1-rho) * g_t^2
rho = 0.9
v = 0.0
v_history = []
for g in grads:
    v = rho * v + (1 - rho) * g ** 2
    v_history.append(v)
v_history = np.array(v_history)
lr_rmsprop = 0.01 / (np.sqrt(v_history) + 1e-8)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Left: Per-step learning rate
ax = axes[0]
ax.plot(lr_adagrad, "r-", linewidth=1.5, label="AdaGrad (累积 → 趋零)")
ax.plot(lr_rmsprop, "b-", linewidth=1.5, label="RMSProp (滑动平均 → 稳定)")
ax.axhline(y=0, color="k", linewidth=0.5)
ax.set_xlabel("训练步数 t", fontsize=11)
ax.set_ylabel("参数的有效学习率", fontsize=11)
ax.set_title("AdaGrad vs RMSProp: 学习率随训练步数变化", fontsize=12, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 0.012)

# Right: 累积统计 vs 滑动平均
ax = axes[1]
ax.plot(G_adagrad, "r-", linewidth=1.5, label="$G_t$ (AdaGrad 累积和, 单调递增)")
ax.plot(v_history, "b-", linewidth=1.5, label="$v_t$ (RMSProp 滑动平均, 收敛到稳态)")
ax.set_xlabel("训练步数 t", fontsize=11)
ax.set_ylabel("梯度平方的累积量", fontsize=11)
ax.set_title("AdaGrad $G_t$ 单调递增 vs RMSProp $v_t$ 收敛稳态", fontsize=12, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.suptitle("AdaGrad 的致命问题: 学习率单调递减到 0 → 训练后期不再更新",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("assets/figures/3-5-adagrad-decay.png", dpi=120, bbox_inches="tight")
print("Saved: assets/figures/3-5-adagrad-decay.png")
