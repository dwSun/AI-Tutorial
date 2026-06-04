"""
3-5-optimizer-convergence.py
Side-by-side convergence comparison of 5 optimizers on a noisy quadratic problem.
Visualizes how fast each optimizer reaches the optimum.

Output: assets/figures/3-5-optimizer-convergence.png
"""
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("assets/figures", exist_ok=True)

np.random.seed(0)
n_steps = 200

# Skewed quadratic: 100x more curvature in one direction (ravine-like)
def grad_fn(theta, noise=0.0):
    g = np.array([0.1 * theta[0], 5.0 * theta[1]])
    if noise > 0:
        g += np.random.randn(2) * noise
    return g


def optimize(name, start, lr, n_steps, beta1=0.9, beta2=0.999, eps=1e-8,
             momentum=0.0, noise=0.05, kind="sgd"):
    theta = start.copy()
    path = [np.linalg.norm(theta)]
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    t = 0
    for _ in range(n_steps):
        g = grad_fn(theta, noise=noise)
        t += 1
        if kind == "sgd":
            theta = theta - lr * g
        elif kind == "momentum":
            m = momentum * m - lr * g
            theta = theta + m
        elif kind == "adagrad":
            v = v + g ** 2
            theta = theta - lr * g / (np.sqrt(v) + eps)
        elif kind == "rmsprop":
            v = 0.9 * v + 0.1 * g ** 2
            theta = theta - lr * g / (np.sqrt(v) + eps)
        elif kind == "adam":
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g ** 2
            mh = m / (1 - beta1 ** t)
            vh = v / (1 - beta2 ** t)
            theta = theta - lr * mh / (np.sqrt(vh) + eps)
        path.append(np.linalg.norm(theta))
    return np.array(path)


start = np.array([3.0, 3.0])
sgd_path = optimize("SGD", start, lr=0.05, n_steps=n_steps, kind="sgd", noise=0.1)
mom_path = optimize("Momentum", start, lr=0.05, n_steps=n_steps, momentum=0.9, kind="momentum", noise=0.1)
adagrad_path = optimize("AdaGrad", start, lr=0.1, n_steps=n_steps, kind="adagrad", noise=0.1)
rmsprop_path = optimize("RMSProp", start, lr=0.1, n_steps=n_steps, kind="rmsprop", noise=0.1)
adam_path = optimize("Adam", start, lr=0.1, n_steps=n_steps, kind="adam", noise=0.1)

fig, ax = plt.subplots(figsize=(13, 6.5))

steps = np.arange(len(sgd_path))
ax.semilogy(steps, sgd_path + 1e-3, "C0-", linewidth=2, label="SGD", alpha=0.85)
ax.semilogy(steps, mom_path + 1e-3, "C1-", linewidth=2, label="SGD + Momentum", alpha=0.85)
ax.semilogy(steps, adagrad_path + 1e-3, "C2-", linewidth=2, label="AdaGrad", alpha=0.85)
ax.semilogy(steps, rmsprop_path + 1e-3, "C3-", linewidth=2, label="RMSProp", alpha=0.85)
ax.semilogy(steps, adam_path + 1e-3, "C4-", linewidth=2.5, label="Adam (工业默认)", alpha=0.95)

ax.set_xlabel("Step", fontsize=12)
ax.set_ylabel("到最优点的距离 $\|\\theta - \\theta^*\|$ (log)", fontsize=12)
ax.set_title("5 种优化器在带噪 ravine 问题上的收敛速度对比",
             fontsize=14, fontweight="bold")
ax.legend(loc="upper right", fontsize=11)
ax.grid(True, alpha=0.3, which="both")

# Annotation: Adam wins early
ax.annotate("Adam 早期下降最快",
            xy=(20, adam_path[20] + 1e-3), xytext=(50, 5),
            fontsize=10, color="purple",
            arrowprops=dict(arrowstyle="->", color="purple"))

# Annotation: SGD steady late
ax.annotate("SGD 后期稳定",
            xy=(180, sgd_path[180] + 1e-3), xytext=(140, 1),
            fontsize=10, color="blue",
            arrowprops=dict(arrowstyle="->", color="blue"))

plt.tight_layout()
plt.savefig("assets/figures/3-5-optimizer-convergence.png", dpi=120, bbox_inches="tight")
print("Saved: assets/figures/3-5-optimizer-convergence.png")
