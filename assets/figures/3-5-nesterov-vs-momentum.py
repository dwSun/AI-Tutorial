"""
3-5-nesterov-vs-momentum.py
Visualize the difference between Standard Momentum and Nesterov Accelerated Gradient (NAG).
Nesterov "looks ahead" before computing the gradient, leading to better convergence.

Output: assets/figures/3-5-nesterov-vs-momentum.png
"""
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("assets/figures", exist_ok=True)


def simple_quad(x, y):
    """A skewed quadratic that's wider in y direction (simulates ravine)."""
    return 0.05 * x ** 2 + 5 * y ** 2


def simple_grad(x, y):
    return np.array([0.1 * x, 10 * y])


def standard_momentum(grad_fn, start, lr, beta, n_iter):
    p = start.copy()
    v = np.zeros_like(p)
    traj = [p.copy()]
    for _ in range(n_iter):
        g = grad_fn(p[0], p[1])
        v = beta * v - lr * g
        p = p + v
        traj.append(p.copy())
    return np.array(traj)


def nesterov(grad_fn, start, lr, beta, n_iter):
    p = start.copy()
    v = np.zeros_like(p)
    traj = [p.copy()]
    for _ in range(n_iter):
        # Look ahead
        p_lookahead = p + beta * v
        g = grad_fn(p_lookahead[0], p_lookahead[1])
        v = beta * v - lr * g
        p = p + v
        traj.append(p.copy())
    return np.array(traj)


start = np.array([3.0, 2.0])
traj_mom = standard_momentum(simple_grad, start, lr=0.1, beta=0.9, n_iter=25)
traj_nes = nesterov(simple_grad, start, lr=0.1, beta=0.9, n_iter=25)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
x = np.linspace(-4, 4, 200)
y = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(x, y)
Z = simple_quad(X, Y)
levels = np.logspace(-1, 2, 20)

# Standard Momentum
ax = axes[0]
ax.contour(X, Y, Z, levels=levels, cmap="viridis", alpha=0.5)
ax.plot(traj_mom[:, 0], traj_mom[:, 1], "b-o", markersize=4, linewidth=1.2, alpha=0.8)
# Highlight the "look ahead" overshoot
ax.annotate("先按速度移动\n再在临时位置算梯度",
            xy=(traj_mom[8, 0], traj_mom[8, 1]),
            xytext=(-3, -1.5), fontsize=10,
            arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
ax.set_title("Standard Momentum\n(边走边看, 容易冲过最优点再返回)",
             fontsize=12)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.plot(traj_mom[0, 0], traj_mom[0, 1], "go", markersize=10, label="起点")
ax.plot(traj_mom[-1, 0], traj_mom[-1, 1], "r*", markersize=15, label="终点")
ax.legend()
ax.grid(True, alpha=0.3)

# Nesterov
ax = axes[1]
ax.contour(X, Y, Z, levels=levels, cmap="viridis", alpha=0.5)
ax.plot(traj_nes[:, 0], traj_nes[:, 1], "g-o", markersize=4, linewidth=1.2, alpha=0.8)
# Highlight the corrected path
ax.annotate("先看一步\n再决定怎么走",
            xy=(traj_nes[8, 0], traj_nes[8, 1]),
            xytext=(-3, -1.5), fontsize=10,
            arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
ax.set_title("Nesterov Accelerated Gradient\n(先看一步, 提前修正, 更精准)",
             fontsize=12)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.plot(traj_nes[0, 0], traj_nes[0, 1], "go", markersize=10, label="起点")
ax.plot(traj_nes[-1, 0], traj_nes[-1, 1], "r*", markersize=15, label="终点")
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle("Momentum vs Nesterov: 同等问题上的轨迹对比",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("assets/figures/3-5-nesterov-vs-momentum.png", dpi=120, bbox_inches="tight")
print("Saved: assets/figures/3-5-nesterov-vs-momentum.png")
