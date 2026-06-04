"""
3-5-gd-sgd-minibatch-trajectories.py
Compare GD / SGD / Mini-batch SGD trajectories on the Rosenbrock function.

Output: assets/figures/3-5-gd-sgd-minibatch-trajectories.png
"""
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("assets/figures", exist_ok=True)


def rosenbrock(x, y):
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def rosenbrock_grad(x, y):
    gx = -2 * (1 - x) - 400 * x * (y - x ** 2)
    gy = 200 * (y - x ** 2)
    return np.array([gx, gy])


def gd(grad_fn, start, lr, n_iter, noise=0.0):
    p = start.copy()
    traj = [p.copy()]
    for _ in range(n_iter):
        g = grad_fn(p[0], p[1])
        if noise > 0:
            g += np.random.randn(2) * noise
        p = p - lr * g
        traj.append(p.copy())
    return np.array(traj)


# 3 trajectories: GD (no noise), SGD (big noise), Mini-batch (small noise)
np.random.seed(42)
start = np.array([-1.5, 1.5])
traj_gd = gd(rosenbrock_grad, start, lr=0.0005, n_iter=500, noise=0.0)
traj_sgd = gd(rosenbrock_grad, start, lr=0.0005, n_iter=500, noise=0.4)
traj_mb = gd(rosenbrock_grad, start, lr=0.0005, n_iter=500, noise=0.05)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
x = np.linspace(-2, 2, 200)
y = np.linspace(-1, 3, 200)
X, Y = np.meshgrid(x, y)
Z = rosenbrock(X, Y)
levels = np.logspace(-1, 3, 15)

for ax, traj, title in zip(axes,
                           [traj_gd, traj_sgd, traj_mb],
                           ["GD (全量梯度)\n稳定但慢, 200 步仍未到最优",
                            "SGD (单样本)\n噪声大, 轨迹抖动剧烈",
                            "Mini-batch GD\n又快又稳, 工业首选"]):
    ax.contour(X, Y, Z, levels=levels, cmap="viridis", alpha=0.5)
    ax.plot(traj[:, 0], traj[:, 1], "r-", linewidth=1.2, alpha=0.7)
    # subsample for points
    idx = np.linspace(0, len(traj) - 1, 25).astype(int)
    ax.plot(traj[idx, 0], traj[idx, 1], "ro", markersize=4)
    ax.plot(traj[0, 0], traj[0, 1], "go", markersize=10, label="起点", zorder=5)
    ax.plot(traj[-1, 0], traj[-1, 1], "b*", markersize=15, label="终点", zorder=5)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

plt.suptitle("Rosenbrock 函数上三种梯度下降的轨迹对比", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("assets/figures/3-5-gd-sgd-minibatch-trajectories.png", dpi=120, bbox_inches="tight")
print("Saved: assets/figures/3-5-gd-sgd-minibatch-trajectories.png")
