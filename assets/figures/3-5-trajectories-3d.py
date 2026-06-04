"""
3-5-trajectories-3d.py
Show optimizer trajectories in 3D: the (w1, w2) position + loss height.
SGD's noisy path is visually obvious in 3D; Mini-batch's smooth glide contrasts clearly.

Output: assets/figures/3-5-trajectories-3d.png
"""
import os
os.makedirs("assets/figures", exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa


def rosenbrock(x, y):
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def rosenbrock_grad(x, y):
    gx = -2 * (1 - x) - 400 * x * (y - x ** 2)
    gy = 200 * (y - x ** 2)
    return np.array([gx, gy])


def optimize(kind, start, lr, n_iter, noise=0.0, beta=0.9):
    p = start.copy()
    v = np.zeros_like(p)
    traj = [p.copy()]
    for _ in range(n_iter):
        g = rosenbrock_grad(p[0], p[1])
        if noise > 0:
            g += np.random.randn(2) * noise
        if kind == "sgd":
            p = p - lr * g
        elif kind == "momentum":
            v = beta * v - lr * g
            p = p + v
        traj.append(p.copy())
    return np.array(traj)


fig = plt.figure(figsize=(18, 6))

x = np.linspace(-2, 2, 150)
y = np.linspace(-1, 3, 150)
X, Y = np.meshgrid(x, y)
Z = rosenbrock(X, Y)
Z = np.clip(Z, 0, 300)

start = np.array([-1.5, 1.5])
np.random.seed(0)

trajs = {
    "GD\n(全量, 稳)":         optimize("sgd",      start, lr=0.0005, n_iter=500,  noise=0.0),
    "SGD\n(单样本, 噪声)":   optimize("sgd",      start, lr=0.0005, n_iter=300,  noise=0.4),
    "Momentum\n(β=0.9, 滑行)": optimize("momentum", start, lr=0.0005, n_iter=100,  noise=0.0),
}

colors = {"GD\n(全量, 稳)": "C0", "SGD\n(单样本, 噪声)": "C3", "Momentum\n(β=0.9, 滑行)": "C2"}

for i, (name, traj) in enumerate(trajs.items(), 1):
    ax = fig.add_subplot(1, 3, i, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.4,
                     linewidth=0, rstride=4, cstride=4, antialiased=True)
    z = np.array([rosenbrock(p[0], p[1]) for p in traj])
    z = np.clip(z, 0, 300)
    ax.plot(traj[:, 0], traj[:, 1], z, color=colors[name], linewidth=1.8, alpha=0.95)
    ax.scatter(traj[0, 0], traj[0, 1], z[0], c="green", s=80, marker="^", zorder=5)
    ax.scatter(traj[-1, 0], traj[-1, 1], z[-1], c="red", s=120, marker="*", zorder=5)
    ax.set_title(f"({chr(96+i)}) {name}", fontsize=12, fontweight="bold")
    ax.set_xlabel("$w_1$", labelpad=2)
    ax.set_ylabel("$w_2$", labelpad=2)
    ax.set_zlabel("Loss", labelpad=2)
    ax.view_init(elev=30, azim=-55)
    ax.set_box_aspect((1, 1, 0.45))
    ax.tick_params(axis="both", labelsize=7)
    ax.tick_params(axis="z", labelsize=7)

plt.suptitle("3 种优化器在 Rosenbrock 谷地上的 3D 轨迹对比\n(高度 = Loss, 高度抖动 = 噪声)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("assets/figures/3-5-trajectories-3d.png", dpi=120, bbox_inches="tight")
print("Saved: assets/figures/3-5-trajectories-3d.png")
