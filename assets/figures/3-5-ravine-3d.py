"""
3-5-ravine-3d.py
Draw the Rosenbrock 'banana' function in 3D and overlay SGD vs Momentum trajectories
on the 3D surface to show how Momentum 'flies' through the valley.

Output: assets/figures/3-5-ravine-3d.png
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


def sgd(grad_fn, start, lr, n_iter):
    p = start.copy()
    traj = [p.copy()]
    for _ in range(n_iter):
        p = p - lr * grad_fn(p[0], p[1])
        traj.append(p.copy())
    return np.array(traj)


def momentum(grad_fn, start, lr, beta, n_iter):
    p = start.copy()
    v = np.zeros_like(p)
    traj = [p.copy()]
    for _ in range(n_iter):
        g = grad_fn(p[0], p[1])
        v = beta * v - lr * g
        p = p + v
        traj.append(p.copy())
    return np.array(traj)


fig = plt.figure(figsize=(18, 7))

# Surface
x = np.linspace(-1.8, 1.8, 200)
y = np.linspace(-0.5, 2.5, 200)
X, Y = np.meshgrid(x, y)
Z = rosenbrock(X, Y)
Z = np.clip(Z, 0, 200)

# 3D surface
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
surf = ax1.plot_surface(X, Y, Z, cmap="viridis", alpha=0.7,
                        linewidth=0, rstride=3, cstride=3, antialiased=True)
# Add contour on the bottom
ax1.contour(X, Y, Z, levels=15, colors="black", linewidths=0.5, alpha=0.4,
            offset=-5)

start = np.array([-1.5, 1.5])
sgd_traj = sgd(rosenbrock_grad, start, lr=0.0008, n_iter=2000)
sgd_z = [rosenbrock(p[0], p[1]) for p in sgd_traj]
ax1.plot(sgd_traj[:, 0], sgd_traj[:, 1], sgd_z, "r-", linewidth=1.5, alpha=0.8,
        label="SGD (2000 步)")
ax1.scatter(sgd_traj[0, 0], sgd_traj[0, 1], sgd_z[0], c="green", s=100, marker="^",
           label="start", zorder=5)
ax1.scatter(sgd_traj[-1, 0], sgd_traj[-1, 1], sgd_z[-1], c="darkred", s=150, marker="*",
           label="end (未到)", zorder=5)
ax1.set_title("(a) SGD 在 Rosenbrock 谷地 (2000 步仍在谷中)\n沿谷壁反复震荡",
              fontsize=11, fontweight="bold")
ax1.set_xlabel("$w_1$", labelpad=2)
ax1.set_ylabel("$w_2$", labelpad=2)
ax1.set_zlabel("Loss", labelpad=2)
ax1.view_init(elev=30, azim=-50)
ax1.legend(loc="upper right", fontsize=9)
ax1.set_box_aspect((1, 1, 0.5))

# Same surface, but with Momentum trajectory
ax2 = fig.add_subplot(1, 2, 2, projection="3d")
ax2.plot_surface(X, Y, Z, cmap="viridis", alpha=0.7,
                 linewidth=0, rstride=3, cstride=3, antialiased=True)
ax2.contour(X, Y, Z, levels=15, colors="black", linewidths=0.5, alpha=0.4,
             offset=-5)

mom_traj = momentum(rosenbrock_grad, start, lr=0.0008, beta=0.9, n_iter=200)
mom_z = [rosenbrock(p[0], p[1]) for p in mom_traj]
ax2.plot(mom_traj[:, 0], mom_traj[:, 1], mom_z, "g-", linewidth=2.0, alpha=0.9,
        label="Momentum (200 步)")
ax2.scatter(mom_traj[0, 0], mom_traj[0, 1], mom_z[0], c="green", s=100, marker="^",
           label="start", zorder=5)
ax2.scatter(mom_traj[-1, 0], mom_traj[-1, 1], mom_z[-1], c="gold", s=200, marker="*",
           label="end (到达!)", zorder=5)
ax2.set_title("(b) Momentum (β=0.9) 沿谷底直接滑到最优点\n(仅 200 步)",
              fontsize=11, fontweight="bold")
ax2.set_xlabel("$w_1$", labelpad=2)
ax2.set_ylabel("$w_2$", labelpad=2)
ax2.set_zlabel("Loss", labelpad=2)
ax2.view_init(elev=30, azim=-50)
ax2.legend(loc="upper right", fontsize=9)
ax2.set_box_aspect((1, 1, 0.5))

plt.suptitle("Rosenbrock 'Banana' 谷地 3D: SGD 反复震荡 vs Momentum 沿主轴滑行",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("assets/figures/3-5-ravine-3d.png", dpi=120, bbox_inches="tight")
print("Saved: assets/figures/3-5-ravine-3d.png")
