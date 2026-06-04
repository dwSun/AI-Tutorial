"""
3-5-ravine-sgd-vs-momentum.py
Visualize how Momentum helps in a narrow valley (ravine) where SGD oscillates.

Output: assets/figures/3-5-ravine-sgd-vs-momentum.png
"""
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("assets/figures", exist_ok=True)


# Rosenbrock is THE classic "narrow curved valley" function
def rosenbrock(x, y):
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def rosenbrock_grad(x, y):
    gx = -2 * (1 - x) - 400 * x * (y - x ** 2)
    gy = 200 * (y - x ** 2)
    return np.array([gx, gy])


def sgd(grad_fn, start, lr, n_iter, noise=0.0):
    p = start.copy()
    traj = [p.copy()]
    for _ in range(n_iter):
        g = grad_fn(p[0], p[1])
        if noise > 0:
            g += np.random.randn(2) * noise
        p = p - lr * g
        traj.append(p.copy())
    return np.array(traj)


def momentum(grad_fn, start, lr, beta, n_iter, noise=0.0):
    p = start.copy()
    v = np.zeros_like(p)
    traj = [p.copy()]
    for _ in range(n_iter):
        g = grad_fn(p[0], p[1])
        if noise > 0:
            g += np.random.randn(2) * noise
        v = beta * v - lr * g
        p = p + v
        traj.append(p.copy())
    return np.array(traj)


np.random.seed(0)
start = np.array([-1.5, 1.5])
sgd_traj = sgd(rosenbrock_grad, start, lr=0.0008, n_iter=2000, noise=0.0)
mom_traj = momentum(rosenbrock_grad, start, lr=0.0008, beta=0.9, n_iter=200, noise=0.0)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
x = np.linspace(-2, 2, 200)
y = np.linspace(-1, 3, 200)
X, Y = np.meshgrid(x, y)
Z = rosenbrock(X, Y)
levels = np.logspace(-1, 3, 20)

# SGD
ax = axes[0]
ax.contour(X, Y, Z, levels=levels, cmap="viridis", alpha=0.5)
ax.plot(sgd_traj[:, 0], sgd_traj[:, 1], "r-", linewidth=1.0, alpha=0.7)
idx = np.linspace(0, len(sgd_traj) - 1, 50).astype(int)
ax.plot(sgd_traj[idx, 0], sgd_traj[idx, 1], "ro", markersize=3)
ax.plot(sgd_traj[0, 0], sgd_traj[0, 1], "go", markersize=10, label="起点")
ax.plot(sgd_traj[-1, 0], sgd_traj[-1, 1], "b*", markersize=15, label="终点 (2000 步)")
ax.set_title("SGD 在 Rosenbrock 狭长谷地\n(2000 步仍未到达最优点, 反复震荡)",
             fontsize=12)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
ax.grid(True, alpha=0.3)

# Momentum
ax = axes[1]
ax.contour(X, Y, Z, levels=levels, cmap="viridis", alpha=0.5)
ax.plot(mom_traj[:, 0], mom_traj[:, 1], "g-", linewidth=1.5, alpha=0.8)
idx = np.linspace(0, len(mom_traj) - 1, 50).astype(int)
ax.plot(mom_traj[idx, 0], mom_traj[idx, 1], "go", markersize=4)
ax.plot(mom_traj[0, 0], mom_traj[0, 1], "go", markersize=10, label="起点")
ax.plot(mom_traj[-1, 0], mom_traj[-1, 1], "r*", markersize=15, label="终点 (200 步)")
ax.set_title("Momentum (β=0.9) 在同一问题上\n(仅 200 步到达最优点, 沿主轴滑行)",
             fontsize=12)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle("狭长谷地 (Rosenbrock): SGD 反复震荡 vs Momentum 沿主轴滑行",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("assets/figures/3-5-ravine-sgd-vs-momentum.png", dpi=120, bbox_inches="tight")
print("Saved: assets/figures/3-5-ravine-sgd-vs-momentum.png")
