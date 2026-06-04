"""
3-5-adam-decomposition.py
Visualize Adam's update as: "direction × step size":
  - direction = momentum_hat = signal (which way to go)
  - step size = 1 / (sqrt(v_hat) + eps) = noise-aware step size

Output: assets/figures/3-5-adam-decomposition.png
"""
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("assets/figures", exist_ok=True)

# Simulate Adam on a skewed + noisy loss
np.random.seed(0)
n_steps = 100
theta = np.array([3.0, 3.0])
lr = 0.1
beta1, beta2, eps = 0.9, 0.999, 1e-8

m = np.zeros_like(theta)
v = np.zeros_like(theta)
directions = []  # unit vectors
step_sizes = []
positions = [theta.copy()]

for t in range(1, n_steps + 1):
    g = np.array([0.1 * theta[0] + np.random.randn() * 0.1,
                  5.0 * theta[1] + np.random.randn() * 0.3])
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g ** 2
    mh = m / (1 - beta1 ** t)
    vh = v / (1 - beta2 ** t)
    update = mh / (np.sqrt(vh) + eps)
    directions.append(mh / (np.linalg.norm(mh) + 1e-8))
    step_sizes.append(lr / (np.sqrt(vh) + eps))
    theta = theta - lr * update
    positions.append(theta.copy())

directions = np.array(directions)
step_sizes = np.array(step_sizes)
positions = np.array(positions)

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# Left: Trajectory
ax = axes[0]
x = np.linspace(-4, 4, 100)
y = np.linspace(-4, 4, 100)
X, Y = np.meshgrid(x, y)
Z = 0.05 * X ** 2 + 2.5 * Y ** 2
ax.contour(X, Y, Z, levels=15, cmap="viridis", alpha=0.4)
ax.plot(positions[:, 0], positions[:, 1], "b-", linewidth=1.5, alpha=0.8)
idx = np.linspace(0, len(positions) - 1, 20).astype(int)
ax.plot(positions[idx, 0], positions[idx, 1], "bo", markersize=5)
ax.plot(positions[0, 0], positions[0, 1], "go", markersize=12, label="start")
ax.plot(positions[-1, 0], positions[-1, 1], "r*", markersize=15, label="end")
ax.set_xlabel("$\\theta_1$")
ax.set_ylabel("$\\theta_2$")
ax.set_title("(a) Adam 优化轨迹\n(沿主轴, 跨步稳定)", fontsize=12, fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect("equal")

# Middle: Direction angle over time
ax = axes[1]
angles = np.degrees(np.arctan2(directions[:, 1], directions[:, 0]))
ax.plot(angles, "g-", linewidth=1.5, alpha=0.8)
ax.set_xlabel("Step", fontsize=11)
ax.set_ylabel("更新方向角度 (°)", fontsize=11)
ax.set_title("(b) $\\hat{m}_t$ 方向 (信号)\n决定往哪走", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

# Right: Step size over time
ax = axes[2]
ax.semilogy(step_sizes[:, 0], "r-", linewidth=1.5, label="参数 1 的步长", alpha=0.85)
ax.semilogy(step_sizes[:, 1], "b-", linewidth=1.5, label="参数 2 的步长", alpha=0.85)
ax.set_xlabel("Step", fontsize=11)
ax.set_ylabel("Per-param 步长 (log)", fontsize=11)
ax.set_title("(c) $1/\\sqrt{\\hat{v}_t}$ 步长 (自适应)\n决定每步走多远", fontsize=12, fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3, which="both")

plt.suptitle("Adam 更新分解: 方向 (momentum) × 步长 (自适应)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("assets/figures/3-5-adam-decomposition.png", dpi=120, bbox_inches="tight")
print("Saved: assets/figures/3-5-adam-decomposition.png")
