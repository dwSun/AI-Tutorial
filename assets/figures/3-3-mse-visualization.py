"""
3-3-mse-visualization.py
=========================
MSE 损失函数可视化：1D 损失曲线 + 2D 损失曲面
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

mpl.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

# 1D MSE
y_true = 2.0
y_pred = np.linspace(-2, 6, 100)
losses = (y_pred - y_true) ** 2

fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5))
axes1[0].plot(y_pred, losses, 'b-', linewidth=2)
axes1[0].axvline(x=y_true, color='r', linestyle='--', label=f'真实值 y={y_true}')
axes1[0].scatter([y_true], [0], color='r', s=100, zorder=5)
axes1[0].set_xlabel('预测值 $\\hat{y}$')
axes1[0].set_ylabel('MSE 损失值')
axes1[0].set_title('MSE 损失函数（一维）')
axes1[0].legend()
axes1[0].grid(True, alpha=0.3)

gradients = 2 * (y_pred - y_true)
axes1[1].plot(y_pred, gradients, 'g-', linewidth=2)
axes1[1].axhline(y=0, color='gray', linestyle=':', alpha=0.5)
axes1[1].axvline(x=y_true, color='r', linestyle='--')
axes1[1].set_xlabel('预测值 $\\hat{y}$')
axes1[1].set_ylabel('梯度值')
axes1[1].set_title('MSE 梯度（与误差成正比）')
axes1[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/david/AI-Tutorial/assets/figures/3-3-mse-1d.png',
            dpi=120, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: 3-3-mse-1d.png")

# 2D MSE Surface
w1_true, w2_true = 2.0, 1.5
w1_range = np.linspace(0, 4, 50)
w2_range = np.linspace(0, 3, 50)
W1, W2 = np.meshgrid(w1_range, w2_range)

np.random.seed(42)
X = np.random.randn(10, 2)
y_true_vals = w1_true * X[:, 0] + w2_true * X[:, 1] + np.random.randn(10) * 0.1

losses_2d = np.zeros_like(W1)
for i in range(W1.shape[0]):
    for j in range(W1.shape[1]):
        y_pred_ij = W1[i, j] * X[:, 0] + W2[i, j] * X[:, 1]
        losses_2d[i, j] = np.mean((y_pred_ij - y_true_vals) ** 2)

fig2 = plt.figure(figsize=(12, 5))
ax1 = fig2.add_subplot(121, projection='3d')
surf = ax1.plot_surface(W1, W2, losses_2d, cmap='viridis', alpha=0.8)
ax1.scatter([w1_true], [w2_true], [np.min(losses_2d)], color='r', s=100, label='最优参数')
ax1.set_xlabel('参数 $w_1$')
ax1.set_ylabel('参数 $w_2$')
ax1.set_zlabel('MSE 损失')
ax1.set_title('MSE 损失曲面（二次抛物面）')
ax1.legend()
fig2.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)

ax2 = fig2.add_subplot(122)
contour = ax2.contour(W1, W2, losses_2d, levels=20, cmap='viridis')
ax2.scatter(w1_true, w2_true, color='r', s=100, label='最优参数')
ax2.set_xlabel('参数 $w_1$')
ax2.set_ylabel('参数 $w_2$')
ax2.set_title('MSE 等高线图')
ax2.legend()
plt.colorbar(contour, ax=ax2)
plt.tight_layout()
plt.savefig('/home/david/AI-Tutorial/assets/figures/3-3-mse-2d-surface.png',
            dpi=120, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: 3-3-mse-2d-surface.png")
