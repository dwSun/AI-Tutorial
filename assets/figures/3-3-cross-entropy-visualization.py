"""
3-3-cross-entropy-visualization.py
====================================
交叉熵损失函数可视化：BCE + CCE + MSE vs CE 对比
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

mpl.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

# BCE
y_true = 1
y_pred = np.linspace(0.001, 0.999, 100)
bce_loss = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5))
axes1[0].plot(y_pred, bce_loss, 'b-', linewidth=2)
axes1[0].axvline(x=0.5, color='gray', linestyle=':', alpha=0.5, label='随机猜测 (p=0.5)')
axes1[0].axvline(x=1.0, color='r', linestyle='--', label='完美预测 (p=1.0)')
axes1[0].scatter([1.0], [0], color='r', s=100, zorder=5)
axes1[0].set_xlabel('预测概率 $\\hat{p}$')
axes1[0].set_ylabel('交叉熵损失')
axes1[0].set_title('二分类交叉熵损失 (真实标签=1)')
axes1[0].legend()
axes1[0].grid(True, alpha=0.3)

axes1[1].plot(y_pred, bce_loss, 'b-', linewidth=2)
axes1[1].set_xlabel('预测概率 $\\hat{p}$')
axes1[1].set_ylabel('交叉熵损失')
axes1[1].set_title('交叉熵的"对数惩罚"特性')
axes1[1].grid(True, alpha=0.3)
axes1[1].set_yscale('log')
axes1[1].set_ylim(1e-2, 10)
plt.tight_layout()
plt.savefig('/home/david/AI-Tutorial/assets/figures/3-3-bce.png',
            dpi=120, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: 3-3-bce.png")

# CCE
y_true_3 = np.array([1.0, 0.0, 0.0])
n_points = 50
p1 = np.linspace(0.001, 0.999, n_points)
p2 = np.linspace(0.001, 0.999, n_points)
P1, P2 = np.meshgrid(p1, p2)
P3 = 1 - P1 - P2
mask = P3 >= 0

losses_3 = np.zeros_like(P1)
for i in range(P1.shape[0]):
    for j in range(P1.shape[1]):
        if mask[i, j]:
            y_pred_ij = np.array([P1[i, j], P2[i, j], P3[i, j]])
            losses_3[i, j] = -np.sum(y_true_3 * np.log(y_pred_ij + 1e-10))
        else:
            losses_3[i, j] = np.nan

fig2 = plt.figure(figsize=(12, 5))
ax1 = fig2.add_subplot(121)
contour = ax1.contourf(P1, P2, losses_3, levels=20, cmap='viridis', alpha=0.8)
ax1.plot([0.33, 0.33, 0], [0.33, 0, 0.33], 'k--', alpha=0.5, label='均匀分布')
ax1.scatter([1.0], [0.0], color='r', s=100, label='最优预测 (p1=1.0)')
ax1.set_xlabel('预测概率 $p_1$ (第一类)')
ax1.set_ylabel('预测概率 $p_2$ (第二类)')
ax1.set_title('三分类交叉熵等高线图 (真实=第一类)')
ax1.legend()
plt.colorbar(contour, ax=ax1, label='交叉熵损失')

ax2 = fig2.add_subplot(122, projection='3d')
surf = ax2.plot_surface(P1, P2, losses_3, cmap='viridis', alpha=0.8)
ax2.scatter([1.0], [0.0], [0.0], color='r', s=100, label='最优预测')
ax2.set_xlabel('$p_1$')
ax2.set_ylabel('$p_2$')
ax2.set_zlabel('交叉熵损失')
ax2.set_title('三分类交叉熵损失曲面')
fig2.colorbar(surf, ax=ax2, shrink=0.5, aspect=5)
plt.tight_layout()
plt.savefig('/home/david/AI-Tutorial/assets/figures/3-3-cce.png',
            dpi=120, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: 3-3-cce.png")

# MSE vs CE
y_pred_raw = np.linspace(-5, 5, 100)
y_pred_sigmoid = 1 / (1 + np.exp(-y_pred_raw))
mse_loss = (y_pred_sigmoid - y_true) ** 2
bce_loss_grad = - (y_true * np.log(y_pred_sigmoid) + (1 - y_true) * np.log(1 - y_pred_sigmoid))
mse_grad = 2 * (y_pred_sigmoid - y_true) * y_pred_sigmoid * (1 - y_pred_sigmoid)
bce_grad = (y_pred_sigmoid - y_true)

fig3, axes3 = plt.subplots(2, 2, figsize=(12, 10))
axes3[0, 0].plot(y_pred_raw, mse_loss, 'r-', linewidth=2, label='MSE')
axes3[0, 0].plot(y_pred_raw, bce_loss_grad, 'b-', linewidth=2, label='Cross Entropy')
axes3[0, 0].axvline(x=0, color='gray', linestyle=':', alpha=0.5, label='决策边界')
axes3[0, 0].set_xlabel('模型原始输出 $z$')
axes3[0, 0].set_ylabel('损失值')
axes3[0, 0].set_title('MSE vs Cross Entropy 损失对比')
axes3[0, 0].legend()
axes3[0, 0].grid(True, alpha=0.3)

axes3[0, 1].plot(y_pred_raw, mse_grad, 'r--', linewidth=2, label='MSE 梯度')
axes3[0, 1].plot(y_pred_raw, bce_grad, 'b--', linewidth=2, label='Cross Entropy 梯度')
axes3[0, 1].axvline(x=0, color='gray', linestyle=':', alpha=0.5)
axes3[0, 1].axhline(y=0, color='gray', linestyle=':', alpha=0.5)
axes3[0, 1].set_xlabel('模型原始输出 $z$')
axes3[0, 1].set_ylabel('梯度值')
axes3[0, 1].set_title('梯度对比（分类任务）')
axes3[0, 1].legend()
axes3[0, 1].grid(True, alpha=0.3)

axes3[1, 0].plot(y_pred_raw, y_pred_sigmoid, 'g-', linewidth=2)
axes3[1, 0].axvline(x=0, color='gray', linestyle=':', alpha=0.5)
axes3[1, 0].axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
axes3[1, 0].set_xlabel('模型原始输出 $z$')
axes3[1, 0].set_ylabel('Sigmoid 概率 $\\sigma(z)$')
axes3[1, 0].set_title('Sigmoid 激活函数')
axes3[1, 0].grid(True, alpha=0.3)

epochs = np.arange(1, 21)
mse_convergence = 1.0 / (epochs ** 0.5)
bce_convergence = 1.0 / (epochs ** 1.5)
axes3[1, 1].plot(epochs, mse_convergence, 'ro-', linewidth=2, label='MSE 收敛')
axes3[1, 1].plot(epochs, bce_convergence, 'bo-', linewidth=2, label='Cross Entropy 收敛')
axes3[1, 1].set_xlabel('训练轮数')
axes3[1, 1].set_ylabel('损失值（模拟）')
axes3[1, 1].set_title('收敛速度对比（模拟）')
axes3[1, 1].legend()
axes3[1, 1].grid(True, alpha=0.3)
axes3[1, 1].set_yscale('log')
plt.tight_layout()
plt.savefig('/home/david/AI-Tutorial/assets/figures/3-3-mse-vs-ce.png',
            dpi=120, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: 3-3-mse-vs-ce.png")
