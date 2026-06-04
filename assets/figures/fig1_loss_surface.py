"""
图1: 损失曲面示例（3D mesh）
展示两个权重 w1, w2 空间中的MSE损失曲面
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 创建权重网格
w1 = np.linspace(-2, 2, 100)
w2 = np.linspace(-2, 2, 100)
W1, W2 = np.meshgrid(w1, w2)

# MSE损失曲面 (简化的碗形曲面)
# 假设最优点在 (0, 0)
loss = W1**2 + W2**2

# 创建图形
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制曲面
surf = ax.plot_surface(W1, W2, loss, cmap='viridis', alpha=0.8,
                        linewidth=0, antialiased=True)

# 标记最优点
ax.scatter([0], [0], [0], color='red', s=150, marker='*', 
           label='全局最优点', zorder=5)

# 标注
ax.set_xlabel('w₁', fontsize=12, labelpad=10)
ax.set_ylabel('w₂', fontsize=12, labelpad=10)
ax.set_zlabel('损失 J(w)', fontsize=12, labelpad=10)
ax.set_title('损失曲面示例（两个权重 w₁, w₂）', fontsize=14, fontweight='bold')

# 添加颜色条
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='损失值')

# 添加图例
ax.legend(loc='upper right', fontsize=10)

# 设置视角
ax.view_init(elev=25, azim=45)

plt.tight_layout()
plt.savefig('fig1_loss_surface.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("fig1_loss_surface saved.")
