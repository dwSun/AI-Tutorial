"""
图3: 二分类交叉熵（BCE）误差曲面
展示预测概率与损失值的关系
"""
import numpy as np
import matplotlib.pyplot as plt

# 预测概率范围 (0.001到0.999避免log(0))
p = np.linspace(0.001, 0.999, 400)

# BCE for y=1 (正类)
bce_y1 = -np.log(p)

# BCE for y=0 (负类)
bce_y0 = -np.log(1 - p)

# 创建图形
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制两条曲线
ax.plot(p, bce_y1, 'b-', linewidth=2.5, label=r'$y=1$: $-\log(\hat{p})$')
ax.plot(p, bce_y0, 'r-', linewidth=2.5, label=r'$y=0$: $-\log(1-\hat{p})$')

# 填充区域
ax.fill_between(p, 0, bce_y1, alpha=0.15, color='blue')
ax.fill_between(p, 0, bce_y0, alpha=0.15, color='red')

# 标记端点
ax.scatter([1.0], [0], color='blue', s=100, zorder=5, marker='*')
ax.scatter([0.0], [0], color='red', s=100, zorder=5, marker='*')

# 添加预测正确/错误的标注
ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
ax.text(0.52, 5, '预测边界', fontsize=10, color='gray')

# 设置坐标轴
ax.set_xlabel('预测概率 p = P(y=1)', fontsize=12)
ax.set_ylabel('交叉熵损失', fontsize=12)
ax.set_title('二分类交叉熵（Binary Cross Entropy）误差曲面', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 7)

# 添加注释框
textstr = '特点:\n• 预测概率越接近真实标签，损失越低\n• 梯度与预测误差成正比\n• 收敛速度快于MSE'
props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('fig3_bce_cross_entropy.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("fig3_bce_cross_entropy saved.")
