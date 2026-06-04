"""
图2: MSE 误差曲面（二维视图）
展示MSE的二次抛物面特性
"""
import numpy as np
import matplotlib.pyplot as plt

# 创建预测值范围
y_true = 0.0
y_pred = np.linspace(-2, 2, 400)

# MSE损失
mse = (y_pred - y_true)**2

# 创建图形
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制MSE曲线
ax.plot(y_pred, mse, 'b-', linewidth=2.5, label=r'$\mathcal{L}_{MSE} = (\hat{y} - y)^2$')

# 标记最优点
ax.scatter([y_true], [0], color='red', s=150, marker='*', zorder=5,
           label=f'最优点 (y={y_true})')

# 填充区域
ax.fill_between(y_pred, 0, mse, alpha=0.2, color='blue')

# 标注梯度
ax.annotate('梯度 = 2(ŷ - y)', xy=(1, 1), xytext=(1.3, 1.5),
            fontsize=11, arrowprops=dict(arrowstyle='->', color='gray'),
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 设置坐标轴
ax.set_xlabel('预测值 ŷ', fontsize=12)
ax.set_ylabel('损失值', fontsize=12)
ax.set_title('MSE 误差曲面（二次抛物面）', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(-2, 2)
ax.set_ylim(0, 4)

# 添加注释框
textstr = '特点:\n• 处处可导，梯度平滑\n• 对异常值极其敏感\n• 唯一全局最优解'
props = dict(boxstyle='round', facecolor='lightblue', alpha=0.7)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('fig2_mse_2d.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("fig2_mse_2d saved.")
