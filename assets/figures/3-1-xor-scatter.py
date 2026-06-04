"""
XOR问题散点图
展示异或问题的线性不可分特性
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path

# XOR 数据点
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])  # XOR 标签

# 创建图形
fig, ax = plt.subplots(figsize=(8, 6))

# 定义颜色
colors = ['#3498db', '#e74c3c']  # 蓝色为0，红色为1
cmap = ListedColormap(colors)

# 绘制散点
scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, s=200, edgecolors='black', linewidths=2, zorder=5)

# 添加网格和轴
ax.set_xlim(-0.3, 1.3)
ax.set_ylim(-0.3, 1.3)
ax.set_xlabel('$x_1$', fontsize=14)
ax.set_ylabel('$x_2$', fontsize=14)
ax.set_title('XOR 问题：线性不可分', fontsize=16, fontweight='bold')

# 添加网格
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

# 添加图例
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#3498db', edgecolor='black', label='y = 0 (0⊕0=0, 1⊕1=0)'),
                  Patch(facecolor='#e74c3c', edgecolor='black', label='y = 1 (0⊕1=1, 1⊕0=1)')]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

# 添加坐标轴箭头
ax.annotate('', xy=(1.3, 0), xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
ax.annotate('', xy=(0, 1.3), xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

# 添加标签
ax.text(1.25, -0.15, '$x_1$', fontsize=12, ha='center')
ax.text(-0.15, 1.2, '$x_2$', fontsize=12, ha='center')

# 标注数据点
for i, (xi, yi) in enumerate(X):
    offset = 0.12
    ax.text(xi + offset, yi + offset, f'({xi},{yi})', fontsize=9, ha='left', va='bottom')

# 添加说明文字
ax.text(0.5, -0.45, '无法用一条直线分开两类点（线性不可分）', fontsize=11, 
         ha='center', style='italic', color='#555')

plt.tight_layout()
output_dir = Path(__file__).resolve().parent
png_path = output_dir / '3-1-xor-scatter.png'
plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"已保存: {png_path}")
