#!/usr/bin/env python3
"""
特征向量几何意义可视化
展示原始向量v经过矩阵变换后方向不变，仅长度缩放λ倍
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

# 设置中文字体
plt.rcParams['axes.unicode_minus'] = False

# 定义矩阵A和特征向量
A = np.array([[2, 0], [0, 1]])  # 对角矩阵，特征值λ1=2, λ2=1
eigenvector = np.array([1, 0])  # 特征向量 v = (1, 0)
eigenvalue = 2  # 特征值 λ = 2

# 变换后的向量
transformed = A @ eigenvector * eigenvalue  # A @ v = λ * v = (2, 0)

# 创建图形
fig, ax = plt.subplots(figsize=(10, 8))

# 设置坐标范围
ax.set_xlim(-0.5, 3.5)
ax.set_ylim(-0.5, 1.5)
ax.set_aspect('equal')

# 绘制网格
ax.grid(True, alpha=0.3, linestyle='--')
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)

# 绘制原始向量 v (蓝色箭头)
arrow_v = FancyArrowPatch((0, 0), (1, 0), 
                           mutation_scale=15, 
                           color='#2196F3', 
                           linewidth=2.5,
                           label='Original vector v⃗')
ax.add_patch(arrow_v)

# 绘制变换后的向量 λ·v (红色箭头)
arrow_av = FancyArrowPatch((0, 0), (2, 0), 
                            mutation_scale=15, 
                            color='#F44336', 
                            linewidth=2.5,
                            label=f'Transformed vector A·v⃗ = {eigenvalue}·v⃗')
ax.add_patch(arrow_av)

# 绘制特征向量方向的虚线辅助线
ax.plot([0, 2.5], [0, 0], 'k--', alpha=0.3, linewidth=1)

# 标注特征值
ax.annotate('λ = 2', xy=(1.5, 0.15), fontsize=14, fontweight='bold', 
            color='#F44336', ha='center')

# 标注原始向量
ax.annotate('v⃗', xy=(0.5, 0.12), fontsize=14, fontweight='bold', 
            color='#2196F3', ha='center')

# 标注变换后向量
ax.annotate('2·v⃗', xy=(1.2, -0.2), fontsize=14, fontweight='bold', 
            color='#F44336', ha='center')

# 添加标题和说明
ax.set_title('Geometric Meaning of Eigenvectors\n(特征向量的几何意义)', 
             fontsize=16, fontweight='bold', pad=20)

# 添加说明文字框
textbox = """Matrix A = [[2, 0], [0, 1]]
Eigenvector v⃗ = (1, 0)
Eigenvalue λ = 2

A·v⃗ = 2·v⃗
Direction unchanged, only scaled"""
ax.text(2.8, 1.0, textbox, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        family='monospace')

# 添加图例
ax.legend(loc='lower right', fontsize=11, framealpha=0.9)

# 添加坐标轴标签
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)

# 移除边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
output_path = Path(__file__).resolve().parent / 'eigenvector_geometry.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print(f"Figure saved to: {output_path}")
