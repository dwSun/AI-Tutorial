"""
图 11-2: 转置卷积（Transposed Convolution）示意图
用于语义分割中的上采样操作
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.patch.set_facecolor('#F8F9FA')

# ==================== 子图1: 普通卷积 ====================
ax1 = axes[0]
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.set_facecolor('#F8F9FA')
ax1.set_title('(a) 普通卷积 (4×4 → 2×2)', fontsize=12, fontweight='bold', pad=10)

# 输入矩阵 (4x4)
input_data = np.array([[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16]])
cell_size = 0.9
start_x, start_y = 0.5, 5.5

for i in range(4):
    for j in range(4):
        color = '#E8E8E8' if input_data[i, j] % 2 == 0 else '#D4E6F1'
        rect = patches.FancyBboxPatch((start_x + j*cell_size, start_y - i*cell_size),
                                        cell_size-0.05, cell_size-0.05,
                                        boxstyle="round,pad=0.02",
                                        facecolor=color, edgecolor='#666666', linewidth=0.8)
        ax1.add_patch(rect)
        ax1.text(start_x + j*cell_size + cell_size/2 - 0.05,
                start_y - i*cell_size + cell_size/2 - 0.05,
                str(input_data[i, j]), ha='center', va='center', fontsize=7)

ax1.text(2.3, 9.5, '输入 (4×4)', ha='center', va='center', fontsize=10, fontweight='bold')
ax1.text(2.3, 5.0, '↓', ha='center', va='center', fontsize=16)
ax1.text(2.3, 4.6, 'Conv 3×3, stride=2', ha='center', va='center', fontsize=8, style='italic')
ax1.text(2.3, 4.2, '↓', ha='center', va='center', fontsize=16)

# 卷积核示意
ax1.add_patch(patches.FancyBboxPatch((1.3, 2.8), 2.1, 0.7,
                                      boxstyle="round,pad=0.05",
                                      facecolor='#4C78A8', edgecolor='white', linewidth=1))
ax1.text(2.35, 3.15, '卷积核 3×3', ha='center', va='center', fontsize=7, color='white', fontweight='bold')

# 输出矩阵 (2x2)
output_data = np.array([[10, 20], [30, 40]])
start_x2, start_y2 = 0.8, 1.5
for i in range(2):
    for j in range(2):
        color = '#FAD7A0'
        rect = patches.FancyBboxPatch((start_x2 + j*1.3, start_y2 - i*1.3),
                                        1.2, 1.2,
                                        boxstyle="round,pad=0.02",
                                        facecolor=color, edgecolor='#E74C3C', linewidth=1.5)
        ax1.add_patch(rect)
        ax1.text(start_x2 + j*1.3 + 0.6, start_y2 - i*1.3 + 0.6,
                f'{output_data[i,j]}', ha='center', va='center', fontsize=10, fontweight='bold')

ax1.text(2.3, 0.6, '输出 (2×2)', ha='center', va='center', fontsize=10, fontweight='bold', color='#E74C3C')

# 箭头
ax1.annotate('', xy=(2.3, 2.5), xytext=(2.3, 2.8),
            arrowprops=dict(arrowstyle='->', color='#666666', lw=2))

# ==================== 子图2: 转置卷积 ====================
ax2 = axes[1]
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.set_facecolor('#F8F9FA')
ax2.set_title('(b) 转置卷积 (2×2 → 4×4)', fontsize=12, fontweight='bold', pad=10)

# 输入矩阵 (2x2)
input_data2 = np.array([[1, 2], [3, 4]])
start_x3, start_y3 = 0.5, 5.5

for i in range(2):
    for j in range(2):
        color = '#FAD7A0'
        rect = patches.FancyBboxPatch((start_x3 + j*1.4, start_y3 - i*1.4),
                                        1.3, 1.3,
                                        boxstyle="round,pad=0.02",
                                        facecolor=color, edgecolor='#E74C3C', linewidth=1.5)
        ax2.add_patch(rect)
        ax2.text(start_x3 + j*1.4 + 0.65, start_y3 - i*1.4 + 0.65,
                str(input_data2[i, j]), ha='center', va='center', fontsize=10, fontweight='bold')

ax2.text(2.3, 9.5, '输入 (2×2)', ha='center', va='center', fontsize=10, fontweight='bold', color='#E74C3C')
ax2.text(2.3, 5.0, '↓', ha='center', va='center', fontsize=16)
ax2.text(2.3, 4.6, '转置Conv 3×3, stride=2', ha='center', va='center', fontsize=8, style='italic')
ax2.text(2.3, 4.2, '↓', ha='center', va='center', fontsize=16)

# 卷积核示意
ax2.add_patch(patches.FancyBboxPatch((1.3, 2.8), 2.1, 0.7,
                                      boxstyle="round,pad=0.05",
                                      facecolor='#54A24B', edgecolor='white', linewidth=1))
ax2.text(2.35, 3.15, '卷积核 3×3', ha='center', va='center', fontsize=7, color='white', fontweight='bold')

# 输出矩阵 (4x4) - 展示稀疏矩阵乘法
output_data2 = np.array([[1, 0, 2, 0],
                          [0, 0, 0, 0],
                          [3, 0, 4, 0],
                          [0, 0, 0, 0]])
start_x4, start_y4 = 0.5, 1.5
for i in range(4):
    for j in range(4):
        if output_data2[i, j] != 0:
            color = '#D4E6F1'
        else:
            color = '#F8F9FA'
        rect = patches.FancyBboxPatch((start_x4 + j*0.7, start_y4 - i*0.7),
                                        0.65, 0.65,
                                        boxstyle="round,pad=0.02",
                                        facecolor=color, edgecolor='#4C78A8', linewidth=0.8)
        ax2.add_patch(rect)
        if output_data2[i, j] != 0:
            ax2.text(start_x4 + j*0.7 + 0.325, start_y4 - i*0.7 + 0.325,
                    str(output_data2[i, j]), ha='center', va='center', fontsize=7, fontweight='bold')

ax2.text(2.3, 0.4, '输出 (4×4, 稀疏)', ha='center', va='center', fontsize=10, fontweight='bold', color='#4C78A8')

# ==================== 子图3: 转置卷积操作可视化 ====================
ax3 = axes[2]
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)
ax3.axis('off')
ax3.set_facecolor('#F8F9FA')
ax3.set_title('(c) 转置卷积计算过程', fontsize=12, fontweight='bold', pad=10)

# 输入
ax3.text(2.0, 9.5, '输入: 2×2', ha='center', va='center', fontsize=10, fontweight='bold')
input_vals = [1, 2, 3, 4]
positions = [(0.5, 8.0), (2.0, 8.0), (0.5, 6.5), (2.0, 6.5)]
for (x, y), v in zip(positions, input_vals):
    rect = patches.FancyBboxPatch((x, y), 1.2, 1.0,
                                    boxstyle="round,pad=0.05",
                                    facecolor='#FAD7A0', edgecolor='#E74C3C', linewidth=1.5)
    ax3.add_patch(rect)
    ax3.text(x+0.6, y+0.5, str(v), ha='center', va='center', fontsize=12, fontweight='bold')

# 箭头向下
ax3.annotate('', xy=(2.0, 5.0), xytext=(2.0, 5.5),
            arrowprops=dict(arrowstyle='->', color='#666666', lw=2))
ax3.text(2.0, 5.3, '滑动 × 核', ha='center', va='center', fontsize=8, style='italic')

# 卷积核
ax3.add_patch(patches.FancyBboxPatch((1.2, 3.5), 1.6, 1.2,
                                      boxstyle="round,pad=0.05",
                                      facecolor='#54A24B', edgecolor='white', linewidth=1.5))
ax3.text(2.0, 4.1, 'w₀ w₁ w₂\nw₃ w₄ w₅\nw₆ w₇ w₈', ha='center', va='center',
        fontsize=7, color='white', fontweight='bold')

# 展开的矩阵乘法说明
ax3.text(5.5, 9.5, '矩阵乘法视角:', ha='center', va='center', fontsize=10, fontweight='bold')
ax3.text(5.5, 8.5, '输入向量: [1, 2, 3, 4]ᵀ', ha='left', va='center', fontsize=9)
ax3.text(5.5, 7.8, '转置卷积核 Cᵀ:', ha='left', va='center', fontsize=9)
ax3.text(5.5, 7.2, '输出 = Cᵀ × 输入', ha='left', va='center', fontsize=9)

# 展示卷积展开
ax3.add_patch(patches.FancyBboxPatch((5.0, 5.5), 3.5, 2.0,
                                      boxstyle="round,pad=0.1",
                                      facecolor='#E8E8E8', edgecolor='#666666', linewidth=1))
ax3.text(6.75, 6.5, 'Cᵀ (16×4 稀疏矩阵)', ha='center', va='center', fontsize=8, style='italic')

# 输出箭头
ax3.annotate('', xy=(6.75, 3.2), xytext=(6.75, 5.3),
            arrowprops=dict(arrowstyle='->', color='#666666', lw=2))

# 输出结果
ax3.text(6.75, 2.8, '输出: 4×4', ha='center', va='center', fontsize=10, fontweight='bold', color='#4C78A8')
ax3.add_patch(patches.FancyBboxPatch((5.8, 1.5), 1.9, 1.0,
                                      boxstyle="round,pad=0.05",
                                      facecolor='#D4E6F1', edgecolor='#4C78A8', linewidth=1.5))
ax3.add_patch(patches.FancyBboxPatch((7.8, 1.5), 1.9, 1.0,
                                      boxstyle="round,pad=0.05",
                                      facecolor='#D4E6F1', edgecolor='#4C78A8', linewidth=1.5))
ax3.text(6.75, 2.0, '空间维度恢复!', ha='center', va='center', fontsize=9, color='#4C78A8', fontweight='bold')

# 关键说明
ax3.text(5.5, 0.8, '注: 转置卷积不是卷积的逆运算,\n而是恢复空间维度的上采样操作',
        ha='left', va='center', fontsize=8, style='italic', color='#666666',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9C4', edgecolor='#F9A825', alpha=0.8))

plt.suptitle('图 11-2(b) 转置卷积（Transposed Convolution）示意图', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
output_dir = Path(__file__).resolve().parent
plt.savefig(output_dir / 'fig11-2-transposed-conv.png', dpi=150, bbox_inches='tight',
            facecolor='#F8F9FA')
print("Transposed convolution diagram saved!")
plt.show()
