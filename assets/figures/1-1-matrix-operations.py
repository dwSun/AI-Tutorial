"""
1-1-matrix-operations.py
==========================
矩阵加法、乘法 + 张量流动图
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

mpl.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False


def draw_matrix(ax, mat, pos, color='lightblue', label=None, fontsize=14):
    """在指定位置绘制矩阵"""
    n_rows, n_cols = mat.shape
    box_w = n_cols * 1.0
    box_h = n_rows * 1.0
    x0, y0 = pos
    rect = FancyBboxPatch((x0, y0), box_w, box_h,
                          boxstyle='round,pad=0.05',
                          facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    for i in range(n_rows):
        for j in range(n_cols):
            ax.text(x0 + j + 0.5, y0 + n_rows - i - 0.5,
                    f'{int(mat[i, j])}', ha='center', va='center',
                    fontsize=fontsize, fontweight='bold')
    if label:
        ax.text(x0 + box_w / 2, y0 - 0.4, label,
                ha='center', va='top', fontsize=11, fontweight='bold')


# ============== 矩阵加法 ==============
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.set_xlim(0, 14)
ax.set_ylim(0, 6)
ax.axis('off')

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = A + B
intermediate = np.array([[1+5, 2+6], [3+7, 4+8]])

draw_matrix(ax, A, (0.5, 2.5), color='#FFE4B5', label='A', fontsize=14)
ax.text(2.7, 3.5, '+', ha='center', va='center', fontsize=24, fontweight='bold')
draw_matrix(ax, B, (3.3, 2.5), color='#FFE4B5', label='B', fontsize=14)
ax.text(5.5, 3.5, '=', ha='center', va='center', fontsize=24, fontweight='bold')
draw_matrix(ax, intermediate, (6.1, 2.5), color='#E0F0FF', label='对应元素相加', fontsize=12)
ax.text(8.3, 3.5, '=', ha='center', va='center', fontsize=24, fontweight='bold')
draw_matrix(ax, C, (8.9, 2.5), color='#90EE90', label='C = A + B', fontsize=14)

ax.set_title('矩阵加法：$C_{ij} = A_{ij} + B_{ij}$（维度必须相同）',
             fontsize=13, fontweight='bold', pad=15)

# ============== 矩阵乘法 ==============
ax = axes[1]
ax.set_xlim(0, 16)
ax.set_ylim(0, 6)
ax.axis('off')

A2 = np.array([[1, 2], [3, 4]])
B2 = np.array([[5, 6], [7, 8]])
C2 = A2 @ B2
intermediate2 = np.array([[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]])

draw_matrix(ax, A2, (0.5, 2.5), color='#FFE4B5', label='A (2×2)', fontsize=14)
ax.text(2.7, 3.5, '×', ha='center', va='center', fontsize=24, fontweight='bold')
draw_matrix(ax, B2, (3.3, 2.5), color='#FFE4B5', label='B (2×2)', fontsize=14)
ax.text(5.5, 3.5, '=', ha='center', va='center', fontsize=24, fontweight='bold')
draw_matrix(ax, intermediate2, (6.1, 2.5), color='#E0F0FF',
            label='内积计算', fontsize=11)
ax.text(8.3, 3.5, '=', ha='center', va='center', fontsize=24, fontweight='bold')
draw_matrix(ax, C2, (8.9, 2.5), color='#90EE90', label='C = A × B', fontsize=14)

ax.set_title('矩阵乘法：$C_{ij} = \\sum_k A_{ik} \\cdot B_{kj}$（A 的列数 = B 的行数）',
             fontsize=13, fontweight='bold', pad=15)

plt.suptitle('矩阵基本运算', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/david/AI-Tutorial/assets/figures/1-1-matrix-operations.png',
            dpi=120, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: 1-1-matrix-operations.png")


# ============== 张量流动 ==============
fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.set_xlim(0, 14)
ax2.set_ylim(0, 8)
ax2.axis('off')

boxes = [
    {'pos': (1, 5.5), 'w': 3, 'h': 1.5, 'color': '#FFE4B5',
     'title': '输入张量', 'detail': '(batch, h, w, c)\n例: (32, 224, 224, 3)'},
    {'pos': (5, 5.5), 'w': 3, 'h': 1.5, 'color': '#90EE90',
     'title': '卷积层 Conv2D', 'detail': 'k×k 滤波器\n权重共享'},
    {'pos': (9, 5.5), 'w': 3, 'h': 1.5, 'color': '#87CEEB',
     'title': '激活层 ReLU', 'detail': 'max(0, x)\n非线性'},
    {'pos': (5, 1.5), 'w': 3, 'h': 1.5, 'color': '#90EE90',
     'title': '池化层 MaxPool', 'detail': '2×2 下采样\n空间维度减半'},
    {'pos': (9, 1.5), 'w': 3, 'h': 1.5, 'color': '#DDA0DD',
     'title': '全连接层 FC', 'detail': '展平 + 线性\n分类输出'},
]

for b in boxes:
    rect = FancyBboxPatch(b['pos'], b['w'], b['h'],
                          boxstyle='round,pad=0.1',
                          facecolor=b['color'], edgecolor='black', linewidth=1.5)
    ax2.add_patch(rect)
    ax2.text(b['pos'][0] + b['w']/2, b['pos'][1] + b['h'] - 0.3,
             b['title'], ha='center', va='top', fontsize=12, fontweight='bold')
    ax2.text(b['pos'][0] + b['w']/2, b['pos'][1] + 0.5,
             b['detail'], ha='center', va='center', fontsize=9)

# 箭头
arrows = [
    ((4, 6.25), (5, 6.25)),   # input → conv
    ((8, 6.25), (9, 6.25)),   # conv → relu
    ((6.5, 5.5), (6.5, 3)),    # conv (or relu) → pool
    ((9, 5.5), (9, 3)),       # relu → FC
]
for (x0, y0), (x1, y1) in arrows:
    if x0 == x1 and y0 == y1:
        # 跳过自环
        continue
    arrow = FancyArrowPatch((x0, y0), (x1, y1),
                            arrowstyle='->', mutation_scale=20,
                            color='black', linewidth=2)
    ax2.add_patch(arrow)

# 输出标签
ax2.text(13, 2.25, '→\n预测', ha='left', va='center', fontsize=10, fontweight='bold')

# 标题
ax2.set_title('卷积神经网络中的张量流动', fontsize=15, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/david/AI-Tutorial/assets/figures/1-1-tensor-flow.png',
            dpi=120, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: 1-1-tensor-flow.png")
