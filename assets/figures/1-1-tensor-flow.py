"""
1-1-tensor-flow.py
====================
卷积神经网络中的张量流动
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

mpl.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')

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
    ax.add_patch(rect)
    ax.text(b['pos'][0] + b['w']/2, b['pos'][1] + b['h'] - 0.3,
             b['title'], ha='center', va='top', fontsize=12, fontweight='bold')
    ax.text(b['pos'][0] + b['w']/2, b['pos'][1] + 0.5,
             b['detail'], ha='center', va='center', fontsize=9)

arrows = [
    ((4, 6.25), (5, 6.25)),   # input -> conv
    ((8, 6.25), (9, 6.25)),   # conv -> relu
    ((6.5, 5.5), (6.5, 3)),    # conv -> pool (down)
    ((9, 5.5), (9, 3)),       # relu -> FC (down)
]
for (x0, y0), (x1, y1) in arrows:
    arrow = FancyArrowPatch((x0, y0), (x1, y1),
                            arrowstyle='->', mutation_scale=20,
                            color='black', linewidth=2)
    ax.add_patch(arrow)

ax.text(13, 2.25, '→\n预测', ha='left', va='center', fontsize=10, fontweight='bold')
ax.set_title('卷积神经网络中的张量流动', fontsize=15, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/david/AI-Tutorial/assets/figures/1-1-tensor-flow.png',
            dpi=120, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: 1-1-tensor-flow.png")
