"""
FPN Family Evolution Timeline
=============================
Compares the structure of 4 feature pyramid / multi-scale fusion methods:
  - FPN (2017)        : Bottom-up + Top-down + Lateral
  - PANet (2018)      : + Bottom-up path augmentation
  - NAS-FPN (2019)    : Neural Architecture Search
  - BiFPN (2020)      : Bidirectional + weighted feature fusion (EfficientDet)

Output: assets/figures/4-4-fpn-family-evolution.png
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def draw_node(ax, x, y, r=0.25, color='#A8DADC', ec='#457B9D', text='', fontsize=8):
    circle = mpatches.Circle((x, y), r, fc=color, ec=ec, linewidth=1.5)
    ax.add_patch(circle)
    if text:
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, fontweight='bold')


def draw_edge(ax, x1, y1, x2, y2, color='#666', style='->', lw=1.2, mutation=10):
    arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, color=color, lw=lw,
                          mutation_scale=mutation, connectionstyle='arc3,rad=0')
    ax.add_patch(arr)


def plot_variant(ax, title, edges, x_left=0.5, x_right=4.5, y_top=4.5, y_bottom=0.5):
    """Plot a feature pyramid variant.
    edges: list of (x1, y1, x2, y2, color, label) for each arrow."""
    ax.set_xlim(-0.2, 6.5)
    ax.set_ylim(-0.5, 5.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)

    # 5 levels (P3-P7 typical) - we'll just show 4 for simplicity
    levels = 4
    for i in range(levels):
        y = y_top - i * (y_top - y_bottom) / (levels - 1)
        # Bottom-up node
        draw_node(ax, 0.5, y, 0.25, '#FFD6A5', '#FF8C00', f'P{i+3}', 9)
        # Top-down node (if exists)
        draw_node(ax, 2.5, y, 0.25, '#A8DADC', '#457B9D', f'P{i+3}', 9)
        # Output node
        draw_node(ax, 4.5, y, 0.25, '#B5E48C', '#52B788', f'Out{i+3}', 9)

    # Draw edges
    for (x1, y1, x2, y2, color, label) in edges:
        draw_edge(ax, x1, y1, x2, y2, color, lw=1.5)
        if label:
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            ax.text(mid_x + 0.1, mid_y, label, fontsize=7, color=color, style='italic')

    # Legend
    legend_elements = [
        mpatches.Patch(color='#FFD6A5', label='Bottom-up (P)'),
        mpatches.Patch(color='#A8DADC', label='Top-down'),
        mpatches.Patch(color='#B5E48C', label='Output (用于检测)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=7, framealpha=0.9)


fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# FPN (2017): Bottom-up + Top-down + Lateral
edges_fpn = []
for i in range(3):
    y1 = 4.5 - i * (4.5 - 0.5) / 3
    y2 = 4.5 - (i + 1) * (4.5 - 0.5) / 3
    edges_fpn.append((0.75, y1, 0.75, y2, '#FF8C00', ''))  # bottom-up
    edges_fpn.append((2.5, y1, 2.5, y2, '#457B9D', '2×'))  # top-down
    edges_fpn.append((0.75, y1, 2.25, y1, '#666', 'lateral'))  # lateral
    edges_fpn.append((2.75, y1, 4.25, y1, '#52B788', ''))  # to output
plot_variant(axes[0, 0], 'FPN (CVPR 2017)\n单向 Top-down + 横向连接', edges_fpn)

# PANet (2018): FPN + Bottom-up path augmentation
edges_panet = []
for i in range(3):
    y1 = 4.5 - i * (4.5 - 0.5) / 3
    y2 = 4.5 - (i + 1) * (4.5 - 0.5) / 3
    edges_panet.append((0.75, y1, 0.75, y2, '#FF8C00', ''))  # bottom-up
    edges_panet.append((2.5, y1, 2.5, y2, '#457B9D', '2×'))  # top-down
    edges_panet.append((0.75, y1, 2.25, y1, '#666', 'lateral'))  # lateral
    edges_panet.append((2.75, y1, 4.25, y1, '#52B788', ''))  # to output
# Add extra bottom-up path
for i in range(3):
    y1 = 4.5 - i * (4.5 - 0.5) / 3
    edges_panet.append((4.5, y1, 4.5, 4.5 - (i + 1) * (4.5 - 0.5) / 3, '#FF1493', 'extra'))  # extra bottom-up
plot_variant(axes[0, 1], 'PANet (CVPR 2018)\nFPN + 额外 Bottom-up 增强路径', edges_panet)

# NAS-FPN (2019): irregular connections
edges_nasfpn = []
for i in range(3):
    y1 = 4.5 - i * (4.5 - 0.5) / 3
    y2 = 4.5 - (i + 1) * (4.5 - 0.5) / 3
    edges_nasfpn.append((0.75, y1, 0.75, y2, '#FF8C00', ''))
    edges_nasfpn.append((2.5, y1, 2.5, y2, '#457B9D', '2×'))
# Add irregular cross-scale connections
edges_nasfpn.append((0.5, 4.5, 2.5, 2.83, '#666', 'skip'))  # cross-scale
edges_nasfpn.append((0.5, 2.83, 2.5, 1.17, '#666', 'skip'))
edges_nasfpn.append((0.5, 1.17, 2.5, 0.5, '#666', 'skip'))
plot_variant(axes[1, 0], 'NAS-FPN (CVPR 2019)\n神经架构搜索, 不规则跨尺度连接', edges_nasfpn)

# BiFPN (2020): Bidirectional + weighted fusion
edges_bifpn = []
for i in range(3):
    y1 = 4.5 - i * (4.5 - 0.5) / 3
    y2 = 4.5 - (i + 1) * (4.5 - 0.5) / 3
    edges_bifpn.append((0.75, y1, 0.75, y2, '#FF8C00', ''))
    edges_bifpn.append((2.5, y1, 2.5, y2, '#457B9D', '2×'))
    edges_bifpn.append((0.75, y1, 2.25, y1, '#666', 'w'))  # weighted lateral
    edges_bifpn.append((2.75, y1, 4.25, y1, '#52B788', ''))
# BiFPN: each node receives from same level + one up + one down
edges_bifpn.append((4.5, 4.5, 2.5, 2.83, '#FF1493', 'extra'))  # extra bottom-up
edges_bifpn.append((4.5, 2.83, 2.5, 1.17, '#FF1493', 'extra'))
plot_variant(axes[1, 1], 'BiFPN (CVPR 2020, EfficientDet)\n双向 + 加权特征融合, 多层堆叠', edges_bifpn)

# Top legend
fig.suptitle('特征金字塔家族演进 (FPN Family Evolution)', fontsize=16, fontweight='bold', y=0.995)
fig.text(0.5, 0.965,
         'FPN 之后的多尺度融合方法: PANet 加深双向, NAS-FPN 搜索连接, BiFPN 加权融合',
         ha='center', fontsize=11, style='italic', color='#444')

plt.tight_layout()
plt.savefig('assets/figures/4-4-fpn-family-evolution.png', dpi=120, bbox_inches='tight', facecolor='white')
print('  ✓ Saved: assets/figures/4-4-fpn-family-evolution.png')
