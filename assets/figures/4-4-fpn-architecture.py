"""
Feature Pyramid Network (FPN) Architecture Diagram
=================================================
Generates a clean side-by-side comparison of:
  (left)   Image Pyramid  (传统多尺度)
  (right)  Feature Pyramid Network (FPN)

Output: assets/figures/4-4-fpn-architecture.png
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def draw_box(ax, x, y, w, h, text, fc='#E8F4FD', ec='#1F77B4', fontsize=10, fontweight='normal'):
    """Draw a rounded box at (x, y) with size (w, h) and centered text."""
    box = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.05',
                          fc=fc, ec=ec, linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight=fontweight, wrap=True)


def draw_arrow(ax, x1, y1, x2, y2, color='#444', style='->', lw=1.5, mutation=15):
    arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, color=color, lw=lw,
                          mutation_scale=mutation, connectionstyle='arc3,rad=0')
    ax.add_patch(arr)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={'width_ratios': [1, 1.2]})

# === Left: Image Pyramid (传统多尺度) ===
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.set_title('传统方案: Image Pyramid (多尺度输入)\n速度慢, 不端到端',
              fontsize=14, fontweight='bold', color='#666')

# 4 image scales (decreasing)
scales = [
    (1.0, '256×256', '#FFE4B5'),
    (0.75, '192×192', '#FFDAB9'),
    (0.55, '128×128', '#FFEBCD'),
    (0.4, '96×96', '#FFEFD5'),
]
for i, (s, sz, color) in enumerate(scales):
    y = 7.5 - i * 1.7
    w = 1.5 * s
    h = 0.6 * s
    x = 2
    draw_box(ax1, x, y, w, h, sz, fc=color, ec='#FF8C00', fontsize=10, fontweight='bold')
    # CNN below
    draw_box(ax1, x - 0.3, y - 1.0, w + 0.6, 0.5, f'CNN', fc='#D3D3D3', ec='#666', fontsize=8)
    # Output
    draw_box(ax1, x - 0.1, y - 1.9, w + 0.2, 0.4, f'特征\n({sz})', fc='#E0E0E0', ec='#666', fontsize=7)
    # Detector
    draw_box(ax1, x - 0.3, y - 2.8, w + 0.6, 0.4, 'Detector', fc='#FFB6C1', ec='#DC143C', fontsize=7)

ax1.text(5, 0.3, 'N×forward = N× cost', ha='center', fontsize=11, color='red', fontweight='bold')

# === Right: FPN ===
ax2.set_xlim(0, 14)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.set_title('FPN: 特征金字塔网络 (Feature Pyramid Network, CVPR 2017)\n单次前向, 多尺度输出, 端到端',
              fontsize=14, fontweight='bold', color='#1F77B4')

# Bottom-up pathway (backbone) - 5 levels
backbone_levels = [
    (1.0, 'C2', 'C2\n1/4', '#1F77B4'),
    (1.5, 'C3', 'C3\n1/8', '#1F77B4'),
    (2.0, 'C4', 'C4\n1/16', '#1F77B4'),
    (2.5, 'C5', 'C5\n1/32', '#1F77B4'),
]
# Input image
draw_box(ax2, 0.5, 0.5, 2, 1.2, '输入图像\nH×W×3', fc='#FFE4B5', ec='#FF8C00', fontsize=11, fontweight='bold')

# C2-C5 (bottom-up)
x_pos = 1.0
for i, (w, name, label, color) in enumerate(backbone_levels):
    y = 0.5
    draw_box(ax2, x_pos, y, 1.5, 1.2, label, fc='#D6E9F8', ec=color, fontsize=10, fontweight='bold')
    if i < len(backbone_levels) - 1:
        draw_arrow(ax2, x_pos + 1.5, y + 0.6, x_pos + 2.2, y + 0.6, color='#1F77B4', lw=2)
    x_pos += 1.7

# 1x1 lateral convs
lat_x = 5.0
for i, (w, name, label, color) in enumerate(backbone_levels):
    y = 0.5
    draw_box(ax2, lat_x + i * 0.5, y, 0.4, 1.2, '1×1', fc='#FFE4B5', ec='#FF8C00', fontsize=7)
    # Arrow from C2-C5 to 1x1
    draw_arrow(ax2, 1.0 + i * 1.7 + 1.5, y + 0.6,
               lat_x + i * 0.5, y + 0.6, color='#888', lw=1)

# Top-down pathway - P2-P5
py_x = 7.5
for i, (w, name, label, color) in enumerate(backbone_levels):
    y = 4.0 + i * 1.0
    pname = f'P{i+2}\n1/{4*(i+1)}'
    draw_box(ax2, py_x, y, 1.8, 0.8, pname, fc='#90EE90', ec='#228B22',
             fontsize=10, fontweight='bold')
    # 3x3 conv after merge
    draw_box(ax2, py_x + 2.0, y, 0.7, 0.8, '3×3', fc='#FFE4B5', ec='#FF8C00', fontsize=7)
    # Final feature
    draw_box(ax2, py_x + 2.9, y, 1.5, 0.8, f'P{i+2}_out\n特征图', fc='#FFB6C1', ec='#DC143C', fontsize=9)
    # Detector head
    draw_box(ax2, py_x + 4.6, y, 1.3, 0.8, 'Detector', fc='#E6E6FA', ec='#663399', fontsize=8)

# Top-down arrows (P5 -> P4 -> P3 -> P2)
for i in range(3):
    y1 = 4.0 + i * 1.0 + 0.4
    y2 = 4.0 + (i+1) * 1.0 + 0.4
    draw_arrow(ax2, py_x + 0.5, y1, py_x + 0.5, y2, color='#228B22', lw=2.5, style='->', mutation=18)

# Lateral connection arrows
for i, (w, name, label, color) in enumerate(backbone_levels):
    y_lateral = 0.5 + 0.6
    y_target = 4.0 + i * 1.0 + 0.4
    # Lateral conv output to top-down
    draw_arrow(ax2, lat_x + i * 0.5 + 0.2, y_lateral,
               py_x + 0.5, y_target, color='#FF8C00', lw=1.2, style='->', mutation=12)

# Annotations
ax2.annotate('Bottom-up\n(下采样)', xy=(3.0, 0.4), fontsize=9, color='#1F77B4', fontweight='bold', ha='center')
ax2.annotate('Top-down\n(上采样)', xy=(7.5, 9.0), fontsize=9, color='#228B22', fontweight='bold', ha='center')
ax2.annotate('Lateral\n(横向连接)', xy=(6.5, 4.5), fontsize=9, color='#FF8C00', fontweight='bold', ha='center', rotation=15)
ax2.annotate('2× Upsample', xy=(8.2, 5.5), fontsize=8, color='#228B22', ha='center', style='italic')

plt.tight_layout()
plt.savefig('assets/figures/4-4-fpn-architecture.png', dpi=120, bbox_inches='tight', facecolor='white')
print('  ✓ Saved: assets/figures/4-4-fpn-architecture.png')
