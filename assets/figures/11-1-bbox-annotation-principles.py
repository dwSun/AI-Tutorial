"""
Bounding Box Annotation Principles
==================================
Visualizes tight box vs loose box, occlusion/truncation rules,
and small object handling for object detection annotation.

Output: assets/figures/11-1-bbox-annotation-principles.png
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import numpy as np


np.random.seed(42)
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('目标检测 Bounding Box 标注原则（BBox Annotation Principles）',
             fontsize=15, fontweight='bold', y=0.995)


def draw_bbox(ax, x, y, w, h, color, lw=2, label='', style='-'):
    rect = Rectangle((x, y), w, h, linewidth=lw, edgecolor=color,
                      facecolor='none', linestyle=style)
    ax.add_patch(rect)
    if label:
        ax.text(x, y - 0.04, label, fontsize=9, color=color, fontweight='bold')


def draw_object(ax, x, y, w, h, color='#888', label=''):
    rect = Rectangle((x, y), w, h, linewidth=0, facecolor=color, alpha=0.6)
    ax.add_patch(rect)
    if label:
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=10, color='white', fontweight='bold')


# Panel 1: Tight box (correct) vs Loose box (wrong)
ax = axes[0, 0]
ax.set_xlim(0, 10); ax.set_ylim(0, 8)
ax.set_aspect('equal'); ax.axis('off')
ax.set_title('原则 1: Tight Box（紧框）\n框要紧贴目标可见边界', fontsize=11, fontweight='bold')
draw_object(ax, 1, 1.5, 2, 2.5, color='#666', label='Cat')
draw_bbox(ax, 1, 1.5, 2, 2.5, color='green', lw=2.5, label='✓ Tight (correct)')
draw_bbox(ax, 0.5, 1.0, 3, 3.5, color='red', lw=2, style='--', label='✗ Loose')
ax.text(5, 0.3, 'Tight box: IoU ≈ 1.0 (与 ground truth 完美匹配)\nLoose box: IoU ≈ 0.6 (训练时引入背景噪声)', fontsize=9, ha='center', color='#444')


# Panel 2: Occlusion handling
ax = axes[0, 1]
ax.set_xlim(0, 10); ax.set_ylim(0, 8)
ax.set_aspect('equal'); ax.axis('off')
ax.set_title('原则 2: 遮挡（Occlusion）\n全框标注 + 类别正确（遮挡不算截断）',
             fontsize=11, fontweight='bold')
draw_object(ax, 1, 1, 3, 4, color='#666', label='Person A')
draw_object(ax, 3.5, 1.5, 2.5, 3.5, color='#888', label='Person B')
draw_bbox(ax, 1, 1, 3, 4, color='green', lw=2.5, label='A: full box')
draw_bbox(ax, 3.5, 1.5, 2.5, 3.5, color='green', lw=2.5, label='B: full box')
ax.text(5, 0.3, '即使 B 被 A 遮挡一半\n仍要标出 B 的完整框（推断位置）',
        fontsize=9, ha='center', color='#444')


# Panel 3: Truncation handling
ax = axes[0, 2]
ax.set_xlim(0, 10); ax.set_ylim(0, 8)
ax.set_aspect('equal'); ax.axis('off')
ax.set_title('原则 3: 截断（Truncation）\n标到图像边界 + 标注 truncated=True',
             fontsize=11, fontweight='bold')
ax.add_patch(Rectangle((0, 0), 10, 8, linewidth=0, facecolor='#F5F5F5', zorder=0))
ax.plot([3.5, 3.5], [0, 8], 'k--', lw=1.5)
draw_object(ax, 1, 2, 3, 4, color='#666', label='Car')
draw_bbox(ax, 1, 2, 3, 4, color='green', lw=2.5, label='Full extent')
ax.text(3.5, 6, 'Image\nEdge', ha='center', fontsize=10, color='black', fontweight='bold')
ax.text(5, 0.3, '目标被图像边界切掉时：\n标到边界 + 标记 truncated',
        fontsize=9, ha='center', color='#444')


# Panel 4: Small object handling
ax = axes[1, 0]
ax.set_xlim(0, 10); ax.set_ylim(0, 8)
ax.set_aspect('equal'); ax.axis('off')
ax.set_title('原则 4: 小目标（Small Object）\n<32×32 px 须谨慎, 单独标注规则',
             fontsize=11, fontweight='bold')
# Big object
draw_object(ax, 0.5, 1, 2, 3, color='#666', label='Big')
draw_bbox(ax, 0.5, 1, 2, 3, color='green', lw=2.5)
# Small object
draw_object(ax, 4, 4, 0.4, 0.4, color='#FF6B6B', label='S')
draw_bbox(ax, 4, 4, 0.4, 0.4, color='green', lw=1.5)
ax.text(4.2, 3.5, '<32×32 px\n边界误差 1px = IoU 损失 15%',
        fontsize=9, color='red')
# 标注员偏好示意
ax.annotate('Skip?', xy=(4.2, 4.2), xytext=(6, 5.5),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=10, color='red', fontweight='bold')
ax.text(7, 6, '✗ 不要跳过小目标\n（它们通常很关键）', fontsize=9, color='red', fontweight='bold')


# Panel 5: Inclusion rules
ax = axes[1, 1]
ax.set_xlim(0, 10); ax.set_ylim(0, 8)
ax.set_aspect('equal'); ax.axis('off')
ax.set_title('原则 5: 包含规则\n目标要被框"包含"（不要切掉可见部分）',
             fontsize=11, fontweight='bold')
draw_object(ax, 2, 2, 4, 3, color='#666', label='Dog')
draw_bbox(ax, 2, 2, 4, 3, color='green', lw=2.5, label='✓ Full extent')
draw_bbox(ax, 3, 2.5, 2, 1.5, color='red', lw=2, style='--', label='✗ Cropped')
ax.text(5, 0.3, '包含 = 框的边界 ≥ 目标可见边界（无遗漏）\n不包含 = 框的边界 < 目标可见边界（切掉）',
        fontsize=9, ha='center', color='#444')


# Panel 6: Self-check checklist
ax = axes[1, 2]
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.set_aspect('equal'); ax.axis('off')
ax.set_title('标注员自检 Checklist', fontsize=11, fontweight='bold')

checklist = """
✓ 1. 框是否紧贴目标？
      → IoU ≥ 0.95 (与"理想"框)

✓ 2. 框是否包含全部可见像素？
      → 框的边界 = max(目标像素)

✓ 3. 遮挡目标是否推断全框？
      → 不能因为遮挡就裁切

✓ 4. 截断目标是否标 truncated？
      → 否则模型学不会"边界目标"

✓ 5. 小目标（<32×32）是否漏标？
      → 训练中难例，但必须标

✓ 6. 类别是否互斥？
      → 一辆车不能同时是 "car" + "vehicle"

✓ 7. 重叠目标是否各标各的？
      → 两个人 = 2 个独立框
"""
ax.text(0.3, 9.5, checklist, fontsize=9, family='monospace', va='top',
        bbox=dict(boxstyle='round', facecolor='#FFF3CD', edgecolor='black', lw=1.5))

plt.tight_layout()
plt.savefig('assets/figures/11-1-bbox-annotation-principles.png', dpi=120, bbox_inches='tight', facecolor='white')
print('  ✓ Saved: assets/figures/11-1-bbox-annotation-principles.png')
