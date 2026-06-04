"""
AI Hardware Selection Decision Tree
=====================================
A flowchart showing decision points for choosing training hardware.

Output: assets/figures/10-8-selection-decision-tree.png
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


fig, ax = plt.subplots(figsize=(16, 12))
ax.set_xlim(0, 16)
ax.set_ylim(0, 12)
ax.axis('off')


def add_box(ax, x, y, w, h, text, fc='#E8F4FD', ec='#1F77B4', fontsize=10, fontweight='normal'):
    box = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05,rounding_size=0.1',
                          fc=fc, ec=ec, linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight=fontweight)


def add_arrow(ax, x1, y1, x2, y2, label='', color='#444'):
    arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', color=color, lw=1.5,
                          mutation_scale=18)
    ax.add_patch(arr)
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.15, mid_y, label, fontsize=9, color=color,
                fontweight='bold', bbox=dict(facecolor='white', edgecolor='none', pad=1))


# Root
add_box(ax, 6.5, 10.5, 3, 1, '训练什么任务？', fc='#FFE4B5', ec='#FF8C00', fontsize=12, fontweight='bold')

# Branches
add_box(ax, 1, 8.5, 3, 1, 'CNN 分类/小目标检测', fc='#90EE90', ec='#228B22', fontsize=10)
add_box(ax, 6.5, 8.5, 3, 1, '大 CNN / 分割 / 大检测', fc='#FFD700', ec='#FF8C00', fontsize=10)
add_box(ax, 12, 8.5, 3, 1, 'LLM / 多模态', fc='#FFB6C1', ec='#DC143C', fontsize=10)

add_arrow(ax, 8, 10.5, 2.5, 9.5)
add_arrow(ax, 8, 10.5, 8, 9.5)
add_arrow(ax, 8, 10.5, 13.5, 9.5)

# L1: 8 GB
add_box(ax, 0.3, 6.5, 2.5, 1, '笔记本 / 入门\n(Mac M2 / RTX 4060)', fc='#E8F5E9', ec='#4CAF50', fontsize=9)
add_box(ax, 0.3, 4.8, 2.5, 1, '云 Colab / Kaggle\n(免费 T4/P100)', fc='#E8F5E9', ec='#4CAF50', fontsize=9)

add_arrow(ax, 2, 8.5, 1.5, 7.5)
add_arrow(ax, 2.5, 7.5, 1.5, 5.8, label='不想买')

# L2: 16-24 GB
add_box(ax, 4, 6.5, 3, 1, 'RTX 4090 (24GB)\n个人首选 ($1600)', fc='#FFF9C4', ec='#FFA000', fontsize=10, fontweight='bold')
add_box(ax, 4, 4.8, 3, 1, '二手 RTX 3090 (24GB)\n$700-900', fc='#FFF9C4', ec='#FFA000', fontsize=9)
add_box(ax, 4, 3.1, 3, 1, '云 Spot 实例\n(Vast.ai / RunPod)', fc='#FFF9C4', ec='#FFA000', fontsize=9)

add_arrow(ax, 7, 8.5, 5.5, 7.5)
add_arrow(ax, 5.5, 6.5, 5.5, 5.8, label='省钱')
add_arrow(ax, 5.5, 5.8, 5.5, 4.1, label='不想买')

# L3: 40-80 GB
add_box(ax, 8, 6.5, 3, 1, 'A100 80G ($15K)\n/ 4090×2 (24+24GB)', fc='#FFE0B2', ec='#FF6F00', fontsize=10, fontweight='bold')
add_box(ax, 8, 4.8, 3, 1, 'H100 80G ($30K)\n或云租', fc='#FFE0B2', ec='#FF6F00', fontsize=9)
add_box(ax, 8, 3.1, 3, 1, '云 AutoDL / 恒源\n($3-6/小时)', fc='#FFE0B2', ec='#FF6F00', fontsize=9)

add_arrow(ax, 12.5, 8.5, 9.5, 7.5)
add_arrow(ax, 9.5, 6.5, 9.5, 5.8, label='需要更快')
add_arrow(ax, 9.5, 5.8, 9.5, 4.1, label='买不起')

# LLM = 个人训练不了
add_box(ax, 12, 6.5, 3, 1, '70B+ 全参训练\n个人训练不了 ⚠', fc='#FFCDD2', ec='#C62828', fontsize=10, fontweight='bold')
add_box(ax, 12, 4.8, 3, 1, '方案: QLoRA 微调\nRTX 4090 即可', fc='#FFCDD2', ec='#C62828', fontsize=9)
add_box(ax, 12, 3.1, 3, 1, '方案: LoRA 推理\n云租 + 开源权重', fc='#FFCDD2', ec='#C62828', fontsize=9)

add_arrow(ax, 13.5, 6.5, 13.5, 5.8, label='妥协')
add_arrow(ax, 13.5, 5.8, 13.5, 4.1, label='降级')

# Bottom: 综合建议
add_box(ax, 1, 0.8, 14, 1.2, '综合建议:\n• 学习/小实验: Mac M2/M3 + 云 Colab (免费)\n• 严肃研究: RTX 4090 (16K 以内 + 24GB VRAM 是甜点)\n• 工业生产: 云租 H100 / 集群 (按需付费, 不维护)',
        fc='#E1BEE7', ec='#6A1B9A', fontsize=11, fontweight='bold')

ax.set_title('AI 训练硬件选型决策树 (2024-2026)',
             fontsize=15, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('assets/figures/10-8-selection-decision-tree.png', dpi=120, bbox_inches='tight', facecolor='white')
print('  ✓ Saved: assets/figures/10-8-selection-decision-tree.png')
