"""
Training Task vs Hardware Requirements
=====================================
Visualizes VRAM/compute requirements for different training tasks
and which hardware tier can handle them.

Output: assets/figures/10-8-training-task-requirements.png
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np


fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(0, 14)
ax.set_ylim(0, 12)
ax.axis('off')

tasks = [
    # (name, y, vram_min, vram_recommended, color, key_concerns)
    ('CNN 分类\n(ResNet-50)', 10.5, 4,  8,   '#90EE90', ['个人笔记本', '~1 天训练']),
    ('CNN 分类\n(ConvNeXt-L)',  8.7,  8,  16,  '#FFD700', ['RTX 4090', '~1 周训练']),
    ('目标检测\n(YOLOv8s)',     9.6,  6,  12,  '#FFD700', ['RTX 4070+', '1-3 天']),
    ('目标检测\n(COCO 完整)',    7.8,  16, 24,  '#FFA500', ['RTX 4090/A5000', '1-2 周']),
    ('图像分割\n(DeepLabV3+)',   6.9,  8,  16,  '#FFA500', ['RTX 4090', '3-7 天']),
    ('Stable Diffusion\n微调',   6.0,  12, 24,  '#FFA500', ['RTX 4090', '~1 周']),
    ('7B LLM LoRA\n(QLoRA)',     5.1,  16, 24,  '#FF6B6B', ['RTX 4090', '~1-3 天']),
    ('13B LLM LoRA',             4.2,  24, 40,  '#FF6B6B', ['A100 40G / 4090×2', '~1 周']),
    ('70B LLM QLoRA',            3.3,  48, 80,  '#FF1493', ['A100 80G × 2', '~2 周']),
    ('7B LLM 全参微调',          2.4,  60, 100, '#FF1493', ['H100 80G × 4+', '~1-2 周']),
    ('70B LLM 全参微调',         1.5,  300, 800, '#8B0000', ['8×H100 / 集群', '数周 + $$$']),
    ('405B LLM 训练',            0.6,  1000, 2000, '#4B0082', ['个人训练不了\n需千卡级集群', '数月 + $1M+']),
]

hardware_zones = [
    (0, 8,    '入门级 (RTX 4060/4070)',  '#E8F5E9'),
    (8, 16,   '消费级 (RTX 4080/4090)',  '#FFF9C4'),
    (16, 32,  '工作站 (A5000/6000)',     '#FFE0B2'),
    (32, 80,  '数据中心 (A100/L40)',     '#FFCDD2'),
    (80, 200, '高端 (H100/H200)',        '#F8BBD0'),
    (200, 1000, '集群 (B200/H100×N)',    '#E1BEE7'),
]

for x_start, x_end, label, color in hardware_zones:
    ax.add_patch(Rectangle((x_start, 0), x_end - x_start, 12, facecolor=color, alpha=0.3))
    ax.text((x_start + x_end) / 2, 11.5, label, ha='center', fontsize=9,
            fontweight='bold', color='#444')

# Draw tasks
for name, y, vram_min, vram_rec, color, concerns in tasks:
    # Min box
    ax.add_patch(Rectangle((1, y), vram_min, 0.5, facecolor=color, alpha=0.7, edgecolor='black'))
    # Recommended box (extends from min to recommended)
    if vram_rec > vram_min:
        ax.add_patch(Rectangle((1 + vram_min, y), vram_rec - vram_min, 0.5,
                                facecolor=color, alpha=0.3, edgecolor='black', hatch='//'))
    ax.text(0.7, y + 0.25, name, ha='right', va='center', fontsize=8.5, fontweight='bold')
    ax.text(1 + vram_rec + 0.2, y + 0.25, ' ← '.join(concerns), va='center', fontsize=8, color='#222')

# Legend
ax.text(7, 0.3,
        '■ 最小可行配置     ▨ 推荐配置     VRAM 数字 = GB 显存',
        ha='center', fontsize=10, style='italic', color='#555',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))

ax.set_title('训练任务 vs 硬件需求\n(横轴 = VRAM 显存, 纵轴 = 任务难度)',
             fontsize=14, fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig('assets/figures/10-8-training-task-requirements.png', dpi=120, bbox_inches='tight', facecolor='white')
print('  ✓ Saved: assets/figures/10-8-training-task-requirements.png')
