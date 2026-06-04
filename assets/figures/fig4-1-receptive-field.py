"""
图 4-1: 感受野扩大示意图
Receptive Field Expansion in CNN Layers
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.patches import FancyBboxPatch

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(1, 1, figsize=(14, 6))
ax.set_xlim(0, 14)
ax.set_ylim(0, 6)
ax.axis('off')
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('#f8f9fa')

# 颜色定义
colors = {
    'input': '#e3f2fd',      # 浅蓝 - 输入层
    'layer1': '#bbdefb',     # 蓝色 - 第1层
    'layer2': '#90caf9',     # 中蓝 - 第2层
    'layer3': '#64b5f6',     # 深蓝 - 第3层
    'kernel': '#fff3e0',     # 浅橙 - 卷积核
    'output': '#ffecb3',     # 橙黄 - 输出点
    'arrow': '#455a64',      # 深灰 - 箭头
    'text': '#263238',       # 深灰 - 文字
    'rf_border': '#d32f2f',  # 红色 - 感受野边框
}

# ========== Layer 1 (左) ==========
L1_center_x, L1_center_y = 2.2, 3.2

# 输入 7x7 网格 (感受野 3x3 覆盖)
for i in range(7):
    for j in range(7):
        x = L1_center_x - 3 + j * 0.5
        y = L1_center_y - 3 + i * 0.5
        color = colors['input']
        # 中心 3x3 高亮为感受野
        if 2 <= i <= 4 and 2 <= j <= 4:
            color = colors['layer1']
        rect = patches.Rectangle((x, y), 0.45, 0.45, 
                                  linewidth=0.5, 
                                  edgecolor='#90a4ae',
                                  facecolor=color)
        ax.add_patch(rect)

# 标注感受野
rf1_rect = patches.Rectangle((L1_center_x - 1.25, L1_center_y - 1.25), 1.5, 1.5,
                               linewidth=2.5, edgecolor=colors['rf_border'],
                               facecolor='none', linestyle='--')
ax.add_patch(rf1_rect)

# 3x3 卷积核标注
kernel1 = patches.FancyBboxPatch((6.2, L1_center_y - 0.9), 0.9, 1.8,
                                   boxstyle="round,pad=0.05",
                                   facecolor=colors['kernel'],
                                   edgecolor='#e65100', linewidth=1.5)
ax.add_patch(kernel1)
ax.text(6.65, L1_center_y, '3×3\nKernel', ha='center', va='center',
        fontsize=8, fontweight='bold', color='#bf360c')

# 输出点 Layer 1
out1 = patches.Circle((L1_center_x, L1_center_y - 2.3), 0.15,
                       facecolor=colors['output'], edgecolor='#f57c00', linewidth=2)
ax.add_patch(out1)

# 箭头 from Layer1 to output
ax.annotate('', xy=(L1_center_x, L1_center_y - 2.1), 
            xytext=(L1_center_x, L1_center_y - 1.5),
            arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))

# Layer 1 标签
ax.text(L1_center_x, L1_center_y + 2.1, 'Layer 1\n(Conv 3×3)', 
        ha='center', va='center', fontsize=10, fontweight='bold', color=colors['text'])
ax.text(L1_center_x, L1_center_y - 2.7, 'Output', ha='center', va='center', fontsize=9, color='#5d4037')
ax.text(L1_center_x, L1_center_y + 3.6, 'Input Feature Map / Receptive Field', 
        ha='center', va='center', fontsize=8, style='italic', color='#546e7a')

# ========== Layer 2 (中) ==========
L2_center_x, L2_center_y = 6.5, 3.2

# 5x5 网格 (Layer 2 的感受野 5x5)
for i in range(5):
    for j in range(5):
        x = L2_center_x - 2 + j * 0.8
        y = L2_center_y - 2 + i * 0.8
        color = colors['input']
        # Layer 1 的输出 3x3 区域
        if 1 <= i <= 3 and 1 <= j <= 3:
            color = colors['layer1']
        # Layer 2 卷积核覆盖区域 3x3
        if 1 <= i <= 3 and 1 <= j <= 3:
            color = colors['layer2']
        rect = patches.Rectangle((x, y), 0.75, 0.75, 
                                  linewidth=0.8, 
                                  edgecolor='#90a4ae',
                                  facecolor=color)
        ax.add_patch(rect)

# Layer 2 卷积核
kernel2 = patches.FancyBboxPatch((9.3, L2_center_y - 1.2), 1.2, 2.4,
                                   boxstyle="round,pad=0.05",
                                   facecolor=colors['kernel'],
                                   edgecolor='#e65100', linewidth=1.5)
ax.add_patch(kernel2)
ax.text(9.9, L2_center_y, '3×3\nKernel', ha='center', va='center',
        fontsize=9, fontweight='bold', color='#bf360c')

# 标注 Layer 2 感受野 (5x5)
rf2_rect = patches.Rectangle((L2_center_x - 2.05, L2_center_y - 2.05), 4.1, 4.1,
                               linewidth=2.5, edgecolor=colors['rf_border'],
                               facecolor='none', linestyle='--')
ax.add_patch(rf2_rect)

# 输出点 Layer 2
out2 = patches.Circle((L2_center_x, L2_center_y - 3.0), 0.18,
                       facecolor=colors['output'], edgecolor='#f57c00', linewidth=2)
ax.add_patch(out2)

# 箭头 from Layer2 to output
ax.annotate('', xy=(L2_center_x, L2_center_y - 2.75), 
            xytext=(L2_center_x, L2_center_y - 2.2),
            arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2))

ax.text(L2_center_x, L2_center_y + 2.6, 'Layer 2\n(Conv 3×3)', 
        ha='center', va='center', fontsize=10, fontweight='bold', color=colors['text'])
ax.text(L2_center_x, L2_center_y - 3.5, 'Output', ha='center', va='center', fontsize=9, color='#5d4037')

# ========== Layer 3 (右) ==========
L3_center_x, L3_center_y = 11.0, 3.2

# 7x7 网格
for i in range(7):
    for j in range(7):
        x = L3_center_x - 3 + j * 0.85
        y = L3_center_y - 3 + i * 0.85
        color = colors['input']
        # Layer 2 输出 5x5
        if 1 <= i <= 5 and 1 <= j <= 5:
            color = colors['layer2']
        # Layer 3 卷积核覆盖 3x3
        if 2 <= i <= 4 and 2 <= j <= 4:
            color = colors['layer3']
        rect = patches.Rectangle((x, y), 0.8, 0.8, 
                                  linewidth=0.8, 
                                  edgecolor='#90a4ae',
                                  facecolor=color)
        ax.add_patch(rect)

# Layer 3 感受野 (7x7)
rf3_rect = patches.Rectangle((L3_center_x - 3.05, L3_center_y - 3.05), 6.1, 6.1,
                               linewidth=2.5, edgecolor=colors['rf_border'],
                               facecolor='none', linestyle='--')
ax.add_patch(rf3_rect)

# Layer 3 卷积核 (右侧)
kernel3 = patches.FancyBboxPatch((13.1, L3_center_y - 1.2), 1.2, 2.4,
                                   boxstyle="round,pad=0.05",
                                   facecolor=colors['kernel'],
                                   edgecolor='#e65100', linewidth=1.5)
ax.add_patch(kernel3)
ax.text(13.7, L3_center_y, '3×3\nKernel', ha='center', va='center',
        fontsize=9, fontweight='bold', color='#bf360c')

# 输出点 Layer 3
out3 = patches.Circle((L3_center_x, L3_center_y - 3.6), 0.2,
                       facecolor=colors['output'], edgecolor='#f57c00', linewidth=2)
ax.add_patch(out3)

ax.annotate('', xy=(L3_center_x, L3_center_y - 3.35), 
            xytext=(L3_center_x, L2_center_y - 2.75),
            arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2, 
                           connectionstyle='arc3,rad=0'))

ax.text(L3_center_x, L3_center_y + 3.6, 'Layer 3\n(Conv 3×3)', 
        ha='center', va='center', fontsize=10, fontweight='bold', color=colors['text'])
ax.text(L3_center_x, L3_center_y - 4.1, 'Output', ha='center', va='center', fontsize=9, color='#5d4037')

# ========== 右侧感受野标注栏 ==========
# RF 尺寸标注
ax.text(0.5, 1.0, 'Receptive Field Size:', fontsize=10, fontweight='bold', color=colors['text'])
ax.text(0.5, 0.5, f'Layer 1: 3×3 → Layer 2: 5×5 → Layer 3: 7×7', 
        fontsize=9, color='#455a64')

# 公式
ax.text(7, 0.5, 'RF_n = (RF_{n-1} - 1) × stride + kernel_size', 
        fontsize=9, style='italic', color='#37474f',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#b0bec5'))

# 标题
fig.suptitle('Receptive Field Expansion in Convolutional Neural Networks', 
             fontsize=14, fontweight='bold', y=0.98, color='#1a237e')
ax.set_title('Each layer\'s output pixel sees an increasingly larger region of the input image', 
             fontsize=10, color='#546e7a', style='italic', pad=10)

# 图例
legend_elements = [
    patches.Patch(facecolor=colors['input'], edgecolor='#90a4ae', label='Input/Background'),
    patches.Patch(facecolor=colors['layer1'], edgecolor='#90a4ae', label='Layer 1 RF (3×3)'),
    patches.Patch(facecolor=colors['layer2'], edgecolor='#90a4ae', label='Layer 2 RF (5×5)'),
    patches.Patch(facecolor=colors['layer3'], edgecolor='#90a4ae', label='Layer 3 RF (7×7)'),
    patches.Patch(facecolor=colors['kernel'], edgecolor='#e65100', label='Conv Kernel'),
    patches.Patch(facecolor=colors['output'], edgecolor='#f57c00', label='Output Point'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=8, 
          framealpha=0.9, edgecolor='#90a4ae')

plt.tight_layout()
plt.savefig('fig4-1-receptive-field.png', dpi=150, bbox_inches='tight', 
            facecolor='#f8f9fa', edgecolor='none')
print("Saved: fig4-1-receptive-field.png")
