"""
ResNet 残差块结构图
用于第4-3章 经典CNN架构
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = script_dir

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.set_aspect('equal')
ax.axis('off')

# 颜色定义
main_path_color = '#4A90D9'    # 蓝色 - 主路径
shortcut_color = '#50C878'     # 绿色 - Shortcut
conv_color = '#FF6B6B'         # 红色 - Conv层
bn_color = '#FFB347'            # 橙色 - BN层
relu_color = '#9B59B6'         # 紫色 - ReLU
add_color = '#F39C12'          # 金色 - Add操作
arrow_color = '#34495E'        # 深灰 - 箭头

def draw_box(ax, x, y, w, h, color, text, fontsize=9, text_color='white'):
    """绘制圆角矩形"""
    box = FancyBboxPatch((x, y), w, h, 
                         boxstyle="round,pad=0.05,rounding_size=0.2",
                         facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', 
            fontsize=fontsize, color=text_color, fontweight='bold')

def draw_arrow(ax, start, end, color=arrow_color, style='->', label='', label_pos=0.5):
    """绘制箭头"""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=2))
    if label:
        mid = ((start[0] + end[0])/2, (start[1] + end[1])/2)
        ax.text(mid[0], mid[1] + 0.25, label, ha='center', va='bottom', 
                fontsize=8, color=color)

def draw_bracketed_arrow(ax, start, end, text, color=arrow_color):
    """绘制带括号的连接线（用于shortcut）"""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='-', color=color, lw=2, linestyle='--'))

# ============== 标题 ==============
ax.text(6, 7.5, 'ResNet 残差块结构', ha='center', va='center', 
        fontsize=16, fontweight='bold', color='#2C3E50')
ax.text(6, 7.0, 'H(x) = F(x) + x', ha='center', va='center', 
        fontsize=12, style='italic', color='#E74C3C')

# ============== 主路径 (Main Path) ==============
# 输入
draw_box(ax, 0.5, 3.5, 1.2, 1, '#95A5A6', 'Input\nx', 10)

# 主路径箭头
ax.annotate('', xy=(1.8, 4), xytext=(1.7, 4),
            arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2))

# Conv3x3 #1
draw_box(ax, 2.0, 3.2, 1.5, 1.6, conv_color, 'Conv\n3x3', 10)
ax.text(2.75, 2.8, '64 ch', ha='center', va='center', fontsize=7, color='#666')

# BN1
draw_box(ax, 3.7, 3.2, 1.2, 1.6, bn_color, 'BN', 10)
ax.text(4.3, 2.8, 'BatchNorm', ha='center', va='center', fontsize=7, color='#666')

# ReLU1
draw_box(ax, 5.1, 3.2, 1.0, 1.6, relu_color, 'ReLU', 10)

# Conv3x3 #2
draw_box(ax, 6.3, 3.2, 1.5, 1.6, conv_color, 'Conv\n3x3', 10)
ax.text(7.05, 2.8, '64 ch', ha='center', va='center', fontsize=7, color='#666')

# BN2
draw_box(ax, 8.0, 3.2, 1.2, 1.6, bn_color, 'BN', 10)

# ============== Shortcut 路径 ==============
# 从输入到加法的虚线箭头
ax.annotate('', xy=(9.4, 4.4), xytext=(1.7, 4.5),
            arrowprops=dict(arrowstyle='-', color=shortcut_color, lw=2.5, 
                          connectionstyle='arc3,rad=-0.2', linestyle='--'))

ax.text(5.5, 5.3, 'Shortcut / Skip Connection', ha='center', va='center', 
        fontsize=10, color=shortcut_color, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F8F5', edgecolor=shortcut_color))

# ============== 加法操作 ==============
# 画一个圈表示加法
circle = plt.Circle((9.7, 4), 0.4, color=add_color, ec='black', lw=2)
ax.add_patch(circle)
ax.text(9.7, 4, '+', ha='center', va='center', fontsize=14, fontweight='bold', color='white')

# 从BN2到加法的箭头
ax.annotate('', xy=(9.3, 4), xytext=(9.2, 4),
            arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2))

# ============== 输出 ReLU ==============
draw_box(ax, 10.3, 3.2, 1.2, 1.6, relu_color, 'ReLU\n(Output)', 9)

# 从加法到输出ReLU的箭头
ax.annotate('', xy=(10.3, 4), xytext=(10.1, 4),
            arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2))

# ============== 最终输出 ==============
draw_box(ax, 5.5, 1.2, 1.5, 0.8, '#2ECC71', 'Output H(x)', 10)

# 从最后一个ReLU到输出的箭头
ax.annotate('', xy=(6.25, 1.2), xytext=(10.3, 3.2),
            arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2,
                          connectionstyle='angle,angleA=0,angleB=-90,rad=0'))
ax.text(8.0, 2.0, 'F(x) + x', ha='center', va='center', fontsize=9, 
        color='#2C3E50', style='italic')

# ============== 标签说明 ==============
# 图例
legend_x = 0.5
legend_y = 0.8
ax.text(legend_x, legend_y + 0.3, 'Legend:', fontsize=9, fontweight='bold', color='#2C3E50')

legend_items = [
    (conv_color, 'Conv3x3: Convolution'),
    (bn_color, 'BN: Batch Normalization'),
    (relu_color, 'ReLU: Activation'),
    (add_color, '+: Element-wise Add'),
    (shortcut_color, '- - ->: Shortcut'),
]

for i, (color, label) in enumerate(legend_items):
    rect = mpatches.Rectangle((legend_x, legend_y - i*0.25 - 0.1), 0.3, 0.2, 
                               facecolor=color, edgecolor='black', lw=1)
    ax.add_patch(rect)
    ax.text(legend_x + 0.4, legend_y - i*0.25, label, fontsize=8, va='center', color='#2C3E50')

# ============== 说明文字 ==============
explanation = """Workflow:
1. Input x goes to both main path and shortcut
2. Main path: x -> Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN
3. Shortcut: x directly passes (may have 1x1 conv for dim adjustment)
4. Two paths are combined via element-wise addition at the + node
5. Result passes through ReLU to output H(x) = F(x) + x"""

ax.text(10.5, 2.0, explanation, fontsize=7.5, va='top', ha='left', 
        color='#2C3E50', linespacing=1.5,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#FDEBD0', edgecolor='#E67E22', alpha=0.9))

plt.tight_layout()

# 保存到 assets/figures 目录
png_path = os.path.join(output_dir, 'resnet_block.png')

plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')

print(f"Saved: {png_path}")