"""
图 11-2: U-Net 架构图
U-Net: Convolutional Networks for Biomedical Image Segmentation (MICCAI 2015)
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path

fig, ax = plt.subplots(1, 1, figsize=(18, 10))
ax.set_xlim(0, 18)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_facecolor('#F8F9FA')
fig.patch.set_facecolor('#F8F9FA')

# 颜色定义
colors = {
    'encoder': '#4C78A8',      # 蓝色 - 编码器
    'decoder': '#E45756',       # 红色 - 解码器
    'bottleneck': '#72B7B0',    # 青色 - 瓶颈层
    'skip': '#F58518',          # 橙色 - 跳跃连接
    'upconv': '#54A24B',        # 绿色 - 上卷积
    'conv': '#B279A8',          # 紫色 - 卷积
    'arrow': '#2D2D2D',
    'text': '#2D2D2D',
    'label': '#444444',
}

def draw_block(ax, x, y, w, h, channels, label, color, text_color='white'):
    """绘制一个卷积块"""
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                    facecolor=color, edgecolor='white', linewidth=1.5, zorder=3)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, f'{label}\n{channels}ch', ha='center', va='center',
            fontsize=8, fontweight='bold', color=text_color, zorder=4)

def draw_upconv(ax, x, y, w, h, color):
    """绘制上卷积/转置卷积块"""
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                    facecolor=color, edgecolor='white', linewidth=1.5, zorder=3)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, f'UpConv\n2×2', ha='center', va='center',
            fontsize=7, fontweight='bold', color='white', zorder=4)

def draw_skip(ax, x1, y1, x2, y2, label=''):
    """绘制跳跃连接线"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=colors['skip'], lw=1.5,
                               connectionstyle='arc3,rad=0.0'))
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my + 0.15, label, ha='center', va='bottom', fontsize=6, color=colors['skip'])

# ==================== 编码器路径（左侧） ====================
# 输入
ax.text(0.3, 5.0, '输入\n572×572', ha='center', va='center', fontsize=8, color=colors['text'])
ax.annotate('', xy=(1.5, 5.0), xytext=(0.8, 5.0),
            arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))

# Level 1: 64 channels
draw_block(ax, 1.5, 4.4, 1.2, 1.2, '64ch', 'Conv3×3\nReLU', colors['encoder'])
draw_block(ax, 1.5, 3.1, 1.2, 1.2, '64ch', 'Conv3×3\nReLU', colors['encoder'])
ax.annotate('', xy=(1.5, 3.1), xytext=(1.5, 4.4),
            arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
ax.text(2.0, 3.75, 'MaxPool\n2×2', ha='left', va='center', fontsize=6, color=colors['text'])
ax.annotate('', xy=(1.5, 2.7), xytext=(1.5, 3.1),
            arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))

# Level 2: 128 channels
draw_block(ax, 1.5, 1.8, 1.2, 1.2, '128ch', 'Conv3×3\nReLU', colors['encoder'])
draw_block(ax, 1.5, 0.5, 1.2, 1.2, '128ch', 'Conv3×3\nReLU', colors['encoder'])
ax.annotate('', xy=(1.5, 0.5), xytext=(1.5, 1.8),
            arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
ax.text(2.0, 1.15, 'MaxPool\n2×2', ha='left', va='center', fontsize=6, color=colors['text'])
ax.annotate('', xy=(1.5, 2.7), xytext=(1.5, 2.9),
            arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))

# Level 3: 256 channels
draw_block(ax, 3.6, 4.4, 1.2, 1.2, '256ch', 'Conv3×3\nReLU', colors['encoder'])
draw_block(ax, 3.6, 3.1, 1.2, 1.2, '256ch', 'Conv3×3\nReLU', colors['encoder'])
ax.annotate('', xy=(3.6, 3.1), xytext=(3.6, 4.4),
            arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
ax.text(4.1, 3.75, 'MaxPool\n2×2', ha='left', va='center', fontsize=6, color=colors['text'])
ax.annotate('', xy=(3.6, 2.7), xytext=(3.6, 3.1),
            arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))

# Level 4: 512 channels
draw_block(ax, 3.6, 1.8, 1.2, 1.2, '512ch', 'Conv3×3\nReLU', colors['encoder'])
draw_block(ax, 3.6, 0.5, 1.2, 1.2, '512ch', 'Conv3×3\nReLU', colors['encoder'])
ax.annotate('', xy=(3.6, 0.5), xytext=(3.6, 1.8),
            arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
ax.text(4.1, 1.15, 'MaxPool\n2×2', ha='left', va='center', fontsize=6, color=colors['text'])
ax.annotate('', xy=(3.6, 2.7), xytext=(3.6, 2.9),
            arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))

# Level 5: 1024 channels (bottleneck)
draw_block(ax, 5.7, 2.5, 1.2, 2.0, '1024ch', 'Conv3×3\nReLU\n×2', colors['bottleneck'])

# ==================== 解码器路径（右侧） ====================
# UpConv + Concat Level 4
draw_upconv(ax, 7.5, 3.0, 1.0, 1.0, colors['upconv'])
# 拼接块
rect = patches.FancyBboxPatch((8.6, 3.0), 0.6, 1.0, boxstyle="round,pad=0.05",
                                facecolor=colors['skip'], edgecolor='white', linewidth=1.5, zorder=3)
ax.add_patch(rect)
ax.text(8.9, 3.5, 'Concat', ha='center', va='center', fontsize=5, fontweight='bold', color='white', zorder=4)

draw_block(ax, 9.3, 3.0, 1.0, 1.0, '512ch', 'Conv3×3\nReLU', colors['decoder'])
draw_block(ax, 9.3, 4.2, 1.0, 1.0, '512ch', 'Conv3×3\nReLU', colors['decoder'])
ax.annotate('', xy=(9.3, 4.2), xytext=(9.3, 4.0),
            arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))

# UpConv + Concat Level 3
draw_upconv(ax, 10.9, 4.4, 1.0, 1.0, colors['upconv'])
rect = patches.FancyBboxPatch((12.0, 4.4), 0.6, 1.0, boxstyle="round,pad=0.05",
                                facecolor=colors['skip'], edgecolor='white', linewidth=1.5, zorder=3)
ax.add_patch(rect)
ax.text(12.3, 4.9, 'Concat', ha='center', va='center', fontsize=5, fontweight='bold', color='white', zorder=4)

draw_block(ax, 12.7, 4.4, 1.0, 1.0, '256ch', 'Conv3×3\nReLU', colors['decoder'])
draw_block(ax, 12.7, 5.6, 1.0, 1.0, '256ch', 'Conv3×3\nReLU', colors['decoder'])
ax.annotate('', xy=(12.7, 5.6), xytext=(12.7, 5.4),
            arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))

# UpConv + Concat Level 2
draw_upconv(ax, 14.3, 5.8, 1.0, 1.0, colors['upconv'])
rect = patches.FancyBboxPatch((15.4, 5.8), 0.6, 1.0, boxstyle="round,pad=0.05",
                                facecolor=colors['skip'], edgecolor='white', linewidth=1.5, zorder=3)
ax.add_patch(rect)
ax.text(15.7, 6.3, 'Concat', ha='center', va='center', fontsize=5, fontweight='bold', color='white', zorder=4)

draw_block(ax, 16.1, 5.8, 1.0, 1.0, '128ch', 'Conv3×3\nReLU', colors['decoder'])
draw_block(ax, 16.1, 7.0, 1.0, 1.0, '128ch', 'Conv3×3\nReLU', colors['decoder'])
ax.annotate('', xy=(16.1, 7.0), xytext=(16.1, 6.8),
            arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))

# UpConv + Concat Level 1
draw_upconv(ax, 14.3, 7.2, 1.0, 1.0, colors['upconv'])
rect = patches.FancyBboxPatch((15.4, 7.2), 0.6, 1.0, boxstyle="round,pad=0.05",
                                facecolor=colors['skip'], edgecolor='white', linewidth=1.5, zorder=3)
ax.add_patch(rect)
ax.text(15.7, 7.7, 'Concat', ha='center', va='center', fontsize=5, fontweight='bold', color='white', zorder=4)

draw_block(ax, 16.1, 7.2, 1.0, 1.0, '64ch', 'Conv3×3\nReLU', colors['decoder'])
draw_block(ax, 16.1, 8.4, 1.0, 1.0, '64ch', 'Conv3×3\nReLU', colors['decoder'])

# Output 1×1 conv
draw_block(ax, 17.3, 7.5, 0.5, 0.6, '2ch', 'Conv1×1', '#2D2D2D')
ax.annotate('', xy=(17.3, 7.8), xytext=(17.1, 8.4),
            arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
ax.text(17.55, 7.8, '输出\n分割图', ha='center', va='center', fontsize=7, color=colors['text'])

# ==================== 跳跃连接 ====================
# Skip from Level 1 to decoder
ax.annotate('', xy=(8.6, 3.5), xytext=(2.7, 3.1),
            arrowprops=dict(arrowstyle='->', color=colors['skip'], lw=1.2,
                           connectionstyle='arc3,rad=-0.2'))
ax.text(5.5, 2.5, 'Skip\nConnection', ha='center', va='center', fontsize=6, color=colors['skip'],
        fontweight='bold')

# Skip from Level 2
ax.annotate('', xy=(12.0, 4.9), xytext=(2.7, 1.2),
            arrowprops=dict(arrowstyle='->', color=colors['skip'], lw=1.2,
                           connectionstyle='arc3,rad=-0.2'))

# Skip from Level 3
ax.annotate('', xy=(15.4, 6.3), xytext=(4.8, 4.4),
            arrowprops=dict(arrowstyle='->', color=colors['skip'], lw=1.2,
                           connectionstyle='arc3,rad=-0.2'))

# ==================== 标注 ====================
# 编码器标签
ax.text(2.1, 5.5, '编码器\n(Encoder)', ha='center', va='center', fontsize=10,
        color=colors['encoder'], fontweight='bold')
ax.add_patch(patches.FancyBboxPatch((1.3, 0.2), 1.6, 5.4, boxstyle="round,pad=0.1",
                                     facecolor='none', edgecolor=colors['encoder'],
                                     linewidth=2, linestyle='--', alpha=0.5, zorder=1))

# 解码器标签
ax.text(13.5, 8.8, '解码器\n(Decoder)', ha='center', va='center', fontsize=10,
        color=colors['decoder'], fontweight='bold')
ax.add_patch(patches.FancyBboxPatch((7.3, 2.8), 10.3, 6.0, boxstyle="round,pad=0.1",
                                     facecolor='none', edgecolor=colors['decoder'],
                                     linewidth=2, linestyle='--', alpha=0.5, zorder=1))

# 瓶颈层标签
ax.text(6.3, 5.0, '瓶颈层\n(Bottleneck)', ha='center', va='center', fontsize=8,
        color=colors['bottleneck'], fontweight='bold')

# 标题
ax.text(9, 9.6, '图 11-2(a) U-Net 架构图', ha='center', va='center',
        fontsize=14, fontweight='bold', color=colors['text'])

# 图例
legend_elements = [
    patches.Patch(facecolor=colors['encoder'], label='编码器 (Encoder)'),
    patches.Patch(facecolor=colors['decoder'], label='解码器 (Decoder)'),
    patches.Patch(facecolor=colors['bottleneck'], label='瓶颈层 (Bottleneck)'),
    patches.Patch(facecolor=colors['skip'], label='跳跃连接 (Skip Connection)'),
    patches.Patch(facecolor=colors['upconv'], label='上卷积 (UpConv/转置卷积)'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=8, framealpha=0.9)

plt.tight_layout()
output_dir = Path(__file__).resolve().parent
plt.savefig(output_dir / 'fig11-2-unet.png', dpi=150, bbox_inches='tight',
            facecolor='#F8F9FA')
print("U-Net architecture diagram saved!")
plt.show()
