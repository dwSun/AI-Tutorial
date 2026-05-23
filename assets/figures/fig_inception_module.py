"""
Inception Module Visualization
Shows the parallel multi-scale convolution structure in GoogLeNet
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import numpy as np

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis('off')
ax.set_title('Inception Module (GoogLeNet)\nParallel Multi-Scale Feature Extraction', 
             fontsize=14, fontweight='bold', pad=20)

# Colors
input_color = '#E3F2FD'
conv1_color = '#FFCDD2'
conv3_color = '#C8E6C9'
conv5_color = '#BBDEFB'
pool_color = '#F3E5F5'
concat_color = '#FFE0B2'

def draw_box(ax, x, y, w, h, color, label, sublabel=''):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05',
                                   linewidth=1.5, edgecolor='#333', facecolor=color)
    ax.add_patch(rect)
    if sublabel:
        ax.text(x+w/2, y+h/2+0.15, label, ha='center', va='center', fontsize=9, fontweight='bold')
        ax.text(x+w/2, y+h/2-0.2, sublabel, ha='center', va='center', fontsize=7, color='#555')
    else:
        ax.text(x+w/2, y+h/2, label, ha='center', va='center', fontsize=9, fontweight='bold')

def draw_arrow(ax, x1, y1, x2, y2, color='#333'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

# Input
draw_box(ax, 0.5, 3.2, 1.5, 1.6, input_color, 'Input', '28×28×192')
ax.annotate('', xy=(2.2, 4), xytext=(2, 4),
            arrowprops=dict(arrowstyle='->', color='#333', lw=2))

# Branch 1: 1×1 Conv
draw_box(ax, 2.5, 3.5, 1.5, 0.8, conv1_color, '1×1 Conv', '64')
ax.annotate('', xy=(7.5, 3.9), xytext=(4, 3.9),
            arrowprops=dict(arrowstyle='->', color='#1976D2', lw=1.5))

# Branch 2: 1×1 Conv → 3×3 Conv
draw_box(ax, 2.5, 2.2, 1.2, 0.6, conv1_color, '1×1 Conv', '96')
draw_box(ax, 2.5, 1.3, 1.2, 0.7, conv3_color, '3×3 Conv', '128')
ax.plot([3.7, 4.5], [2.5, 1.65], 'k-', lw=1)
ax.annotate('', xy=(4.5, 1.65), xytext=(4.5, 2.5),
            arrowprops=dict(arrowstyle='->', color='#388E3C', lw=1.5))
ax.annotate('', xy=(7.5, 1.65), xytext=(5.5, 1.65),
            arrowprops=dict(arrowstyle='->', color='#388E3C', lw=1.5))

# Branch 3: 1×1 Conv → 5×5 Conv
draw_box(ax, 5.5, 2.5, 1.2, 0.6, conv1_color, '1×1 Conv', '16')
draw_box(ax, 5.5, 1.3, 1.5, 0.7, conv5_color, '5×5 Conv', '32')
ax.plot([6.7, 7.3], [2.8, 1.65], 'k-', lw=1)
ax.annotate('', xy=(7.3, 1.65), xytext=(7.3, 2.8),
            arrowprops=dict(arrowstyle='->', color='#1976D2', lw=1.5))
ax.annotate('', xy=(7.5, 1.65), xytext=(7.3, 1.65),
            arrowprops=dict(arrowstyle='->', color='#1976D2', lw=1.5))

# Branch 4: 3×3 MaxPool → 1×1 Conv
draw_box(ax, 5.5, 3.7, 1.5, 0.7, pool_color, '3×3 MaxPool', 'stride=1')
draw_box(ax, 5.5, 4.6, 1.2, 0.6, conv1_color, '1×1 Conv', '32')
ax.plot([7, 7.3], [4.35, 4.9], 'k-', lw=1)
ax.annotate('', xy=(7.3, 4.9), xytext=(7, 4.35),
            arrowprops=dict(arrowstyle='->', color='#7B1FA2', lw=1.5))
ax.annotate('', xy=(7.5, 4.9), xytext=(7.5, 4.9),
            arrowprops=dict(arrowstyle='->', color='#7B1FA2', lw=1.5))

# Concat
draw_box(ax, 7.5, 1.2, 1.8, 4.4, concat_color, 'Concat', '256 channels')
ax.annotate('', xy=(9.3, 3.4), xytext=(9.3, 3.4),
            arrowprops=dict(arrowstyle='->', color='#333', lw=2))

# Output
draw_box(ax, 9.5, 2.4, 2, 2, input_color, 'Output', '28×28×256')

# Arrow from concat to output
ax.annotate('', xy=(11.5, 3.4), xytext=(9.3, 3.4),
            arrowprops=dict(arrowstyle='->', color='#333', lw=2))

# Annotations
ax.text(4, 5.5, '1×1 Conv\n(Bottleneck)', ha='center', fontsize=8, 
        bbox=dict(boxstyle='round', facecolor='#FFF9C4', alpha=0.8))

# Legend
legend_y = 0.4
ax.add_patch(patches.Rectangle((0.5, legend_y), 0.4, 0.3, facecolor=conv1_color, edgecolor='#333'))
ax.text(1.1, legend_y+0.15, '1×1 Conv', fontsize=8, va='center')

ax.add_patch(patches.Rectangle((2.5, legend_y), 0.4, 0.3, facecolor=conv3_color, edgecolor='#333'))
ax.text(3.1, legend_y+0.15, '3×3 Conv', fontsize=8, va='center')

ax.add_patch(patches.Rectangle((4.5, legend_y), 0.4, 0.3, facecolor=conv5_color, edgecolor='#333'))
ax.text(5.1, legend_y+0.15, '5×5 Conv', fontsize=8, va='center')

ax.add_patch(patches.Rectangle((6.5, legend_y), 0.4, 0.3, facecolor=pool_color, edgecolor='#333'))
ax.text(7.1, legend_y+0.15, 'MaxPool', fontsize=8, va='center')

ax.add_patch(patches.Rectangle((8.5, legend_y), 0.4, 0.3, facecolor=concat_color, edgecolor='#333'))
ax.text(9.1, legend_y+0.15, 'Concat', fontsize=8, va='center')

plt.tight_layout()
plt.savefig('fig_inception_module.png', dpi=150, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("Saved: fig_inception_module.png")