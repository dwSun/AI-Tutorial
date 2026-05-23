"""
GoogLeNet Architecture Visualization
Shows the complete network structure with Inception modules
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('GoogLeNet Architecture (22 Layers)\nInception V1 - Multi-Scale Feature Extraction', 
             fontsize=14, fontweight='bold', pad=15)

# Colors
stem_color = '#E3F2FD'
inception_color = '#C8E6C9'
pool_color = '#F3E5F5'
fc_color = '#FFCDD2'
output_color = '#BBDEFB'

def draw_layer(ax, x, y, w, h, color, label, sublabel=''):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.03',
                                  linewidth=1, edgecolor='#333', facecolor=color)
    ax.add_patch(rect)
    if sublabel:
        ax.text(x+w/2, y+h/2+0.12, label, ha='center', va='center', fontsize=7, fontweight='bold')
        ax.text(x+w/2, y+h/2-0.15, sublabel, ha='center', va='center', fontsize=6, color='#555')
    else:
        ax.text(x+w/2, y+h/2, label, ha='center', va='center', fontsize=7, fontweight='bold')

def draw_arrow(ax, x1, y1, x2, y2, color='#333', rad=0):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1, 
                               connectionstyle=f'arc3,rad={rad}'))

# Vertical positions for main flow
main_y = 4.5
aux_y1 = 2.2
aux_y2 = 7.2

# === INPUT ===
draw_layer(ax, 0.3, 3.8, 1.2, 1.4, stem_color, 'Input', '224×224×3')

# === STEM ===
draw_arrow(ax, 1.5, 4.5, 1.8, 4.5)
draw_layer(ax, 1.8, 4.0, 1.0, 1.0, stem_color, 'Conv7×7', '64, s2')
draw_layer(ax, 1.8, 2.7, 1.0, 0.8, pool_color, 'MaxPool', '3×3, s2')
draw_layer(ax, 1.8, 1.5, 1.0, 0.7, stem_color, 'LocalRespNorm', '')
draw_layer(ax, 1.8, 0.4, 1.0, 0.7, stem_color, 'Conv3×3', '192, s1')

draw_arrow(ax, 2.8, 0.75, 3.1, 0.75)
draw_layer(ax, 3.1, 0.4, 1.0, 0.7, pool_color, 'MaxPool', '3×3, s2')

# === INCEPTION MODULES (3 groups) ===
inception_x = 4.4
inception_w = 1.8

# Group 1: Inception A (96→64→64→128)
y_pos = 0.3
draw_layer(ax, inception_x, y_pos, inception_w, 0.5, inception_color, 'Inception A', '96→256')
ax.annotate('', xy=(inception_x+inception_w, y_pos+0.25), xytext=(inception_x, y_pos+0.25),
            arrowprops=dict(arrowstyle='->', color='#333', lw=1))

# Group 2: Inception A
inception_x = 4.4 + 2.1
draw_layer(ax, inception_x, y_pos, inception_w, 0.5, inception_color, 'Inception A', '128→256')
ax.annotate('', xy=(inception_x+inception_w, y_pos+0.25), xytext=(inception_x, y_pos+0.25),
            arrowprops=dict(arrowstyle='->', color='#333', lw=1))

# Pool between groups
inception_x = 4.4 + 4.2
draw_layer(ax, inception_x, y_pos, 1.0, 0.5, pool_color, 'MaxPool', '3×3, s2')
ax.annotate('', xy=(inception_x+1.0, y_pos+0.25), xytext=(inception_x, y_pos+0.25),
            arrowprops=dict(arrowstyle='->', color='#333', lw=1))

# Group 3: Inception B (192→128→192→384)
inception_x = 4.4 + 5.5
draw_layer(ax, inception_x, y_pos, inception_w, 0.5, inception_color, 'Inception B', '192→480')
ax.annotate('', xy=(inception_x+inception_w, y_pos+0.25), xytext=(inception_x, y_pos+0.25),
            arrowprops=dict(arrowstyle='->', color='#333', lw=1))

# More inception modules
for i in range(4):
    inception_x = 4.4 + 5.5 + 2.1*(i+1)
    if i == 2:
        # Pool
        draw_layer(ax, inception_x, y_pos, 1.0, 0.5, pool_color, 'MaxPool', '3×3, s2')
    else:
        draw_layer(ax, inception_x, y_pos, inception_w, 0.5, inception_color, 'Inception B', '192→480')
    ax.annotate('', xy=(inception_x+inception_w if i != 2 else inception_x+1.0, y_pos+0.25), 
                xytext=(inception_x, y_pos+0.25),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1))

# === GLOBAL AVERAGE POOLING ===
gaps_x = 15.5
draw_layer(ax, gaps_x, y_pos, 1.2, 0.5, pool_color, 'GlobalAvgPool', '7×7→1×1')
ax.annotate('', xy=(gaps_x+1.2, y_pos+0.25), xytext=(gaps_x, y_pos+0.25),
            arrowprops=dict(arrowstyle='->', color='#333', lw=1))

# === DROPOUT & FC ===
draw_layer(ax, gaps_x+1.5, y_pos, 1.0, 0.5, fc_color, 'Dropout', '40%')
draw_arrow(ax, gaps_x+1.2, y_pos+0.25, gaps_x+1.5, y_pos+0.25)

draw_layer(ax, gaps_x+2.8, y_pos, 1.0, 0.5, fc_color, 'FC', '1000')
draw_arrow(ax, gaps_x+2.5, y_pos+0.25, gaps_x+2.8, y_pos+0.25)

draw_layer(ax, gaps_x+4.0, y_pos, 0.8, 0.5, output_color, 'Softmax', '')
draw_arrow(ax, gaps_x+3.8, y_pos+0.25, gaps_x+4.0, y_pos+0.25)

# === AUXILIARY CLASSIFIERS ===
# Aux 1 (right side, lower)
draw_layer(ax, 10, aux_y1, 1.5, 0.4, fc_color, 'Aux Classifier 1', '')
draw_arrow(ax, 10.6, aux_y1+0.4, 10.6, aux_y1+0.8, color='#1976D2')
draw_layer(ax, 10.3, aux_y1+0.8, 0.7, 0.4, pool_color, 'AvgPool', '5×5, s3')
draw_arrow(ax, 10.65, aux_y1+1.2, 10.65, aux_y1+1.6, color='#1976D2')
draw_layer(ax, 10.3, aux_y1+1.6, 0.7, 0.4, fc_color, 'FC', '1024')
draw_arrow(ax, 10.65, aux_y1+2.0, 10.65, aux_y2-0.2, color='#1976D2')

# Aux 2 (right side, upper)
draw_layer(ax, 10, aux_y2, 1.5, 0.4, fc_color, 'Aux Classifier 2', '')
draw_arrow(ax, 10.6, aux_y2-0.4, 10.6, aux_y2-0.8, color='#1976D2')
draw_layer(ax, 10.3, aux_y2-1.6, 0.7, 0.4, pool_color, 'AvgPool', '5×5, s3')
draw_arrow(ax, 10.65, aux_y2-2.0, 10.65, aux_y2-2.4, color='#1976D2')
draw_layer(ax, 10.3, aux_y2-2.8, 0.7, 0.4, fc_color, 'FC', '1024')
draw_arrow(ax, 10.65, aux_y2-3.2, 10.65, 0.75, color='#1976D2')

# Connection from inception to aux
ax.annotate('', xy=(10, aux_y1+0.2), xytext=(14.5, 0.55),
            arrowprops=dict(arrowstyle='->', color='#1976D2', lw=0.8, 
                           connectionstyle='arc3,rad=0.3', linestyle='dashed'))

# === STAGE LABELS ===
ax.text(2.8, 5.5, 'Stem', fontsize=9, fontweight='bold', color='#1565C0')
ax.text(6.5, 5.5, 'Inception × 3', fontsize=9, fontweight='bold', color='#2E7D32')
ax.text(9.5, 5.5, 'Inception × 6', fontsize=9, fontweight='bold', color='#2E7D32')
ax.text(13.5, 5.5, 'Output', fontsize=9, fontweight='bold', color='#7B1FA2')

# === LEGEND ===
legend_y = 9.3
ax.add_patch(patches.Rectangle((0.5, legend_y), 0.3, 0.25, facecolor=stem_color, edgecolor='#333'))
ax.text(0.95, legend_y+0.12, 'Conv/Pool', fontsize=7, va='center')

ax.add_patch(patches.Rectangle((2.5, legend_y), 0.3, 0.25, facecolor=inception_color, edgecolor='#333'))
ax.text(2.95, legend_y+0.12, 'Inception', fontsize=7, va='center')

ax.add_patch(patches.Rectangle((4.5, legend_y), 0.3, 0.25, facecolor=pool_color, edgecolor='#333'))
ax.text(4.95, legend_y+0.12, 'Pooling', fontsize=7, va='center')

ax.add_patch(patches.Rectangle((6.5, legend_y), 0.3, 0.25, facecolor=fc_color, edgecolor='#333'))
ax.text(6.95, legend_y+0.12, 'FC/Dropout', fontsize=7, va='center')

ax.add_patch(patches.Rectangle((8.8, legend_y), 0.3, 0.25, facecolor=output_color, edgecolor='#333'))
ax.text(9.25, legend_y+0.12, 'Output', fontsize=7, va='center')

# Stats
ax.text(12, legend_y+0.12, 'Parameters: ~5M (vs VGG 138M)', fontsize=7, color='#666')
ax.text(12, legend_y-0.3, 'Note: Auxiliary classifiers used during training only', fontsize=6, color='#999')

plt.tight_layout()
plt.savefig('fig_googlenet_architecture.png', dpi=150, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("Saved: fig_googlenet_architecture.png")