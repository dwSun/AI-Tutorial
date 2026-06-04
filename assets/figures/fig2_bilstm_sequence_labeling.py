"""
图2: BiLSTM 序列标注示意图
展示双向LSTM如何同时利用前向和后向上下文进行序列标注
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(1, 1, figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')
ax.set_title("BiLSTM Sequence Labeling (BiLSTM 序列标注)", fontsize=16, fontweight='bold', pad=20)

# Colors
color_fwd = '#3498DB'   # Blue for forward
color_bwd = '#E74C3C'    # Red for backward
color_combine = '#27AE60'  # Green for combined
color_label = '#9B59B6'  # Purple for labels

def draw_box(ax, x, y, w, h, text, color, fontsize=10):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                         facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.85)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color='white')

def draw_arrow(ax, start, end, color='#7F8C8D', lw=2):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=lw))

# Input sequence - characters
input_chars = ['马', '克', '访', '问', '了', '北', '京']
input_x_start = 1.5
box_w = 0.8
spacing = 1.0

for i, char in enumerate(input_chars):
    x = input_x_start + i * spacing
    draw_box(ax, x, 6.5, box_w, 0.6, char, '#34495E', 12)
    ax.text(x + box_w/2, 7.2, f'x$_{i}$', ha='center', fontsize=9)

# Forward LSTM arrows
for i in range(len(input_chars) - 1):
    x1 = input_x_start + i * spacing + box_w/2
    x2 = input_x_start + (i + 1) * spacing + box_w/2
    ax.annotate('', xy=(x2, 5.8), xytext=(x1, 5.8),
                arrowprops=dict(arrowstyle='->', color=color_fwd, lw=2))

# Backward LSTM arrows (reversed)
for i in range(len(input_chars) - 1):
    x1 = input_x_start + (len(input_chars) - 1 - i) * spacing + box_w/2
    x2 = input_x_start + (len(input_chars) - 2 - i) * spacing + box_w/2
    ax.annotate('', xy=(x2, 4.5), xytext=(x1, 4.5),
                arrowprops=dict(arrowstyle='->', color=color_bwd, lw=2))

# Forward hidden states
ax.text(0.5, 5.8, r'$\overrightarrow{h}$', fontsize=12, color=color_fwd)
ax.text(0.5, 4.5, r'$\overleftarrow{h}$', fontsize=12, color=color_bwd)

for i in range(len(input_chars)):
    x = input_x_start + i * spacing
    # Forward h
    ax.text(x + box_w/2, 5.8 + 0.2, f'h{i}', ha='center', fontsize=8, color=color_fwd)
    draw_box(ax, x + 0.1, 5.5, box_w - 0.2, 0.5, f'h$_{i}$→', color_fwd, 8)
    # Backward h
    draw_box(ax, x + 0.1, 4.2, box_w - 0.2, 0.5, f'←h$_{i}$', color_bwd, 8)

# Labels
label_chars = ['O', 'O', 'O', 'O', 'O', 'B-LOC', 'I-LOC']
for i, label in enumerate(label_chars):
    x = input_x_start + i * spacing
    draw_box(ax, x + 0.1, 2.0, box_w - 0.2, 0.6, label, color_label, 9)

# Combined vectors to labels
for i in range(len(input_chars)):
    x = input_x_start + i * spacing + box_w/2
    # Forward to label
    ax.annotate('', xy=(x, 2.6), xytext=(x, 5.5),
                arrowprops=dict(arrowstyle='->', color=color_fwd, lw=1.5, linestyle='dashed'))
    # Backward to label
    ax.annotate('', xy=(x, 2.6), xytext=(x, 4.7),
                arrowprops=dict(arrowstyle='->', color=color_bwd, lw=1.5, linestyle='dashed'))

# Combination arrow
ax.annotate('', xy=(8.5, 2.6), xytext=(8.5, 5.0),
            arrowprops=dict(arrowstyle='->', color=color_combine, lw=2))
ax.text(8.7, 3.8, 'concat', fontsize=9, color=color_combine)

# Combined state
draw_box(ax, 8, 4.7, 1, 0.5, r'[h_f, h_b]', color_combine, 8)

# Labels row label
ax.text(0.3, 2.3, 'Labels:', fontsize=10, ha='right')

# Example sentence
ax.text(5, 0.5, 'Example: "马克访问了北京" → O O O O O B-LOC I-LOC (Named Entity Recognition)',
         ha='center', fontsize=10, style='italic')

# Legend
legend_elements = [
    mpatches.Patch(color=color_fwd, label='Forward LSTM (left→right)'),
    mpatches.Patch(color=color_bwd, label='Backward LSTM (right←left)'),
    mpatches.Patch(color=color_combine, label='Hidden State Combination'),
    mpatches.Patch(color=color_label, label='Output Labels'),
]
ax.legend(handles=legend_elements, loc='upper center', ncol=2, fontsize=9)

plt.tight_layout()
plt.savefig(Path(__file__).resolve().parent / 'fig2_bilstm_sequence_labeling.png', dpi=150, bbox_inches='tight')
plt.close()
print("fig2_bilstm_sequence_labeling.png saved.")
