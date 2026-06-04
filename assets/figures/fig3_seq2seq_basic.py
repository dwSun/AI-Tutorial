"""
图3: Seq2Seq 基础架构
展示编码器-解码器的基本结构和工作流程
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(1, 1, figsize=(14, 9))
ax.set_xlim(0, 14)
ax.set_ylim(0, 9)
ax.axis('off')
ax.set_title("Seq2Seq Basic Architecture (Seq2Seq 基础架构)", fontsize=16, fontweight='bold', pad=20)

# Colors
color_encoder = '#3498DB'   # Blue for encoder
color_decoder = '#27AE60'   # Green for decoder
color_vector = '#E74C3C'    # Red for context vector
color_input = '#F39C12'     # Orange for input
color_output = '#9B59B6'    # Purple for output

def draw_box(ax, x, y, w, h, text, color, fontsize=10):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                         facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.85)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color='white')

# Input sequence labels
input_chars = ['x₁', 'x₂', 'x₃', '...', 'x_T']
input_x_start = 1.5
spacing = 0.9
box_w = 0.6

for i, char in enumerate(input_chars):
    x = input_x_start + i * spacing
    draw_box(ax, x, 6.8, box_w, 0.5, char, color_input, 10)

# Encoder label
ax.text(1.2, 7.5, 'Input:', fontsize=11, ha='right')
draw_box(ax, 5.5, 6.5, 3, 1.5, 'Encoder\n(Encoder)', color_encoder, 12)

# Encoder arrows from input
for i in range(5):
    x1 = input_x_start + i * spacing + box_w/2
    x2 = 7
    ax.annotate('', xy=(x2, 7.1), xytext=(x1, 7.15),
                arrowprops=dict(arrowstyle='->', color=color_input, lw=1.5))

# Context vector
draw_box(ax, 5.5, 4.2, 3, 1, r'Context Vector C', color_vector, 11)
ax.annotate('', xy=(7, 5.2), xytext=(7, 6.5),
            arrowprops=dict(arrowstyle='->', color=color_vector, lw=2))
ax.text(7.2, 5.8, '', fontsize=9)

# Decoder label
draw_box(ax, 5.5, 2.2, 3, 1.5, 'Decoder\n(Decoder)', color_decoder, 12)
ax.annotate('', xy=(7, 3.7), xytext=(7, 4.2),
            arrowprops=dict(arrowstyle='->', color=color_decoder, lw=2))

# Output sequence
output_chars = ['y₁', 'y₂', 'y₃', '...', "y_T'"]
for i, char in enumerate(output_chars):
    x = input_x_start + i * spacing
    draw_box(ax, x, 0.8, box_w, 0.5, char, color_output, 10)

# Decoder arrows to output
for i in range(5):
    x1 = 7
    x2 = input_x_start + i * spacing + box_w/2
    ax.annotate('', xy=(x2, 1.3), xytext=(x1, 2.2),
                arrowprops=dict(arrowstyle='->', color=color_output, lw=1.5))

ax.text(1.2, 0.8 + 0.25, 'Output:', fontsize=11, ha='right')

# Autoregressive arrow inside decoder
ax.annotate('', xy=(4.5, 2.2), xytext=(3.5, 2.2),
            arrowprops=dict(arrowstyle='->', color='#7F8C8D', lw=1.5, connectionstyle='arc3,rad=0'))
ax.text(4.0, 1.9, 'autoregressive', fontsize=8, ha='center', color='#7F8C8D')

# Description texts
ax.text(10, 7.3, '• Reads entire input sequence\n• Compresses to fixed vector C\n• Uses LSTM/GRU', fontsize=9, va='top')
ax.text(10, 3.5, '• Generates output step-by-step\n• Each step input = prev output\n• Until <EOS> or max len', fontsize=9, va='top')

# Flow arrows - detailed
ax.text(2.5, 5.5, 'Encoder:\nh₁, h₂, ..., h_T', fontsize=10, ha='center', color=color_encoder)

# Legend
legend_elements = [
    mpatches.Patch(color=color_encoder, label='Encoder'),
    mpatches.Patch(color=color_decoder, label='Decoder'),
    mpatches.Patch(color=color_vector, label='Context Vector C'),
    mpatches.Patch(color=color_input, label='Input Sequence'),
    mpatches.Patch(color=color_output, label='Output Sequence'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

# Machine translation example
ax.text(7, 0.1, 'Example: "Hello world" → "你好世界"', ha='center', fontsize=10, style='italic')

plt.tight_layout()
plt.savefig(Path(__file__).resolve().parent / 'fig3_seq2seq_basic.png', dpi=150, bbox_inches='tight')
plt.close()
print("fig3_seq2seq_basic.png saved.")
