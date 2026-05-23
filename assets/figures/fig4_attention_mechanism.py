"""
图4: 注意力机制示意图
展示注意力机制如何在生成每个输出时关注输入的不同部分
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title("Attention Mechanism (注意力机制)", fontsize=16, fontweight='bold', pad=20)

# Colors
color_encoder = '#3498DB'
color_decoder = '#27AE60'
color_attention = '#E74C3C'
color_h = '#9B59B6'

def draw_box(ax, x, y, w, h, text, color, fontsize=10):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                         facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.85)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color='white')

# Encoder hidden states (top row)
encoder_states = [r'$h_1$', r'$h_2$', r'$h_3$', r'$h_4$']
encoder_x_start = 2.0
spacing = 1.2
box_w = 0.8

ax.text(0.5, 7.8, 'Encoder\nHidden\nStates', fontsize=9, ha='center', va='center')
for i, state in enumerate(encoder_states):
    x = encoder_x_start + i * spacing
    draw_box(ax, x, 7.3, box_w, 0.7, state, color_encoder, 11)

# Attention layer (middle)
ax.text(0.5, 5.3, 'Attention\nLayer', fontsize=9, ha='center', va='center')
draw_box(ax, 5.5, 4.5, 3, 1.5, 'Attention\nCalculation', color_attention, 11)

# Decoder hidden states (bottom row)
decoder_states = [r'$s_0$', r'$s_1$', r'$s_2$', r'$s_3$']
for i, state in enumerate(decoder_states):
    x = encoder_x_start + i * spacing
    draw_box(ax, x, 2.5, box_w, 0.7, state, color_decoder, 11)

ax.text(0.5, 2.85, 'Decoder\nHidden\nStates', fontsize=9, ha='center', va='center')

# Output
output_labels = [r'$y_1$', r'$y_2$', r'$y_3$', r'$y_4$']
for i, label in enumerate(output_labels):
    x = encoder_x_start + i * spacing
    draw_box(ax, x, 1.0, box_w, 0.6, label, color_h, 11)

# Arrows from encoder to attention
for i in range(4):
    x1 = encoder_x_start + i * spacing + box_w/2
    ax.annotate('', xy=(7, 5.3), xytext=(x1, 7.3),
                arrowprops=dict(arrowstyle='->', color=color_encoder, lw=1.2))

# Arrows from attention to decoder
for i in range(4):
    x2 = encoder_x_start + i * spacing + box_w/2
    ax.annotate('', xy=(x2, 3.2), xytext=(7, 5.2),
                arrowprops=dict(arrowstyle='->', color=color_decoder, lw=1.2))

# Arrows from decoder to output
for i in range(4):
    x = encoder_x_start + i * spacing + box_w/2
    ax.annotate('', xy=(x, 1.6), xytext=(x, 2.5),
                arrowprops=dict(arrowstyle='->', color=color_h, lw=1.5))

# Attention weight visualization (heatmap-style)
ax.text(8.5, 7.8, 'Attention Weights:', fontsize=10, fontweight='bold')

# Simulate attention weights for each output step
# y1="Machine" attends mostly to 机 (h1)
# y2="Learning" attends to 学 (h3), 习 (h4)
attention_data = [
    [0.70, 0.15, 0.10, 0.05],  # y1 attends h1
    [0.05, 0.10, 0.50, 0.35],  # y2 attends h3, h4
]

for step, weights in enumerate(attention_data):
    y_base = 6.5 - step * 1.5
    ax.text(8.5, y_base, f'$y_{step+1}$:', fontsize=10)
    for j, w in enumerate(weights):
        x = 9.5 + j * 0.8
        alpha = 0.3 + 0.7 * w
        rect = plt.Rectangle((x, y_base - 0.3), 0.6, 0.5,
                              facecolor=color_attention, alpha=alpha)
        ax.add_patch(rect)
        ax.text(x + 0.3, y_base - 0.05, f'{w:.2f}', ha='center', va='center', fontsize=8)
    # Arrow to encoder
    ax.annotate('', xy=(encoder_x_start + (step * 1.2) + box_w/2, 7.5),
                xytext=(12.2, y_base),
                arrowprops=dict(arrowstyle='->', color=color_attention, lw=1,
                               connectionstyle='arc3,rad=0', linestyle='dashed'))

# Chinese labels for encoder states
chinese_labels = ['机', '器', '学', '习']
for i, label in enumerate(chinese_labels):
    x = encoder_x_start + i * spacing
    ax.text(x + box_w/2, 8.2, label, ha='center', fontsize=10, color='#2C3E50')

# English labels for decoder outputs
english_labels = ['Machine', 'Learning', '', '']
for i, label in enumerate(english_labels[:2]):
    x = encoder_x_start + i * spacing
    ax.text(x + box_w/2, 0.4, label, ha='center', fontsize=8, color='#2C3E50', style='italic')

# Formula box
formula_text = (
    r"$e_{ti} = \text{score}(s_{t-1}, h_i)$" + "\n"
    r"$\alpha_{ti} = \frac{\exp(e_{ti})}{\sum_j \exp(e_{tj})}$" + "\n"
    r"$c_t = \sum_i \alpha_{ti} h_i$"
)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.5, 0.5, formula_text, fontsize=9, verticalalignment='bottom',
        bbox=props, family='monospace')

# Legend
legend_elements = [
    mpatches.Patch(color=color_encoder, label='Encoder Hidden States'),
    mpatches.Patch(color=color_decoder, label='Decoder Hidden States'),
    mpatches.Patch(color=color_attention, label='Attention'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig(Path(__file__).resolve().parent / 'fig4_attention_mechanism.png', dpi=150, bbox_inches='tight')
plt.close()
print("fig4_attention_mechanism.png saved.")
