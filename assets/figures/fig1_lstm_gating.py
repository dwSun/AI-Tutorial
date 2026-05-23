"""
图1: LSTM 门控机制示意图
展示LSTM的三个门（遗忘门、输入门、输出门）和细胞状态更新流程
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title("LSTM Gating Mechanism (LSTM 门控机制)", fontsize=16, fontweight='bold', pad=20)

# Colors
color_gate = '#E74C3C'      # Red for gates
color_cell = '#3498DB'      # Blue for cell state
color_hidden = '#2ECC71'    # Green for hidden state
color_input = '#F39C12'     # Orange for inputs
color_arrow = '#7F8C8D'     # Gray for arrows

def draw_box(ax, x, y, w, h, text, color, fontsize=10):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                         facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.85)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color='white')

def draw_arrow(ax, start, end, color=color_arrow, label='', label_pos=0.5):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=2))
    if label:
        mid = (start[0] + (end[0] - start[0]) * label_pos,
               start[1] + (end[1] - start[1]) * label_pos)
        ax.text(mid[0], mid[1] + 0.15, label, ha='center', va='bottom',
                fontsize=9, color=color)

# Input labels
ax.text(1, 6.5, r'$h^{(t-1)}, x^{(t)}$', fontsize=12, ha='center')
draw_arrow(ax, (1, 6.3), (2.5, 6.3))

# Forget Gate
draw_box(ax, 2.5, 5.5, 2, 1.2, 'Forget Gate\n$f^{(t)}$', color_gate, 9)
draw_arrow(ax, (4.5, 6), (4.5, 5.5))
ax.text(5.2, 5.85, r'$\sigma$', fontsize=11, color=color_gate)

# Input Gate
draw_box(ax, 2.5, 3.8, 2, 1.2, 'Input Gate\n$i^{(t)}$', color_input, 9)
draw_arrow(ax, (4.5, 6), (3.7, 4.3))
ax.text(4.0, 5.0, r'$\sigma$', fontsize=11, color=color_input)

# Candidate Cell
draw_box(ax, 5.5, 3.8, 2, 1.2, r'Candidate $\tilde{c}^{(t)}$', '#9B59B6', 9)
draw_arrow(ax, (4.5, 6), (6.5, 4.3))
ax.text(5.8, 5.0, r'$\tanh$', fontsize=11, color='#9B59B6')

# Cell State Box
draw_box(ax, 7.5, 4.5, 2.5, 1.8, r'Cell State $c^{(t)}$\n(Long-term Memory)', color_cell, 10)

# Connections to cell state
draw_arrow(ax, (4.5, 5.5), (7.5, 5.3))
ax.text(5.8, 5.5, r'$f^{(t)} \odot c^{(t-1)}$', fontsize=9, color=color_cell)

draw_arrow(ax, (7.5, 4.3), (7.5, 4.5))
ax.text(8.5, 4.0, r'$i^{(t)} \odot \tilde{c}^{(t)}$', fontsize=9, color=color_input)

# Previous cell state
ax.text(3.5, 7.5, r'$c^{(t-1)}$', fontsize=11, ha='center')
draw_arrow(ax, (3.5, 7.3), (4.5, 6.7), color_cell)
ax.text(4.2, 7.0, r'$\odot$', fontsize=10, color=color_cell)

# Output Gate
draw_box(ax, 10.5, 4.5, 2, 1.8, 'Output Gate\n$o^{(t)}$', color_gate, 9)
draw_arrow(ax, (10, 6.5), (10.5, 5.5))
ax.text(10.0, 6.0, r'$\sigma$', fontsize=11, color=color_gate)

# tanh to output gate
ax.text(11.8, 5.0, r'$\tanh(c^{(t)})$', fontsize=10, ha='center')
draw_arrow(ax, (10, 5.3), (11.2, 5.1))

# Hidden State
draw_box(ax, 10.5, 2.2, 2, 1.2, r'$h^{(t)}$\n(Short-term)', color_hidden, 10)
draw_arrow(ax, (11.5, 4.5), (11.5, 3.4))
ax.text(11.8, 4.0, r'$\odot$', fontsize=10, color=color_hidden)

# Output
ax.text(11.5, 1.5, r'$h^{(t)}$ output', fontsize=10, ha='center', color=color_hidden)
draw_arrow(ax, (11.5, 2.2), (11.5, 1.8))

# Legend
legend_elements = [
    mpatches.Patch(color=color_gate, label='Gate (sigmoid)'),
    mpatches.Patch(color=color_cell, label='Cell State'),
    mpatches.Patch(color=color_hidden, label='Hidden State'),
    mpatches.Patch(color=color_input, label='Input/Candidate'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

# Formula box
formula_text = (
    r"$f^{(t)} = \sigma(W_f \cdot [h^{(t-1)}, x^{(t)}] + b_f)$" + "\n"
    r"$i^{(t)} = \sigma(W_i \cdot [h^{(t-1)}, x^{(t)}] + b_i)$" + "\n"
    r"$\tilde{c}^{(t)} = \tanh(W_c \cdot [h^{(t-1)}, x^{(t)}] + b_c)$" + "\n"
    r"$c^{(t)} = f^{(t)} \odot c^{(t-1)} + i^{(t)} \odot \tilde{c}^{(t)}$" + "\n"
    r"$o^{(t)} = \sigma(W_o \cdot [h^{(t-1)}, x^{(t)}] + b_o)$" + "\n"
    r"$h^{(t)} = o^{(t)} \odot \tanh(c^{(t)})$"
)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.5, 2.5, formula_text, fontsize=9, verticalalignment='top',
        bbox=props, family='monospace')

plt.tight_layout()
plt.savefig(Path(__file__).resolve().parent / 'fig1_lstm_gating.png', dpi=150, bbox_inches='tight')
plt.close()
print("fig1_lstm_gating.png saved.")
