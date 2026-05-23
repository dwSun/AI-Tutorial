"""
Draw a multi-layer fully-connected neural network (MLP) structure diagram.
4 layers: Input(5) -> Hidden1(4) -> Hidden2(3) -> Output(2)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')
ax.set_title('Multi-Layer Perceptron (MLP) Structure', fontsize=16, fontweight='bold', pad=20)

# Layer config: (label, n_nodes, x_position)
layers = [
    ('Input\nLayer', 5, 1.5),
    ('Hidden\nLayer 1', 4, 5.5),
    ('Hidden\nLayer 2', 3, 9.5),
    ('Output\nLayer', 2, 13.0),
]

layer_bg_colors = ['#E3F2FD', '#E8F5E9', '#FFF3E0', '#FCE4EC']
node_colors = ['#1976D2', '#388E3C', '#F57C00', '#D81B60']

y_positions = {}
for i, (name, n_nodes, x) in enumerate(layers):
    spacing = 1.2
    total_height = (n_nodes - 1) * spacing
    y_start = 4 + total_height / 2
    ys = [y_start - k * spacing for k in range(n_nodes)]
    y_positions[i] = ys

    # Layer background box
    rect = mpatches.FancyBboxPatch(
        (x - 0.9, y_start + 0.3), 1.8, -total_height - 0.6,
        boxstyle="round,pad=0.1",
        facecolor=layer_bg_colors[i], edgecolor='gray', linewidth=1.5, alpha=0.6
    )
    ax.add_patch(rect)

    # Layer label
    ax.text(x + 0.0, 7.2, name, ha='center', va='center', fontsize=10, fontweight='bold')

    # Draw nodes
    for j, y in enumerate(ys):
        circle = plt.Circle((x, y), 0.22, color=node_colors[i], ec='white', linewidth=2, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, f'a$_{{{j+1}}}$', ha='center', va='center', fontsize=7, color='white', zorder=4)

# Draw connections (representative lines only to avoid clutter)
for i in range(len(layers) - 1):
    n_curr = layers[i][1]
    n_next = layers[i + 1][1]
    ys_curr = y_positions[i]
    ys_next = y_positions[i + 1]

    lines_to_draw = min(n_curr, n_next, 3)
    indices_curr = np.linspace(0, n_curr - 1, lines_to_draw, dtype=int)
    indices_next = np.linspace(0, n_next - 1, lines_to_draw, dtype=int)

    for ic, in_ in zip(indices_curr, indices_next):
        y1 = ys_curr[ic]
        y2 = ys_next[in_]
        ax.annotate('',
                    xy=(layers[i + 1][2] - 0.25, y2),
                    xytext=(layers[i][2] + 0.25, y1),
                    arrowprops=dict(arrowstyle='->', color='#90A4AE',
                                    connectionstyle='arc3,rad=0.0', lw=1.2),
                    zorder=1)

# Bottom labels: node count per layer
for i, (name, n_nodes, x) in enumerate(layers):
    ax.text(x, 0.5, f'{n_nodes} neurons', ha='center', va='center', fontsize=8, color='gray')

# Legend
legend_x = 0.5
ax.text(legend_x, 1.0, 'Legend:', fontsize=9, fontweight='bold')
for i, color in enumerate(node_colors):
    circle = plt.Circle((legend_x + 0.8 + i * 2.8, 0.5), 0.15, color=color, zorder=3)
    ax.add_patch(circle)
    ax.text(legend_x + 1.1 + i * 2.8, 0.5, layers[i][0].split('\n')[0], ha='left', va='center', fontsize=8)

# Layer count annotation
ax.text(13.8, 4, '4 layers\n(3 hidden)', ha='center', va='center', fontsize=8, color='#666666',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
output_path = Path(__file__).resolve().parent / '3-1-mlp-structure.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f'Saved: {output_path}')
plt.close()
