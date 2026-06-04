"""
RNN Cell Structure
Single timestep RNN neuron: input x^(t), hidden state h^(t), previous hidden state h^(t-1)
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

plt.rcParams['font.family'] = 'DejaVu Sans'

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis('off')
ax.set_title('RNN Cell Structure', fontsize=18, fontweight='bold', pad=20)

# Input node x^(t)
input_box = FancyBboxPatch((1, 3.5), 2, 1.2, boxstyle="round,pad=0.05",
                            facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=2)
ax.add_patch(input_box)
ax.text(2, 4.1, r'$x^{(t)}$', fontsize=16, ha='center', va='center', fontweight='bold')
ax.text(2, 3.6, 'Input', fontsize=11, ha='center', va='center', color='#1565C0')

# Arrow: input -> hidden
ax.annotate('', xy=(4.5, 4.2), xytext=(3, 4.1),
            arrowprops=dict(arrowstyle='->', color='#1976D2', lw=2))

# Previous hidden state h^(t-1)
prev_h_box = FancyBboxPatch((1, 0.8), 2, 1.2, boxstyle="round,pad=0.05",
                             facecolor='#FFF3E0', edgecolor='#E65100', linewidth=2)
ax.add_patch(prev_h_box)
ax.text(2, 1.4, r'$h^{(t-1)}$', fontsize=16, ha='center', va='center', fontweight='bold')
ax.text(2, 0.9, 'Prev Hidden', fontsize=11, ha='center', va='center', color='#E65100')

# Arrow: h^(t-1) -> hidden layer (recurrent connection)
ax.annotate('', xy=(4.5, 3.8), xytext=(3, 1.4),
            arrowprops=dict(arrowstyle='->', color='#FF8F00', lw=2, 
                          connectionstyle="arc3,rad=0.3"))

# RNN hidden unit (core)
hidden_box = FancyBboxPatch((4.5, 2.5), 3, 3, boxstyle="round,pad=0.1",
                             facecolor='#E8F5E9', edgecolor='#2E7D32', linewidth=3)
ax.add_patch(hidden_box)
ax.text(6, 5.0, 'RNN', fontsize=20, ha='center', va='center', fontweight='bold', color='#2E7D32')
ax.text(6, 4.4, r'$\tanh$', fontsize=14, ha='center', va='center')
ax.text(6, 3.6, r'$h^{(t)} = \tanh(W_{xh}x^{(t)} + W_{hh}h^{(t-1)} + b_h)$', 
        fontsize=10, ha='center', va='center')

# Weight labels
ax.text(3.2, 4.5, r'$W_{xh}$', fontsize=12, ha='center', va='center', color='#1976D2')
ax.text(3.5, 2.5, r'$W_{hh}$', fontsize=12, ha='center', va='center', color='#FF8F00')

# Output node y^(t)
output_box = FancyBboxPatch((8.5, 3.5), 2, 1.2, boxstyle="round,pad=0.05",
                             facecolor='#FCE4EC', edgecolor='#C2185B', linewidth=2)
ax.add_patch(output_box)
ax.text(9.5, 4.1, r'$y^{(t)}$', fontsize=16, ha='center', va='center', fontweight='bold')
ax.text(9.5, 3.6, 'Output', fontsize=11, ha='center', va='center', color='#C2185B')

# Arrow: hidden -> output
ax.annotate('', xy=(8.5, 4.1), xytext=(7.5, 4.1),
            arrowprops=dict(arrowstyle='->', color='#C2185B', lw=2))
ax.text(8, 4.5, r'$W_{hy}$', fontsize=12, ha='center', va='center', color='#C2185B')

# Arrow: hidden -> next timestep (recurrent output)
ax.annotate('', xy=(10, 1.5), xytext=(7, 2.5),
            arrowprops=dict(arrowstyle='->', color='#388E3C', lw=2,
                          connectionstyle="arc3,rad=-0.3"))
ax.text(9.5, 1.8, r'$h^{(t)} \to h^{(t+1)}$', fontsize=11, ha='center', va='center', color='#388E3C')
ax.text(9.5, 0.5, '(Recurrent Connection)', fontsize=10, ha='center', va='center', color='#388E3C', style='italic')

# Legend
legend_elements = [
    mpatches.Patch(facecolor='#E3F2FD', edgecolor='#1565C0', label=r'$x^{(t)}$ Input'),
    mpatches.Patch(facecolor='#FFF3E0', edgecolor='#E65100', label=r'$h^{(t-1)}$ Prev Hidden'),
    mpatches.Patch(facecolor='#E8F5E9', edgecolor='#2E7D32', label='RNN Cell (Hidden)'),
    mpatches.Patch(facecolor='#FCE4EC', edgecolor='#C2185B', label=r'$y^{(t)}$ Output'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)

plt.tight_layout()
plt.savefig(Path(__file__).resolve().parent / 'rnn_cell.png', dpi=150, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.close()
print("Saved: rnn_cell.png")
