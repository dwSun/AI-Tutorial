"""
RNN Unrolled Graph (Time Steps)
Chain structure showing RNN along time axis: x^(0)...x^(T) inputs, hidden states, outputs
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

plt.rcParams['font.family'] = 'DejaVu Sans'

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 18)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('RNN Unrolled Graph (Time Steps)', fontsize=18, fontweight='bold', pad=20)

num_steps = 5
step_width = 2.8
start_x = 1.5

input_color = '#1565C0'
hidden_color = '#2E7D32'
output_color = '#7B1FA2'
arrow_color = '#616161'

for i in range(num_steps):
    x_center = start_x + i * step_width
    
    # Input node
    input_box = FancyBboxPatch((x_center - 0.6, 6), 1.2, 0.8, 
                                boxstyle="round,pad=0.03",
                                facecolor='#E3F2FD', edgecolor=input_color, linewidth=2)
    ax.add_patch(input_box)
    ax.text(x_center, 6.4, rf'$x^{({i})}$', fontsize=12, ha='center', va='center', fontweight='bold')
    
    # Hidden node
    hidden_box = FancyBboxPatch((x_center - 0.7, 3.5), 1.4, 1.4,
                                 boxstyle="round,pad=0.05",
                                 facecolor='#C8E6C9', edgecolor=hidden_color, linewidth=2.5)
    ax.add_patch(hidden_box)
    ax.text(x_center, 4.3, rf'$h^{({i})}$', fontsize=13, ha='center', va='center', fontweight='bold', color='#1B5E20')
    
    # Output node
    output_box = FancyBboxPatch((x_center - 0.5, 1), 1, 0.7,
                                  boxstyle="round,pad=0.03",
                                  facecolor='#E1BEE7', edgecolor=output_color, linewidth=2)
    ax.add_patch(output_box)
    ax.text(x_center, 1.35, rf'$y^{({i})}$', fontsize=11, ha='center', va='center', fontweight='bold')
    
    # Connection: input -> hidden
    ax.annotate('', xy=(x_center, 4.9), xytext=(x_center, 6),
                arrowprops=dict(arrowstyle='->', color=input_color, lw=1.5))
    
    # Connection: hidden -> output
    ax.annotate('', xy=(x_center, 1.7), xytext=(x_center, 3.5),
                arrowprops=dict(arrowstyle='->', color=output_color, lw=1.5))
    
    # Timestep label
    ax.text(x_center, 7.5, f'Time Step {i}', fontsize=10, ha='center', va='center', 
            color='#424242', fontweight='bold')

# Draw recurrent connections along the bottom
for i in range(num_steps - 1):
    x1 = start_x + i * step_width
    x2 = start_x + (i + 1) * step_width
    
    ax.annotate('', xy=(x2, 4.0), xytext=(x1 + 0.5, 3.5),
                arrowprops=dict(arrowstyle='->', color=hidden_color, lw=2,
                              connectionstyle="arc3,rad=0.4"))
    
    mid_x = (x1 + x2) / 2
    ax.text(mid_x, 2.8, r'$W_{hh}$', fontsize=9, ha='center', va='center', color=hidden_color)

# Side labels
ax.text(0.3, 6.4, 'Input\nSequence', fontsize=10, ha='center', va='center', color='#424242')
ax.text(0.3, 4.3, 'Hidden\nState', fontsize=10, ha='center', va='center', color='#424242')
ax.text(0.3, 1.35, 'Output', fontsize=10, ha='center', va='center', color='#424242')

# Right side annotations
ax.text(16.5, 6.4, r'$\rightarrow$', fontsize=14, ha='left', va='center')
ax.text(16.8, 6.4, r'$W_{xh}$: Input->Hidden', fontsize=10, ha='left', va='center', color=input_color)

ax.text(16.5, 4.3, r'$\rightarrow$', fontsize=14, ha='left', va='center')
ax.text(16.8, 4.3, r'$W_{hh}$: Hidden->Hidden', fontsize=10, ha='left', va='center', color=hidden_color)

ax.text(16.5, 1.35, r'$\rightarrow$', fontsize=14, ha='left', va='center')
ax.text(16.8, 1.35, r'$W_{hy}$: Hidden->Output', fontsize=10, ha='left', va='center', color=output_color)

# Bottom note
ax.text(9, 0.2, 'Same weight matrices across all time steps (Parameter Sharing)', 
        fontsize=11, ha='center', va='center', style='italic', color='#616161')

# Bottom connection arrow
ax.annotate('', xy=(start_x + 0.2, 0.6), xytext=(start_x + 0.2, 0.9),
            arrowprops=dict(arrowstyle='->', color='#757575', lw=1.5))
ax.annotate('', xy=(start_x - 0.3, 0.9), xytext=(start_x + 0.2, 0.6),
            arrowprops=dict(arrowstyle='->', color='#757575', lw=1.5))
ax.text(start_x + 1.0, 0.5, 'Information flows along time axis', fontsize=9, ha='center', va='center', color='#757575')

plt.tight_layout()
plt.savefig(Path(__file__).resolve().parent / 'rnn_unrolled.png', dpi=150, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.close()
print("Saved: rnn_unrolled.png")
