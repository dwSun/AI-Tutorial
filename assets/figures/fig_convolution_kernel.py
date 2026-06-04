"""
Convolution Kernel Visualization
Shows how a 3x3 kernel slides over input to produce output feature map
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

fig = plt.figure(figsize=(14, 6))

# Color scheme
input_color = '#E3F2FD'
kernel_color = '#FFECB3'
output_color = '#C8E6C9'
arrow_color = '#666'

# === Input Matrix ===
ax1 = fig.add_axes([0.05, 0.25, 0.25, 0.5])
ax1.set_xlim(0, 5)
ax1.set_ylim(0, 5)
ax1.set_title('Input Image (5×5)', fontsize=11, fontweight='bold')
ax1.axis('off')

input_data = [
    [1, 2, 3, 2, 1],
    [0, 1, 2, 3, 2],
    [0, 0, 1, 2, 3],
    [1, 0, 0, 1, 2],
    [2, 1, 0, 0, 1]
]

for i in range(5):
    for j in range(5):
        rect = patches.Rectangle((j, 4-i), 1, 1, linewidth=0.5, 
                                   edgecolor='#333', facecolor=input_color)
        ax1.add_patch(rect)
        ax1.text(j+0.5, 4-i+0.5, str(input_data[i][j]), 
                 ha='center', va='center', fontsize=9)

# === Kernel ===
ax2 = fig.add_axes([0.38, 0.35, 0.18, 0.3])
ax2.set_xlim(0, 3)
ax2.set_ylim(0, 3)
ax2.set_title('Kernel (3×3)', fontsize=11, fontweight='bold')
ax2.axis('off')

kernel_data = [
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
]

for i in range(3):
    for j in range(3):
        rect = patches.Rectangle((j, 2-i), 1, 1, linewidth=1, 
                                   edgecolor='#333', facecolor=kernel_color)
        ax2.add_patch(rect)
        val = kernel_data[i][j]
        color = 'blue' if val > 0 else ('red' if val < 0 else 'black')
        ax2.text(j+0.5, 2-i+0.5, str(val), 
                 ha='center', va='center', fontsize=10, color=color, fontweight='bold')

# Arrow between input and kernel
ax1.annotate('', xy=(0.42, 0.5), xytext=(0.32, 0.5),
             arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2),
             xycoords='axes fraction')

# === Output Matrix ===
ax3 = fig.add_axes([0.62, 0.25, 0.22, 0.5])
ax3.set_xlim(0, 3)
ax3.set_ylim(0, 3)
ax3.set_title('Feature Map (3×3)', fontsize=11, fontweight='bold')
ax3.axis('off')

output_data = [
    [4, 2, -4],
    [2, 0, -2],
    [0, -2, 4]
]

for i in range(3):
    for j in range(3):
        rect = patches.Rectangle((j, 2-i), 1, 1, linewidth=0.5, 
                                   edgecolor='#333', facecolor=output_color)
        ax3.add_patch(rect)
        val = output_data[i][j]
        color = 'blue' if val > 0 else ('red' if val < 0 else 'black')
        ax3.text(j+0.5, 2-i+0.5, str(val), 
                 ha='center', va='center', fontsize=10, color=color, fontweight='bold')

# Arrow between kernel and output
ax2.annotate('', xy=(0.6, 0.5), xytext=(0.58, 0.5),
             arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2),
             xycoords='figure fraction')

# === Formula Box ===
ax4 = fig.add_axes([0.05, 0.02, 0.9, 0.15])
ax4.axis('off')
ax4.text(0.5, 0.7, 'Convolution: Element-wise multiply → Sum', 
         ha='center', fontsize=11, fontweight='bold', transform=ax4.transAxes)
ax4.text(0.5, 0.3, 
         'Example: output[0,0] = 1×(-1) + 2×0 + 3×1 + 0×(-1) + 1×0 + 2×1 + 0×(-1) + 0×0 + 1×1 = 4',
         ha='center', fontsize=10, transform=ax4.transAxes,
         bbox=dict(boxstyle='round', facecolor='#FFF9C4', alpha=0.8))

# Highlight the first kernel position on input
ax1.add_patch(patches.Rectangle((0, 2), 3, 3, linewidth=2, 
                                  edgecolor='#D32F2F', facecolor='none', 
                                  linestyle='--', transform=ax1.transData))

plt.savefig('fig_convolution_kernel.png', dpi=150, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("Saved: fig_convolution_kernel.png")