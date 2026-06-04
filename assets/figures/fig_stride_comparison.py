"""
Stride Comparison Visualization
Demonstrates how stride=1 vs stride=2 affect convolution output size
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Color scheme
input_color = '#E3F2FD'
kernel_color = '#FF7043'
output_color = '#C8E6C9'

# === Stride = 1 ===
ax1 = axes[0]
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.set_title('Stride = 1\nInput 5×5 → Output 3×3', fontsize=12, fontweight='bold')
ax1.axis('off')

# Draw input grid (5x5)
for i in range(5):
    for j in range(5):
        rect = patches.Rectangle((j, 4-i), 1, 1, linewidth=0.5, 
                                   edgecolor='#333', facecolor=input_color)
        ax1.add_patch(rect)

# Draw kernel positions (stride=1, so 3 positions)
kernel_positions = [(0, 0), (1, 0), (0, 1)]
for pos in kernel_positions:
    rect = patches.Rectangle((pos[0], pos[1]), 3, 3, linewidth=2, 
                               edgecolor='#D32F2F', facecolor=kernel_color, 
                               alpha=0.5, linestyle='--')
    ax1.add_patch(rect)

# Labels
ax1.text(2.5, -0.5, 'Input 5×5', ha='center', fontsize=10)
ax1.text(8, 5, '3×3\nKernel', ha='center', fontsize=9, color='#D32F2F')

# Arrows
ax1.annotate('', xy=(6, 3.5), xytext=(5.5, 3.5),
             arrowprops=dict(arrowstyle='->', color='#333', lw=2))
ax1.text(6.5, 3.5, '→', fontsize=16)

# Output
for i in range(3):
    for j in range(3):
        rect = patches.Rectangle((7.5+j, 3.5-i), 1, 1, linewidth=0.5, 
                                   edgecolor='#333', facecolor=output_color)
        ax1.add_patch(rect)
ax1.text(8, 0.3, 'Output 3×3', ha='center', fontsize=10)

# === Stride = 2 ===
ax2 = axes[1]
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.set_title('Stride = 2\nInput 5×5 → Output 2×2', fontsize=12, fontweight='bold')
ax2.axis('off')

# Draw input grid (5x5)
for i in range(5):
    for j in range(5):
        rect = patches.Rectangle((j, 4-i), 1, 1, linewidth=0.5, 
                                   edgecolor='#333', facecolor=input_color)
        ax2.add_patch(rect)

# Draw kernel positions (stride=2, so 2 positions)
kernel_positions = [(0, 0), (2, 0)]
for pos in kernel_positions:
    rect = patches.Rectangle((pos[0], pos[1]), 3, 3, linewidth=2, 
                               edgecolor='#D32F2F', facecolor=kernel_color, 
                               alpha=0.5, linestyle='--')
    ax2.add_patch(rect)

# Labels
ax2.text(2.5, -0.5, 'Input 5×5', ha='center', fontsize=10)

# Output (2x2)
for i in range(2):
    for j in range(2):
        rect = patches.Rectangle((7.5+j, 3.5-i), 1, 1, linewidth=0.5, 
                                   edgecolor='#333', facecolor=output_color)
        ax2.add_patch(rect)
ax2.text(8, 1.3, 'Output\n2×2', ha='center', fontsize=10)

# Arrows
ax2.annotate('', xy=(6, 3), xytext=(5.5, 3),
             arrowprops=dict(arrowstyle='->', color='#333', lw=2))

# Annotation box
props = dict(boxstyle='round', facecolor='#FFF9C4', alpha=0.9)
ax2.text(5, -1.2, 'Stride = 2: Skip 1 pixel between steps\nOutput ≈ Input/2', 
         ha='center', fontsize=9, bbox=props)

plt.tight_layout()
plt.savefig('fig_stride_comparison.png', dpi=150, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("Saved: fig_stride_comparison.png")