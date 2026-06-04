#!/usr/bin/env python3
"""
Gradient Flow Comparison - Chapter 4-3 ResNet Section
Replaces ASCII diagram with matplotlib figure
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Gradient Flow Comparison: Plain Network vs ResNet', fontsize=16, fontweight='bold', y=0.98)

# ===== Left: Plain Network =====
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 8)
ax1.set_aspect('equal')
ax1.axis('off')
ax1.set_title('Plain Network', fontsize=13, fontweight='bold', pad=15, color='#d32f2f')

layers = ['Input', 'Conv+BN\n+ReLU', 'Conv+BN\n+ReLU', 'Conv+BN\n+ReLU', 'Conv+BN\n+ReLU', 'Output']
y_positions = [1, 2.2, 3.4, 4.6, 5.8, 7]
layer_colors = ['#E3F2FD', '#BBDEFB', '#90CAF9', '#64B5F6', '#42A5F5', '#E3F2FD']

for i, (layer, y) in enumerate(zip(layers, y_positions)):
    rect = FancyBboxPatch((3.5, y - 0.4), 3, 0.8,
                           boxstyle="round,pad=0.05,rounding_size=0.1",
                           facecolor=layer_colors[i], edgecolor='#1976D2', linewidth=2)
    ax1.add_patch(rect)
    ax1.text(5, y, layer, ha='center', va='center', fontsize=9, fontweight='bold')

# Gradient backprop arrows with fading
for i in range(4):
    alpha = 1.0 - i * 0.25
    ax1.annotate('', xy=(7.2 - i * 0.2, 4.8 - i * 0.4), xytext=(3.8 + i * 0.2, 5.2 - i * 0.4),
                arrowprops=dict(arrowstyle='->', color='#d32f2f', alpha=alpha, lw=2))

ax1.text(5, 0.3, 'Gradient Backprop', ha='center', va='center', fontsize=10, 
         style='italic', color='#d32f2f', bbox=dict(boxstyle='round', facecolor='#FFEBEE', edgecolor='#d32f2f'))

ax1.annotate('Long path\nMany nonlinearities\nGradient Vanishing',
             xy=(5, 4), xytext=(0.8, 4),
             fontsize=9, ha='left', va='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE', edgecolor='#d32f2f', alpha=0.9),
             arrowprops=dict(arrowstyle='->', color='#d32f2f', connectionstyle='arc3,rad=0.2'))

# ===== Right: ResNet =====
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 8)
ax2.set_aspect('equal')
ax2.axis('off')
ax2.set_title('ResNet (Residual Network)', fontsize=13, fontweight='bold', pad=15, color='#388E3C')

for i, (layer, y) in enumerate(zip(layers, y_positions)):
    rect = FancyBboxPatch((3.5, y - 0.4), 3, 0.8,
                           boxstyle="round,pad=0.05,rounding_size=0.1",
                           facecolor=layer_colors[i], edgecolor='#1976D2', linewidth=2)
    ax2.add_patch(rect)
    ax2.text(5, y, layer, ha='center', va='center', fontsize=9, fontweight='bold')

# Shortcut connections (thick green lines)
for i in range(5):
    y_bottom = y_positions[i] + 0.4
    y_top = y_positions[i + 1] - 0.4
    ax2.plot([7.2, 7.2], [y_bottom, y_top], 
             color='#4CAF50', linewidth=4, solid_capstyle='round', alpha=0.8)
    ax2.annotate('', xy=(7.2, y_top + 0.1), xytext=(7.2, y_bottom - 0.1),
                arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=3))

# F(x) path arrows (normal gradient)
for i in range(4):
    ax2.annotate('', xy=(6.6 - i * 0.15, 4.8 - i * 0.4), xytext=(4.4 + i * 0.15, 5.2 - i * 0.4),
                arrowprops=dict(arrowstyle='->', color='#388E3C', alpha=0.5, lw=1.5, 
                               connectionstyle='arc3,rad=-0.1'))

ax2.text(8.3, 4, 'Shortcut\n(Identity)', ha='left', va='center', fontsize=9, 
         fontweight='bold', color='#4CAF50',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', edgecolor='#4CAF50', alpha=0.9))

ax2.annotate('Gradient Highway\nDirect path to shallow layers',
             xy=(7.2, 4), xytext=(1.2, 4),
             fontsize=9, ha='left', va='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', edgecolor='#4CAF50', alpha=0.9),
             arrowprops=dict(arrowstyle='->', color='#4CAF50', connectionstyle='arc3,rad=0.2'))

# Legend
normal_patch = mpatches.Patch(color='#d32f2f', alpha=0.7, label='Plain: Gradient Decay Path')
resnet_patch = mpatches.Patch(color='#4CAF50', alpha=0.7, label='ResNet: Shortcut Highway')
ax2.legend(handles=[normal_patch, resnet_patch], loc='lower center', 
           ncol=2, fontsize=9, framealpha=0.9, edgecolor='gray')

plt.tight_layout(rect=[0, 0.08, 1, 0.95])

output_path = Path(__file__).resolve().parent / 'gradient_flow_comparison.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"Saved: {output_path}")
