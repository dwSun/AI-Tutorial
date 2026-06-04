"""
4-1-receptive-field.py
========================
感受野（Receptive Field）随卷积层数增加而扩大
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

def visualize_receptive_field(num_layers=3, kernel_size=3):
    fig, axes = plt.subplots(1, num_layers, figsize=(15, 5))
    input_size = 15
    input_image = np.zeros((input_size, input_size))
    input_image[input_size//2, input_size//2] = 1
    current_rf = 1
    for layer in range(num_layers):
        current_rf = current_rf + (kernel_size - 1) * (2 ** layer)
        rf_visualization = np.zeros((input_size, input_size))
        center = input_size // 2
        half_rf = current_rf // 2
        start_row = max(0, center - half_rf)
        end_row = min(input_size, center + half_rf + 1)
        start_col = max(0, center - half_rf)
        end_col = min(input_size, center + half_rf + 1)
        rf_visualization[start_row:end_row, start_col:end_col] = 0.5
        rf_visualization[center, center] = 1
        ax = axes[layer]
        im = ax.imshow(rf_visualization, cmap='YlOrRd')
        ax.set_title(f'第{layer+1}层卷积后\n感受野: {current_rf}×{current_rf}')
        ax.set_xticks([])
        ax.set_yticks([])
        rect = patches.Rectangle((start_col-0.5, start_row-0.5),
                                end_col-start_col, end_row-start_row,
                                linewidth=2, edgecolor='blue', facecolor='none')
        ax.add_patch(rect)
        ax.plot(center, center, 'bo', markersize=10)
    plt.suptitle(f'感受野随卷积层数增加而扩大 (kernel_size={kernel_size})', fontsize=14)
    plt.tight_layout()
    plt.savefig('/home/david/AI-Tutorial/assets/figures/4-1-receptive-field.png',
                dpi=120, bbox_inches='tight', facecolor='white')
    print("Saved: 4-1-receptive-field.png")

visualize_receptive_field(num_layers=3, kernel_size=3)
