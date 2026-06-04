"""
4-2-cnn-core-operations.py
============================
CNN 核心操作可视化：池化 + BN + GAP vs FC + 架构尺寸变化
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle

mpl.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. 原始特征图
feature_map = np.array([[2, 4, 1, 3], [8, 6, 5, 7], [1, 3, 9, 2], [4, 2, 3, 5]], dtype=float)
max_pooled = np.zeros((2, 2))
avg_pooled = np.zeros((2, 2))
max_pool_positions = []

for i in range(0, 4, 2):
    for j in range(0, 4, 2):
        region = feature_map[i:i+2, j:j+2]
        max_pooled[i//2, j//2] = np.max(region)
        max_pos = np.argwhere(region == np.max(region))[0]
        max_pool_positions.append((i + max_pos[0], j + max_pos[1]))
        avg_pooled[i//2, j//2] = np.mean(region)

ax1 = axes[0, 0]
im1 = ax1.imshow(feature_map, cmap='YlOrRd', vmin=0, vmax=10)
ax1.set_title('原始特征图 (4×4)', fontsize=12, fontweight='bold')
ax1.set_xticks([])
ax1.set_yticks([])
for i in range(4):
    for j in range(4):
        ax1.text(j, i, f'{feature_map[i, j]:.0f}', ha='center', va='center',
                color='black', fontsize=12, fontweight='bold')
for i in range(0, 4, 2):
    for j in range(0, 4, 2):
        rect = Rectangle((j-0.5, i-0.5), 2, 2, fill=False, edgecolor='blue', linewidth=2, linestyle='--')
        ax1.add_patch(rect)

# 2. Max Pooling 结果
ax2 = axes[0, 1]
im2 = ax2.imshow(max_pooled, cmap='YlOrRd', vmin=0, vmax=10)
ax2.set_title('Max Pooling 结果 (2×2)', fontsize=12, fontweight='bold')
ax2.set_xticks([])
ax2.set_yticks([])
for i in range(2):
    for j in range(2):
        ax2.text(j, i, f'{max_pooled[i, j]:.0f}', ha='center', va='center',
                color='black', fontsize=12, fontweight='bold')
for (i, j), pool_pos in zip([(0,0), (0,1), (1,0), (1,1)], max_pool_positions):
    orig_i, orig_j = pool_pos
    ax2.text(j, i, f'←({orig_i},{orig_j})', fontsize=8, color='blue', ha='right', va='top')

# 3. Average Pooling 结果
ax3 = axes[0, 2]
im3 = ax3.imshow(avg_pooled, cmap='YlOrRd', vmin=0, vmax=10)
ax3.set_title('Average Pooling 结果 (2×2)', fontsize=12, fontweight='bold')
ax3.set_xticks([])
ax3.set_yticks([])
for i in range(2):
    for j in range(2):
        ax3.text(j, i, f'{avg_pooled[i, j]:.1f}', ha='center', va='center',
                color='black', fontsize=12, fontweight='bold')

# 4. Batch Normalization 效果
ax4 = axes[1, 0]
np.random.seed(42)
activations_before_bn = []
activations_after_bn = []
for layer in range(4):
    mean = np.random.uniform(-2, 2)
    std = np.random.uniform(0.5, 3)
    layer_activations = np.random.normal(mean, std, 1000)
    activations_before_bn.append(layer_activations)
    bn_activations = (layer_activations - np.mean(layer_activations)) / np.std(layer_activations)
    activations_after_bn.append(bn_activations)

colors = ['red', 'blue', 'green', 'orange']
for i, (before, after, color) in enumerate(zip(activations_before_bn, activations_after_bn, colors)):
    ax4.hist(before, bins=50, alpha=0.3, color=color,
            label=f'层{i+1}前: μ={np.mean(before):.2f}')
    ax4.hist(after, bins=50, alpha=0.6, color=color, linestyle='--',
            label=f'层{i+1}后: μ={np.mean(after):.2f}')
ax4.set_title('Batch Normalization 效果对比', fontsize=12, fontweight='bold')
ax4.set_xlabel('激活值')
ax4.set_ylabel('频数')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# 5. GAP vs FC 参数对比
ax5 = axes[1, 1]
conv_output_shape = (7, 7, 512)
num_classes = 1000
gap_params = 512 * num_classes
fc_params = (7*7*512) * 4096 + 4096 * 4096 + 4096 * num_classes
gap_memory = gap_params * 4 / (1024**2)
fc_memory = fc_params * 4 / (1024**2)

methods = ['GAP', 'FC (VGG风格)']
params = [gap_params, fc_params]
memory = [gap_memory, fc_memory]

x = np.arange(len(methods))
width = 0.35
bars1 = ax5.bar(x - width/2, params, width, label='参数量', color='skyblue')
ax5.set_ylabel('参数量（百万）', color='skyblue')
ax5.tick_params(axis='y', labelcolor='skyblue')
ax5.set_xticks(x)
ax5.set_xticklabels(methods)
ax6 = ax5.twinx()
bars2 = ax6.bar(x + width/2, memory, width, label='内存占用', color='salmon')
ax6.set_ylabel('内存占用（MB）', color='salmon')
ax6.tick_params(axis='y', labelcolor='salmon')
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    ax5.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.5,
            f'{params[i]/1e6:.1f}M', ha='center', va='bottom', color='blue', fontsize=9)
    ax6.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.5,
            f'{memory[i]:.1f}MB', ha='center', va='bottom', color='red', fontsize=9)
ax5.set_title('GAP vs FC 参数和内存对比', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')
lines1, labels1 = ax5.get_legend_handles_labels()
lines2, labels2 = ax6.get_legend_handles_labels()
ax5.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)

# 6. CNN 架构特征图尺寸变化
ax7 = axes[1, 2]
layers = ['Input', 'Conv1', 'Pool1', 'Conv2', 'Pool2', 'Conv3', 'Pool3', 'GAP', 'Output']
heights = [224, 224, 112, 112, 56, 56, 28, 1, 1]
channels = [3, 64, 64, 128, 128, 256, 256, 256, 1000]
x_idx = np.arange(len(layers))
ax7_secondary = ax7.twinx()
ax7_secondary.spines['right'].set_position(('outward', 60))
line1, = ax7.plot(x_idx, heights, 'o-', color='blue', linewidth=2, markersize=8, label='高度')
line2, = ax7_secondary.plot(x_idx, channels, '^-', color='red', linewidth=2, markersize=8, label='通道数')
ax7.set_xticks(x_idx)
ax7.set_xticklabels(layers, rotation=45, ha='right')
ax7.set_ylabel('高度', color='blue')
ax7_secondary.set_ylabel('通道数', color='red')
ax7.tick_params(axis='y', labelcolor='blue')
ax7_secondary.tick_params(axis='y', labelcolor='red')
ax7.set_title('CNN 架构特征图尺寸变化', fontsize=12, fontweight='bold')
ax7.grid(True, alpha=0.3)
ax7.legend([line1, line2], [l.get_label() for l in [line1, line2]], loc='upper right', fontsize=8)

plt.suptitle('CNN 核心操作可视化（池化 / BN / GAP / 架构）', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/david/AI-Tutorial/assets/figures/4-2-cnn-core-operations.png',
            dpi=120, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: 4-2-cnn-core-operations.png")
