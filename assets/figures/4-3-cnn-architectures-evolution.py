"""
4-3-cnn-architectures-evolution.py
=====================================
经典 CNN 架构演进：深度/参数量/准确率/卷积核尺寸/时间线
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle

mpl.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

architectures = [
    {'name': 'LeNet-5', 'year': 1998, 'layers': 5, 'params': 0.06, 'acc': 99.0, 'color': '#FF6B6B'},
    {'name': 'AlexNet', 'year': 2012, 'layers': 8, 'params': 60, 'acc': 84.7, 'color': '#4ECDC4'},
    {'name': 'VGG-16', 'year': 2014, 'layers': 16, 'params': 138, 'acc': 92.5, 'color': '#45B7D1'},
    {'name': 'GoogLeNet', 'year': 2014, 'layers': 22, 'params': 6.8, 'acc': 93.3, 'color': '#96CEB4'},
    {'name': 'ResNet-50', 'year': 2015, 'layers': 50, 'params': 25.6, 'acc': 96.4, 'color': '#FFEAA7'},
    {'name': 'ResNet-152', 'year': 2015, 'layers': 152, 'params': 60.2, 'acc': 96.8, 'color': '#DDA0DD'},
]

names = [a['name'] for a in architectures]
years = [a['year'] for a in architectures]
layers = [a['layers'] for a in architectures]
params = [a['params'] for a in architectures]
accs = [a['acc'] for a in architectures]
colors = [a['color'] for a in architectures]

fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

ax1 = fig.add_subplot(gs[0, 0])
bars1 = ax1.barh(names, layers, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_xlabel('网络深度（层数）', fontsize=11, fontweight='bold')
ax1.set_title('CNN 架构深度演进', fontsize=12, fontweight='bold')
ax1.set_xlim(0, 160)
for i, (bar, layer) in enumerate(zip(bars1, layers)):
    ax1.text(layer + 3, bar.get_y() + bar.get_height()/2,
            f'{layer}层', va='center', fontsize=9, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

ax2 = fig.add_subplot(gs[0, 1])
bars2 = ax2.barh(names, params, color=colors, edgecolor='black', linewidth=1.5)
ax2.set_xlabel('参数量（百万）', fontsize=11, fontweight='bold')
ax2.set_title('CNN 架构参数量对比', fontsize=12, fontweight='bold')
ax2.set_xscale('log')
for i, (bar, param) in enumerate(zip(bars2, params)):
    if param >= 1:
        ax2.text(param * 1.2, bar.get_y() + bar.get_height()/2,
                f'{param:.1f}M', va='center', fontsize=9, fontweight='bold')
    else:
        ax2.text(param * 1.5, bar.get_y() + bar.get_height()/2,
                f'{param*1000:.0f}K', va='center', fontsize=9, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

ax3 = fig.add_subplot(gs[1, 0])
bars3 = ax3.barh(names, accs, color=colors, edgecolor='black', linewidth=1.5)
ax3.set_xlabel('ImageNet Top-5 准确率 (%)', fontsize=11, fontweight='bold')
ax3.set_title('CNN 架构准确率演进', fontsize=12, fontweight='bold')
ax3.set_xlim(80, 100)
for i, (bar, acc) in enumerate(zip(bars3, accs)):
    ax3.text(acc + 0.3, bar.get_y() + bar.get_height()/2,
            f'{acc}%', va='center', fontsize=9, fontweight='bold')
ax3.grid(axis='x', alpha=0.3)

ax4 = fig.add_subplot(gs[1, 1])
for i, arch in enumerate(architectures):
    ax4.scatter(arch['params'], arch['acc'], s=arch['layers']*15,
               c=arch['color'], edgecolors='black', linewidth=2,
               label=arch['name'], alpha=0.7)
    ax4.annotate(arch['name'], (arch['params'], arch['acc']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=9, fontweight='bold')
ax4.set_xlabel('参数量（百万）', fontsize=11, fontweight='bold')
ax4.set_ylabel('ImageNet Top-5 准确率 (%)', fontsize=11, fontweight='bold')
ax4.set_title('准确率 vs 参数量（气泡大小=网络深度）', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.set_xscale('log')

ax5 = fig.add_subplot(gs[2, 0])
kernel_evolution = [
    ('LeNet-5', [5, 2, 'avg']),
    ('AlexNet', [11, 5, 3, 'max']),
    ('VGG-16', [3, 3, 3, 3, 'max']),
    ('GoogLeNet', [1, 3, 5, 'max']),
    ('ResNet', [1, 3, 1, 'avg'])
]

y_pos = np.arange(len(kernel_evolution))
for i, (name, kernels) in enumerate(kernel_evolution):
    x_pos = 0
    for j, k in enumerate(kernels):
        if k == 'max':
            color = '#FF6B6B'
            width = 1
        elif k == 'avg':
            color = '#4ECDC4'
            width = 1
        else:
            color = plt.cm.Blues(0.3 + 0.7 * (k / 11))
            width = k / 3
        rect = Rectangle((x_pos, i-0.3), width, 0.6,
                       facecolor=color, edgecolor='black', linewidth=1)
        ax5.add_patch(rect)
        if isinstance(k, int):
            ax5.text(x_pos + width/2, i, f'{k}×{k}',
                    ha='center', va='center', fontsize=8, fontweight='bold')
        x_pos += width + 0.2
ax5.set_yticks(y_pos)
ax5.set_yticklabels([name for name, _ in kernel_evolution])
ax5.set_xlabel('架构组成', fontsize=11, fontweight='bold')
ax5.set_title('卷积核尺寸演进（从大到小）', fontsize=12, fontweight='bold')
ax5.set_xlim(0, 12)
legend_elements = [
    mpatches.Patch(facecolor=plt.cm.Blues(0.7), edgecolor='black', label='卷积层'),
    mpatches.Patch(facecolor='#FF6B6B', edgecolor='black', label='Max Pooling'),
    mpatches.Patch(facecolor='#4ECDC4', edgecolor='black', label='Avg Pooling')
]
ax5.legend(handles=legend_elements, loc='lower right', fontsize=8)

ax6 = fig.add_subplot(gs[2, 1])
ax6.axis('off')
milestones = [
    (1998, 'LeNet-5', 'CNN奠基', '#FF6B6B'),
    (2012, 'AlexNet', '深度学习复兴\nReLU+GPU+Dropout', '#4ECDC4'),
    (2014, 'VGGNet', '小卷积核堆叠\n3×3标准范式', '#45B7D1'),
    (2014, 'GoogLeNet', 'Inception多尺度\n1×1降维', '#96CEB4'),
    (2015, 'ResNet', '残差连接\n解决深度退化', '#FFEAA7'),
    (2017, 'DenseNet', '密集连接\n特征复用', '#DDA0DD'),
    (2018, 'SE-Net', '通道注意力\n自适应特征重标定', '#F8B500')
]
ax6.axhline(y=0.5, color='gray', linestyle='-', linewidth=2, alpha=0.5)
for i, (year, name, desc, color) in enumerate(milestones):
    y_offset = 0.7 if i % 2 == 0 else 0.3
    ax6.scatter(year, 0.5, s=200, c=color, edgecolors='black', linewidth=2, zorder=5)
    ax6.text(year, 0.5, str(year), ha='center', va='center',
            fontsize=9, fontweight='bold', zorder=6)
    ax6.plot([year, year], [0.5, y_offset], color=color, linewidth=2, linestyle='--', alpha=0.7)
    ax6.text(year, y_offset, f'{name}\n{desc}', ha='center',
            va='center' if i % 2 == 0 else 'top',
            fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))
ax6.set_xlim(1995, 2020)
ax6.set_ylim(0, 1)
ax6.set_title('CNN 架构关键技术突破时间线', fontsize=12, fontweight='bold')

plt.suptitle('经典 CNN 架构演进全景图', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/david/AI-Tutorial/assets/figures/4-3-cnn-architectures.png',
            dpi=120, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: 4-3-cnn-architectures.png")

# ResNet vs Plain
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
layers_arr = np.arange(1, 21)
plain_gradients = np.exp(-layers_arr * 0.15) + 0.05
resnet_gradients = np.clip(np.ones_like(layers_arr) * 0.6 + np.random.normal(0, 0.05, len(layers_arr)), 0.4, 0.8)

ax1 = axes2[0]
ax1.plot(layers_arr, plain_gradients, 'o-', color='red', linewidth=2,
        markersize=8, label='普通深度网络', alpha=0.8)
ax1.plot(layers_arr, resnet_gradients, 's-', color='green', linewidth=2,
        markersize=8, label='ResNet（残差连接）', alpha=0.8)
ax1.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5, label='有效学习阈值')
ax1.fill_between(layers_arr, 0, 0.1, alpha=0.2, color='gray')
ax1.set_xlabel('网络层数', fontsize=11, fontweight='bold')
ax1.set_ylabel('梯度大小', fontsize=11, fontweight='bold')
ax1.set_title('梯度流动对比：普通网络 vs ResNet', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1)

ax2 = axes2[1]
depths = [20, 32, 44, 56, 110, 152]
plain_errors = [25, 28, 35, 42, 55, 65]
resnet_errors = [25, 22, 20, 18, 15, 13]
ax2.plot(depths, plain_errors, 'o-', color='red', linewidth=2, markersize=10, label='普通深度网络', alpha=0.8)
ax2.plot(depths, resnet_errors, 's-', color='green', linewidth=2, markersize=10, label='ResNet（残差连接）', alpha=0.8)
ax2.set_xlabel('网络深度', fontsize=11, fontweight='bold')
ax2.set_ylabel('测试误差 (%)', fontsize=11, fontweight='bold')
ax2.set_title('网络深度 vs 测试误差（CIFAR-10）', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10, loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.annotate('退化问题：\n深度增加反而变差', xy=(110, 55), xytext=(80, 60),
            fontsize=9, bbox=dict(boxstyle='round', facecolor='red', alpha=0.3),
            arrowprops=dict(arrowstyle='->', color='red'))
ax2.annotate('ResNet 解决退化\n深度增加性能提升', xy=(152, 13), xytext=(130, 20),
            fontsize=9, bbox=dict(boxstyle='round', facecolor='green', alpha=0.3),
            arrowprops=dict(arrowstyle='->', color='green'))
plt.suptitle('ResNet 如何解决深度网络退化问题', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/david/AI-Tutorial/assets/figures/4-3-resnet-vs-plain.png',
            dpi=120, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: 4-3-resnet-vs-plain.png")
