"""
Confusion Matrix for Error Analysis
===================================
Visualizes how to use confusion matrix to find the model's weakest classes,
then drill into specific misclassified samples.

Output: assets/figures/2-3-confusion-error-analysis.png
"""
import matplotlib.pyplot as plt
import numpy as np


np.random.seed(42)

# Simulate a 5-class confusion matrix (e.g., CIFAR-10 subset: cat, dog, car, plane, bird)
classes = ['Cat', 'Dog', 'Car', 'Plane', 'Bird']
n = len(classes)

# Build a realistic confusion matrix
# Strong diagonal, but systematic confusions: cat<->dog, car<->plane
cm = np.array([
    [85, 10, 1, 1, 3],   # cat: confuses 10% with dog, 3% with bird
    [8, 82, 2, 1, 7],    # dog: confuses 8% with cat, 7% with bird
    [2, 1, 88, 6, 3],    # car: confuses 6% with plane
    [1, 1, 5, 90, 3],    # plane: confuses 5% with car
    [3, 5, 4, 2, 86],    # bird: 5% with dog
], dtype=float)

# Normalize per row to get per-class recall
cm_norm = cm / cm.sum(axis=1, keepdims=True) * 100

# Compute per-class metrics
recall = np.diag(cm_norm)
precision = np.diag(cm / cm.sum(axis=0, keepdims=True) * 100)
errors_per_class = 100 - recall

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3, width_ratios=[1.4, 1, 1], height_ratios=[1, 0.7])

# Plot 1: Confusion matrix heatmap
ax1 = fig.add_subplot(gs[:, 0])
im = ax1.imshow(cm_norm, cmap='Blues', vmin=0, vmax=100)
ax1.set_xticks(range(n))
ax1.set_yticks(range(n))
ax1.set_xticklabels(classes, fontsize=11)
ax1.set_yticklabels(classes, fontsize=11)
ax1.set_xlabel('Predicted', fontsize=12, fontweight='bold')
ax1.set_ylabel('Actual', fontsize=12, fontweight='bold')
ax1.set_title('Normalized Confusion Matrix\n(Recall per row, %)',
              fontsize=13, fontweight='bold', pad=10)

# Annotate cells
for i in range(n):
    for j in range(n):
        color = 'white' if cm_norm[i, j] > 50 else 'black'
        weight = 'bold' if i == j else 'normal'
        ax1.text(j, i, f'{cm_norm[i, j]:.0f}%', ha='center', va='center',
                 color=color, fontweight=weight, fontsize=11)

# Highlight systematic confusions
ax1.add_patch(plt.Rectangle((-0.5, 0.5), 1, 1, fill=False, edgecolor='red', lw=3))
ax1.add_patch(plt.Rectangle((0.5, -0.5), 1, 1, fill=False, edgecolor='red', lw=3))
ax1.add_patch(plt.Rectangle((2.5, 3.5), 1, 1, fill=False, edgecolor='orange', lw=3))
ax1.add_patch(plt.Rectangle((3.5, 2.5), 1, 1, fill=False, edgecolor='orange', lw=3))
ax1.text(0.5, 0.5, '', color='red')
plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label='Recall (%)')

# Plot 2: Per-class error rate
ax2 = fig.add_subplot(gs[0, 1])
sorted_idx = np.argsort(errors_per_class)[::-1]
colors = ['#dc3545' if e > 15 else '#fd7e14' if e > 10 else '#28a745' for e in errors_per_class[sorted_idx]]
bars = ax2.barh(range(n), errors_per_class[sorted_idx], color=colors, edgecolor='black')
ax2.set_yticks(range(n))
ax2.set_yticklabels([classes[i] for i in sorted_idx], fontsize=11)
ax2.set_xlabel('Error Rate (%)', fontsize=11, fontweight='bold')
ax2.set_title('Per-Class Error Rate\n(找出最差的类别)', fontsize=12, fontweight='bold')
ax2.invert_yaxis()
ax2.grid(axis='x', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, errors_per_class[sorted_idx])):
    ax2.text(val + 0.5, bar.get_y() + bar.get_height()/2, f'{val:.0f}%',
             va='center', fontsize=10, fontweight='bold')

# Plot 3: Top-K confusions
ax3 = fig.add_subplot(gs[0, 2])
confusions = []
for i in range(n):
    for j in range(n):
        if i != j and cm[i, j] > 3:
            confusions.append((cm[i, j], i, j))
confusions.sort(reverse=True)
top_k = confusions[:5]

ax3.axis('off')
ax3.set_title('Top 5 Systematic Confusions\n(对角线外的最大错分)',
              fontsize=12, fontweight='bold', pad=10)
ax3_text = '\n'.join([f'{count:3.0f} | {classes[i]:5s} → {classes[j]:5s}'
                       for count, i, j in top_k])
ax3.text(0.1, 0.95, ax3_text, fontsize=11, family='monospace', va='top',
         transform=ax3.transAxes,
         bbox=dict(boxstyle='round', facecolor='#FFE4B5', edgecolor='black', lw=1.5))

# Plot 4: Recommendation flow
ax4 = fig.add_subplot(gs[1, 1:])
ax4.axis('off')
ax4.set_title('从混淆矩阵到数据清洗 / 模型改进的流程',
              fontsize=12, fontweight='bold', pad=10)

flow = """
┌─────────────────────────────────────┐
│ Step 1: 训练模型，画出 Confusion Matrix │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│ Step 2: 计算 Per-Class Recall / Error │
│         → 找出 Recall < 80% 的类       │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│ Step 3: 提取 Top-K Systematic 错分   │
│         (对角线外最大值对)             │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│ Step 4: 三选一行动                    │
│  • 数据清洗：检查错分样本的标签        │
│  • 数据增强：对困难类做更多增广        │
│  • 模型调整：增加类权重 / Focal Loss  │
└─────────────────────────────────────┘
"""
ax4.text(0.5, 0.95, flow, fontsize=10, family='monospace', va='top', ha='center',
         transform=ax4.transAxes,
         bbox=dict(boxstyle='round', facecolor='#E8F4FD', edgecolor='#1F77B4', lw=1.5))

fig.suptitle('用混淆矩阵做错误分析 (Error Analysis) — 数据清洗的起点',
             fontsize=15, fontweight='bold', y=0.995)

plt.tight_layout()
plt.savefig('assets/figures/2-3-confusion-error-analysis.png', dpi=120, bbox_inches='tight', facecolor='white')
print('  ✓ Saved: assets/figures/2-3-confusion-error-analysis.png')
