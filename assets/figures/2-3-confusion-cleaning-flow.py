"""
Confusion Matrix Before/After Data Cleaning
===========================================
Visualizes how a confusion matrix improves after cleaning noisy labels
in the training set. Shows that cleaning can lift the diagonal (correct
predictions) and reduce off-diagonal noise.

Output: assets/figures/2-3-confusion-cleaning-flow.png
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


np.random.seed(42)
classes = ['Cat', 'Dog', 'Car', 'Plane', 'Bird']
n = len(classes)

# Before cleaning: noisy labels
cm_before = np.array([
    [78, 10, 2, 1, 9],
    [8, 75, 3, 2, 12],
    [3, 2, 82, 8, 5],
    [2, 1, 7, 84, 6],
    [6, 8, 4, 5, 77],
], dtype=float)
cm_before = cm_before / cm_before.sum(axis=1, keepdims=True) * 100
recall_before = np.diag(cm_before)
acc_before = np.trace(cm_before) / n

# After cleaning: cleaner diagonal
cm_after = np.array([
    [88, 6, 1, 1, 4],
    [5, 87, 2, 1, 5],
    [2, 1, 90, 5, 2],
    [1, 1, 5, 91, 2],
    [3, 5, 2, 2, 88],
], dtype=float)
cm_after = cm_after / cm_after.sum(axis=1, keepdims=True) * 100
recall_after = np.diag(cm_after)
acc_after = np.trace(cm_after) / n

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Top: confusion matrices
for ax, cm, title, acc in [
    (axes[0, 0], cm_before, 'Before Cleaning (Noisy Labels)\n平均 Recall = {:.1f}%'.format(recall_before.mean()), acc_before),
    (axes[0, 1], cm_after, 'After Cleaning (Labels Verified)\n平均 Recall = {:.1f}%'.format(recall_after.mean()), acc_after)
]:
    im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=100)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_yticklabels(classes, fontsize=11)
    ax.set_xlabel('Predicted', fontsize=11, fontweight='bold')
    ax.set_ylabel('True', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    for i in range(n):
        for j in range(n):
            color = 'white' if cm[i, j] > 50 else 'black'
            weight = 'bold' if i == j else 'normal'
            ax.text(j, i, f'{cm[i, j]:.0f}', ha='center', va='center',
                    color=color, fontweight=weight, fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='%')

# Bottom: per-class improvement
x = np.arange(n)
width = 0.35
axes[1, 0].bar(x - width/2, recall_before, width, label='Before', color='#dc3545', alpha=0.8)
axes[1, 0].bar(x + width/2, recall_after, width, label='After', color='#28a745', alpha=0.8)
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(classes, fontsize=11)
axes[1, 0].set_ylabel('Recall (%)', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Per-Class Recall Improvement\n(清洗后的提升)', fontsize=12, fontweight='bold')
axes[1, 0].legend(fontsize=11)
axes[1, 0].set_ylim(60, 100)
axes[1, 0].grid(axis='y', alpha=0.3)
for i, (b, a) in enumerate(zip(recall_before, recall_after)):
    axes[1, 0].text(i - width/2, b + 0.5, f'{b:.0f}', ha='center', fontsize=9)
    axes[1, 0].text(i + width/2, a + 0.5, f'{a:.0f}', ha='center', fontsize=9, fontweight='bold', color='green')

# Bottom right: 4-step cleaning flow
axes[1, 1].axis('off')
axes[1, 1].set_title('4-Step Confusion-Matrix-Driven Data Cleaning',
                     fontsize=12, fontweight='bold', pad=10)

steps = [
    ('1. 训练 Baseline\n   + 画 Confusion Matrix', '#FFE4B5'),
    ('2. 识别 Symmetric Pairs\n   (对角线外的高对称对)', '#FFB6C1'),
    ('3. 提取错分样本 + High-Confidence 错分\n   → 优先清洗列表', '#E0FFFF'),
    ('4. 人工 / 半自动核查\n   + 重训练 → 提升 {0:.1f}%'.format(acc_after - acc_before), '#90EE90'),
]

for i, (text, color) in enumerate(steps):
    y = 0.85 - i * 0.22
    box = FancyBboxPatch((0.05, y - 0.07), 0.9, 0.15,
                          boxstyle='round,pad=0.02', fc=color, ec='black', lw=1.5)
    axes[1, 1].add_patch(box)
    axes[1, 1].text(0.5, y, text, ha='center', va='center', fontsize=10, fontweight='bold')
    if i < 3:
        arrow = FancyArrowPatch((0.5, y - 0.075), (0.5, y - 0.145),
                                 arrowstyle='->', mutation_scale=20, color='black', lw=2)
        axes[1, 1].add_patch(arrow)

fig.suptitle('用混淆矩阵驱动数据清洗 (Confusion-Matrix-Driven Data Cleaning)',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('assets/figures/2-3-confusion-cleaning-flow.png', dpi=120, bbox_inches='tight', facecolor='white')
print('  ✓ Saved: assets/figures/2-3-confusion-cleaning-flow.png')
