"""
Confusion Matrix for Label Noise Detection
==========================================
Visualizes how a confusion matrix can reveal mislabeled training data:
- High "off-diagonal but symmetric" pairs (A<->B) often indicate label noise
- Asymmetric confusions suggest genuine class similarity

Output: assets/figures/2-3-confusion-label-noise.png
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


np.random.seed(42)
classes = ['Cat', 'Dog', 'Car', 'Plane', 'Bird', 'Fish']
n = len(classes)

# Simulate a confusion matrix with INTENTIONAL label noise:
# 8% of "Cat" images are actually "Dog" (mislabeled in training)
# 6% of "Plane" images are actually "Car"
cm_noisy = np.array([
    [82, 8, 1, 1, 5, 3],    # Cat: 8% labeled-as-dog, 5% labeled-as-bird
    [6, 80, 2, 1, 8, 3],    # Dog: symmetric with Cat
    [2, 1, 84, 6, 4, 3],    # Car: 6% mislabeled as Plane
    [1, 1, 5, 88, 3, 2],    # Plane: symmetric with Car
    [4, 6, 3, 2, 80, 5],    # Bird
    [2, 2, 2, 1, 4, 89],    # Fish
], dtype=float)

# Normalize per row
cm_norm = cm_noisy / cm_noisy.sum(axis=1, keepdims=True) * 100

# Symmetric pairs indicator
sym_mask = np.zeros_like(cm_norm)
for i in range(n):
    for j in range(i+1, n):
        avg = (cm_norm[i, j] + cm_norm[j, i]) / 2
        sym_mask[i, j] = avg
        sym_mask[j, i] = avg

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# Left: confusion matrix
im = axes[0].imshow(cm_norm, cmap='Reds', vmin=0, vmax=20)
axes[0].set_xticks(range(n))
axes[0].set_yticks(range(n))
axes[0].set_xticklabels(classes, fontsize=11)
axes[0].set_yticklabels(classes, fontsize=11)
axes[0].set_xlabel('Predicted', fontsize=12, fontweight='bold')
axes[0].set_ylabel('True (in dataset)', fontsize=12, fontweight='bold')
axes[0].set_title('Confusion Matrix with Label Noise',
                  fontsize=12, fontweight='bold')
for i in range(n):
    for j in range(n):
        color = 'white' if cm_norm[i, j] > 10 else 'black'
        axes[0].text(j, i, f'{cm_norm[i, j]:.0f}', ha='center', va='center',
                     color=color, fontsize=10)
plt.colorbar(im, ax=axes[0], fraction=0.046, label='%')

# Middle: symmetric matrix - highlights "noisy pairs"
im2 = axes[1].imshow(sym_mask, cmap='Greens', vmin=0, vmax=10)
axes[1].set_xticks(range(n))
axes[1].set_yticks(range(n))
axes[1].set_xticklabels(classes, fontsize=11)
axes[1].set_yticklabels(classes, fontsize=11)
axes[1].set_xlabel('Predicted', fontsize=12, fontweight='bold')
axes[1].set_ylabel('True', fontsize=12, fontweight='bold')
axes[1].set_title('Symmetric Confusion\nmin(CM[i,j], CM[j,i])\n高对称 → 疑似标签噪声',
                  fontsize=12, fontweight='bold')
for i in range(n):
    for j in range(n):
        if i != j and sym_mask[i, j] > 2:
            color = 'white' if sym_mask[i, j] > 5 else 'black'
            axes[1].text(j, i, f'{sym_mask[i, j]:.1f}', ha='center', va='center',
                         color=color, fontsize=10, fontweight='bold')
plt.colorbar(im2, ax=axes[1], fraction=0.046, label='%')

# Right: noise detection rules
axes[2].axis('off')
axes[2].set_title('Label Noise Detection Heuristics',
                  fontsize=12, fontweight='bold', pad=10)

rules = """
1. SYMMETRIC PAIRS (高对称错分)
   CM[A,B] ≈ CM[B,A]  (双向都有)
   → 多半是标签错标 / 类别定义模糊
   → 例: Cat ↔ Dog (8%/6%) 提示猫狗混淆
   
2. ASYMMETRIC PAIRS (单向错分)
   CM[A,B] >> CM[B,A]  (只有 A→B 多)
   → 可能是真实类别相似性
   → 例: Plane → Car 多于 Car → Plane
   → 这是 "Hard Example"，不是噪声

3. HIGH-OFF-DIAGONAL ROW SUMS
   row i 的非对角线总和 > 25%
   → 该类的标签质量可能有问题
   → 需要人工核查该类的训练样本

4. CONFIDENCE-WEIGHTED ANALYSIS
   模型非常确信但错分（高 probs）
   → 强烈提示标签错误
   → 用作 "清洗候选" 优先列表
"""
axes[2].text(0.05, 0.95, rules, fontsize=10, family='monospace', va='top',
             transform=axes[2].transAxes,
             bbox=dict(boxstyle='round', facecolor='#FFF3CD', edgecolor='black'))

fig.suptitle('用混淆矩阵识别数据集中的标签噪声 (Label Noise Detection)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('assets/figures/2-3-confusion-label-noise.png', dpi=120, bbox_inches='tight', facecolor='white')
print('  ✓ Saved: assets/figures/2-3-confusion-label-noise.png')
