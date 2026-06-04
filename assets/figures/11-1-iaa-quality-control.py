"""
Inter-Annotator Agreement (IAA) for Label Quality
=================================================
Visualizes how Cohen's Kappa and other metrics measure annotation quality.

Output: assets/figures/11-1-iaa-quality-control.png
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


np.random.seed(42)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('标注质量控制：Inter-Annotator Agreement (IAA)',
             fontsize=15, fontweight='bold', y=1.02)


# === Panel 1: 混淆矩阵 - 双标注员 ===
ax = axes[0]
classes = ['cat', 'dog', 'car']
n = len(classes)
# Simulate 2 annotators
cm = np.array([
    [85, 5, 2],   # cat
    [6, 88, 3],   # dog
    [3, 2, 92],   # car
], dtype=float)
cm_norm = cm / cm.sum(axis=1, keepdims=True) * 100
im = ax.imshow(cm_norm, cmap='Greens', vmin=0, vmax=100)
ax.set_xticks(range(n)); ax.set_yticks(range(n))
ax.set_xticklabels(classes, fontsize=11)
ax.set_yticklabels(classes, fontsize=11)
ax.set_xlabel('Annotator B', fontsize=11, fontweight='bold')
ax.set_ylabel('Annotator A', fontsize=11, fontweight='bold')
ax.set_title('Annotator A vs B 混淆矩阵\n(对角线 = 一致, 偏离 = 争议)',
             fontsize=11, fontweight='bold')
for i in range(n):
    for j in range(n):
        color = 'white' if cm_norm[i, j] > 50 else 'black'
        weight = 'bold' if i == j else 'normal'
        ax.text(j, i, f'{cm_norm[i, j]:.0f}', ha='center', va='center',
                color=color, fontweight=weight, fontsize=11)
plt.colorbar(im, ax=ax, fraction=0.046, label='%')


# === Panel 2: Cohen's Kappa 解释 ===
ax = axes[1]
ax.axis('off')
ax.set_title("Cohen's Kappa (κ) 解释", fontsize=12, fontweight='bold')

kappa_text = """
公式: κ = (p_o - p_e) / (1 - p_e)
  p_o: 实际一致率
  p_e: 随机一致率

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

κ = 1.00  完美一致
κ = 0.81 - 0.99 几乎完美 (Substantial)
κ = 0.61 - 0.80 实质性一致 (Moderate) ← 工业标准
κ = 0.41 - 0.60 中等 (Fair) ← 需要核查
κ = 0.21 - 0.40 一般 (Slight) ← 标注员培训
κ < 0.20   一致性差 (Poor) ← 重新定义类别

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

工业门槛: κ ≥ 0.7
对困难任务: κ ≥ 0.5 也可接受
"""
ax.text(0.05, 0.95, kappa_text, fontsize=10, family='monospace', va='top',
        transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='#FFF3CD', edgecolor='black', lw=1.5))


# === Panel 3: 标注质量控制流程 ===
ax = axes[2]
ax.set_xlim(0, 10); ax.set_ylim(0, 12)
ax.set_aspect('equal'); ax.axis('off')
ax.set_title('3 阶段质量控制流程', fontsize=12, fontweight='bold')

steps = [
    ('1. 标注员培训\n   类别定义 + 样例标注\n   试标 κ 检验', '#FFE4B5', 10.5),
    ('2. 双标注 + 仲裁\n   2 人独立标同一批\n   不一致由专家仲裁', '#FFB6C1', 7.5),
    ('3. 持续抽样审计\n   随机抽 5%-10% 复检\n   IAA 监控 + 反馈', '#90EE90', 4.5),
]

for i, (text, color, y) in enumerate(steps):
    box = FancyBboxPatch((0.5, y - 1.0), 9, 1.7,
                          boxstyle='round,pad=0.05', fc=color, ec='black', lw=1.5)
    ax.add_patch(box)
    ax.text(5, y - 0.15, text, ha='center', va='center', fontsize=10, fontweight='bold')
    if i < 2:
        arrow = FancyArrowPatch((5, y - 1.0), (5, y - 1.7),
                                 arrowstyle='->', mutation_scale=20, color='black', lw=2)
        ax.add_patch(arrow)

ax.text(5, 0.5, '质量门槛: 双标注 IAA κ ≥ 0.7\n否则: 重新培训 + 重新定义类别',
        ha='center', va='center', fontsize=10, color='red', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#FFE0E0', edgecolor='red', lw=1.5))

plt.tight_layout()
plt.savefig('assets/figures/11-1-iaa-quality-control.png', dpi=120, bbox_inches='tight', facecolor='white')
print('  ✓ Saved: assets/figures/11-1-iaa-quality-control.png')
