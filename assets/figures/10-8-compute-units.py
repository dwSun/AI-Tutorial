"""
Compute Units Comparison: TOPS / FLOPS / FLOPs / MAC
===================================================
Visualizes the differences between various compute units and their
relationships, addressing common confusion (TOPS vs TFLOPS).

Output: assets/figures/10-8-compute-units.png
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


fig, axes = plt.subplots(1, 2, figsize=(18, 9))
fig.suptitle('算力单位详解：TOPS / FLOPS / FLOPs / MAC 关系',
             fontsize=15, fontweight='bold', y=0.97)


# === Left: Compute Unit Pyramid ===
ax = axes[0]
ax.set_xlim(0, 10); ax.set_ylim(0, 14)
ax.axis('off')

units = [
    (11.5, 'OPs (Operations)\n原始操作次数', '#FFE4B5', 9, '最基础：1 次 mul 或 1 次 add'),
    (9.5, 'GOPS / MOPS / TOPS\n(G=10⁹, M=10⁶, T=10¹²)', '#FFD700', 9, '通用名："每秒多少万亿次操作"\n⚠️ 不一定是浮点！'),
    (7.5, 'FLOPS (浮点操作/秒)\nMFLOPS / GFLOPS / TFLOPS / PFLOPS', '#FFA500', 10, '必须是浮点，精度相关'),
    (5.5, 'FLOPs (浮点操作总量)\n1 FLOPs = 1 FP32 op', '#87CEEB', 9, '表示模型计算量，不是速度'),
    (3.5, 'MAC (Multiply Accumulate)\n1 MAC = 1 mul + 1 add = 2 OPs', '#90EE90', 9, '矩阵乘法的基本单位'),
    (1.5, 'DMIPS (CPU 整数性能)\nDhrystone Million Instructions/s', '#FFB6C1', 9, 'CPU 整数性能专用'),
]

for y, name, color, fs, desc in units:
    box = FancyBboxPatch((0.5, y - 0.7), 9, 1.4, boxstyle='round,pad=0.05',
                          fc=color, ec='black', linewidth=1.5)
    ax.add_patch(box)
    ax.text(5, y + 0.2, name, ha='center', va='center', fontsize=fs, fontweight='bold')
    ax.text(5, y - 0.3, desc, ha='center', va='center', fontsize=8, style='italic', color='#333')

# Arrows
for y_pair in [(11, 10), (9, 8), (7, 6), (5, 4), (3, 2)]:
    arr = FancyArrowPatch((5, y_pair[0]), (5, y_pair[1]),
                          arrowstyle='->', color='#444', lw=1.5, mutation_scale=15)
    ax.add_patch(arr)

ax.text(5, 13.5, '⚠️ 核心陷阱：TOPS 算的是操作，TFLOPS 算的是浮点', ha='center',
        fontsize=11, color='red', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#FFE0E0', edgecolor='red'))


# === Right: Precision vs Compute Power (same chip, different precision) ===
ax = axes[1]
ax.set_xlim(0, 10); ax.set_ylim(0, 10)

precisions = ['INT4', 'INT8', 'FP16/BF16', 'TF32', 'FP32', 'FP64']
# 假设一个"基准 1 TOPS @ FP32" 的芯片
ratios = [4, 2, 1, 0.5, 1, 0.0625]  # 相对 FP32 算力
labels_text = [
    '4× (量化加速)',
    '2× (典型 INT8 加速)',
    '1× (FP16 Tensor Core 满速)',
    '0.5× (TF32 折损)',
    '1× (FP32 基准)',
    '0.0625× (1/64, 消费卡)',
]
colors_p = ['#90EE90', '#FFD700', '#FFA500', '#FFB6C1', '#87CEEB', '#FF6B6B']

bars = ax.barh(range(len(precisions)), ratios, color=colors_p, edgecolor='black', linewidth=1.5)
ax.set_yticks(range(len(precisions)))
ax.set_yticklabels(precisions, fontsize=11, fontweight='bold')
ax.set_xlabel('相对算力 (FP32 = 1.0)', fontsize=11, fontweight='bold')
ax.set_title('同一颗芯片, 不同精度的算力\n(同 1 颗 GPU 在 6 种精度下的相对性能)',
             fontsize=12, fontweight='bold')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)
ax.set_xscale('log')
ax.set_xlim(0.05, 10)

for i, (bar, val, label) in enumerate(zip(bars, ratios, labels_text)):
    ax.text(val * 1.1, bar.get_y() + bar.get_height()/2, label,
            va='center', fontsize=9, fontweight='bold')

ax.axvline(1.0, color='red', linestyle='--', lw=1.5, alpha=0.7)
ax.text(1.0, -0.7, 'FP32 基线', ha='center', fontsize=10, color='red', fontweight='bold')

ax.text(5, 9.3,
        '⚠️ 厂商陷阱：很多 AI 芯片只标 INT8 TOPS，不标 FP16/TFLOPS\nFP16 实际算力可能只有 INT8 的 25-50%！',
        ha='center', fontsize=10, color='red', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#FFE0E0', edgecolor='red', lw=1.5))

plt.tight_layout()
plt.savefig('assets/figures/10-8-compute-units.png', dpi=120, bbox_inches='tight', facecolor='white')
print('  ✓ Saved: assets/figures/10-8-compute-units.png')
