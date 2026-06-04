"""
GPU Hardware Comparison for AI Training
========================================
Compares 2024-2026 NVIDIA GPUs across price, VRAM, FP16 TFLOPS,
and what training tasks they can handle.

Output: assets/figures/10-8-gpu-comparison.png
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


gpus = [
    # (name, vram_GB, fp16_tflops, price_usd, segment)
    ('RTX 4060',  8,  15.1,  300,  'entry'),
    ('RTX 4070',  12, 29.1,  600,  'entry'),
    ('RTX 4080',  16, 48.7,  1200, 'mid'),
    ('RTX 4090',  24, 82.6,  1600, 'mid-high'),
    ('RTX 5090',  32, 105.0, 2500, 'high'),
    ('A4000',     16, 38.7,  1000, 'workstation'),
    ('A5000',     24, 77.4,  1700, 'workstation'),
    ('A6000 Ada', 48, 91.1,  6800, 'workstation'),
    ('L40',       48, 90.5,  10000, 'datacenter'),
    ('L40S',      48, 91.6,  12000, 'datacenter'),
    ('A100 40G',  40, 77.9,  10000, 'datacenter'),
    ('A100 80G',  80, 77.9,  15000, 'datacenter'),
    ('H100 80G',  80, 197.9, 30000, 'datacenter'),
    ('H200 141G', 141, 197.9, 40000, 'datacenter'),
    ('B200',      192, 2250.0, 40000, 'datacenter'),
]

names = [g[0] for g in gpus]
vrams = [g[1] for g in gpus]
tflops = [g[2] for g in gpus]
prices = [g[3] for g in gpus]
segments = [g[4] for g in gpus]

seg_colors = {
    'entry':       '#90EE90',
    'mid':         '#FFD700',
    'mid-high':    '#FFA500',
    'workstation': '#87CEEB',
    'datacenter':  '#FF6B6B',
    'high':        '#FF1493',
}

fig, axes = plt.subplots(1, 2, figsize=(18, 9))

# === Left: VRAM vs TFLOPS (bubble = price) ===
ax = axes[0]
for i, g in enumerate(gpus):
    color = seg_colors[g[4]]
    ax.scatter(g[1], g[2], s=g[3] * 4, c=color, alpha=0.6,
                edgecolors='black', linewidths=1)
    ax.annotate(g[0], (g[1], g[2]), xytext=(5, 5),
                textcoords='offset points', fontsize=8, fontweight='bold')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('VRAM (GB, log scale)', fontsize=11, fontweight='bold')
ax.set_ylabel('FP16 TFLOPS (log scale)', fontsize=11, fontweight='bold')
ax.set_title('GPU 算力 vs 显存 (气泡 = 价格)\n右上方 = 性价比之选', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, which='both')

# Add zone labels
ax.axvspan(8, 16, alpha=0.05, color='green', zorder=-1)
ax.axvspan(16, 32, alpha=0.05, color='yellow', zorder=-1)
ax.axvspan(32, 80, alpha=0.05, color='orange', zorder=-1)
ax.axvspan(80, 200, alpha=0.05, color='red', zorder=-1)
ax.text(12, 5, 'CNN/分类/检测', fontsize=10, color='green', fontweight='bold')
ax.text(24, 5, '大 CNN/小 LLM', fontsize=10, color='orange', fontweight='bold')
ax.text(60, 5, '中 LLM/QLoRA', fontsize=10, color='red', fontweight='bold')
ax.text(150, 5, 'LLM 全参微调', fontsize=10, color='darkred', fontweight='bold')

# Legend
legend_elements = [mpatches.Patch(color=c, label=s) for s, c in seg_colors.items()]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9, title='市场定位')

# === Right: 性价比 (TFLOPS per $) ===
ax = axes[1]
perf_per_dollar = [g[2] / g[3] * 1000 for g in gpus]  # TFLOPS per $1000
bars = ax.barh(range(len(names)), perf_per_dollar,
                 color=[seg_colors[g[4]] for g in gpus], edgecolor='black')
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=10)
ax.set_xlabel('Performance per Dollar (TFLOPS per $1000)', fontsize=11, fontweight='bold')
ax.set_title('GPU 性价比 (算力 / 价格)\n越长 = 越划算', fontsize=12, fontweight='bold')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars, perf_per_dollar)):
    ax.text(val + 0.3, bar.get_y() + bar.get_height()/2, f'{val:.1f}',
            va='center', fontsize=9, fontweight='bold')

fig.suptitle('2024-2026 AI 训练 GPU 全景对比', fontsize=15, fontweight='bold', y=1.005)
plt.tight_layout()
plt.savefig('assets/figures/10-8-gpu-comparison.png', dpi=120, bbox_inches='tight', facecolor='white')
print('  ✓ Saved: assets/figures/10-8-gpu-comparison.png')
