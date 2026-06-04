"""
FLOPs vs Actual Inference Speed (Non-Linear!)
=============================================
Visualizes the famous "EfficientNet paradox" - low FLOPs but slow inference.
Demonstrates Roofline Model: compute-bound vs memory-bound.

Output: assets/figures/10-8-flops-vs-speed.png
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np


fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle('FLOPs vs 实际推理速度：非线性关系 (Roofline Model)',
             fontsize=15, fontweight='bold', y=0.98)


# === Left: Famous case - ResNet50 vs EfficientNet-B3 ===
ax = axes[0]

models = ['ResNet-50', 'EfficientNet-B3']
flops_g = [4.1, 1.8]      # GFLOPs
inf_time_ms = [25, 50]    # ms per image (lower = faster)
qps = [40, 20]            # queries per second
params_m = [25.6, 12.0]   # million params

x = np.arange(len(models))
width = 0.2

# 4 metrics
ax2 = ax.twinx()

# GFLOPs (lower is better)
bars1 = ax.bar(x - 1.5*width, flops_g, width, label='GFLOPs (越低越好)', color='#90EE90', edgecolor='black')
# Inference time
bars2 = ax.bar(x - 0.5*width, inf_time_ms, width, label='推理时间 ms (越低越好)', color='#FF6B6B', edgecolor='black')
# QPS
bars3 = ax.bar(x + 0.5*width, qps, width, label='QPS 吞吐 (越高越好)', color='#FFD700', edgecolor='black')
# Params
bars4 = ax.bar(x + 1.5*width, params_m, width, label='参数量 M (越低越好)', color='#87CEEB', edgecolor='black')

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12, fontweight='bold')
ax.set_ylabel('数值', fontsize=11, fontweight='bold')
ax.set_title('反直觉: EfficientNet-B3 FLOPs < ResNet50 一半\n但推理时间反而是 ResNet50 的 2 倍!',
             fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars, vals in [(bars1, flops_g), (bars2, inf_time_ms), (bars3, qps), (bars4, params_m)]:
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val}', ha='center', fontsize=9, fontweight='bold')

# Add the key insight
ax.text(0.5, 38,
        'EfficientNet 用了大量 depthwise 卷积 (低 FLOPs, 高访存)\n'
        'GPU 性能瓶颈在访存带宽, 不在算力 → depthwise 把 GPU "喂不饱"',
        ha='center', fontsize=10, color='red', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#FFE0E0', edgecolor='red', lw=1.5))


# === Right: Roofline Model ===
ax = axes[1]
ax.set_xlim(-0.5, 11)
ax.set_ylim(0, 1.1)

# X axis: Operational Intensity (FLOPs / Byte)
x = np.linspace(0.1, 10, 100)
# Peak compute = 1 (normalized)
peak_compute = np.ones_like(x)
# Memory bandwidth line: compute = bandwidth * intensity
# bandwidth normalized such that ridge point is at x=1
ridge_point = 1
bandwidth_line = x / ridge_point

# Memory-bound region (left of ridge)
memory_bound_mask = x <= ridge_point
# Compute-bound region (right of ridge)
compute_bound_mask = x > ridge_point

# Plot the roofline
ax.plot(x, np.minimum(peak_compute, bandwidth_line), 'b-', lw=3, label='Roofline')

# Fill regions
ax.fill_between(x[memory_bound_mask], 0, peak_compute[memory_bound_mask],
                  color='#FFE0B2', alpha=0.5, label='Memory-bound (受带宽限制)')
ax.fill_between(x[compute_bound_mask], 0, peak_compute[compute_bound_mask],
                  color='#90EE90', alpha=0.5, label='Compute-bound (受算力限制)')

# Mark some typical models
models = [
    ('ResNet-50', 8, 0.5, '#1F77B4'),
    ('EfficientNet-B3', 3, 0.3, '#FF6B6B'),
    ('Depthwise Conv (MobileNet)', 1.5, 0.15, '#FFD700'),
    ('MatMul (Transformer)', 30, 0.95, '#4CAF50'),
]

for name, op_int, compute, color in models:
    ax.scatter([op_int], [compute], s=200, c=color, edgecolor='black', linewidth=2, zorder=5)
    ax.annotate(name, (op_int, compute), xytext=(5, 5), textcoords='offset points',
                fontsize=9, fontweight='bold')

# Ridge point
ax.axvline(ridge_point, color='gray', linestyle='--', lw=1.5, alpha=0.7)
ax.text(ridge_point, 0.05, 'Ridge Point\n(拐点)', ha='center', fontsize=9, color='gray')

ax.set_xlabel('Operational Intensity (FLOPs / Byte)', fontsize=11, fontweight='bold')
ax.set_ylabel('可达到的计算性能 (TFLOPS)', fontsize=11, fontweight='bold')
ax.set_title('Roofline Model: 算力 vs 访存\n区分 Compute-bound vs Memory-bound',
             fontsize=12, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3)

# Add insight
ax.text(0.3, 0.85,
        'depthwise 卷积操作\n强度低 → memory-bound\nGPU 算力无法用满!',
        fontsize=9, color='#E65100', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#FFE0B2', edgecolor='#E65100', lw=1.5))

ax.text(8, 0.4,
        '矩阵乘法 / 普通卷积\n操作强度高 → compute-bound\nGPU 算力充分利用',
        fontsize=9, color='#1B5E20', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#C8E6C9', edgecolor='#1B5E20', lw=1.5))

plt.tight_layout()
plt.savefig('assets/figures/10-8-flops-vs-speed.png', dpi=120, bbox_inches='tight', facecolor='white')
print('  ✓ Saved: assets/figures/10-8-flops-vs-speed.png')
