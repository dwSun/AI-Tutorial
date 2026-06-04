"""
Hardware Trap: Why A100 Trains Faster Despite Lower FP TFLOPS
=============================================================
Visualizes the "A100 paradox" - A100 FP32 is only 19.5 TFLOPS, lower
than RTX 3090 (35.6 TFLOPS), but A100 trains models 5-10x faster.

Reasons:
- Tensor Cores (BF16 with Tensor Core: 312 TFLOPS, vs 35 TFLOPS FP16 CUDA)
- HBM2 bandwidth (1555 GB/s vs 936 GB/s for GDDR6X)
- NVLink for multi-GPU
- ECC for stability (24/7 training)

Output: assets/figures/10-8-a100-paradox.png
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle('A100 训练速度之谜：FP 算力不高, 但训练却快 5-10 倍',
             fontsize=14, fontweight='bold', y=0.97)


# === Left: Different precisions comparison ===
ax = axes[0]
gpus = ['RTX 3090', 'RTX 4090', 'A100 80G', 'H100 80G']
fp32 = [35.6, 82.6, 19.5, 51.0]
fp16 = [71.2, 165.2, 77.9, 151.0]  # without tensor core
bf16_tensor = [0, 165.2, 312.0, 756.0]  # with tensor core (RTX 3090 has no tensor core BF16)
hbm_bw = [936, 1008, 2039, 3350]  # GB/s

x = np.arange(len(gpus))
width = 0.2

ax2 = ax.twinx()

bars1 = ax.bar(x - 1.5*width, fp32, width, label='FP32 (CUDA)', color='#87CEEB', edgecolor='black')
bars2 = ax.bar(x - 0.5*width, fp16, width, label='FP16 (CUDA)', color='#FFD700', edgecolor='black')
bars3 = ax.bar(x + 0.5*width, bf16_tensor, width, label='BF16 (Tensor Core)', color='#FF6B6B', edgecolor='black')

ax.set_xticks(x)
ax.set_xticklabels(gpus, fontsize=11, fontweight='bold')
ax.set_ylabel('算力 (TFLOPS)', fontsize=11, fontweight='bold')
ax.set_title('同一颗 GPU, 不同精度的算力差异巨大\nRTX 3090 根本没有 Tensor Core BF16!',
             fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars, vals in [(bars1, fp32), (bars2, fp16), (bars3, bf16_tensor)]:
    for bar, val in zip(bars, vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    f'{val}', ha='center', fontsize=8, fontweight='bold')

# Highlight A100 Tensor Core vs FP32
ax.annotate('A100 Tensor Core\n是 FP32 的 16×!',
            xy=(2, 312), xytext=(2, 400),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=10, color='red', fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='#FFE0E0', edgecolor='red'))


# === Right: A100 advantages ===
ax = axes[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

ax.set_title('A100/H100 训练快的 5 大原因 (不只是算力)',
             fontsize=12, fontweight='bold')

reasons = [
    ('1. Tensor Core 加速',
     'A100 BF16 Tensor Core: 312 TFLOPS (16× FP32)\nH100 BF16 Tensor Core: 756 TFLOPS (15× FP32)',
     '#FF6B6B', 10),
    ('2. HBM 高带宽',
     'A100: HBM2e 2039 GB/s (2× RTX 3090 的 GDDR6X)\nH100: HBM3 3350 GB/s (3.6× RTX 3090)\n→ 数据加载不再是瓶颈',
     '#FFD700', 8),
    ('3. NVLink 互联',
     'A100: 600 GB/s NVLink3 (12× PCIe Gen4)\nH100: 900 GB/s NVLink4 (18× PCIe)\n→ 多卡训练线性加速',
     '#90EE90', 6),
    ('4. ECC 显存 + 24/7 稳定',
     '消费卡无 ECC, 训练 1 周可能遇到 1 次静默错误\n数据中心 GPU ECC 保证 100% 精度, 适合长时间训练',
     '#87CEEB', 4),
    ('5. 软件生态',
     'cuDNN / NCCL / Apex / DeepSpeed 优先优化数据中心卡\n实际训练速度 = 理论算力 × 80% (A100) vs 60% (消费卡)',
     '#DDA0DD', 2),
]

for title, desc, color, y in reasons:
    box = mpatches.FancyBboxPatch((0.3, y - 0.8), 9.4, 1.5,
                                    boxstyle='round,pad=0.05', fc=color, ec='black', linewidth=1, alpha=0.5)
    ax.add_patch(box)
    ax.text(0.5, y + 0.3, title, fontsize=11, fontweight='bold')
    ax.text(0.5, y - 0.3, desc, fontsize=8.5, va='top', style='italic', color='#333')

plt.tight_layout()
plt.savefig('assets/figures/10-8-a100-paradox.png', dpi=120, bbox_inches='tight', facecolor='white')
print('  ✓ Saved: assets/figures/10-8-a100-paradox.png')
