"""
AI Chip Classification Tree
============================
Visualizes the taxonomy of compute chips: CPU/GPU/DSP/FPGA/ASIC/MCU/SOC
plus AI-specific chips: NPU/TPU/BPU/DPU/IPU/VPU and the user's
specific pain point about hardware traps.

Output: assets/figures/10-8-ai-chip-taxonomy.png
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


fig, ax = plt.subplots(figsize=(16, 12))
ax.set_xlim(0, 16)
ax.set_ylim(0, 14)
ax.axis('off')


def add_box(ax, x, y, w, h, text, fc='#E8F4FD', ec='#1F77B4', fontsize=10, fontweight='normal', text_color='black'):
    box = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05,rounding_size=0.1',
                          fc=fc, ec=ec, linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight=fontweight, color=text_color)


# Root
add_box(ax, 6.5, 12.5, 3, 1, '计算芯片\n(Compute Chip)', fc='#FFE4B5', ec='#FF8C00', fontsize=13, fontweight='bold')

# Layer 1: 通用 vs 专用
add_box(ax, 1, 10.5, 3.5, 1, '通用芯片\n(灵活, 低效)', fc='#90EE90', ec='#228B22', fontsize=11, fontweight='bold')
add_box(ax, 5.5, 10.5, 3.5, 1, '半定制芯片\n(可编程)', fc='#FFD700', ec='#FF8C00', fontsize=11, fontweight='bold')
add_box(ax, 10, 10.5, 5, 1, '专用芯片 (ASIC)\n(高效, 不可改)', fc='#FFB6C1', ec='#DC143C', fontsize=11, fontweight='bold')

# Arrows
for x in [2.75, 7.25, 12.5]:
    arr = FancyArrowPatch((8, 12.5), (x, 11.5), arrowstyle='->', color='#444', lw=1.5, mutation_scale=12)
    ax.add_patch(arr)

# Layer 2: 具体类型
add_box(ax, 0.5, 8.5, 2, 0.8, 'CPU', fc='#E8F5E9', ec='#4CAF50', fontsize=11, fontweight='bold')
add_box(ax, 2.7, 8.5, 2, 0.8, 'GPU', fc='#E8F5E9', ec='#4CAF50', fontsize=11, fontweight='bold')
add_box(ax, 0.5, 7.5, 2, 0.8, 'DSP', fc='#E8F5E9', ec='#4CAF50', fontsize=9)
add_box(ax, 2.7, 7.5, 2, 0.8, 'MCU', fc='#E8F5E9', ec='#4CAF50', fontsize=9)
add_box(ax, 0.5, 6.5, 2, 0.8, 'SOC/APU', fc='#E8F5E9', ec='#4CAF50', fontsize=9)
add_box(ax, 2.7, 6.5, 2, 0.8, 'MPU', fc='#E8F5E9', ec='#4CAF50', fontsize=9)

add_box(ax, 4.7, 8.5, 2, 0.8, 'FPGA', fc='#FFF9C4', ec='#FFA000', fontsize=11, fontweight='bold')
add_box(ax, 6.9, 8.5, 2, 0.8, 'DSP 可编程', fc='#FFF9C4', ec='#FFA000', fontsize=9)
add_box(ax, 4.7, 7.5, 2, 0.8, 'CGRA', fc='#FFF9C4', ec='#FFA000', fontsize=9)
add_box(ax, 6.9, 7.5, 2, 0.8, 'eFPGA', fc='#FFF9C4', ec='#FFA000', fontsize=9)

# ASIC AI chips
add_box(ax, 9.5, 8.5, 1.5, 0.8, 'TPU\n(Google)', fc='#FFCDD2', ec='#C62828', fontsize=9, fontweight='bold')
add_box(ax, 11.2, 8.5, 1.5, 0.8, 'NPU\n(华为昇腾)', fc='#FFCDD2', ec='#C62828', fontsize=9, fontweight='bold')
add_box(ax, 12.9, 8.5, 1.5, 0.8, 'BPU\n(地平线)', fc='#FFCDD2', ec='#C62828', fontsize=9)
add_box(ax, 14.6, 8.5, 1.4, 0.8, 'DPU\n(Xilinx)', fc='#FFCDD2', ec='#C62828', fontsize=9)

add_box(ax, 9.5, 7.5, 1.5, 0.8, 'IPU\n(Graphcore)', fc='#FFCDD2', ec='#C62828', fontsize=9)
add_box(ax, 11.2, 7.5, 1.5, 0.8, 'VPU\n(Intel Movidius)', fc='#FFCDD2', ec='#C62828', fontsize=9)
add_box(ax, 12.9, 7.5, 1.5, 0.8, 'APU\n(AMD)', fc='#FFCDD2', ec='#C62828', fontsize=9)
add_box(ax, 14.6, 7.5, 1.4, 0.8, 'HPU\n(HoloLens)', fc='#FFCDD2', ec='#C62828', fontsize=9)

add_box(ax, 9.5, 6.5, 1.5, 0.8, 'QPU\n(量子)', fc='#F8BBD0', ec='#880E4F', fontsize=9)
add_box(ax, 11.2, 6.5, 1.5, 0.8, 'XPU\n(百度)', fc='#F8BBD0', ec='#880E4F', fontsize=9)
add_box(ax, 12.9, 6.5, 1.5, 0.8, 'KPU\n(嘉楠耘智)', fc='#F8BBD0', ec='#880E4F', fontsize=9)
add_box(ax, 14.6, 6.5, 1.4, 0.8, 'EPU\n(情感)', fc='#F8BBD0', ec='#880E4F', fontsize=9)

# Bottom: 算力陷阱
add_box(ax, 1, 4.0, 14, 1.8, '''⚠️ 算力陷阱 (Marketing Tricks):
• 很多 AI 芯片厂商只标 INT8 TOPS，不标 FP16/TFLOPS → 实际训练性能可能只有 1/4
• TOPS 算的是操作（mul+add 各算 1），不是浮点；INT8 TOPS 不能直接转 TFLOPS
• 实际可用算力 ≈ 理论 TOPS × 50%（受 SRAM/DRAM 带宽、指令集、模型优化限制）
• NPU 标 "100 TOPS" 但 FP16 跑 LLaMA 可能只有 25 TFLOPS''', fc='#FFE0E0', ec='red', fontsize=10, fontweight='bold', text_color='red')

# 算力与显存
add_box(ax, 1, 1.5, 14, 1.8, '''⚠️ 算力 vs 显存/内存 不是线性关系 (非直觉):
• FLOPs 低的模型 ≠ 推理快 (EfficientNet-B3: FLOPs < ResNet50 一半, 但实际慢 2 倍)
• 原因: 算力瓶颈常是访存带宽, 不是 FLOPS; depthwise 卷积是 "低 FLOPs, 高访存"
• 内存占用 (≈模型大小) 与 FLOPS 不成正比: 参数量小但结构复杂 (如 MoE) 反而算力高
• 工业界更看 QPS (打满 GPU 时的吞吐), 不是 FLOPs
• Roofline Model: 区分 compute-bound (受算力限制) vs memory-bound (受带宽限制)''', fc='#FFF3CD', ec='black', fontsize=10, fontweight='bold')

ax.set_title('AI 算力硬件分类 + 算力陷阱', fontsize=15, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('assets/figures/10-8-ai-chip-taxonomy.png', dpi=120, bbox_inches='tight', facecolor='white')
print('  ✓ Saved: assets/figures/10-8-ai-chip-taxonomy.png')
