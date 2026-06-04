"""
CUDA Compute Capability Coverage Visualization
=============================================
Visualizes the relationship between CUDA Compute Capability versions
and which features (Tensor Cores, FP16, BF16, FP8) are supported.

Output: assets/figures/10-8-cuda-compute-capability.png
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np


fig, ax = plt.subplots(figsize=(18, 10))
ax.set_xlim(0, 18)
ax.set_ylim(0, 12)
ax.axis('off')


# Compute Capability versions and their key features
cc_data = [
    # (version, year, arch, tensor_core, fp16, bf16, fp8, int8, color, x_pos)
    (1.0, 2006, 'Tesla',  '', '', '', '', '', '#E0E0E0', 0.5),
    (2.0, 2008, 'Fermi',  '', '', '', '', '', '#D3D3D3', 2.0),
    (2.1, 2009, 'Fermi',  '', '', '', '', '', '#D3D3D3', 2.0),
    (3.0, 2012, 'Kepler', '', '', '', '', '', '#C0C0C0', 3.5),
    (3.5, 2012, 'Kepler', '', '', '', '', '', '#C0C0C0', 3.5),
    (3.7, 2013, 'Kepler', '', '', '', '', '', '#C0C0C0', 3.5),
    (5.0, 2014, 'Maxwell','', '', '', '', '', '#B0B0B0', 5.0),
    (5.2, 2014, 'Maxwell','', '', '', '', '', '#B0B0B0', 5.0),
    (5.3, 2015, 'Maxwell','', '', '', '', '', '#B0B0B0', 5.0),
    (6.0, 2016, 'Pascal', '', '✓', '', '', '', '#FFD700', 6.5),
    (6.1, 2016, 'Pascal', '', '✓', '', '', '', '#FFD700', 6.5),
    (6.2, 2017, 'Pascal', '', '✓', '', '', '', '#FFD700', 6.5),
    (7.0, 2017, 'Volta',  '✓(v1)', '✓', '', '', '✓', '#FFA500', 8.0),
    (7.2, 2018, 'Xavier', '✓', '✓', '', '', '✓', '#FFA500', 8.0),
    (7.5, 2018, 'Turing', '✓(v2)', '✓', '', '', '✓', '#FFA500', 8.0),
    (8.0, 2020, 'Ampere (A100)', '✓(v3)','✓', '✓', '', '✓', '#90EE90', 10.0),
    (8.6, 2020, 'Ampere (RTX 30)', '✓(v3)','✓', '✓', '', '✓', '#90EE90', 10.0),
    (8.7, 2022, 'Orin',  '✓(v3)','✓', '✓', '', '✓', '#90EE90', 10.0),
    (8.9, 2022, 'Ada Lovelace (RTX 40)', '✓(v4)','✓', '✓', '✓(部分)','✓', '#87CEEB', 12.0),
    (9.0, 2022, 'Hopper (H100)', '✓(v4)','✓', '✓', '✓', '✓', '#FF6B6B', 14.0),
    (10.0, 2024, 'Blackwell (B200)', '✓(v5)','✓', '✓', '✓', '✓', '#FF1493', 15.5),
    (12.0, 2025, 'Blackwell (RTX 50)', '✓(v5)','✓', '✓', '✓', '✓', '#FF1493', 17.0),
]

# Draw each version as a row
y_start = 10
row_h = 0.42
for i, (cc, year, arch, tc, fp16, bf16, fp8, int8, color, x) in enumerate(cc_data):
    y = y_start - i * row_h
    # Version box
    ax.add_patch(FancyBboxPatch((0.2, y - 0.18), 1.2, 0.36,
                                  boxstyle='round,pad=0.02', fc=color, ec='black', linewidth=1))
    ax.text(0.8, y, f'CC {cc}', ha='center', va='center', fontsize=8.5, fontweight='bold')
    # Year
    ax.text(1.7, y, str(year), ha='left', va='center', fontsize=8)
    # Architecture
    ax.text(2.6, y, arch, ha='left', va='center', fontsize=8, style='italic')
    # Tensor Core
    tc_color = 'green' if '✓' in tc else 'red'
    tc_text = tc if tc else '✗'
    ax.text(6.5, y, tc_text, ha='center', va='center', fontsize=8,
            color=tc_color, fontweight='bold' if '✓' in tc else 'normal')
    # FP16
    ax.text(8.0, y, '✓' if fp16 else '✗', ha='center', va='center', fontsize=8,
            color='green' if fp16 else 'red', fontweight='bold')
    # BF16
    ax.text(9.5, y, '✓' if bf16 else '✗', ha='center', va='center', fontsize=8,
            color='green' if bf16 else 'red', fontweight='bold')
    # FP8
    ax.text(11.0, y, '✓' if fp8 else '✗', ha='center', va='center', fontsize=8,
            color='green' if fp8 else 'red', fontweight='bold')
    # INT8
    ax.text(12.5, y, '✓' if int8 else '✗', ha='center', va='center', fontsize=8,
            color='green' if int8 else 'red', fontweight='bold')

# Headers
headers = ['CC 版本', '年份', '架构', '', 'Tensor Core', 'FP16', 'BF16', 'FP8', 'INT8']
header_x = [0.8, 1.7, 2.6, 5.0, 6.5, 8.0, 9.5, 11.0, 12.5]
for h, hx in zip(headers, header_x):
    weight = 'bold' if h else 'normal'
    if h:
        ax.text(hx, y_start + 0.5, h, ha='center', va='bottom', fontsize=9.5,
                fontweight='bold', color='#1F77B4')

# Separator line
ax.plot([0.2, 13.0], [y_start + 0.35, y_start + 0.35], 'b-', lw=1.5)

# Right side: Key insight
ax.text(14, 10.5, '关键洞察:', fontsize=12, fontweight='bold', color='red')
ax.text(14, 9.8, '• CC ≥ 5.0 才能跑神经网络', fontsize=10, color='#222')
ax.text(14, 9.3, '• CC ≥ 7.0 才有 Tensor Core', fontsize=10, color='#222')
ax.text(14, 8.8, '• CC ≥ 8.0 才有 BF16 Tensor Core', fontsize=10, color='#222')
ax.text(14, 8.3, '• CC ≥ 9.0 才有 FP8 (H100)', fontsize=10, color='#222')
ax.text(14, 7.8, '• CC 10/12 = Blackwell (B200/RTX 50)', fontsize=10, color='#222')
ax.text(14, 7.3, '• CC 版本号 X.Y:', fontsize=10, color='#222', fontweight='bold')
ax.text(14, 6.8, '  X = major (架构代)', fontsize=9.5, color='#222')
ax.text(14, 6.3, '  Y = minor (增量改进)', fontsize=9.5, color='#222')

# Bottom: Timeline
ax.text(9, 1.2, 'CUDA Compute Capability 演进时间线 (2006-2025)', ha='center', fontsize=12, fontweight='bold')

timeline_y = 0.5
ax.plot([0.5, 17.5], [timeline_y, timeline_y], 'k-', lw=2)

milestones = [
    (1.0, 2006, 'Tesla'),
    (2.0, 2008, 'Fermi'),
    (3.0, 2012, 'Kepler'),
    (5.0, 2014, 'Maxwell'),
    (6.0, 2016, 'Pascal'),
    (7.0, 2017, 'Volta\n+Tensor Core'),
    (8.0, 2020, 'Ampere\n+BF16'),
    (8.9, 2022, 'Ada\n+FP8'),
    (9.0, 2022, 'Hopper\n+FP8'),
    (10.0, 2024, 'Blackwell\n+FP4'),
    (12.0, 2025, 'Blackwell\nConsumer'),
]

for cc, year, label in milestones:
    x_pos = 0.5 + (cc - 0.5) * (17.0 / 12.0)  # Spread across x-axis
    x_pos = max(0.5, min(17.0, x_pos))
    ax.plot([x_pos, x_pos], [timeline_y, timeline_y + 0.15], 'k-', lw=2)
    ax.text(x_pos, timeline_y - 0.15, f'{cc}', ha='center', fontsize=8, fontweight='bold')
    ax.text(x_pos, timeline_y + 0.18, label, ha='center', fontsize=7.5, rotation=30, va='bottom')

ax.set_title('CUDA Compute Capability 与硬件特性矩阵 (2006-2025)', fontsize=15, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('assets/figures/10-8-cuda-compute-capability.png', dpi=120, bbox_inches='tight', facecolor='white')
print('  ✓ Saved: assets/figures/10-8-cuda-compute-capability.png')
