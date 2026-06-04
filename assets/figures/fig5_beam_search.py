"""
图5: Beam Search 搜索策略
展示Beam Search如何在每一步保留top-k候选，并最终选择最优序列
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(1, 1, figsize=(14, 9))
ax.set_xlim(0, 14)
ax.set_ylim(0, 9)
ax.axis('off')
ax.set_title("Beam Search (束搜索)", fontsize=16, fontweight='bold', pad=20)

# Colors
color_beam = '#3498DB'
color_candidate = '#95A5A6'
color_selected = '#27AE60'
color_best = '#E74C3C'

def draw_box(ax, x, y, w, h, text, color, fontsize=10, bold=False):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                         facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.85)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight=weight, color='white')

def draw_arrow(ax, start, end, color='#7F8C8D', lw=2):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=lw))

# Beam width label
ax.text(7, 8.3, 'beam_width = 3', fontsize=12, ha='center',
        bbox=dict(boxstyle='round', facecolor='#ECF0F1', edgecolor='#BDC3C7'))

# Step 1
ax.text(1.5, 7.2, 'Step 1', fontsize=11, fontweight='bold')
draw_box(ax, 1.5, 6.0, 1.2, 0.7, '我\n0.6', color_beam, 9)
draw_box(ax, 3.0, 6.0, 1.2, 0.7, '爱\n0.3', color_beam, 9)
draw_box(ax, 4.5, 6.0, 1.2, 0.7, '机\n0.1', color_beam, 9)

ax.text(5.5, 6.0 + 0.35, '→', fontsize=14)
ax.text(6.0, 6.0, 'Keep top-3', fontsize=9, ha='left', va='center', color=color_beam)

# Step 2
ax.text(1.5, 4.7, 'Step 2', fontsize=11, fontweight='bold')

# Expanded candidates (cross products shown as labels)
candidates_t2 = [
    ('我爱', 0.30, True), ('我机', 0.06, False), ('爱学', 0.12, True),
    ('爱机', 0.05, False), ('机器', 0.08, True), ('爱学', 0.12, False),
    ('我', 0.6, False), ('爱', 0.3, False), ('机', 0.1, False),
]
# Actually show just the top ones with arrows
positions_t2 = [
    (1.5, 3.5, '我 爱', 0.30, True),
    (3.0, 3.5, '爱 学', 0.12, True),
    (4.5, 3.5, '机 器', 0.08, True),
]
for x, y, text, prob, sel in positions_t2:
    col = color_selected if sel else color_candidate
    draw_box(ax, x, y, 1.2, 0.7, f'{text}\n{prob:.2f}', col, 9)

# Arrows from step 1 to step 2
ax.annotate('', xy=(1.5 + 0.6, 3.5 + 0.7), xytext=(1.5 + 0.6, 6.0),
            arrowprops=dict(arrowstyle='->', color='#7F8C8D', lw=1.2))
ax.annotate('', xy=(3.0 + 0.6, 3.5 + 0.7), xytext=(1.5 + 0.6, 6.0),
            arrowprops=dict(arrowstyle='->', color='#7F8C7D', lw=1.2))
ax.annotate('', xy=(4.5 + 0.6, 3.5 + 0.7), xytext=(3.0 + 0.6, 6.0),
            arrowprops=dict(arrowstyle='->', color='#7F8C8D', lw=1.2))

# Step 3
ax.text(1.5, 2.2, 'Step 3', fontsize=11, fontweight='bold')
positions_t3 = [
    (1.5, 1.0, '我 爱 学', 0.12, True),
    (3.0, 1.0, '爱 学 习', 0.07, True),
    (4.5, 1.0, '机 器 学', 0.04, True),
]
for x, y, text, prob, sel in positions_t3:
    col = color_selected if sel else color_candidate
    draw_box(ax, x, y, 1.4, 0.7, f'{text}\n{prob:.2f}', col, 8)

# Star the best
ax.text(1.5 + 1.4 + 0.2, 1.35, '★ Best', fontsize=10, color=color_best, fontweight='bold')

# Calculation annotation
ax.text(7.5, 4.0,
        'Beam Search Calculation:\n\n'
        'Step 1:\n'
        '  Candidates: [我, 爱, 机]\n'
        '  Probs: [0.6, 0.3, 0.1]\n'
        '  Keep: top-3 = [我, 爱, 机]\n\n'
        'Step 2:\n'
        '  Expand each candidate\n'
        '  "我" + 爱 → 0.6×0.5 = 0.30\n'
        '  "爱" + 学 → 0.3×0.4 = 0.12\n'
        '  "机" + 器 → 0.1×0.8 = 0.08\n'
        '  Keep: top-3 = [我爱, 爱学, 机器]\n\n'
        'Step 3:\n'
        '  Continue expanding...\n'
        '  Final: "我爱学" (highest prob)',
        fontsize=9, va='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='#F8F9FA', edgecolor='#DEE2E6'))

# Trade-off diagram
ax.text(11.5, 7.5, 'Search Strategy Trade-off', fontsize=11, fontweight='bold', ha='center')

# Greedy
draw_box(ax, 10.5, 6.3, 2, 0.7, 'Greedy\n(k=1)', '#95A5A6', 9)
ax.text(11.5, 6.0, 'Fast, may miss\nglobal optimum', fontsize=8, ha='center')

# Beam
draw_box(ax, 10.5, 4.8, 2, 0.7, 'Beam\n(k=5~10)', color_beam, 9)
ax.text(11.5, 4.5, 'Balanced speed\nand quality', fontsize=8, ha='center')

# Exhaustive
draw_box(ax, 10.5, 3.3, 2, 0.7, 'Exhaustive\n(k=V^T)', color_best, 9)
ax.text(11.5, 3.0, 'Optimal but\ncomputationally\nimpossible', fontsize=8, ha='center')

# Arrow showing trade-off
ax.annotate('', xy=(11.5, 3.3 + 0.7), xytext=(11.5, 4.8),
            arrowprops=dict(arrowstyle='<->', color='#7F8C8D', lw=1.5))
ax.text(12.2, 4.1, 'Better\nquality', fontsize=8, color='#27AE60')

plt.tight_layout()
plt.savefig(Path(__file__).resolve().parent / 'fig5_beam_search.png', dpi=150, bbox_inches='tight')
plt.close()
print("fig5_beam_search.png saved.")
