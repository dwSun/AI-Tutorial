"""
1-3-derivative-geometric.py
============================
导数的几何意义：函数曲线 + 切线 + 割线动画
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyArrowPatch

mpl.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 1. 切线斜率 = 导数
ax = axes[0]
x = np.linspace(-1, 4, 200)
f = x**2
x0 = 2.0
f0 = f0 = x0**2
slope = 2 * x0  # 2x

ax.plot(x, f, 'b-', linewidth=2.5, label='$f(x) = x^2$')
# 切线
x_tan = np.linspace(1.2, 3.2, 50)
tangent = slope * (x_tan - x0) + f0
ax.plot(x_tan, tangent, 'r--', linewidth=2, label='切线 (斜率=4)')
# 切点
ax.scatter([x0], [f0], s=120, c='red', zorder=5, edgecolors='black', linewidth=1.5)
slope_label = "f'(x0) = {}".format(slope)
ax.annotate(f"切点 ({x0}, {f0:.0f})\n切线斜率 = {slope_label}",
            xy=(x0, f0), xytext=(0.3, 8),
            fontsize=11, ha='left',
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
# 割线（h=1）
h = 1.0
x1 = x0 + h
f1 = x1**2
ax.plot([x0, x1], [f0, f1], 'g-', linewidth=2, marker='o', markersize=8, label=f'割线 (h={h:.0f})')
ax.annotate(f'割线斜率 = ({f1:.0f}-{f0:.0f})/{h:.0f} = {(f1-f0)/h:.0f}',
            xy=((x0+x1)/2, (f0+f1)/2), xytext=(2.5, 1.5),
            fontsize=10, color='green',
            arrowprops=dict(arrowstyle='->', color='green', lw=1.2),
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
# 坐标轴
ax.axhline(y=0, color='gray', linewidth=0.5)
ax.axvline(x=0, color='gray', linewidth=0.5)
ax.grid(True, alpha=0.3)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('f(x)', fontsize=12)
ax.set_title('导数的几何意义：切线斜率', fontsize=13, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.set_ylim(-1, 16)
ax.set_xlim(-0.5, 4)

# 2. h→0 割线逼近切线
ax = axes[1]
x0 = 2.0
f0 = x0**2
slope_true = 2 * x0
hs = [1.5, 0.8, 0.3, 0.05]
colors = ['#FF6B6B', '#FFA94D', '#FFD93D', '#6BCB77']
for h, c in zip(hs, colors):
    x1 = x0 + h
    f1 = x1**2
    secant_slope = (f1 - f0) / h
    x_sec = np.linspace(x0 - 0.3, x0 + h + 0.3, 50)
    secant = secant_slope * (x_sec - x0) + f0
    ax.plot(x_sec, secant, color=c, linewidth=2,
            label=f'h={h}, 割线斜率={secant_slope:.2f}')
# 真实切线
x_tan = np.linspace(1.2, 3.2, 50)
tangent = slope_true * (x_tan - x0) + f0
ax.plot(x_tan, tangent, 'k-', linewidth=2.5, label=f'切线 (真值=4)')
# 原函数
x = np.linspace(-0.5, 4, 200)
ax.plot(x, x**2, 'b-', linewidth=2.5, alpha=0.5, label='$f(x) = x^2$')
ax.scatter([x0], [f0], s=100, c='black', zorder=5)
ax.axhline(y=0, color='gray', linewidth=0.5)
ax.axvline(x=0, color='gray', linewidth=0.5)
ax.grid(True, alpha=0.3)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('f(x)', fontsize=12)
ax.set_title('当 h → 0 时，割线斜率 → 切线斜率', fontsize=13, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.set_xlim(0.5, 4)
ax.set_ylim(2, 14)

title_math = "导数的几何意义：$f'(x_0) = \\lim_{h \\to 0} \\frac{f(x_0+h) - f(x_0)}{h}$"
ax.set_title(title_math, fontsize=13)
plt.tight_layout()
plt.savefig('/home/david/AI-Tutorial/assets/figures/1-3-derivative-geometric.png',
            dpi=120, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: 1-3-derivative-geometric.png")
