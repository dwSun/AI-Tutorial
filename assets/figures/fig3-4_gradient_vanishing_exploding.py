"""
图 3-4：梯度消失与梯度爆炸示意图
生成清晰的 matplotlib 图表，保存为 PNG
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
import matplotlib.patheffects as pe
from pathlib import Path as FilePath

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 创建画布
fig, ax = plt.subplots(figsize=(12, 8))

# 设置背景
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('white')

# 层深度 (1 到 10 层)
layers = np.arange(1, 11)

# ===== 梯度消失曲线 (sigmoid-like decay) =====
# 模拟梯度随层深度指数衰减
gradient_vanish = 1.0 * (0.3 ** (layers - 1))

# ===== 梯度爆炸曲线 =====
# 模拟梯度随层深度指数增长
gradient_explode = 0.01 * (1.8 ** (layers - 1))

# ===== 正常梯度曲线 (稳定范围) =====
# 在 0.8-1.2 之间小幅波动
np.random.seed(42)
gradient_normal = 1.0 + 0.15 * np.sin(layers * 0.8) + 0.05 * np.random.randn(len(layers))

# ===== 绘制区域填充 =====
# 消失区域 (底部，0 以下)
ax.fill_between(layers, 0, -0.15, alpha=0.15, color='#e74c3c', label='_nolegend_')
# 爆炸区域 (顶部)
ax.fill_between(layers, 2.5, 3.5, alpha=0.15, color='#e74c3c', label='_nolegend_')
# 正常区域
ax.fill_between(layers, 0.3, 1.8, alpha=0.1, color='#27ae60', label='_nolegend_')

# ===== 绘制三条主要曲线 =====

# 梯度消失曲线 - 红色虚线
ax.plot(layers, gradient_vanish, 'o-', color='#c0392b', linewidth=2.5,
        markersize=8, label='Vanishing Gradient', path_effects=[
            pe.Stroke(linewidth=4, foreground='#c0392b'),
            pe.Normal()
        ])

# 梯度爆炸曲线 - 红色实线
ax.plot(layers, gradient_explode, 's-', color='#e74c3c', linewidth=2.5,
        markersize=8, label='Exploding Gradient', path_effects=[
            pe.Stroke(linewidth=4, foreground='#e74c3c'),
            pe.Normal()
        ])

# 正常梯度曲线 - 绿色实线
ax.plot(layers, gradient_normal, '^-', color='#27ae60', linewidth=2.5,
        markersize=8, label='Healthy Gradient', path_effects=[
            pe.Stroke(linewidth=4, foreground='#27ae60'),
            pe.Normal()
        ])

# ===== 添加区域标注 =====

# 爆炸区域标注
ax.annotate('Exploding\n(NaN / Divergence)', xy=(9.5, 3.2), fontsize=11,
            ha='center', va='center', color='#c0392b', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffebee', edgecolor='#e74c3c', linewidth=1.5))

# 消失区域标注
ax.annotate('Vanishing\n(Training Stalls)', xy=(9.5, 0.08), fontsize=11,
            ha='center', va='center', color='#7f8c8d', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f5f5f5', edgecolor='#95a5a6', linewidth=1.5))

# 正常区域标注
ax.annotate('Healthy Range', xy=(1.5, 1.0), fontsize=10,
            ha='left', va='center', color='#27ae60', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f8f0', edgecolor='#27ae60', linewidth=1.5))

# ===== 添加关键点标注 =====

# 消失曲线的最后一点
ax.annotate(f'{gradient_vanish[-1]:.2e}', xy=(10, gradient_vanish[-1]),
            xytext=(10.3, 0.15), fontsize=9, color='#c0392b',
            arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.5),
            ha='left', va='center')

# 爆炸曲线的最后一点
ax.annotate(f'{gradient_explode[-1]:.2e}', xy=(10, gradient_explode[-1]),
            xytext=(10.3, 2.5), fontsize=9, color='#e74c3c',
            arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5),
            ha='left', va='center')

# ===== 设置坐标轴 =====

ax.set_xlim(0.5, 11.5)
ax.set_ylim(-0.3, 3.6)

ax.set_xlabel('Layer Depth (from input to output)', fontsize=13, fontweight='bold', labelpad=10)
ax.set_ylabel('Gradient Magnitude (log scale concept)', fontsize=13, fontweight='bold', labelpad=10)

ax.set_title('Gradient Vanishing vs Exploding in Deep Networks', fontsize=15, fontweight='bold', pad=15)

# X轴刻度
ax.set_xticks(layers)
ax.set_xticklabels([f'Layer {i}' for i in layers], rotation=45, ha='right', fontsize=9)

# Y轴刻度 - 使用对数感知的标注
yticks = [0, 0.1, 0.5, 1, 2, 3]
yticklabels = ['0', '0.1', '0.5', '1', '2', '3']
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels, fontsize=10)

# ===== 添加网格 =====

ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax.set_axisbelow(True)

# ===== 添加图例 =====

legend = ax.legend(loc='upper center', ncol=3, fontsize=11,
                   framealpha=0.95, edgecolor='gray',
                   bbox_to_anchor=(0.5, -0.12))

# ===== 添加说明文本框 =====

textbox_content = (
    "Causes:\n"
    "• Vanishing: σ'(z) < 1 (e.g., Sigmoid)\n"
    "• Exploding: Large weights or σ'(z) > 1\n\n"
    "Solutions:\n"
    "• ReLU / He init / BatchNorm / Residual"
)

props = dict(boxstyle='round,pad=0.5', facecolor='#fff9e6', edgecolor='#f39c12', linewidth=1.5)
ax.text(0.02, 0.97, textbox_content, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=props, family='monospace')

# ===== 绘制分隔线 =====

# 标记正常范围的上下界
ax.axhline(y=0.3, color='#27ae60', linestyle=':', alpha=0.5, linewidth=1)
ax.axhline(y=1.8, color='#27ae60', linestyle=':', alpha=0.5, linewidth=1)

plt.tight_layout()
plt.subplots_adjust(bottom=0.18)

# 保存图片
output_path = FilePath(__file__).resolve().parent / 'fig3-4_gradient_vanishing_exploding.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"图片已保存到: {output_path}")

plt.close()
