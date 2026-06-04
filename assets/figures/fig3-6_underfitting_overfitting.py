"""Generate a matplotlib figure for underfitting vs overfitting."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def true_function(x):
    return 0.5 * np.sin(1.2 * x) + 0.08 * x**3 - 0.35 * x


np.random.seed(42)
x_train = np.linspace(-2.8, 2.8, 18)
y_true = true_function(x_train)
y_train = y_true + np.random.normal(scale=0.18, size=len(x_train))

x_plot = np.linspace(-3.2, 3.2, 400)
y_plot = true_function(x_plot)


degrees = [1, 4, 12]
titles = ['Underfitting', 'Good Fit', 'Overfitting']
descriptions = [
    'High error on both train and validation',
    'Captures the main trend and generalizes well',
    'Almost memorizes training points, unstable on new data'
]
colors = ['#d62728', '#2ca02c', '#ff7f0e']


fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
fig.patch.set_facecolor('white')

for ax, degree, title, desc, color in zip(axes, degrees, titles, descriptions, colors):
    coeffs = np.polyfit(x_train, y_train, degree)
    poly = np.poly1d(coeffs)
    y_pred = poly(x_plot)

    ax.scatter(x_train, y_train, s=42, color='#1f77b4', edgecolors='white', linewidth=0.8, label='Training samples', zorder=3)
    ax.plot(x_plot, y_plot, '--', color='#7f7f7f', linewidth=2.0, label='True pattern')
    ax.plot(x_plot, y_pred, color=color, linewidth=2.6, label='Model fit')

    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.text(0.5, -0.20, desc, ha='center', va='top', transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#f8f9fa', edgecolor='#cccccc'))
    ax.set_xlabel('Input x', fontsize=10)
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.set_facecolor('#fcfcfc')

axes[0].set_ylabel('Output y', fontsize=10)
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, frameon=True, bbox_to_anchor=(0.5, -0.02))
fig.suptitle('Underfitting vs Good Fit vs Overfitting', fontsize=15, fontweight='bold', y=1.02)

plt.tight_layout()
plt.subplots_adjust(bottom=0.26)

output_path = Path(__file__).resolve().parent / 'fig3-6_underfitting_overfitting.png'
plt.savefig(output_path, dpi=160, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f'图片已保存到: {output_path}')
plt.close()
