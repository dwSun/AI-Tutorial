import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

x = np.linspace(-6, 6, 400)
y = 1 / (1 + np.exp(-x))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, y, 'b-', linewidth=2, label=r'$\sigma(x) = \frac{1}{1+e^{-x}}$')
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)
ax.axhline(y=1, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

# 标注关键点
ax.annotate('(0, 0.5)', xy=(0, 0.5), xytext=(0.8, 0.55),
            arrowprops=dict(arrowstyle='->', color='gray'), fontsize=10)
ax.annotate('(0, 1)', xy=(0, 1), xytext=(1.5, 0.92),
            arrowprops=dict(arrowstyle='->', color='gray'), fontsize=10)

ax.set_xlim(-6, 6)
ax.set_ylim(-0.1, 1.1)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('σ(x)', fontsize=12)
ax.set_title('Sigmoid 函数', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sigmoid.png', dpi=150, bbox_inches='tight')
print("sigmoid.png 已保存")
