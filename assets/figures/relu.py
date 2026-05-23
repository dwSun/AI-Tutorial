import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

x = np.linspace(-5, 5, 400)
y = np.maximum(0, x)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, y, 'g-', linewidth=2, label=r'$\text{ReLU}(x) = \max(0, x)$')
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)

# 标注
ax.annotate('(0, 0)', xy=(0, 0), xytext=(0.5, -0.3),
            arrowprops=dict(arrowstyle='->', color='gray'), fontsize=10)
ax.annotate('右侧: y=x', xy=(3, 3), xytext=(2, 2),
            arrowprops=dict(arrowstyle='->', color='gray'), fontsize=10)
ax.annotate('左侧: y=0\n(梯度=0, 神经元死亡)', xy=(-2, 0), xytext=(-4.5, 0.5),
            arrowprops=dict(arrowstyle='->', color='gray'), fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

ax.set_xlim(-5, 5)
ax.set_ylim(-0.5, 5.5)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('ReLU(x)', fontsize=12)
ax.set_title('ReLU 函数', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('relu.png', dpi=150, bbox_inches='tight')
print("relu.png 已保存")
