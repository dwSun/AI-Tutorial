import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

x = np.linspace(-5, 5, 400)
y_relu = np.maximum(0, x)
y_leaky = np.where(x > 0, x, 0.01 * x)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 左图: Dying ReLU 问题
ax1.fill_between(x, 0, y_relu, alpha=0.3, color='red', label='梯度=0 区域')
ax1.plot(x, y_relu, 'r-', linewidth=2, label='ReLU')
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)
ax1.set_xlim(-5, 5)
ax1.set_ylim(-0.5, 5.5)
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('输出', fontsize=12)
ax1.set_title('Dying ReLU 问题', fontsize=13, fontweight='bold')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# 添加文字说明
ax1.text(-4, 3, '梯度=0\n神经元永久死亡', fontsize=10, 
         bbox=dict(boxstyle='round', facecolor='red', alpha=0.2), ha='center')

# 右图: Leaky ReLU 解决方案
ax2.plot(x, y_leaky, 'b-', linewidth=2, label='Leaky ReLU: max(0.01x, x)')
ax2.plot(x, y_relu, 'r--', linewidth=1.5, alpha=0.7, label='ReLU (对比)')
ax2.axhline(y=0, color='k', linewidth=0.5)
ax2.axvline(x=0, color='k', linewidth=0.5)
ax2.set_xlim(-5, 5)
ax2.set_ylim(-0.5, 5.5)
ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('输出', fontsize=12)
ax2.set_title('Leaky ReLU 解决方案', fontsize=13, fontweight='bold')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

# 添加文字说明
ax2.text(-4, 0.5, '梯度=0.01\n永不死亡', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='blue', alpha=0.2), ha='center')

plt.suptitle('Dying ReLU 问题与解决方案', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('dying_relu.png', dpi=150, bbox_inches='tight')
print("dying_relu.png 已保存")
