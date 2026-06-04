"""
3-4-gradient-issues.py
=======================
梯度消失/爆炸 + 激活函数导数 + 链式法则 + 梯度流动热力图
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(12, 8))

# 1. 梯度消失/爆炸（不同初始化）
ax1 = plt.subplot(2, 2, 1)
np.random.seed(42)
n_layers = 20
init_methods = ['标准正态', '过小权重', '过大权重']
init_scales = [1.0, 0.1, 10.0]
for method, scale in zip(init_methods, init_scales):
    weights = [np.random.randn(10, 10) * scale / np.sqrt(10) for _ in range(n_layers)]
    gradient = np.ones((10, 10))
    norms = []
    for i in reversed(range(n_layers)):
        activation_derivative = 0.25
        gradient = np.dot(gradient, weights[i].T) * activation_derivative
        norms.append(np.linalg.norm(gradient))
    norms = list(reversed(norms))
    layers = list(range(1, n_layers + 1))
    ax1.plot(layers, norms, 'o-', label=method, linewidth=2)
ax1.set_xlabel('网络层数（从输入层到输出层）', fontsize=10)
ax1.set_ylabel('梯度范数（对数坐标）', fontsize=10)
ax1.set_title('梯度消失/爆炸（不同初始化）', fontsize=12, fontweight='bold')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9)

# 2. 激活函数导数对比
ax2 = plt.subplot(2, 2, 2)
x = np.linspace(-5, 5, 1000)
sigmoid = 1 / (1 + np.exp(-x))
sigmoid_derivative = sigmoid * (1 - sigmoid)
relu = np.maximum(0, x)
relu_derivative = (x > 0).astype(float)
tanh = np.tanh(x)
tanh_derivative = 1 - tanh**2
ax2.plot(x, sigmoid_derivative, 'b-', label='Sigmoid 导数', linewidth=2)
ax2.plot(x, relu_derivative, 'r-', label='ReLU 导数', linewidth=2)
ax2.plot(x, tanh_derivative, 'g-', label='Tanh 导数', linewidth=2)
ax2.set_xlabel('输入值', fontsize=10)
ax2.set_ylabel('导数值', fontsize=10)
ax2.set_title('不同激活函数的导数对比', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=1, color='k', linestyle='--', alpha=0.3)
ax2.axhline(y=0.25, color='k', linestyle='--', alpha=0.3, label='Sigmoid 最大导数=0.25')
ax2.legend(fontsize=8)

# 3. 链式法则可视化
ax3 = plt.subplot(2, 2, 3)
x_vals = np.linspace(-2, 2, 400)
u = x_vals**2
y = np.sin(u)
dy_du = np.cos(u)
du_dx = 2 * x_vals
dy_dx = dy_du * du_dx
ax3.plot(x_vals, y, 'b-', label='y = sin(x²)', linewidth=2)
ax3.plot(x_vals, dy_dx, 'r-', label='dy/dx = cos(x²)·2x', linewidth=2)
ax3.set_xlabel('x', fontsize=10)
ax3.set_ylabel('y / dy/dx', fontsize=10)
ax3.set_title('链式法则示例：y = sin(x²)', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=9)

# 4. 梯度流动热力图
ax4 = plt.subplot(2, 2, 4)
n_layers_sim = 5
gradient_flow = np.zeros((n_layers_sim, 10))
gradient_flow[0, :] = 1.0
for layer in range(1, n_layers_sim):
    decay_factor = 0.5
    gradient_flow[layer, :] = gradient_flow[layer-1, :] * decay_factor
im = ax4.imshow(gradient_flow, cmap='viridis', aspect='auto')
plt.colorbar(im, ax=ax4, label='梯度强度')
ax4.set_xlabel('神经元索引', fontsize=10)
ax4.set_ylabel('网络层数（从输入到输出）', fontsize=10)
ax4.set_title('梯度流动热力图（梯度消失）', fontsize=12, fontweight='bold')
ax4.set_yticks(range(n_layers_sim), [f'L{i+1}' for i in range(n_layers_sim)])

plt.suptitle('梯度消失与爆炸的 4 个视角', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/david/AI-Tutorial/assets/figures/3-4-gradient-issues.png',
            dpi=120, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: 3-4-gradient-issues.png")
