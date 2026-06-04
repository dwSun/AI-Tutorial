"""
5-1-rnn-visualization.py
=========================
RNN 原理可视化：RNN 展开 + 序列预测 + 梯度消失/爆炸
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle

mpl.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

# 1. RNN 展开 + 序列预测 + 梯度消失/爆炸
fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

ax1 = fig.add_subplot(gs[0, :])
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 3)
ax1.axis('off')
time_steps = 5
x_positions = np.linspace(1, 9, time_steps)
for i, x in enumerate(x_positions):
    circle = Circle((x, 2), 0.3, facecolor='lightblue', edgecolor='blue', linewidth=2)
    ax1.add_patch(circle)
    ax1.text(x, 2, f'h_{i}', ha='center', va='center', fontsize=10, fontweight='bold')
    ax1.plot([x, x], [1.2, 1.7], 'k-', linewidth=2)
    ax1.text(x, 1, f'x_{i}', ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow'))
    ax1.plot([x, x], [2.3, 2.8], 'k-', linewidth=2)
    ax1.text(x, 3, f'y_{i}', ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightgreen'))
    if i < time_steps - 1:
        ax1.annotate('', xy=(x_positions[i+1]-0.3, 2), xytext=(x+0.3, 2),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))
        ax1.text((x + x_positions[i+1])/2, 2.2, 'W_h', fontsize=9, color='red')
ax1.set_title('RNN 展开结构（时间序列处理）', fontsize=13, fontweight='bold', pad=20)

ax2 = fig.add_subplot(gs[1, 0])
t = np.linspace(0, 4*np.pi, 100)
sequence = np.sin(t)
predicted = np.convolve(sequence, np.ones(5)/5, mode='same')
ax2.plot(t, sequence, 'b-', linewidth=2, label='真实序列', alpha=0.7)
ax2.plot(t, predicted, 'r--', linewidth=2, label='RNN 预测', alpha=0.7)
ax2.set_xlabel('时间步')
ax2.set_ylabel('数值')
ax2.set_title('序列预测示例', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[1, 1])
seq_lengths = np.arange(1, 51)
weights = [0.5, 0.8, 0.95, 1.0, 1.05, 1.2]
colors_r = plt.cm.RdYlGn(np.linspace(0, 1, len(weights)))
for w, color in zip(weights, colors_r):
    gradients = np.clip(w ** seq_lengths, 1e-10, 1e10)
    ax3.semilogy(seq_lengths, gradients, linewidth=2,
                label=f'W={w}', color=color, alpha=0.8)
ax3.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='梯度=1')
ax3.fill_between(seq_lengths, 1e-10, 0.01, alpha=0.2, color='blue', label='梯度消失区')
ax3.fill_between(seq_lengths, 100, 1e10, alpha=0.2, color='red', label='梯度爆炸区')
ax3.set_xlabel('时间步（序列长度）')
ax3.set_ylabel('梯度大小（对数刻度）')
ax3.set_title('梯度消失与梯度爆炸', fontsize=12, fontweight='bold')
ax3.legend(fontsize=8, loc='upper left')
ax3.grid(True, alpha=0.3)
ax3.set_ylim(1e-10, 1e10)

plt.suptitle('RNN 原理可视化', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/david/AI-Tutorial/assets/figures/5-1-rnn-unfolding.png',
            dpi=120, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: 5-1-rnn-unfolding.png")

# 2. BPTT
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
ax1 = axes2[0]
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 6)
ax1.axis('off')
time_steps_bptt = 4
x_positions_bptt = np.linspace(2, 8, time_steps_bptt)
for i, x in enumerate(x_positions_bptt):
    ax1.annotate('', xy=(x, 5-i*0.8), xytext=(x, 4-i*0.8),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax1.text(x, 5.2-i*0.8, f'h_{i}', ha='center', fontsize=9, color='blue')
for i in range(time_steps_bptt-1, 0, -1):
    ax1.annotate('', xy=(x_positions_bptt[i-1], 4.4-(i-1)*0.8),
                xytext=(x_positions_bptt[i], 4.4-i*0.8),
                arrowprops=dict(arrowstyle='->', color='red', lw=2, ls='--'))
    mid_x = (x_positions_bptt[i-1] + x_positions_bptt[i]) / 2
    mid_y = (4.4-(i-1)*0.8 + 4.4-i*0.8) / 2
    ax1.text(mid_x, mid_y, '∂L/∂h', fontsize=8, color='red', ha='center')
ax1.set_title('BPTT: 随时间反向传播', fontsize=12, fontweight='bold')
ax1.text(5, 0.5, '损失 L 对早期隐藏状态的梯度\n需要通过链式法则\n穿越多个时间步相乘',
        ha='center', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

ax2 = axes2[1]
distances = np.arange(1, 31)
grad_vanilla = np.exp(-distances * 0.3)
grad_lstm = np.exp(-distances * 0.05) + 0.1
grad_gru = np.exp(-distances * 0.08) + 0.1
ax2.plot(distances, grad_vanilla, 'o-', color='red', linewidth=2, markersize=6, label='普通RNN', alpha=0.7)
ax2.plot(distances, grad_lstm, 's-', color='green', linewidth=2, markersize=6, label='LSTM', alpha=0.7)
ax2.plot(distances, grad_gru, '^-', color='blue', linewidth=2, markersize=6, label='GRU', alpha=0.7)
ax2.axhline(y=0.01, color='gray', linestyle='--', alpha=0.5, label='有效梯度阈值')
ax2.fill_between(distances, 0, 0.01, alpha=0.2, color='gray')
ax2.set_xlabel('依赖距离（时间步）')
ax2.set_ylabel('梯度大小')
ax2.set_title('长期依赖问题与解决方案', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 1.2)
ax2.annotate('普通RNN梯度快速消失\n难以学习长期依赖',
            xy=(15, grad_vanilla[14]), xytext=(20, 0.5),
            fontsize=9, bbox=dict(boxstyle='round', facecolor='red', alpha=0.3),
            arrowprops=dict(arrowstyle='->', color='red'))
ax2.annotate('LSTM/GRU通过门控机制\n保持梯度稳定',
            xy=(20, grad_lstm[19]), xytext=(22, 0.8),
            fontsize=9, bbox=dict(boxstyle='round', facecolor='green', alpha=0.3),
            arrowprops=dict(arrowstyle='->', color='green'))
plt.suptitle('RNN 训练中的梯度问题', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/david/AI-Tutorial/assets/figures/5-1-bptt.png',
            dpi=120, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: 5-1-bptt.png")

# 3. 权重初始化
fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))
hidden_dim = 100
seq_length = 50
num_trials = 20

init_methods = [
    ('Xavier初始化', lambda h: np.random.randn(h, h) * np.sqrt(1/h)),
    ('He初始化', lambda h: np.random.randn(h, h) * np.sqrt(2/h)),
    ('正交初始化', lambda h: np.linalg.qr(np.random.randn(h, h))[0]),
    ('随机初始化(过大)', lambda h: np.random.randn(h, h) * 0.1),
]
colors_i = ['blue', 'green', 'purple', 'red']

ax1 = axes3[0, 0]
for (name, init_fn), color in zip(init_methods, colors_i):
    grad_norms = []
    for trial in range(num_trials):
        W = init_fn(hidden_dim)
        grads = [np.linalg.norm(np.linalg.matrix_power(W.T, t))
                for t in range(1, seq_length+1)]
        grad_norms.append(grads)
    grad_norms = np.array(grad_norms)
    mean_grads = np.mean(grad_norms, axis=0)
    std_grads = np.std(grad_norms, axis=0)
    t = np.arange(1, seq_length+1)
    ax1.semilogy(t, mean_grads, color=color, linewidth=2, label=name)
    ax1.fill_between(t, mean_grads-std_grads, mean_grads+std_grads, color=color, alpha=0.2)
ax1.axhline(y=1e-7, color='gray', linestyle='--', alpha=0.5)
ax1.axhline(y=1e7, color='gray', linestyle='--', alpha=0.5)
ax1.fill_between(t, 1e-10, 1e-7, alpha=0.2, color='blue')
ax1.fill_between(t, 1e7, 1e10, alpha=0.2, color='red')
ax1.set_xlabel('时间步')
ax1.set_ylabel('梯度范数（对数）')
ax1.set_title('不同初始化方法的梯度稳定性', fontsize=11, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

ax2 = axes3[0, 1]
activations_data = []
labels = []
for name, init_fn in init_methods:
    W = init_fn(hidden_dim)
    x = np.random.randn(hidden_dim)
    activations = []
    for t in range(seq_length):
        x = np.tanh(W @ x)
        activations.extend(x)
    activations_data.append(activations)
    labels.append(name)
ax2.boxplot(activations_data, labels=[l.split('初始化')[0] for l in labels])
ax2.set_ylabel('激活值分布')
ax2.set_title('不同初始化的激活值分布', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

ax3 = axes3[1, 0]
for (name, init_fn), color in zip(init_methods[:3], colors_i[:3]):
    W = init_fn(hidden_dim)
    eigenvalues = np.linalg.eigvals(W)
    ax3.scatter(np.real(eigenvalues), np.imag(eigenvalues),
               c=color, alpha=0.5, s=20, label=name)
theta = np.linspace(0, 2*np.pi, 100)
ax3.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=2, label='单位圆')
ax3.axhline(y=0, color='k', linewidth=0.5)
ax3.axvline(x=0, color='k', linewidth=0.5)
ax3.set_xlabel('实部')
ax3.set_ylabel('虚部')
ax3.set_title('权重矩阵特征值分布', fontsize=11, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_aspect('equal')

ax4 = axes3[1, 1]
dependencies = [5, 10, 20, 30, 40, 50]
success_vanilla = [95, 80, 50, 25, 10, 5]
success_lstm = [98, 95, 85, 75, 65, 55]
success_gru = [97, 93, 82, 72, 62, 52]
success_attention = [99, 98, 95, 92, 88, 85]
ax4.plot(dependencies, success_vanilla, 'o-', color='red', linewidth=2, markersize=8, label='普通RNN')
ax4.plot(dependencies, success_lstm, 's-', color='green', linewidth=2, markersize=8, label='LSTM')
ax4.plot(dependencies, success_gru, '^-', color='blue', linewidth=2, markersize=8, label='GRU')
ax4.plot(dependencies, success_attention, 'd-', color='purple', linewidth=2, markersize=8, label='Transformer')
ax4.set_xlabel('依赖距离（时间步）')
ax4.set_ylabel('任务成功率 (%)')
ax4.set_title('长期依赖任务性能对比', fontsize=11, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.set_ylim(0, 105)
plt.suptitle('RNN 权重初始化与长期依赖分析', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/david/AI-Tutorial/assets/figures/5-1-weight-init.png',
            dpi=120, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: 5-1-weight-init.png")
