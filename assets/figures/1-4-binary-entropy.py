"""
1-4 二元熵与交叉熵曲线
========================
- 左图：二元熵 H(p) 随概率 p 的变化，0.5 处最大为 log(2) ≈ 0.693
- 右图：交叉熵 H(p, q) 当真实分布固定为 p=0.3 时，预测 q 在 [0, 1] 上的损失
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

p = np.linspace(0.001, 0.999, 500)
# 二元熵
H_p = -p * np.log2(p) - (1 - p) * np.log2(1 - p)

# 交叉熵 (KL 部分 = H(p, q) - H(p))
# 真实分布固定为 p_true=0.3
p_true = 0.3
H_true = -p_true * np.log2(p_true) - (1 - p_true) * np.log2(1 - p_true)

q = np.linspace(0.001, 0.999, 500)
H_pq = -p_true * np.log2(q) - (1 - p_true) * np.log2(1 - q)
KL = H_pq - H_true  # KL 散度

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：二元熵
ax = axes[0]
ax.plot(p, H_p, 'b-', lw=2.5, label='$H(p) = -p\\log p - (1-p)\\log(1-p)$')
# 标注最大值
ax.axvline(0.5, color='r', linestyle='--', alpha=0.5)
ax.scatter([0.5], [1.0], color='red', s=100, zorder=5,
           label='最大值 $H(0.5) = \\log 2 \\approx 0.693$')
# 阴影
ax.fill_between(p, 0, H_p, alpha=0.15, color='blue')
ax.set_xlabel('概率 $p$', fontsize=12)
ax.set_ylabel('熵 $H(p)$  [bit]', fontsize=12)
ax.set_title('二元熵 Binary Entropy', fontsize=13)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.1)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)

# 标注解释
ax.annotate('分布越均匀\n不确定性越大',
            xy=(0.5, 0.7), xytext=(0.65, 0.5),
            fontsize=10, ha='left',
            arrowprops=dict(arrowstyle='->', color='black', lw=1))
ax.annotate('分布越确定\n不确定性越小',
            xy=(0.05, 0.05), xytext=(0.15, 0.3),
            fontsize=10, ha='left',
            arrowprops=dict(arrowstyle='->', color='black', lw=1))

# 右图：交叉熵 + KL
ax = axes[1]
ax.plot(q, H_pq, 'g-', lw=2.5, label='$H(p,q)$ 交叉熵')
ax.plot(q, KL, 'r-', lw=2.5, label='$D_{KL}(p \\| q)$ KL 散度')
# 标注最优预测点
ax.axvline(p_true, color='blue', linestyle='--', alpha=0.5)
ax.scatter([p_true], [H_true], color='blue', s=100, zorder=5,
           label=f'最优预测 $q^* = p = {p_true}$')
# 当 q = 0 或 1 时的爆炸
ax.set_ylim(-0.5, 5)
ax.set_xlim(0, 1)
ax.set_xlabel('预测概率 $q$', fontsize=12)
ax.set_ylabel('损失 [bit]', fontsize=12)
ax.set_title(f'交叉熵 vs KL 散度 (真实 $p={p_true}$)', fontsize=13)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)

# 标注
ax.annotate('$q \\to 0$ 时\n$H \\to +\\infty$\n(梯度爆炸)',
            xy=(0.02, 4.5), xytext=(0.15, 3.5),
            fontsize=10, ha='left',
            arrowprops=dict(arrowstyle='->', color='black', lw=1))

fig.suptitle('信息论核心曲线 (Information Theory Essentials)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/home/david/AI-Tutorial/assets/figures/1-4-binary-entropy.png',
            dpi=120, bbox_inches='tight', facecolor='white')
print("Saved: 1-4-binary-entropy.png")
