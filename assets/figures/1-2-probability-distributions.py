"""
1-2-probability-distributions
=============================
常见概率分布可视化
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False
from scipy.stats import bernoulli, binom, norm
plt.style.use('default')

# 设置中文字体（如果系统支持中文）
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 1. 伯努利分布 (p=0.7)
p_bernoulli = 0.7
samples_bernoulli = bernoulli.rvs(p_bernoulli, size=1000)
success_rate = np.mean(samples_bernoulli)

axes[0, 0].bar([0, 1], [1-p_bernoulli, p_bernoulli], 
               width=0.6, color=['skyblue', 'lightcoral'], alpha=0.7)
axes[0, 0].scatter([0, 1], [1-p_bernoulli, p_bernoulli], 
                   color=['blue', 'red'], s=100, zorder=3)
axes[0, 0].set_xticks([0, 1])
axes[0, 0].set_xticklabels(['失败 (0)', '成功 (1)'])
axes[0, 0].set_ylabel('概率')
axes[0, 0].set_title(f'伯努利分布 (p={p_bernoulli})')
axes[0, 0].text(0.5, 0.8, f'理论概率: P(成功)={p_bernoulli}\n模拟概率: {success_rate:.3f}',
                transform=axes[0, 0].transAxes, ha='center', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 2. 二项分布 (n=10, p=0.4)
n_binom, p_binom = 10, 0.4
x_binom = np.arange(0, n_binom+1)
pmf_binom = binom.pmf(x_binom, n_binom, p_binom)
samples_binom = binom.rvs(n_binom, p_binom, size=5000)

axes[0, 1].bar(x_binom, pmf_binom, alpha=0.7, color='lightgreen')
axes[0, 1].set_xlabel('成功次数')
axes[0, 1].set_ylabel('概率')
axes[0, 1].set_title(f'二项分布 (n={n_binom}, p={p_binom})')
axes[0, 1].grid(True, alpha=0.3)

# 添加统计信息
mean_binom = np.mean(samples_binom)
var_binom = np.var(samples_binom)
axes[0, 1].text(0.05, 0.95, 
                f'理论期望: {n_binom*p_binom:.1f}\n理论方差: {n_binom*p_binom*(1-p_binom):.2f}',
                transform=axes[0, 1].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

# 3. 正态分布 (μ=0, σ=1)
mu_norm, sigma_norm = 0, 1
x_norm = np.linspace(mu_norm - 4*sigma_norm, mu_norm + 4*sigma_norm, 1000)
pdf_norm = norm.pdf(x_norm, mu_norm, sigma_norm)
samples_norm = norm.rvs(mu_norm, sigma_norm, size=5000)

axes[1, 0].plot(x_norm, pdf_norm, 'b-', linewidth=2, label='概率密度函数')
axes[1, 0].hist(samples_norm, bins=30, density=True, alpha=0.4, color='orange', label='样本分布')
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('概率密度')
axes[1, 0].set_title(f'正态分布 N({mu_norm}, {sigma_norm}²)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. 不同参数的正态分布对比
x_compare = np.linspace(-5, 5, 1000)
params = [(0, 0.5, 'N(0, 0.5²)', 'blue'), 
          (0, 1, 'N(0, 1²)', 'green'),
          (0, 2, 'N(0, 2²)', 'red'),
          (1.5, 1, 'N(1.5, 1²)', 'purple')]

for mu, sigma, label, color in params:
    pdf = norm.pdf(x_compare, mu, sigma)
    axes[1, 1].plot(x_compare, pdf, label=label, color=color, linewidth=2)

axes[1, 1].set_xlabel('x')
axes[1, 1].set_ylabel('概率密度')
axes[1, 1].set_title('不同参数的正态分布对比')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.tight_layout()
plt.savefig('/home/david/AI-Tutorial/assets/figures/1-2-probability-distributions.png', dpi=120, bbox_inches='tight', facecolor='white')
# plt.show() # removed for non-interactive rendering
