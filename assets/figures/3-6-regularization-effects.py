"""
3-6-regularization-effects
==========================
L1/L2 正则化对系数的影响对比（无 sklearn 依赖，手写实现）
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

np.random.seed(42)
n_samples = 100
n_features = 20

X = np.random.randn(n_samples, n_features)
true_coef = np.zeros(n_features)
true_coef[:5] = [1.5, -1.2, 0.8, -0.5, 0.3]
y = X @ true_coef + np.random.randn(n_samples) * 0.1

def soft_threshold(x, threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

def coord_descent_lasso(X, y, alpha, n_iter=200):
    n, p = X.shape
    coef = np.zeros(p)
    XtX = X.T @ X
    Xty = X.T @ y
    for _ in range(n_iter):
        for j in range(p):
            rho = XtX[j, j]
            z = Xty[j] - (XtX[j, :] @ coef) + rho * coef[j]
            coef[j] = soft_threshold(z / rho, alpha / rho)
    return coef

def ridge_closed_form(X, y, alpha):
    n, p = X.shape
    I = np.eye(p)
    return np.linalg.solve(X.T @ X + alpha * I, X.T @ y)

l1_coef = coord_descent_lasso(X, y, alpha=0.1)
l2_coef = ridge_closed_form(X, y, alpha=0.1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.stem(range(n_features), true_coef, linefmt='b-', markerfmt='bo', basefmt=' ', label='真实系数')
ax.stem(range(n_features), l1_coef, linefmt='r--', markerfmt='rx', basefmt=' ', label='L1 估计 (Lasso)')
ax.stem(range(n_features), l2_coef, linefmt='g:', markerfmt='gs', basefmt=' ', label='L2 估计 (Ridge)')
ax.set_xlabel('特征索引')
ax.set_ylabel('系数值')
ax.set_title('Lasso vs Ridge 系数估计对比')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.bar(np.arange(n_features) - 0.2, true_coef, width=0.4, label='真实', color='blue', alpha=0.7)
ax.bar(np.arange(n_features) + 0.2, l1_coef, width=0.4, label='L1', color='red', alpha=0.7)
n_zero = np.sum(np.abs(l1_coef) < 1e-6)
ax.set_title(f'L1 稀疏化（{n_zero}/{n_features} 个系数被压至 0）')
ax.set_xlabel('特征索引')
ax.set_ylabel('系数值')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/david/AI-Tutorial/assets/figures/3-6-regularization-effects.png', dpi=120, bbox_inches='tight', facecolor='white')
# plt.show() # removed for non-interactive rendering

