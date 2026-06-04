"""
2-3 ROC 曲线、PR 曲线与 AUC
============================
- 左图：3 个分类器的 ROC 曲线 + AUC
- 右图：3 个分类器的 Precision-Recall 曲线 + AP
- 底部：展示阈值移动的混淆矩阵
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

np.random.seed(42)
n_samples = 1000

def make_scores(n, pos_rate, sep, noise=0.1):
    n_pos = int(n * pos_rate)
    n_neg = n - n_pos
    pos_scores = np.clip(sep + np.random.randn(n_pos) * noise, 0, 1)
    neg_scores = np.clip(np.random.rand(n_neg) * (1 - sep), 0, 1)
    y_true = np.array([1]*n_pos + [0]*n_neg)
    y_score = np.concatenate([pos_scores, neg_scores])
    idx = np.random.permutation(n)
    return y_true[idx], y_score[idx]

y1, s1 = make_scores(n_samples, 0.3, sep=0.85)
y2, s2 = make_scores(n_samples, 0.3, sep=0.70)
y3, s3 = make_scores(n_samples, 0.3, sep=0.55)

classifiers = [
    ('Classifier A (好)', s1, '#1f77b4'),
    ('Classifier B (中)', s2, '#2ca02c'),
    ('Classifier C (差)', s3, '#d62728'),
]

def roc_curve_manual(y_true, y_score):
    thresholds = np.sort(np.unique(y_score))[::-1]
    tpr_list, fpr_list = [0.0], [0.0]
    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)
    for thr in thresholds:
        y_pred = (y_score >= thr).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tpr_list.append(tp / P if P > 0 else 0)
        fpr_list.append(fp / N if N > 0 else 0)
    tpr_list.append(1.0)
    fpr_list.append(1.0)
    return np.array(fpr_list), np.array(tpr_list)

def auc_manual(fpr, tpr):
    return float(np.trapezoid(tpr, fpr))

def pr_curve_manual(y_true, y_score):
    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]
    tp_cum = np.cumsum(y_true_sorted == 1)
    fp_cum = np.cumsum(y_true_sorted == 0)
    total_pos = np.sum(y_true == 1)
    precision = tp_cum / (tp_cum + fp_cum + 1e-10)
    recall = tp_cum / (total_pos + 1e-10)
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    return recall, precision

def ap_manual(recall, precision):
    return float(np.trapezoid(precision, recall))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for name, score, color in classifiers:
    fpr, tpr = roc_curve_manual(y1, score)
    roc_auc = auc_manual(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2, label=f'{name}  AUC={roc_auc:.3f}')
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='随机基线 AUC=0.5')
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
ax.set_ylabel('True Positive Rate (TPR)', fontsize=12)
ax.set_title('ROC Curve (受试者工作特征曲线)', fontsize=13)
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)

ax = axes[1]
for name, score, color in classifiers:
    recall, precision = pr_curve_manual(y1, score)
    ap = ap_manual(recall, precision)
    ax.plot(recall, precision, color=color, lw=2, label=f'{name}  AP={ap:.3f}')
baseline = 0.3
ax.axhline(baseline, color='k', linestyle='--', lw=1, label=f'随机基线 AP={baseline:.2f}')
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([0, 1.05])
ax.set_xlabel('Recall (召回率)', fontsize=12)
ax.set_ylabel('Precision (精确率)', fontsize=12)
ax.set_title('Precision-Recall Curve (精确率-召回率曲线)', fontsize=13)
ax.legend(loc='lower left', fontsize=10)
ax.grid(True, alpha=0.3)

fig.suptitle('ROC vs PR 曲线对比 (Imbalanced Classification)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/home/david/AI-Tutorial/assets/figures/2-3-roc-pr-curves.png',
            dpi=120, bbox_inches='tight', facecolor='white')
print("Saved: 2-3-roc-pr-curves.png")
