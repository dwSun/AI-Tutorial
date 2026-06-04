"""
6-1 注意力热力图 (Attention Heatmap)
=====================================
可视化 Self-Attention 权重矩阵：
- 左图：源句子与目标句子的注意力对齐（机器翻译场景）
- 右图：多头注意力的多头热力图（4 个 head）
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 中文字体
mpl.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

np.random.seed(7)

# ============== 左图：英→中 翻译注意力对齐 ==============
src_words = ['The', 'cat', 'sat', 'on', 'the', 'mat']
tgt_words = ['猫', '坐', '在', '垫子', '上']

# 构造一个可解释的注意力矩阵（"mat" 应主要对齐 "垫子"）
attn_align = np.array([
    [0.05, 0.05, 0.05, 0.05, 0.05, 0.75],  # 猫  -> 关注 The/cat
    [0.20, 0.60, 0.10, 0.05, 0.03, 0.02],  # 坐  -> 关注 sat
    [0.10, 0.10, 0.50, 0.10, 0.10, 0.10],  # 在  -> 关注 sat/on
    [0.05, 0.05, 0.10, 0.10, 0.65, 0.05],  # 垫子 -> 关注 the/mat
    [0.10, 0.10, 0.10, 0.60, 0.05, 0.05],  # 上  -> 关注 on
])
attn_align = attn_align / attn_align.sum(axis=1, keepdims=True)

# ============== 右图：多头注意力的 4 个头 ==============
tgt_words_4 = ['我', '爱', '自然', '语言', '处理']
n_tokens = len(tgt_words_4)
heads = []
# Head 1: 对角线（自对齐）
h1 = np.eye(n_tokens) * 0.5 + np.random.rand(n_tokens, n_tokens) * 0.1
h1 = h1 / h1.sum(axis=1, keepdims=True)
# Head 2: 偏向 "我"（位置 0）
h2 = np.zeros((n_tokens, n_tokens))
h2[:, 0] = 0.6
h2 += np.random.rand(n_tokens, n_tokens) * 0.05
h2 = h2 / h2.sum(axis=1, keepdims=True)
# Head 3: 局部窗口（前一个 token）
h3 = np.eye(n_tokens) * 0.3
for i in range(n_tokens - 1):
    h3[i, i+1] = 0.5
h3 += np.random.rand(n_tokens, n_tokens) * 0.05
h3 = h3 / h3.sum(axis=1, keepdims=True)
# Head 4: 偏向最后两个 token
h4 = np.zeros((n_tokens, n_tokens))
h4[:, -2:] = 0.4
h4 += np.random.rand(n_tokens, n_tokens) * 0.05
h4 = h4 / h4.sum(axis=1, keepdims=True)
heads = [h1, h2, h3, h4]

# ============== 绘图 ==============
fig, axes = plt.subplots(1, 5, figsize=(18, 4), gridspec_kw={'width_ratios': [5, 4, 4, 4, 4]})

# 左图：跨语言注意力
ax = axes[0]
im = ax.imshow(attn_align, cmap='YlOrRd', vmin=0, vmax=0.8)
ax.set_xticks(range(len(src_words)))
ax.set_xticklabels(src_words, fontsize=11)
ax.set_yticks(range(len(tgt_words)))
ax.set_yticklabels(tgt_words, fontsize=11)
ax.set_xlabel('Source (English)', fontsize=12)
ax.set_ylabel('Target (Chinese)', fontsize=12)
ax.set_title('Cross-lingual Attention\n(EN→ZH, "the mat" → "垫子")', fontsize=12)
# 标注每格数值
for i in range(len(tgt_words)):
    for j in range(len(src_words)):
        ax.text(j, i, f'{attn_align[i, j]:.2f}', ha='center', va='center',
                color='white' if attn_align[i, j] > 0.4 else 'black', fontsize=9)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# 右图：4 个 head
for idx, (head, ax) in enumerate(zip(heads, axes[1:]), 1):
    im = ax.imshow(head, cmap='Blues', vmin=0, vmax=0.6)
    ax.set_xticks(range(n_tokens))
    ax.set_xticklabels(tgt_words_4, fontsize=10)
    ax.set_yticks(range(n_tokens))
    ax.set_yticklabels(tgt_words_4, fontsize=10)
    ax.set_title(f'Head {idx}', fontsize=12)
    for i in range(n_tokens):
        for j in range(n_tokens):
            ax.text(j, i, f'{head[i, j]:.2f}', ha='center', va='center',
                    color='white' if head[i, j] > 0.3 else 'black', fontsize=8)

fig.suptitle('Self-Attention 权重可视化（Self-Attention Weight Visualization）',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/home/david/AI-Tutorial/assets/figures/6-1-attention-heatmap.png',
            dpi=120, bbox_inches='tight', facecolor='white')
print("Saved: 6-1-attention-heatmap.png")
