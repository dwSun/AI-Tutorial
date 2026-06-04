"""
9-2 CLIP N×N 相似度矩阵
========================
展示 CLIP 的核心思想：图像-文本在共享 embedding 空间的对齐
- 4 张图像 × 4 段文本描述
- 对角线最亮（正确配对）
- 使用 softmax 后展示概率分布
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

mpl.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

np.random.seed(123)

# 4 张图像的描述文本
captions = [
    "a photo of a cat",
    "a photo of a dog",
    "a photo of a car",
    "a photo of a tree",
]

image_labels = ['猫', '狗', '汽车', '树']

# 构造相似度矩阵：对角线高，其他位置低
n = 4
sim = np.array([
    [4.2, 1.8, 0.5, 0.3],   # 猫 -> 猫描述最匹配
    [1.5, 4.0, 0.4, 0.2],   # 狗 -> 狗描述最匹配
    [0.4, 0.3, 4.1, 1.0],   # 汽车 -> 汽车描述最匹配
    [0.3, 0.2, 0.8, 3.9],   # 树 -> 树描述最匹配
])

# 对每一行做 softmax 得到概率分布（每张图像的文本匹配概率）
def softmax(x, T=1.0):
    e = np.exp(x / T)
    return e / e.sum(axis=1, keepdims=True)

probs = softmax(sim, T=0.5)

# ============== 绘图 ==============
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：原始 cosine similarity
ax = axes[0]
im = ax.imshow(sim, cmap='RdYlGn', vmin=0, vmax=5)
ax.set_xticks(range(n))
ax.set_xticklabels(captions, rotation=30, ha='right', fontsize=10)
ax.set_yticks(range(n))
ax.set_yticklabels(image_labels, fontsize=11)
ax.set_xlabel('Text', fontsize=12)
ax.set_ylabel('Image', fontsize=12)
ax.set_title('Raw Cosine Similarity\n余弦相似度（对角线最亮）', fontsize=12)
for i in range(n):
    for j in range(n):
        ax.text(j, i, f'{sim[i, j]:.1f}', ha='center', va='center',
                color='white' if sim[i, j] > 2.5 else 'black', fontsize=11)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# 右图：softmax 后的概率分布
ax = axes[1]
im = ax.imshow(probs, cmap='Blues', vmin=0, vmax=0.9)
ax.set_xticks(range(n))
ax.set_xticklabels(captions, rotation=30, ha='right', fontsize=10)
ax.set_yticks(range(n))
ax.set_yticklabels(image_labels, fontsize=11)
ax.set_xlabel('Text', fontsize=12)
ax.set_ylabel('Image', fontsize=12)
ax.set_title('Softmax Probabilities (T=0.5)\nsoftmax 后匹配概率', fontsize=12)
for i in range(n):
    for j in range(n):
        ax.text(j, i, f'{probs[i, j]:.2f}', ha='center', va='center',
                color='white' if probs[i, j] > 0.4 else 'black', fontsize=11)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle('CLIP Image-Text Matching Matrix (N×N)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/home/david/AI-Tutorial/assets/figures/9-2-clip-similarity.png',
            dpi=120, bbox_inches='tight', facecolor='white')
print("Saved: 9-2-clip-similarity.png")
