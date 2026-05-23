"""
图5: 损失函数选择决策树
以树形结构展示如何根据任务类型选择损失函数
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# 创建图形
fig, ax = plt.subplots(figsize=(14, 10))

# 禁用坐标轴
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# 定义节点颜色
root_color = '#4ECDC4'      # 青色 - 根节点
regression_color = '#FF6B6B'  # 红色 - 回归
binary_color = '#95E1D3'      # 浅绿 - 二分类
multiclass_color = '#F38181' # 粉红 - 多分类
metric_color = '#AA96DA'     # 紫色 - 度量学习
leaf_color = '#FFEAA7'       # 黄色 - 叶子节点

def draw_box(ax, x, y, width, height, text, color, fontsize=10):
    """绘制圆角矩形框"""
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle="round,pad=0.05,rounding_size=0.2",
                         facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            fontweight='bold', wrap=True)

def draw_arrow(ax, start, end, color='black'):
    """绘制连接箭头"""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=2))

# 第0层: 根节点
draw_box(ax, 7, 9, 3.5, 0.8, '你的任务是什么？', root_color, fontsize=12)

# 第1层: 任务类型
draw_box(ax, 2.5, 7, 2.5, 0.7, '回归任务\n(预测连续值)', regression_color, fontsize=10)
draw_box(ax, 7, 7, 2.5, 0.7, '二分类任务\n(正/负二类)', binary_color, fontsize=10)
draw_box(ax, 11.5, 7, 2.5, 0.7, '多分类任务\n(K个类别)', multiclass_color, fontsize=10)

# 第2层: 具体损失函数
# 回归分支
draw_box(ax, 1, 4.5, 2, 0.7, 'MSE\n(L2损失)', leaf_color, fontsize=9)
draw_box(ax, 4, 4.5, 2, 0.7, 'MAE\n(L1损失)', leaf_color, fontsize=9)

# 二分类分支
draw_box(ax, 6, 4.5, 2.2, 0.7, 'BCE\n(Sigmoid+CE)', leaf_color, fontsize=9)

# 多分类分支
draw_box(ax, 10, 4.5, 2.2, 0.7, 'Softmax\n+ Cross Entropy', leaf_color, fontsize=9)

# 度量学习分支
draw_box(ax, 11.5, 5.5, 2.2, 0.7, '度量学习\n(距离/相似度)', metric_color, fontsize=10)
draw_box(ax, 9, 2.5, 2.2, 0.7, 'Triplet Loss', leaf_color, fontsize=9)
draw_box(ax, 12, 2.5, 2.2, 0.7, 'Contrastive\nLoss', leaf_color, fontsize=9)
draw_box(ax, 14.5, 2.5, 2.2, 0.7, 'ArcFace\nAM-Softmax', leaf_color, fontsize=9)

# 回归第三层
draw_box(ax, 2.5, 2, 2.2, 0.7, 'Huber Loss\n(Smooth L1)', leaf_color, fontsize=9)

# 第3层连接
# 回归
draw_arrow(ax, (2.5, 4.15), (1, 4.85))
draw_arrow(ax, (2.5, 4.15), (4, 4.85))
draw_arrow(ax, (1, 4.15), (2.5, 2.35))
draw_arrow(ax, (4, 4.15), (2.5, 2.35))

# 度量学习
draw_arrow(ax, (11.5, 5.15), (9, 2.85))
draw_arrow(ax, (11.5, 5.15), (12, 2.85))
draw_arrow(ax, (11.5, 5.15), (14.5, 2.85))

# 第1层连接
draw_arrow(ax, (5.75, 8.6), (2.5, 7.35))
draw_arrow(ax, (7, 8.6), (7, 7.35))
draw_arrow(ax, (8.25, 8.6), (11.5, 7.35))

# 度量学习连接
draw_arrow(ax, (8.5, 9), (11.5, 5.85))

# 第2层连接（实际是第二层连接到第一层的子分支）
draw_arrow(ax, (6, 4.15), (6, 4.85))
draw_arrow(ax, (10, 4.15), (10, 4.85))

# 添加标题
ax.text(7, 10.3, '损失函数选择决策树', ha='center', va='center',
         fontsize=16, fontweight='bold')

# 添加图例说明
legend_text = """选择指南:
• 回归任务 → MSE / MAE / Huber
• 二分类 → BCE (Sigmoid + Cross Entropy)  
• 多分类 → Softmax + Cross Entropy
• 人脸识别/度量学习 → Triplet / ArcFace"""

props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9)
ax.text(0.3, 1.2, legend_text, transform=ax.transData, fontsize=9,
        verticalalignment='top', bbox=props, family='monospace')

# 添加连接线装饰（虚线表示可选分支）
ax.plot([5.75, 8.5], [8.6, 9], 'k--', alpha=0.2, linewidth=1)

plt.tight_layout()
plt.savefig('fig5_loss_decision_tree.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("fig5_loss_decision_tree saved.")
