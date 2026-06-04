"""
图4: Triplet Loss（三元组损失）示意图
展示 anchor, positive, negative 三元组的关系
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 创建图形
fig, ax = plt.subplots(figsize=(12, 6))

# 定义三元组的二维嵌入位置
anchor = np.array([0.5, 0.5])
positive = np.array([0.8, 0.7])
negative = np.array([1.8, 1.6])

# 计算距离
dist_ap = np.linalg.norm(anchor - positive)
dist_an = np.linalg.norm(anchor - negative)

# 绘制样本点
ax.scatter(*anchor, color='blue', s=300, marker='o', zorder=5, edgecolors='black', linewidths=2)
ax.scatter(*positive, color='green', s=300, marker='o', zorder=5, edgecolors='black', linewidths=2)
ax.scatter(*negative, color='red', s=300, marker='o', zorder=5, edgecolors='black', linewidths=2)

# 标注样本类型
ax.annotate('Anchor\n(锚点)', anchor, textcoords="offset points", xytext=(-30, -25),
            fontsize=11, ha='center', fontweight='bold')
ax.annotate('Positive\n(正例)', positive, textcoords="offset points", xytext=(15, 15),
            fontsize=11, ha='center', fontweight='bold', color='green')
ax.annotate('Negative\n(负例)', negative, textcoords="offset points", xytext=(15, 10),
            fontsize=11, ha='center', fontweight='bold', color='red')

# 绘制连线
# Anchor 到 Positive (实线，表示pull拉近)
ax.plot([anchor[0], positive[0]], [anchor[1], positive[1]], 
        'g-', linewidth=2.5, label=f'd(a,p) = {dist_ap:.2f} (需减小)')

# Anchor 到 Negative (虚线，表示push推远)
ax.plot([anchor[0], negative[0]], [anchor[1], negative[1]], 
        'r--', linewidth=2.5, label=f'd(a,n) = {dist_an:.2f} (需增大)')

# 添加距离标注（中间点）
mid_ap = (anchor + positive) / 2
mid_an = (anchor + negative) / 2

# 绘制margin圆弧（以anchor为中心）
margin_circle = plt.Circle(anchor, dist_ap + 0.3, fill=False, 
                            color='orange', linestyle=':', linewidth=2, alpha=0.7)
ax.add_patch(margin_circle)

# 添加向量标注
ax.annotate('', xy=positive, xytext=anchor,
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax.annotate('', xy=negative, xytext=anchor,
            arrowprops=dict(arrowstyle='<->', color='red', lw=2))

# 添加Triplet Loss公式
formula = r'$\mathcal{L} = \max\left( \|f_a - f_p\|^2 - \|f_a - f_n\|^2 + m, 0 \right)$'
ax.text(0.5, 0.02, formula, transform=ax.transAxes, fontsize=13,
        ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# 设置坐标轴
ax.set_xlim(-0.2, 2.5)
ax.set_ylim(-0.2, 2.2)
ax.set_xlabel('嵌入维度 1', fontsize=12)
ax.set_ylabel('嵌入维度 2', fontsize=12)
ax.set_title('Triplet Loss（三元组损失）示意图', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# 添加说明框
textstr = '目标:\n• 拉近 Anchor-Positive 距离\n• 推远 Anchor-Negative 距离\n• 间隔至少为 margin'
props = dict(boxstyle='round', facecolor='lightcyan', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('fig4_triplet_loss.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("fig4_triplet_loss saved.")
