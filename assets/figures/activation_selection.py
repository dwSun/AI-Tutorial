import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(10, 7))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('激活函数选择流程图', fontsize=16, fontweight='bold', pad=20)

# 定义方框样式
box_style = dict(boxstyle='round,pad=0.5', facecolor='lightblue', edgecolor='navy', linewidth=2)
green_style = dict(boxstyle='round,pad=0.5', facecolor='lightgreen', edgecolor='darkgreen', linewidth=2)
yellow_style = dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='orange', linewidth=2)
purple_style = dict(boxstyle='round,pad=0.5', facecolor='lavender', edgecolor='purple', linewidth=2)

# 起始节点
ax.text(5, 9.2, '开始: 选择激活函数', ha='center', va='center', fontsize=12, fontweight='bold')

# 层级 1
ax.text(5, 8, '网络类型?', ha='center', va='center', fontsize=12, bbox=box_style)

# 分支
ax.annotate('', xy=(2.5, 7.2), xytext=(4.5, 7.7),
            arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
ax.text(2.2, 7.9, 'CNN / MLP', fontsize=10, ha='center', color='blue')

ax.annotate('', xy=(7.5, 7.2), xytext=(5.5, 7.7),
            arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
ax.text(7.8, 7.9, 'RNN / 序列', fontsize=10, ha='center', color='blue')

# CNN/MLP 分支
ax.text(2.5, 6.8, 'ReLU?', ha='center', va='center', fontsize=11, bbox=box_style)

ax.annotate('', xy=(1, 6), xytext=(2, 6.5),
            arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
ax.text(0.5, 6.6, 'Yes', fontsize=9, color='green')

ax.annotate('', xy=(4, 6), xytext=(3, 6.5),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
ax.text(4.3, 6.6, 'No', fontsize=9, color='red')

ax.text(1, 5.6, '[OK] ReLU', ha='center', va='center', fontsize=11, 
        bbox=green_style, color='darkgreen', fontweight='bold')

ax.text(4, 5.6, 'Dying?', ha='center', va='center', fontsize=11, bbox=yellow_style)

ax.annotate('', xy=(3, 4.8), xytext=(3.7, 5.3),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
ax.text(2.5, 5.1, 'Yes', fontsize=9, color='red')

ax.text(3, 4.4, '→ Leaky ReLU\n   或 ELU', ha='center', va='center', fontsize=10,
        bbox=green_style, color='darkgreen')

# RNN 分支
ax.text(7.5, 6.8, 'Tanh', ha='center', va='center', fontsize=11,
        bbox=purple_style, color='purple', fontweight='bold')

# Transformer
ax.annotate('', xy=(7.5, 6), xytext=(5.5, 6.5),
            arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
ax.text(6.5, 6.6, 'Transformer?', fontsize=10, ha='center', color='blue')

ax.text(7.5, 5.6, 'GELU', ha='center', va='center', fontsize=11,
        bbox=purple_style, color='purple', fontweight='bold')

# 输出层
ax.annotate('', xy=(5, 4.8), xytext=(5, 5.6),
            arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
ax.text(5.2, 5.2, '输出层?', fontsize=10, color='blue')

ax.text(5, 4.4, '二分类 → Sigmoid\n多分类 → Softmax', ha='center', va='center',
        fontsize=10, bbox=box_style)

# 添加图例
legend_elements = [
    mpatches.Patch(facecolor='lightgreen', edgecolor='darkgreen', label='推荐首选'),
    mpatches.Patch(facecolor='lightyellow', edgecolor='orange', label='需注意'),
    mpatches.Patch(facecolor='lavender', edgecolor='purple', label='特定场景'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig('activation_selection.png', dpi=150, bbox_inches='tight')
print("activation_selection.png 已保存")
