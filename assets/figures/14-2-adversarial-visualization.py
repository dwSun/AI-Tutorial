"""
14-2-adversarial-visualization
================================
对抗样本可视化（原始 / 对抗 / 扰动 / 预测）
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

def visualize_adversarial(original, adversarial, perturbation, pred_orig='猫', pred_adv='狗'):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original)
    axes[0].set_title(f'原始样本  预测: {pred_orig}', fontsize=12)
    axes[0].axis('off')
    axes[1].imshow(adversarial)
    axes[1].set_title(f'对抗样本  预测: {pred_adv}', fontsize=12)
    axes[1].axis('off')
    im = axes[2].imshow(perturbation, cmap='RdBu', vmin=-0.3, vmax=0.3)
    axes[2].set_title('对抗扰动 (×10)', fontsize=12)
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig('/home/david/AI-Tutorial/assets/figures/14-2-adversarial-visualization.png', dpi=120, bbox_inches='tight', facecolor='white')
    # plt.show() # removed for non-interactive rendering


# 模拟示例
np.random.seed(0)
size = 64
original = np.random.rand(size, size, 3) * 0.3
yy, xx = np.mgrid[:size, :size]
mask = ((xx - size//2)**2 + (yy - size//2)**2) < (size//3)**2
original[mask] = [0.9, 0.7, 0.3]
perturbation = np.random.randn(size, size, 3) * 0.05
adversarial = np.clip(original + perturbation * 5, 0, 1)
visualize_adversarial(original, adversarial, perturbation * 5, pred_orig='猫 (92%)', pred_adv='狗 (88%)')
