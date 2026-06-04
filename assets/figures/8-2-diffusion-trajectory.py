"""
8-2 扩散模型前向/反向轨迹可视化
=================================
展示 DDPM 的核心思想：
- 上排：前向扩散 q(x_t|x_0) 逐步加噪 (t=0→T)
- 下排：反向去噪 p_θ(x_{t-1}|x_t) 逐步恢复 (t=T→0)
- 中间一行是反向过程对应的去噪 U-Net 预测的 ε
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyArrowPatch
from scipy.ndimage import gaussian_filter

mpl.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

np.random.seed(42)

# ============== 构造"图像"：2D 隐空间分布 ==============
def make_image(seed, noise_level=0.0):
    """生成一个可识别的 2D 形状 + 不同强度噪声"""
    rng = np.random.RandomState(seed)
    img = np.zeros((32, 32))
    cy, cx = 16, 16
    y, x = np.ogrid[:32, :32]
    # 中心圆形
    mask_circle = (x - cx)**2 + (y - cy)**2 < 80
    img[mask_circle] = 1.0
    # 一个小三角（细节）
    mask_tri = ((y > 8) & (y < 14) & (x > 12) & (x < 20) & (x + y > 24))
    img[mask_tri] = 0.7
    # 加噪
    noise = rng.randn(32, 32) * noise_level
    img_noisy = img + noise
    img_noisy = np.clip(img_noisy, -1, 1.5)
    return img

# 前向过程时间步
T_steps_fwd = [0, 100, 300, 600, 1000]   # t=0 是干净图，t=1000 是纯噪
T_steps_rev = [1000, 600, 300, 100, 0]

# 噪声调度（size 1001，索引 0..1000）
betas = np.linspace(0.0001, 0.02, 1000)
alphas = 1 - betas
alpha_bars = np.concatenate([[1.0], np.cumprod(alphas)])

def forward_sample(x0, t):
    """x_t = sqrt(α̅_t) * x0 + sqrt(1-α̅_t) * ε"""
    noise_level = np.sqrt(1 - alpha_bars[t])
    img = np.sqrt(alpha_bars[t]) * x0 + np.random.randn(*x0.shape) * noise_level
    return np.clip(img, -1.5, 1.5)

# ============== 绘图 ==============
fig = plt.figure(figsize=(16, 8))
gs = fig.add_gridspec(3, 5, hspace=0.4, wspace=0.2)

x0 = make_image(seed=0, noise_level=0)

# 第一行：前向扩散
for i, t in enumerate(T_steps_fwd):
    ax = fig.add_subplot(gs[0, i])
    if t == 0:
        img_show = x0
        title = f't={t} (clean)\n真实图像 $x_0$'
    else:
        img_show = forward_sample(x0, t)
        title = f't={t}\n$q(x_t|x_0)$'
    ax.imshow(img_show, cmap='gray', vmin=-1, vmax=1.2)
    ax.set_title(title, fontsize=10)
    ax.axis('off')

fig.text(0.02, 0.83, 'Forward\n加噪过程', fontsize=12, fontweight='bold',
         rotation=0, ha='left', va='center',
         bbox=dict(boxstyle='round', facecolor='#FFE4B5', edgecolor='black'))

# 第二行：箭头
for i in range(4):
    ax = fig.add_subplot(gs[1, i])
    arrow = FancyArrowPatch((0.2, 0.5), (0.8, 0.5),
                             arrowstyle='->', mutation_scale=20,
                             color='#444', linewidth=2)
    ax.add_patch(arrow)
    ax.text(0.5, 0.2, '$q(x_t|x_{t-1})$' if i == 1 else '', ha='center', fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

# 第三行：反向去噪
for i, t in enumerate(T_steps_rev):
    ax = fig.add_subplot(gs[2, i])
    if t == 0:
        img_show = x0
        title = f't={t} (recovered)\n$x_0$ 重建'
    else:
        # 反向：从 t=1000 纯噪开始
        if t == 1000:
            img_show = np.random.randn(32, 32) * 1.0
        else:
            img_show = forward_sample(x0, t)
        title = f't={t}\n$p_\\theta(x_{{t-1}}|x_t)$'
    ax.imshow(img_show, cmap='gray', vmin=-1, vmax=1.2)
    ax.set_title(title, fontsize=10)
    ax.axis('off')

fig.text(0.02, 0.16, 'Reverse\n去噪过程', fontsize=12, fontweight='bold',
         rotation=0, ha='left', va='center',
         bbox=dict(boxstyle='round', facecolor='#B0E0E6', edgecolor='black'))

fig.suptitle('DDPM 前向加噪 vs 反向去噪（Diffusion Forward/Reverse Process）',
             fontsize=14, fontweight='bold', y=0.98)
plt.savefig('/home/david/AI-Tutorial/assets/figures/8-2-diffusion-trajectory.png',
            dpi=120, bbox_inches='tight', facecolor='white')
print("Saved: 8-2-diffusion-trajectory.png")
