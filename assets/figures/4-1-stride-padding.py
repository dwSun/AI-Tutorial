"""
4-1-stride-padding.py
======================
步幅（Stride）和填充（Padding）对卷积输出尺寸的影响
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

test_image = np.array([
    [1, 2, 3, 4, 5],
    [2, 3, 4, 5, 6],
    [3, 4, 5, 6, 7],
    [4, 5, 6, 7, 8],
    [5, 6, 7, 8, 9]
])

kernel_identity = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

def manual_convolution(image, kernel, stride=1, padding=0):
    if padding > 0:
        image_padded = np.pad(image, ((padding, padding), (padding, padding)), mode='constant')
    else:
        image_padded = image
    h_in, w_in = image_padded.shape
    h_kernel, w_kernel = kernel.shape
    h_out = (h_in - h_kernel) // stride + 1
    w_out = (w_in - w_kernel) // stride + 1
    output = np.zeros((h_out, w_out))
    for i in range(0, h_out):
        for j in range(0, w_out):
            image_patch = image_padded[i*stride:i*stride+h_kernel, j*stride:j*stride+w_kernel]
            output[i, j] = np.sum(image_patch * kernel)
    return output

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

ax = axes[0, 0]
im = ax.imshow(test_image, cmap='gray')
ax.set_title('原始图像 (5x5)')
ax.set_xticks(range(5))
ax.set_yticks(range(5))
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
for i in range(5):
    for j in range(5):
        ax.text(j, i, str(test_image[i, j]), ha='center', va='center', color='white')

configs = [
    {'stride': 1, 'padding': 0, 'title': 'stride=1, padding=0\n输出尺寸: (3x3)'},
    {'stride': 2, 'padding': 0, 'title': 'stride=2, padding=0\n输出尺寸: (2x2)'},
    {'stride': 1, 'padding': 1, 'title': 'stride=1, padding=1\n输出尺寸: (5x5)'},
    {'stride': 2, 'padding': 1, 'title': 'stride=2, padding=1\n输出尺寸: (3x3)'},
    {'stride': 3, 'padding': 0, 'title': 'stride=3, padding=0\n输出尺寸: (1x1)'}
]

for idx, config in enumerate(configs):
    row = 0 if idx < 2 else 1
    col = idx if idx < 2 else idx - 2
    h_in, w_in = test_image.shape
    h_kernel, w_kernel = kernel_identity.shape
    padding = config['padding']
    stride = config['stride']
    h_out = (h_in + 2*padding - h_kernel) // stride + 1
    w_out = (w_in + 2*padding - w_kernel) // stride + 1
    result = manual_convolution(test_image, kernel_identity, stride=stride, padding=padding)
    ax = axes[row, col]
    im = ax.imshow(result, cmap='viridis')
    ax.set_title(config['title'])
    ax.set_xticks(range(w_out))
    ax.set_yticks(range(h_out))
    for i in range(h_out):
        for j in range(w_out):
            ax.text(j, i, f'{result[i, j]:.0f}', ha='center', va='center',
                   color='white' if result[i, j] > 4 else 'black')
    ax.text(0, -0.5, f'公式: ⌊({h_in}+2×{padding}-{h_kernel})/{stride}⌋+1 = {h_out}',
            transform=ax.transAxes, fontsize=9)

plt.tight_layout()
plt.suptitle('步幅(Stride)和填充(Padding)对输出尺寸的影响', fontsize=16, y=1.02)
plt.savefig('/home/david/AI-Tutorial/assets/figures/4-1-stride-padding.png',
            dpi=120, bbox_inches='tight', facecolor='white')
print("Saved: 4-1-stride-padding.png")
