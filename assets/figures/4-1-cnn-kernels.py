"""
4-1-cnn-kernels.py
===================
不同卷积核效果可视化（边缘检测、模糊、锐化、Sobel 等 7 种核）
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

np.random.seed(42)
example_image = np.zeros((8, 8))
example_image[2:6, 2:6] = 1

kernels = {
    '边缘检测（垂直）': np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
    '边缘检测（水平）': np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
    '锐化': np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
    '高斯模糊': np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16,
    '均值模糊': np.ones((3, 3)) / 9,
    'Sobel（x方向）': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
    'Sobel（y方向）': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
}

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
            i_start = i * stride
            j_start = j * stride
            image_patch = image_padded[i_start:i_start+h_kernel, j_start:j_start+w_kernel]
            output[i, j] = np.sum(image_patch * kernel)
    return output

fig, axes = plt.subplots(3, 4, figsize=(16, 12))

ax = axes[0, 0]
im = ax.imshow(example_image, cmap='gray', vmin=0, vmax=1)
ax.set_title('原始图像 (8x8)')
ax.set_xticks([])
ax.set_yticks([])
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

for idx, (kernel_name, kernel) in enumerate(list(kernels.items())[:3]):
    ax = axes[0, idx+1]
    im = ax.imshow(kernel, cmap='coolwarm')
    ax.set_title(f'{kernel_name}\n{kernel.shape[0]}x{kernel.shape[1]}')
    ax.set_xticks(range(kernel.shape[1]))
    ax.set_yticks(range(kernel.shape[0]))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            ax.text(j, i, f'{kernel[i, j]:.2f}', ha='center', va='center',
                   color='white' if abs(kernel[i, j]) > 0.5 else 'black')

for idx, (kernel_name, kernel) in enumerate(kernels.items()):
    row = 1 + idx // 4
    col = idx % 4
    result = manual_convolution(example_image, kernel, stride=1, padding=1)
    ax = axes[row, col]
    im = ax.imshow(result, cmap='viridis')
    ax.set_title(f'{kernel_name}结果')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            ax.text(j, i, f'{result[i, j]:.1f}', ha='center', va='center',
                   color='white' if abs(result[i, j]) > 2 else 'black')

plt.tight_layout()
plt.suptitle('不同卷积核效果可视化 (stride=1, padding=1)', fontsize=16, y=1.02)
plt.savefig('/home/david/AI-Tutorial/assets/figures/4-1-cnn-kernels.png',
            dpi=120, bbox_inches='tight', facecolor='white')
print("Saved: 4-1-cnn-kernels.png")
