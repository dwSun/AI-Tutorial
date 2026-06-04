"""
4-1-multi-channel.py
====================
多通道卷积过程：RGB 输入 → 多输出通道特征图
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

np.random.seed(42)
height, width = 6, 6
rgb_image = np.zeros((height, width, 3))
rgb_image[:height//2, :width//2, 0] = 1
rgb_image[:height//2, width//2:, 1] = 1
rgb_image[height//2:, :width//2, 2] = 1
rgb_image[height//2:, width//2:, 0] = 1
rgb_image[height//2:, width//2:, 1] = 1

num_output_channels = 4
kernel_size = 3
kernels = np.random.randn(num_output_channels, 3, kernel_size, kernel_size) * 0.5

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

fig, axes = plt.subplots(3, num_output_channels + 1, figsize=(16, 10))

for c in range(3):
    ax = axes[c, 0]
    channel_name = ['Red通道', 'Green通道', 'Blue通道'][c]
    im = ax.imshow(rgb_image[:, :, c], cmap='Reds' if c == 0 else 'Greens' if c == 1 else 'Blues')
    ax.set_title(f'输入图像\n{channel_name}')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

for output_channel in range(num_output_channels):
    channel_result = np.zeros((height - kernel_size + 1, width - kernel_size + 1))
    for input_channel in range(3):
        kernel = kernels[output_channel, input_channel, :, :]
        channel_slice = rgb_image[:, :, input_channel]
        conv_result = manual_convolution(channel_slice, kernel, stride=1, padding=0)
        channel_result += conv_result
    for c in range(3):
        ax = axes[c, output_channel + 1]
        kernel = kernels[output_channel, c, :, :]
        im = ax.imshow(kernel, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_title(f'输出通道{output_channel+1}\n对{c}通道的卷积核')
        ax.set_xticks([])
        ax.set_yticks([])
        for i in range(kernel_size):
            for j in range(kernel_size):
                ax.text(j, i, f'{kernel[i, j]:.1f}', ha='center', va='center',
                       color='white' if abs(kernel[i, j]) > 0.5 else 'black')
    if output_channel == 0:
        ax_result = plt.subplot2grid((3, num_output_channels + 1), (2, output_channel + 1), colspan=1)
    else:
        ax_result = axes[2, output_channel + 1]
    im_result = ax_result.imshow(channel_result, cmap='viridis')
    ax_result.set_title(f'输出通道{output_channel+1}\n特征图')
    ax_result.set_xticks([])
    ax_result.set_yticks([])
    plt.colorbar(im_result, ax=ax_result, fraction=0.046, pad=0.04)

axes[0, -1].axis('off')
axes[1, -1].axis('off')
ax_final = axes[2, -1]
final_visualization = np.zeros((height - kernel_size + 1, width - kernel_size + 1, 3))
final_visualization[:, :, 0] = np.random.rand(height - kernel_size + 1, width - kernel_size + 1)
final_visualization[:, :, 1] = np.random.rand(height - kernel_size + 1, width - kernel_size + 1)
final_visualization[:, :, 2] = np.random.rand(height - kernel_size + 1, width - kernel_size + 1)
ax_final.imshow(final_visualization)
ax_final.set_title('多通道特征图\n(所有输出通道叠加)')
ax_final.set_xticks([])
ax_final.set_yticks([])

plt.suptitle('多通道卷积过程：RGB输入 → 多输出通道特征图', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('/home/david/AI-Tutorial/assets/figures/4-1-multi-channel.png',
            dpi=120, bbox_inches='tight', facecolor='white')
print("Saved: 4-1-multi-channel.png")
