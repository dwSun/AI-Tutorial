"""
4-1-real-image.py
==================
实际图像上的卷积操作示例（边缘检测、模糊）
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import ndimage

mpl.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

np.random.seed(7)
size = 128
x = np.linspace(0, 4*np.pi, size)
y = np.linspace(0, 4*np.pi, size)
X, Y = np.meshgrid(x, y)
img = (np.sin(X) * np.cos(Y) * 0.5 + 0.5)
img = img + np.random.rand(size, size) * 0.1
img = np.clip(img, 0, 1)

edge_kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
blur_kernel = np.ones((5, 5)) / 25

edge_result = ndimage.convolve(img, edge_kernel)
blur_result = ndimage.convolve(img, blur_kernel)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img, cmap='gray')
axes[0].set_title('原始图像 (合成)')
axes[0].axis('off')
axes[1].imshow(edge_result, cmap='gray')
axes[1].set_title('边缘检测 (Sobel)')
axes[1].axis('off')
axes[2].imshow(blur_result, cmap='gray')
axes[2].set_title('均值模糊 (5x5)')
axes[2].axis('off')
plt.suptitle('实际图像上的卷积操作示例 (合成测试图)', fontsize=14)
plt.tight_layout()
plt.savefig('/home/david/AI-Tutorial/assets/figures/4-1-real-image.png',
            dpi=120, bbox_inches='tight', facecolor='white')
print("Saved: 4-1-real-image.png")
