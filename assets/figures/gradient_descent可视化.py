#!/usr/bin/env python3
"""
梯度下降可视化脚本
生成梯度下降路径图（loss曲面+下降路径箭头）
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

def gradient_descent_path():
    """生成梯度下降3D路径图"""
    
    # 定义损失函数 (Rosenbrock函数简化版: f(x,y) = (1-x)^2 + 100(y-x^2)^2)
    def loss_func(x, y):
        return (1 - x)**2 + 100 * (y - x**2)**2
    
    # 梯度
    def gradient(x, y):
        dx = -2 * (1 - x) - 400 * x * (y - x**2)
        dy = 200 * (y - x**2)
        return np.array([dx, dy])
    
    # 梯度下降
    def compute_gradient_descent(start, lr=0.001, max_iter=1000, tol=1e-6):
        path = [start]
        x, y = start
        for _ in range(max_iter):
            grad = gradient(x, y)
            if np.linalg.norm(grad) < tol:
                break
            x = x - lr * grad[0]
            y = y - lr * grad[1]
            path.append([x, y])
        return np.array(path)
    
    # 起始点 (从远处开始)
    start_point = np.array([-0.5, -0.8])
    path = compute_gradient_descent(start_point, lr=0.0008, max_iter=2000)
    
    # 创建网格用于绘制曲面
    x = np.linspace(-1.5, 1.5, 200)
    y = np.linspace(-1.5, 1.5, 200)
    X, Y = np.meshgrid(x, y)
    Z = loss_func(X, Y)
    
    # 绘图
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制损失曲面 (使用对数刻度更好地显示)
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, edgecolor='none')
    
    # 绘制梯度下降路径
    path_z = loss_func(path[:, 0], path[:, 1])
    ax.plot(path[:, 0], path[:, 1], path_z, 'r.-', linewidth=2, markersize=4, label='Gradient Descent Path')
    
    # 标记起点和终点
    ax.scatter(*path[0], color='green', s=150, marker='o', label=f'Start: ({path[0, 0]:.2f}, {path[0, 1]:.2f})', zorder=5)
    ax.scatter(*path[-1], color='red', s=150, marker='*', label=f'End: ({path[-1, 0]:.2f}, {path[-1, 1]:.2f})', zorder=5)
    
    ax.set_xlabel('w₁', fontsize=12)
    ax.set_ylabel('w₂', fontsize=12)
    ax.set_zlabel('Loss', fontsize=12)
    ax.set_title('Gradient Descent on Loss Surface (Rosenbrock Function)', fontsize=14)
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig('gradient_descent_3d.png', dpi=150, bbox_inches='tight')
    print("Saved: gradient_descent_3d.png")
    plt.close()


def gradient_descent_contour():
    """生成梯度下降等高线图"""
    
    def loss_func(x, y):
        return (1 - x)**2 + 100 * (y - x**2)**2
    
    def gradient(x, y):
        dx = -2 * (1 - x) - 400 * x * (y - x**2)
        dy = 200 * (y - x**2)
        return np.array([dx, dy])
    
    def compute_gradient_descent(start, lr=0.001, max_iter=1000, tol=1e-6):
        path = [start]
        x, y = start
        for _ in range(max_iter):
            grad = gradient(x, y)
            if np.linalg.norm(grad) < tol:
                break
            x = x - lr * grad[0]
            y = y - lr * grad[1]
            path.append([x, y])
        return np.array(path)
    
    # 多个起始点的梯度下降
    start_points = [np.array([-0.5, -0.8]), np.array([0.5, -0.8]), np.array([-0.8, 0.5])]
    colors = ['red', 'blue', 'green']
    
    x = np.linspace(-1.5, 1.5, 300)
    y = np.linspace(-1.5, 1.5, 300)
    X, Y = np.meshgrid(x, y)
    Z = loss_func(X, Y)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制等高线
    levels = np.logspace(-1, 3, 20)
    contour = ax.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.8)
    ax.contour(X, Y, Z, levels=levels, colors='white', alpha=0.3, linewidths=0.5)
    
    # 绘制每个起始点的下降路径
    for start, color in zip(start_points, colors):
        path = compute_gradient_descent(start, lr=0.0008, max_iter=2000)
        ax.plot(path[:, 0], path[:, 1], 'o-', color=color, markersize=3, linewidth=1.5, alpha=0.8)
        ax.scatter(*path[0], color=color, s=100, marker='o', edgecolors='white', linewidths=2)
        ax.scatter(*path[-1], color=color, s=100, marker='*', edgecolors='white', linewidths=2)
    
    # 标记最优点
    ax.scatter(1, 1, color='white', s=200, marker='*', edgecolors='black', linewidths=2, label='Optimal (1, 1)', zorder=10)
    
    ax.set_xlabel('w₁', fontsize=12)
    ax.set_ylabel('w₂', fontsize=12)
    ax.set_title('Gradient Descent Paths on Loss Contour', fontsize=14)
    ax.set_aspect('equal')
    
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Loss', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('gradient_descent_contour.png', dpi=150, bbox_inches='tight')
    print("Saved: gradient_descent_contour.png")
    plt.close()


def saddle_point_visualization():
    """生成鞍点可视化图"""
    
    def saddle_func(x, y):
        return x**2 - y**2
    
    x = np.linspace(-2, 2, 200)
    y = np.linspace(-2, 2, 200)
    X, Y = np.meshgrid(x, y)
    Z = saddle_func(X, Y)
    
    fig = plt.figure(figsize=(12, 5))
    
    # 3D图
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.8, edgecolor='none')
    ax1.scatter(0, 0, 0, color='black', s=100, marker='o', label='Saddle Point (0,0)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f(x,y)')
    ax1.set_title('Saddle Point: f(x,y) = x² - y²')
    ax1.legend()
    
    # 等高线图
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(X, Y, Z, levels=30, cmap='coolwarm', alpha=0.8)
    ax2.contour(X, Y, Z, levels=20, colors='white', alpha=0.5, linewidths=0.5)
    ax2.scatter(0, 0, color='black', s=100, marker='o', label='Saddle Point (0,0)')
    
    # 标记方向
    ax2.annotate('', xy=(1, 0), xytext=(0.3, 0),
                arrowprops=dict(arrowstyle='->', color='white', lw=2))
    ax2.annotate('', xy=(-1, 0), xytext=(-0.3, 0),
                arrowprops=dict(arrowstyle='->', color='white', lw=2))
    ax2.annotate('', xy=(0, 0.5), xytext=(0, 0.1),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax2.annotate('', xy=(0, -0.5), xytext=(0, -0.1),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Contour Plot with Directions')
    ax2.legend()
    ax2.set_aspect('equal')
    
    plt.colorbar(contour, ax=ax2, label='f(x,y)')
    
    plt.tight_layout()
    plt.savefig('saddle_point.png', dpi=150, bbox_inches='tight')
    print("Saved: saddle_point.png")
    plt.close()


if __name__ == '__main__':
    print("Generating gradient descent visualizations...")
    gradient_descent_path()
    gradient_descent_contour()
    saddle_point_visualization()
    print("Done!")
