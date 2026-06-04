"""
3-5-rosenbrock-lr-schedules
===========================
Rosenbrock 优化器轨迹 + 4 种学习率策略（与 3-5-optimizer-convergence.py / 3-5-lr-schedules.py 互补）
"""
import numpy as np
import matplotlib.pyplot as plt

def rosenbrock(x, y):
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

def rosenbrock_grad(x, y):
    gx = -2 * (1 - x) - 400 * x * (y - x ** 2)
    gy = 200 * (y - x ** 2)
    return np.array([gx, gy])

def gradient_descent(grad_func, start_point, lr=0.001, n_iter=100):
    point = start_point.copy()
    trajectory = [point.copy()]
    for _ in range(n_iter):
        grad = grad_func(point[0], point[1])
        point -= lr * grad
        trajectory.append(point.copy())
    return np.array(trajectory)

def momentum_gd(grad_func, start_point, lr=0.001, beta=0.9, n_iter=100):
    point = start_point.copy()
    velocity = np.zeros_like(point)
    trajectory = [point.copy()]
    for _ in range(n_iter):
        grad = grad_func(point[0], point[1])
        velocity = beta * velocity - lr * grad
        point += velocity
        trajectory.append(point.copy())
    return np.array(trajectory)

def adam_optimizer(grad_func, start_point, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, n_iter=100):
    point = start_point.copy()
    m = np.zeros_like(point)
    v = np.zeros_like(point)
    trajectory = [point.copy()]

    for t in range(1, n_iter + 1):
        grad = grad_func(point[0], point[1])
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        point -= lr * m_hat / (np.sqrt(v_hat) + eps)
        trajectory.append(point.copy())

    return np.array(trajectory)

start_point = np.array([-1.5, 1.5])
gd_traj = gradient_descent(rosenbrock_grad, start_point, lr=0.0005, n_iter=200)
momentum_traj = momentum_gd(rosenbrock_grad, start_point, lr=0.0005, beta=0.9, n_iter=200)
adam_traj = adam_optimizer(rosenbrock_grad, start_point, lr=0.01, n_iter=200)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
x = np.linspace(-2, 2, 100)
y = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x, y)
Z = rosenbrock(X, Y)
levels = np.logspace(-1, 3, 20)

for ax, traj, title in zip(
    axes,
    [gd_traj, momentum_traj, adam_traj],
    ['标准梯度下降 (GD)', '带动量梯度下降 (Momentum)', 'Adam优化器']
):
    ax.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.5)
    ax.plot(traj[:, 0], traj[:, 1], 'ro-', markersize=3, linewidth=1.5)
    ax.plot(traj[0, 0], traj[0, 1], 'go', markersize=8, label='起点')
    ax.plot(traj[-1, 0], traj[-1, 1], 'bo', markersize=8, label='终点')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.tight_layout()
plt.savefig('/home/david/AI-Tutorial/assets/figures/3-5-rosenbrock-lr-schedules.png', dpi=120, bbox_inches='tight', facecolor='white')
plt.show()
epochs = 200

def step_lr(initial_lr, step_size=50, gamma=0.5):
    return [initial_lr * (gamma ** (epoch // step_size)) for epoch in range(epochs)]

def cosine_lr(initial_lr, t_max=epochs):
    return [0.5 * initial_lr * (1 + np.cos(np.pi * epoch / t_max)) for epoch in range(epochs)]

def exp_lr(initial_lr, gamma=0.99):
    return [initial_lr * (gamma ** epoch) for epoch in range(epochs)]

def onecycle_lr(initial_lr, max_lr=0.1, pct_start=0.3):
    lr = []
    steps_up = int(epochs * pct_start)
    steps_down = epochs - steps_up

    for step in range(steps_up):
        lr.append(initial_lr + (max_lr - initial_lr) * (step / steps_up))

    for step in range(steps_down):
        lr.append(max_lr - (max_lr - initial_lr / 100) * (step / steps_down))

    return lr

plt.figure(figsize=(10, 5))
plt.plot(step_lr(0.1), label='Step衰减', linewidth=2)
plt.plot(cosine_lr(0.1), label='Cosine衰减', linewidth=2)
plt.plot(exp_lr(0.1), label='Exponential衰减', linewidth=2)
plt.plot(onecycle_lr(0.01, max_lr=0.1), label='OneCycleLR', linewidth=2)
plt.xlabel('训练轮次 (Epoch)')
plt.ylabel('学习率')
plt.title('不同学习率策略对比')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()