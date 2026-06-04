"""
9-3-rl-training-curves.py
Visualize typical RL training curves: episode reward, episode length, loss.
Demonstrates the "Sparrow RL" pattern: noisy but trending up.

Output: assets/figures/9-3-rl-training-curves.png
"""
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("assets/figures", exist_ok=True)

np.random.seed(0)
n_episodes = 500

# Simulate PPO training on CartPole
# Episode reward: noisy upward trend, +100 is max
def smooth_curve(target, noise, decay, n):
    """Trending toward target with noise that decays over time."""
    base = target * (1 - np.exp(-decay * np.arange(n)))
    return base + np.random.randn(n) * noise * np.exp(-decay * np.arange(n) * 0.5)

episodes = np.arange(n_episodes)
rewards = smooth_curve(target=475, noise=200, decay=0.012, n=n_episodes)
rewards = np.clip(rewards, 0, 500)
lengths = smooth_curve(target=475, noise=80, decay=0.012, n=n_episodes) / 5.0

# Losses (multiple components)
actor_loss = -smooth_curve(target=0.01, noise=0.4, decay=0.008, n=n_episodes)
critic_loss = smooth_curve(target=0.05, noise=0.3, decay=0.01, n=n_episodes)
entropy = smooth_curve(target=0.005, noise=0.05, decay=0.015, n=n_episodes)
total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

# Smoothed (rolling average)
def rolling_mean(x, w=20):
    return np.convolve(x, np.ones(w) / w, mode="valid")


fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Left: Episode reward
ax = axes[0]
ax.plot(episodes, rewards, "b-", alpha=0.3, linewidth=0.8, label="原始 (噪声)")
ax.plot(episodes[19:], rolling_mean(rewards), "r-", linewidth=2, label="20-ep 滑动平均")
ax.axhline(y=475, color="green", linestyle="--", alpha=0.6, label="理论最大值")
ax.set_xlabel("Episode")
ax.set_ylabel("Episode Reward")
ax.set_title("(a) Episode Reward\n收敛到最大值 475", fontsize=12, fontweight="bold")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)

# Middle: Episode length (CartPole specific)
ax = axes[1]
ax.plot(episodes, lengths, "g-", alpha=0.3, linewidth=0.8)
ax.plot(episodes[19:], rolling_mean(lengths), "b-", linewidth=2)
ax.set_xlabel("Episode")
ax.set_ylabel("Episode Length")
ax.set_title("(b) Episode Length\n(杆子保持直立的时间)", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3)

# Right: Losses
ax = axes[2]
ax.plot(episodes[19:], rolling_mean(actor_loss), label="Actor Loss", linewidth=1.5)
ax.plot(episodes[19:], rolling_mean(critic_loss), label="Critic Loss", linewidth=1.5)
ax.plot(episodes[19:], rolling_mean(total_loss), label="Total Loss", linewidth=1.5, linestyle="--")
ax.set_xlabel("Episode")
ax.set_ylabel("Loss")
ax.set_title("(c) PPO 损失曲线\nActor-Critic 联合优化", fontsize=12, fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle("RL 训练典型曲线 (PPO on CartPole-v1, 模拟)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("assets/figures/9-3-rl-training-curves.png", dpi=120, bbox_inches="tight")
print("Saved: assets/figures/9-3-rl-training-curves.png")
