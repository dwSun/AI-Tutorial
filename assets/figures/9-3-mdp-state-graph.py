"""
9-3-mdp-state-graph.py
Visualize a Markov Decision Process (MDP) state transition graph with
state nodes, action labels, and reward values on edges.

Output: assets/figures/9-3-mdp-state-graph.png
"""
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("assets/figures", exist_ok=True)


def draw_mdp(ax):
    """Draw a small MDP with 3 states and 2 actions per state."""
    # State positions
    states = {
        "s1": (1, 2),
        "s2": (3, 3),
        "s3": (5, 2),
        "s_T": (5, 0),  # terminal
    }

    # Draw states
    for name, (x, y) in states.items():
        if name == "s_T":
            ax.add_patch(plt.Circle((x, y), 0.4, color="lightgreen", ec="darkgreen", lw=2, zorder=3))
            ax.text(x, y, "Terminal", ha="center", va="center", fontsize=10, fontweight="bold", zorder=4)
        else:
            ax.add_patch(plt.Circle((x, y), 0.4, color="lightblue", ec="navy", lw=2, zorder=3))
            ax.text(x, y, name, ha="center", va="center", fontsize=11, fontweight="bold", zorder=4)

    # Draw transitions
    # From s1
    ax.annotate("", xy=(3, 3), xytext=(1, 2),
                arrowprops=dict(arrowstyle="->", lw=2, color="green", alpha=0.7),
                zorder=2)
    ax.text(2, 2.6, "a₁: r=+1", fontsize=9, color="green", fontweight="bold")
    ax.annotate("", xy=(1.4, 1.6), xytext=(1, 2),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="red", alpha=0.5, connectionstyle="arc3,rad=0.3"),
                zorder=1)
    ax.text(0.5, 1.3, "a₂: r=0", fontsize=8, color="red")

    # From s2
    ax.annotate("", xy=(5, 2), xytext=(3, 3),
                arrowprops=dict(arrowstyle="->", lw=2, color="green", alpha=0.7),
                zorder=2)
    ax.text(4, 2.7, "a₁: r=+2", fontsize=9, color="green", fontweight="bold")
    ax.annotate("", xy=(5, 0), xytext=(3, 3),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="orange", alpha=0.5, connectionstyle="arc3,rad=0.3"),
                zorder=1)
    ax.text(3.5, 1.5, "a₃: r=+10\n(终止)", fontsize=8, color="orange", fontweight="bold")

    # From s3
    ax.annotate("", xy=(5, 0), xytext=(5, 2),
                arrowprops=dict(arrowstyle="->", lw=2, color="blue", alpha=0.7),
                zorder=2)
    ax.text(5.5, 1.0, "a₁: r=+5\n(终止)", fontsize=8, color="blue", fontweight="bold")

    # Style
    ax.set_xlim(0, 6.5)
    ax.set_ylim(-1, 4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("MDP 状态转移图: S × A → S' × R", fontsize=13, fontweight="bold")


def draw_bellman(ax):
    """Show the Bellman equation and value function flow."""
    ax.text(0.5, 0.85, r"$V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V^\pi(s')]$",
            ha="center", va="center", fontsize=14, fontweight="bold",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", ec="orange"))

    ax.annotate("", xy=(0.5, 0.65), xytext=(0.5, 0.75),
                arrowprops=dict(arrowstyle="->", lw=2, color="red"),
                transform=ax.transAxes)
    ax.text(0.5, 0.55, "Bellman 期望方程:\n状态价值 = 即时奖励 + 折扣未来价值",
            ha="center", va="center", fontsize=11, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))

    # Show 3 elements
    elements = [
        (0.15, 0.25, "Policy\n$\\pi(a|s)$", "lightyellow"),
        (0.5, 0.25, "Reward\n$R(s,a,s')$", "lightgreen"),
        (0.85, 0.25, "Transition\n$P(s'|s,a)$", "lightcoral"),
    ]
    for x, y, text, color in elements:
        ax.text(x, y, text, ha="center", va="center", fontsize=10,
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, ec="gray"))
    ax.text(0.5, 0.05, "三个核心要素: 策略 + 奖励 + 转移概率",
            ha="center", fontsize=11, transform=ax.transAxes,
            fontweight="bold", color="darkred")


fig = plt.figure(figsize=(18, 6))
ax1 = fig.add_subplot(121)
draw_mdp(ax1)
ax2 = fig.add_subplot(122)
draw_bellman(ax2)
ax2.axis("off")

plt.suptitle("Markov Decision Process (MDP) 与 Bellman 方程",
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("assets/figures/9-3-mdp-state-graph.png", dpi=120, bbox_inches="tight")
print("Saved: assets/figures/9-3-mdp-state-graph.png")
