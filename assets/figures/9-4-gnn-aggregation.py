"""
9-4-gnn-aggregation.py
Visualize the 3 main GNN aggregation schemes:
  - GCN: mean of normalized neighbors
  - GraphSAGE: concat(self, mean(neighbors))
  - GAT: attention-weighted aggregation

Output: assets/figures/9-4-gnn-aggregation.png
"""
import os
os.makedirs("assets/figures", exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
np.random.seed(0)


def draw_aggregation(ax, title, weights, formula, description):
    """Draw a small star graph with center node and 4 neighbors."""
    G = nx.Graph()
    G.add_node("v", node_color="red", node_size=1200)
    neighbors = ["n₁", "n₂", "n₃", "n₄"]
    colors = ["#87CEEB", "#90EE90", "#FFD700", "#DDA0DD"]
    for i, n in enumerate(neighbors):
        G.add_node(n, node_color=colors[i], node_size=600)
        G.add_edge("v", n)

    pos = {"v": (0, 0), "n₁": (1, 0.8), "n₂": (-1, 0.8), "n₃": (-1, -0.8), "n₄": (1, -0.8)}

    # Draw edges with varying width to indicate attention/weight
    for i, n in enumerate(neighbors):
        ax.plot([0, pos[n][0]], [0, pos[n][1]], "-",
                color="gray", linewidth=1 + weights[i] * 5, alpha=0.7)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax,
                          node_color=[G.nodes[n].get("node_color", "lightblue") for n in G.nodes()],
                          node_size=[G.nodes[n].get("node_size", 500) for n in G.nodes()],
                          edgecolors="black", linewidths=1.5)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=14, font_weight="bold")

    # Show weights next to edges
    for i, n in enumerate(neighbors):
        mid_x, mid_y = (pos[n][0] / 2), (pos[n][1] / 2)
        ax.text(mid_x * 1.3, mid_y * 1.3, f"α={weights[i]:.2f}",
                fontsize=10, color="navy", fontweight="bold")

    # Show formula
    ax.text(0.5, -0.12, formula, ha="center", fontsize=10, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", ec="gray"))
    ax.text(0.5, -0.22, description, ha="center", fontsize=9, transform=ax.transAxes,
            style="italic", color="gray")

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")


# GCN: equal weights
draw_aggregation(axes[0],
    title="(a) GCN\n(均匀加权平均)",
    weights=[0.25, 0.25, 0.25, 0.25],
    formula=r"$h_v = \sigma\left(\sum_{u \in N(v)} \frac{1}{\sqrt{\hat{d}_v \hat{d}_u}} W x_u\right)$",
    description="所有邻居权重相等, 用度归一化")

# GraphSAGE: weighted by importance
draw_aggregation(axes[1],
    title="(b) GraphSAGE\n(拼接 self + 加权邻居)",
    weights=[0.40, 0.20, 0.30, 0.10],
    formula=r"$h_v = \sigma\left(W_1 x_v + W_2 \cdot \text{mean}_{u \in N(v)} x_u\right)$",
    description="邻居聚合后与自身拼接, 支持归纳学习")

# GAT: attention-based
draw_aggregation(axes[2],
    title="(c) GAT\n(注意力加权)",
    weights=[0.50, 0.15, 0.25, 0.10],
    formula=r"$h_v = \sigma\left(\sum_{u \in N(v)} \alpha_{vu} W x_u\right)$",
    description="注意力系数 $\\alpha_{vu}$ 自动学习邻居重要性")

plt.suptitle("GNN 三大聚合范式: 中心节点 v 如何聚合邻居信息",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("assets/figures/9-4-gnn-aggregation.png", dpi=120, bbox_inches="tight")
print("Saved: assets/figures/9-4-gnn-aggregation.png")
