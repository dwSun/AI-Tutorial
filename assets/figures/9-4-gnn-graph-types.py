"""
9-4-gnn-graph-types.py
Visualize different graph types used in GNN research:
  - Citation graph (heterogeneous)
  - Social network
  - Molecule (chemistry)
  - Knowledge graph

Uses NetworkX for layout and Matplotlib for drawing.
Output: assets/figures/9-4-gnn-graph-types.png
"""
import os
os.makedirs("assets/figures", exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def draw_graph(ax, G, pos, node_colors, node_sizes, edge_colors, title,
               with_labels=True, label_fontsize=8):
    """Draw a NetworkX graph on the given axis."""
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                          node_size=node_sizes, alpha=0.85,
                          edgecolors="black", linewidths=0.8)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors,
                          width=1.2, alpha=0.6, arrows=True,
                          arrowsize=10, connectionstyle="arc3,rad=0.1")
    if with_labels:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=label_fontsize,
                              font_weight="bold")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")


fig, axes = plt.subplots(1, 4, figsize=(20, 5))
np.random.seed(0)

# 1. Citation network (small)
G1 = nx.DiGraph()
papers = ["P₁", "P₂", "P₃", "P₄", "P₅", "P₆"]
for p in papers:
    G1.add_node(p)
edges_1 = [("P₁", "P₂"), ("P₁", "P₃"), ("P₂", "P₄"),
           ("P₃", "P₄"), ("P₄", "P₅"), ("P₅", "P₆"),
           ("P₃", "P₆"), ("P₆", "P₁")]
G1.add_edges_from(edges_1)
pos1 = nx.spring_layout(G1, seed=0)
draw_graph(axes[0], G1, pos1,
           node_colors=["#FFB6C1", "#87CEEB", "#90EE90", "#FFD700", "#DDA0DD", "#F0E68C"],
           node_sizes=600, edge_colors="gray",
           title="(a) 论文引用网络\n(异构图, 边有方向)")

# 2. Social network
G2 = nx.Graph()
people = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Hank"]
G2.add_nodes_from(people)
edges_2 = [("Alice", "Bob"), ("Alice", "Carol"), ("Bob", "Dave"),
           ("Bob", "Eve"), ("Carol", "Frank"), ("Dave", "Eve"),
           ("Eve", "Frank"), ("Frank", "Grace"), ("Grace", "Hank"),
           ("Eve", "Hank"), ("Alice", "Dave")]
G2.add_edges_from(edges_2)
pos2 = nx.spring_layout(G2, seed=1)
# Color by community (simple heuristic)
community_colors = ["#FFB6C1"] * 3 + ["#87CEEB"] * 3 + ["#90EE90"] * 2
draw_graph(axes[1], G2, pos2,
           node_colors=community_colors,
           node_sizes=400, edge_colors="darkgray",
           title="(b) 社交网络\n(节点 = 用户, 边 = 好友)")

# 3. Molecule (simple)
G3 = nx.Graph()
# Benzene ring with substituents
atoms = [("C1", {"element": "C"}), ("C2", {"element": "C"}), ("C3", {"element": "C"}),
         ("C4", {"element": "C"}), ("C5", {"element": "C"}), ("C6", {"element": "C"}),
         ("O", {"element": "O"}), ("H", {"element": "H"})]
for node, attrs in atoms:
    G3.add_node(node, **attrs)
bonds = [("C1", "C2"), ("C2", "C3"), ("C3", "C4"), ("C4", "C5"),
         ("C5", "C6"), ("C6", "C1"), ("C1", "O"), ("C2", "H")]
G3.add_edges_from(bonds)
pos3 = nx.spring_layout(G3, seed=2)

# Color atoms: C=gray, O=red, H=white
atom_colors = []
for n, d in G3.nodes(data=True):
    if d["element"] == "C":
        atom_colors.append("#404040")
    elif d["element"] == "O":
        atom_colors.append("#FF4136")
    else:
        atom_colors.append("#FFFFFF")
draw_graph(axes[2], G3, pos3,
           node_colors=atom_colors,
           node_sizes=600, edge_colors="#333333",
           title="(c) 分子图\n(节点 = 原子, 边 = 化学键)",
           with_labels=True, label_fontsize=9)

# 4. Knowledge graph (heterogeneous)
G4 = nx.Graph()
entities = [
    ("Einstein", {"type": "Person"}), ("Physics", {"type": "Field"}),
    ("Nobel", {"type": "Award"}), ("Relativity", {"type": "Theory"}),
    ("1921", {"type": "Year"}), ("Germany", {"type": "Place"})
]
for node, attrs in entities:
    G4.add_node(node, **attrs)
relations = [
    ("Einstein", "Physics"), ("Einstein", "Nobel"), ("Einstein", "Relativity"),
    ("Einstein", "Germany"), ("Nobel", "1921"), ("Relativity", "Physics")
]
G4.add_edges_from(relations)
pos4 = nx.spring_layout(G4, seed=3)
entity_colors = []
for n, d in G4.nodes(data=True):
    t = d["type"]
    if t == "Person":
        entity_colors.append("#FFB6C1")
    elif t == "Field":
        entity_colors.append("#90EE90")
    elif t == "Award":
        entity_colors.append("#FFD700")
    elif t == "Theory":
        entity_colors.append("#DDA0DD")
    else:
        entity_colors.append("#87CEEB")
draw_graph(axes[3], G4, pos4,
           node_colors=entity_colors,
           node_sizes=700, edge_colors="gray",
           title="(d) 知识图谱\n(异构实体 + 关系)",
           with_labels=True, label_fontsize=8)

plt.suptitle("GNN 研究中常见的 4 类图结构 (NetworkX 真实渲染)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("assets/figures/9-4-gnn-graph-types.png", dpi=120, bbox_inches="tight")
print("Saved: assets/figures/9-4-gnn-graph-types.png")
