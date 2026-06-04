"""
Word Embedding Illustration
Shows how words are mapped to dense vectors and semantic relationships in vector space
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

plt.rcParams['font.family'] = 'DejaVu Sans'

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# ==================== Left: Word Embedding Mapping Process ====================
ax1 = axes[0]
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.set_title('Word Embedding Mapping Process', fontsize=14, fontweight='bold', pad=15)

words = ['today', 'weather', 'good', 'learn', 'machine']
y_positions = [7, 5.5, 4, 2.5, 1]
box_width = 1.5

for word, y in zip(words, y_positions):
    word_box = FancyBboxPatch((1, y - 0.4), box_width, 0.8,
                               boxstyle="round,pad=0.05",
                               facecolor='#FFF9C4', edgecolor='#F9A825', linewidth=2)
    ax1.add_patch(word_box)
    ax1.text(1.75, y, word, fontsize=11, ha='center', va='center', fontweight='bold')

# One-hot vectors
ax1.text(4, 8.5, 'One-Hot Vector', fontsize=10, ha='center', va='center', color='#757575')
for i, (word, y) in enumerate(zip(words, y_positions)):
    vector_str = ''.join(['1' if j == i else '0' for j in range(5)])
    ax1.text(4.2, y, f'[{vector_str}]', fontsize=8, ha='left', va='center', 
             family='monospace', color='#1565C0')

# Mapping arrow and layer
ax1.annotate('', xy=(6.2, 4.5), xytext=(2.8, 4.5),
             arrowprops=dict(arrowstyle='->', color='#E91E63', lw=2.5))
ax1.text(4.5, 5, 'Embedding\nMapping', fontsize=9, ha='center', va='center', color='#E91E63', fontweight='bold')

# Embedding vectors (dense vectors)
embedding_dim = 4
ax1.text(7.5, 8.5, f'Embedding Vector (dim={embedding_dim})', fontsize=10, ha='center', va='center', color='#757575')
for i, (word, y) in enumerate(zip(words, y_positions)):
    np.random.seed(i * 10 + 5)
    vec = np.random.randn(embedding_dim) * 0.3 + 0.5
    vec = np.clip(vec, -1, 1)
    vec_str = '[' + ', '.join([f'{v:.1f}' for v in vec]) + ']'
    ax1.text(7.5, y, vec_str, fontsize=7.5, ha='left', va='center', 
             family='monospace', color='#2E7D32')

# ==================== Right: Semantic Relationships in Vector Space ====================
ax2 = axes[1]
ax2.set_xlim(-2, 2)
ax2.set_ylim(-2, 2)
ax2.set_title('Semantic Relationships in Vector Space', fontsize=14, fontweight='bold', pad=15)
ax2.set_xlabel('Semantic Dimension (e.g., Gender)', fontsize=10)
ax2.set_ylabel('Semantic Dimension (e.g., Royal/Humble)', fontsize=10)
ax2.axhline(y=0, color='#BDBDBD', linestyle='--', linewidth=1)
ax2.axvline(x=0, color='#BDBDBD', linestyle='--', linewidth=1)
ax2.grid(True, alpha=0.3)

# Word vectors (simplified 2D representation)
word_vectors = {
    'King': (-1.2, 1.0),
    'Man': (-1.0, -0.3),
    'Woman': (1.0, -0.3),
    'Princess': (1.2, 1.0),
}

colors = ['#1565C0', '#1976D2', '#E91E63', '#C2185B']

for (word, (x, y)), color in zip(word_vectors.items(), colors):
    ax2.scatter(x, y, s=200, c=color, zorder=5, edgecolors='white', linewidth=2)
    ax2.annotate(word, (x, y), xytext=(10, 10), textcoords='offset points',
                fontsize=12, fontweight='bold', color=color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor=color))

# Semantic relationship arrows
ax2.annotate('', xy=(1.2, 1.0), xytext=(-1.2, 1.0),
             arrowprops=dict(arrowstyle='<->', color='#7B1FA2', lw=2))
ax2.text(0, 1.25, 'Gender+', fontsize=9, ha='center', color='#7B1FA2')

ax2.annotate('', xy=(1.0, -0.3), xytext=(-1.0, -0.3),
             arrowprops=dict(arrowstyle='<->', color='#388E3C', lw=2))
ax2.text(0, -0.5, 'Gender-', fontsize=9, ha='center', color='#388E3C')

# Bottom formula
ax2.text(0, -1.7, r'$\vec{King} - \vec{Man} + \vec{Woman} \approx \vec Princess$', 
         fontsize=11, ha='center', va='center', 
         bbox=dict(facecolor='#F5F5F5', edgecolor='#BDBDBD', pad=8))

ax2.legend(['Word Vectors'], loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig(Path(__file__).resolve().parent / 'word_embedding.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Saved: word_embedding.png")
