"""
Visualize the LinearHypernetwork architecture using NetworkX.

Architecture (hidden_dim=50, num_classes=10):
  Embedding(10→50) -> Linear(50→100) -> Linear(100→200) -> Linear(200→100) -> Linear(100→W)
  Each linear block: Linear + LayerNorm + LeakyReLU + Dropout
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Config ────────────────────────────────────────────────────────────────────
NUM_CLASSES    = 10
HIDDEN_DIM     = 50
WEIGHT_VEC_LEN = 2698   # LeNet-5 total params; adjust if different

# Scale node count proportionally to actual dim, relative to the largest
# hidden layer (200). Output layer is capped at the same max.
REF_DIM  = HIDDEN_DIM * 4   # 200 — largest hidden dim
MAX_SHOW = 20               # nodes drawn for the reference layer

def n_show(actual_dim, force=None):
    if force is not None:
        return force
    n = round(actual_dim / REF_DIM * MAX_SHOW)
    return max(3, min(MAX_SHOW, n))

# Layer definitions: (label, actual_dim, color, forced_n_show or None)
LAYERS = [
    ("Embedding input\n(one node per class)",
     NUM_CLASSES,      "#4C72B0", NUM_CLASSES),   # show all 10 class tokens

    ("Embedding output\n(hidden_dim = 50)",
     HIDDEN_DIM,       "#4C72B0", None),

    ("Linear 1 output\n(100)\n+ LayerNorm\n+ LeakyReLU + Dropout",
     HIDDEN_DIM * 2,   "#DD8452", None),

    ("Linear 2 output\n(200)\n+ LayerNorm\n+ LeakyReLU + Dropout",
     HIDDEN_DIM * 4,   "#55A868", None),

    ("Linear 3 output\n(100)\n+ LayerNorm\n+ LeakyReLU + Dropout",
     HIDDEN_DIM * 2,   "#C44E52", None),

    (f"Output layer\n({WEIGHT_VEC_LEN:,} weights)\n+ Tanh",
     WEIGHT_VEC_LEN,   "#8172B3", MAX_SHOW),       # cap output, add ellipsis
]

# Pre-compute how many nodes to draw per layer
layer_n = [n_show(dim, force) for _, dim, _, force in LAYERS]

# ── Build graph ───────────────────────────────────────────────────────────────
G = nx.DiGraph()
pos = {}
node_colors = []
layer_node_lists = []

max_n = max(layer_n)

for col, ((label, actual_dim, color, _), n) in enumerate(zip(LAYERS, layer_n)):
    node_ids = [f"L{col}_n{r}" for r in range(n)]
    layer_node_lists.append(node_ids)

    # Center each column vertically within the global max height
    y_positions = [r - (n - 1) / 2.0 for r in range(n)]
    for node, y in zip(node_ids, y_positions):
        G.add_node(node)
        pos[node] = (col * 3.5, y)
        node_colors.append(color)

# Fully-connected edges between adjacent layers
for col in range(len(LAYERS) - 1):
    for src in layer_node_lists[col]:
        for dst in layer_node_lists[col + 1]:
            G.add_edge(src, dst)

# ── Draw ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(26, 12))

nx.draw_networkx_nodes(G, pos, ax=ax,
                       node_color=node_colors, node_size=200, alpha=0.92)
nx.draw_networkx_edges(G, pos, ax=ax,
                       edge_color="black", alpha=0.10,
                       arrows=False, width=0.4)

# ── Per-layer annotations ─────────────────────────────────────────────────────
y_label_top    =  max_n / 2.0 + 0.8
y_label_bottom = -max_n / 2.0 - 0.6

for col, ((label, actual_dim, color, _), n) in enumerate(zip(LAYERS, layer_n)):
    x = col * 3.5

    # Header box above the column
    ax.text(x, y_label_top, label,
            ha='center', va='bottom', fontsize=7.5, color='white', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.35', facecolor=color, alpha=0.85, edgecolor='none'))

    # Actual dimension below the column
    ax.text(x, y_label_bottom, f"dim = {actual_dim:,}\n(showing {n})",
            ha='center', va='top', fontsize=7.5, color='#333333')

    # Ellipsis in the middle if we're truncating
    if actual_dim > n:
        ax.text(x, 0, "·  ·  ·",
                ha='center', va='center', fontsize=16, color=color, alpha=0.85)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(color="#4C72B0", label=f"Embedding  (10 classes → {HIDDEN_DIM})"),
    mpatches.Patch(color="#DD8452", label="Block 1  (50 → 100)"),
    mpatches.Patch(color="#55A868", label="Block 2  (100 → 200)  ← widest"),
    mpatches.Patch(color="#C44E52", label="Block 3  (200 → 100)"),
    mpatches.Patch(color="#8172B3", label=f"Output  (100 → {WEIGHT_VEC_LEN:,})  + Tanh"),
]
ax.legend(handles=legend_items, loc='lower center', ncol=5, fontsize=8,
          framealpha=0.4, bbox_to_anchor=(0.5, -0.12))

ax.axis('off')
plt.tight_layout()
plt.savefig("linear_hnet_architecture.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved → linear_hnet_architecture.png")
