import sys
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# 0. Setup
# =========================
# §3.3/§6.3/§9.4 repeatedly describe the sampled graph as "sparse, star-like"
# (a consequence of Top-100-counterparty sampling: each seed's counterparties
# are captured, but the counterparties are almost never connected to each
# other). A raw edge-list CSV cannot show this visually -- this script draws
# the actual ego-network (seed + its direct neighbours + edges among them)
# for a few example seeds, so the star shape can be seen directly rather than
# taken on faith from a clustering-coefficient number.
#
# Optional CLI args: edges_file out_subdir file_suffix -- reruns on the Plan-B
# reconstructed graph (see progress_summary.md §13) for a before/after visual
# comparison, e.g.
# `python visualize_graph_topology.py raw_edges_reconstructed.csv eda_outputs _reconstructed`
EDGES_FILE = sys.argv[1] if len(sys.argv) > 1 else "raw_edges.csv"
OUT_SUBDIR = sys.argv[2] if len(sys.argv) > 2 else "eda_outputs"
FILE_SUFFIX = sys.argv[3] if len(sys.argv) > 3 else ""

OUT_DIR = Path(OUT_SUBDIR)
OUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "font.size": 9,
})


def clean_address(x):
    if pd.isna(x):
        return None
    return str(x).strip().lower()


edges = pd.read_csv(EDGES_FILE)
edges["source"] = edges["source"].apply(clean_address)
edges["target"] = edges["target"].apply(clean_address)
edges = edges.dropna(subset=["source", "target"])

G = nx.from_pandas_edgelist(edges, source="source", target="target", create_using=nx.DiGraph())
G_undirected = G.to_undirected()

# =========================
# 1. Pick example seeds: a couple of high-degree "hub" seeds and a couple of
#    typical/median-degree seeds, across different classes, so both the "big
#    star" and "small star" pattern are visible.
# =========================

seeds = pd.read_csv("final_seed_address500.csv")
seeds["address"] = seeds["address"].apply(clean_address)
label_of = seeds.set_index("address")["label"].to_dict()

EXAMPLE_SEEDS = [
    "0x111111125421ca6dc452d289314280a0f8842a65",  # DeFi, degree 119 (high-degree hub)
    "0x28c6c06298d514db089934071355e5743bf21d60",  # CEX, degree 118 (high-degree hub)
    "0x331d9a049d496385998067abf6cbb6371c8d2466",  # Malicious_MEV, degree 105 (high-degree hub)
    "0x5ae028a6478ca2d3b881c51440674ff9af21341f",  # Malicious_MEV, degree 5 (typical/median)
]

fig, axes = plt.subplots(2, 2, figsize=(13, 13))

for ax, seed in zip(axes.flat, EXAMPLE_SEEDS):
    ego = nx.ego_graph(G_undirected, seed, radius=1)
    n_neighbours = ego.number_of_nodes() - 1
    # edges that exist AMONG the neighbours themselves (excluding the seed) --
    # this is the number that clustering_coeff is measuring: a "true star"
    # has zero of these.
    neighbour_subgraph = ego.subgraph([n for n in ego.nodes() if n != seed])
    n_neighbour_edges = neighbour_subgraph.number_of_edges()

    pos = nx.spring_layout(ego, seed=42, k=0.6)
    node_colors = ["#d64545" if n == seed else "#2a78d6" for n in ego.nodes()]
    node_sizes = [260 if n == seed else 60 for n in ego.nodes()]

    nx.draw_networkx_edges(ego, pos, ax=ax, edge_color="#c3c2b7", width=0.8)
    nx.draw_networkx_nodes(ego, pos, ax=ax, node_color=node_colors, node_size=node_sizes, linewidths=0)

    label = label_of.get(seed, "?")
    ax.set_title(
        f"{label} seed  ({seed[:10]}...)\n"
        f"{n_neighbours} direct neighbours (blue), {n_neighbour_edges} edges among them (red = seed)",
        fontsize=9,
    )
    ax.axis("off")

suptitle_label = (
    "before Plan-B reconstruction: the \"star\" pattern (few/no edges among neighbours)"
    if not FILE_SUFFIX else
    "after Plan-B reconstruction: same seeds, missing counterparty-counterparty edges filled in"
)
fig.suptitle(f"Ego-networks of example seeds ({suptitle_label})", fontsize=12)
fig.tight_layout()
fig.savefig(OUT_DIR / f"graph_topology_examples{FILE_SUFFIX}.png", dpi=150)
plt.close(fig)
print(f"Saved {OUT_DIR / f'graph_topology_examples{FILE_SUFFIX}.png'}")

# =========================
# 2. Whole-graph overview (sampled -- the full 8,029-node graph is an
#    unreadable hairball if plotted directly). Sample a connected chunk
#    around several seeds to show how these stars tile together.
# =========================

sample_nodes = set()
for seed in EXAMPLE_SEEDS:
    sample_nodes.update(nx.ego_graph(G_undirected, seed, radius=1).nodes())
sample_graph = G_undirected.subgraph(sample_nodes)

fig2, ax2 = plt.subplots(figsize=(11, 11))
pos2 = nx.spring_layout(sample_graph, seed=42, k=0.25, iterations=50)
node_colors2 = ["#d64545" if n in EXAMPLE_SEEDS else "#2a78d6" for n in sample_graph.nodes()]
node_sizes2 = [180 if n in EXAMPLE_SEEDS else 20 for n in sample_graph.nodes()]
nx.draw_networkx_edges(sample_graph, pos2, ax=ax2, edge_color="#c3c2b7", width=0.5, alpha=0.6)
nx.draw_networkx_nodes(sample_graph, pos2, ax=ax2, node_color=node_colors2, node_size=node_sizes2, linewidths=0)
multi_title_tail = "separate stars, rarely touching" if not FILE_SUFFIX else "after reconstruction: cross-links between neighbours filled in"
ax2.set_title(f"4 example seeds' ego-networks combined ({sample_graph.number_of_nodes()} nodes) -- {multi_title_tail}")
ax2.axis("off")
fig2.tight_layout()
fig2.savefig(OUT_DIR / f"graph_topology_multi_seed_sample{FILE_SUFFIX}.png", dpi=150)
plt.close(fig2)
print(f"Saved {OUT_DIR / f'graph_topology_multi_seed_sample{FILE_SUFFIX}.png'}")

print("\n=== Graph topology visualization complete ===")
