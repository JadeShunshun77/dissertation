import pandas as pd
import numpy as np
import networkx as nx

# =========================
# 0. Read data
# =========================

seeds = pd.read_csv("final_seed_address500.csv")
edges = pd.read_csv("raw_edges.csv")

SEED_ADDRESS_COL = "address"
LABEL_COL = "label"
SOURCE_COL = "source"
TARGET_COL = "target"
TX_COUNT_COL = "tx_count"
VALUE_COL = "total_eth_value"
GAS_COL = "avg_gas_fee"


def clean_address(x):
    if pd.isna(x):
        return np.nan
    return str(x).strip().lower()


seeds[SEED_ADDRESS_COL] = seeds[SEED_ADDRESS_COL].apply(clean_address)
edges[SOURCE_COL] = edges[SOURCE_COL].apply(clean_address)
edges[TARGET_COL] = edges[TARGET_COL].apply(clean_address)
edges = edges.dropna(subset=[SOURCE_COL, TARGET_COL])

print(f"Seeds: {len(seeds)}, Edges: {len(edges)}")

# =========================
# 1. Build directed graph
# =========================

G = nx.from_pandas_edgelist(
    edges,
    source=SOURCE_COL,
    target=TARGET_COL,
    edge_attr=[TX_COUNT_COL, VALUE_COL, GAS_COL],
    create_using=nx.DiGraph(),
)
print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# =========================
# 2. Transaction behaviour features (per seed)
# =========================

out_edges = edges.groupby(SOURCE_COL).agg(
    out_tx_count=(TX_COUNT_COL, "sum"),
    out_eth_value=(VALUE_COL, "sum"),
    out_gas_fee_mean=(GAS_COL, "mean"),
    out_zero_value_edges=(VALUE_COL, lambda s: (s == 0).sum()),
    out_edge_count=(VALUE_COL, "size"),
)
in_edges = edges.groupby(TARGET_COL).agg(
    in_tx_count=(TX_COUNT_COL, "sum"),
    in_eth_value=(VALUE_COL, "sum"),
    in_gas_fee_mean=(GAS_COL, "mean"),
    in_zero_value_edges=(VALUE_COL, lambda s: (s == 0).sum()),
    in_edge_count=(VALUE_COL, "size"),
)

feat = seeds.set_index(SEED_ADDRESS_COL).join(out_edges, how="left").join(in_edges, how="left")

count_fill_cols = [
    "out_tx_count", "out_eth_value", "out_zero_value_edges", "out_edge_count",
    "in_tx_count", "in_eth_value", "in_zero_value_edges", "in_edge_count",
]
feat[count_fill_cols] = feat[count_fill_cols].fillna(0)

feat["total_tx_count"] = feat["in_tx_count"] + feat["out_tx_count"]
feat["total_eth_value"] = feat["in_eth_value"] + feat["out_eth_value"]
feat["total_edge_count"] = feat["in_edge_count"] + feat["out_edge_count"]
feat["total_zero_value_edges"] = feat["in_zero_value_edges"] + feat["out_zero_value_edges"]
feat["zero_value_tx_ratio"] = np.where(
    feat["total_edge_count"] > 0,
    feat["total_zero_value_edges"] / feat["total_edge_count"],
    0.0,
)
feat["is_active"] = feat["total_edge_count"] > 0

# out_gas_fee_mean / in_gas_fee_mean are NaN exactly when a seed has no
# outgoing / incoming edges to average over (out_edge_count == 0 / in_edge_count == 0).
# Fill with 0 and keep explicit flags so a filled 0 is distinguishable from a
# true zero-fee edge.
feat["has_out_edge"] = feat["out_edge_count"] > 0
feat["has_in_edge"] = feat["in_edge_count"] > 0
feat["out_gas_fee_mean"] = feat["out_gas_fee_mean"].fillna(0.0)
feat["in_gas_fee_mean"] = feat["in_gas_fee_mean"].fillna(0.0)

# log1p transforms for heavy-tailed variables (see sanity_check: std >> mean, max in millions)
for col in [
    "in_tx_count", "out_tx_count", "total_tx_count",
    "in_eth_value", "out_eth_value", "total_eth_value",
]:
    feat[f"log_{col}"] = np.log1p(feat[col])

# =========================
# 3. Graph structural features (computed on full graph, then joined to seeds)
# =========================

print("Computing degree features...")
in_degree = dict(G.in_degree())
out_degree = dict(G.out_degree())
weighted_in_degree = dict(G.in_degree(weight=TX_COUNT_COL))
weighted_out_degree = dict(G.out_degree(weight=TX_COUNT_COL))

print("Computing PageRank...")
pagerank_unweighted = nx.pagerank(G)
pagerank_weighted = nx.pagerank(G, weight=TX_COUNT_COL)

print("Computing clustering coefficient (undirected view)...")
G_undirected = G.to_undirected()
clustering = nx.clustering(G_undirected)

print("Computing weakly connected component sizes...")
node_to_wcc_size = {}
for component in nx.weakly_connected_components(G):
    size = len(component)
    for node in component:
        node_to_wcc_size[node] = size

struct_df = pd.DataFrame({
    "in_degree": pd.Series(in_degree),
    "out_degree": pd.Series(out_degree),
    "weighted_in_degree": pd.Series(weighted_in_degree),
    "weighted_out_degree": pd.Series(weighted_out_degree),
    "pagerank": pd.Series(pagerank_unweighted),
    "pagerank_weighted": pd.Series(pagerank_weighted),
    "clustering_coeff": pd.Series(clustering),
    "wcc_size": pd.Series(node_to_wcc_size),
})
struct_df["degree"] = struct_df["in_degree"] + struct_df["out_degree"]
struct_df["weighted_degree"] = struct_df["weighted_in_degree"] + struct_df["weighted_out_degree"]

feat = feat.join(struct_df, how="left")

struct_cols = [
    "in_degree", "out_degree", "degree",
    "weighted_in_degree", "weighted_out_degree", "weighted_degree",
    "pagerank", "pagerank_weighted", "clustering_coeff", "wcc_size",
]
feat[struct_cols] = feat[struct_cols].fillna(0)

for col in ["degree", "weighted_degree", "pagerank", "pagerank_weighted"]:
    feat[f"log_{col}"] = np.log1p(feat[col])

# clustering_coeff is 0 for 80-99% of seeds in every class (sparse, star-like
# ego-network from top-100 counterparty sampling), so also keep a binary flag
# capturing rare closed-triangle structure separately from the mostly-zero magnitude.
feat["has_closed_triangle"] = (feat["clustering_coeff"] > 0).astype(int)

# =========================
# 4. Save
# =========================

feat = feat.reset_index().rename(columns={"index": SEED_ADDRESS_COL})

print("\n=== Feature matrix summary ===")
print(f"Rows: {len(feat)}, Columns: {len(feat.columns)}")
print(feat.groupby(LABEL_COL)["is_active"].agg(["sum", "count", "mean"]))

feat.to_csv("seed_feature_matrix.csv", index=False)
print("\nSaved feature matrix to seed_feature_matrix.csv")

# =========================
# 5. Feature group assignment (for the Week 9 ablation study)
# =========================
# weighted_in_degree/weighted_out_degree/weighted_degree are edge weight (tx_count)
# sums, and since raw_edges.csv has no duplicate source-target rows, they are
# numerically IDENTICAL to in_tx_count/out_tx_count/total_tx_count (verified:
# 100% exact match). They are therefore assigned to the transaction-behaviour
# group only, so the behaviour-vs-structural ablation compares non-overlapping
# signal instead of the same quantity counted twice.

TRANSACTION_BEHAVIOUR_FEATURES = [
    "in_tx_count", "out_tx_count", "total_tx_count",
    "in_eth_value", "out_eth_value", "total_eth_value",
    "in_gas_fee_mean", "out_gas_fee_mean",
    "zero_value_tx_ratio",
    "weighted_in_degree", "weighted_out_degree", "weighted_degree",
    "log_in_tx_count", "log_out_tx_count", "log_total_tx_count",
    "log_in_eth_value", "log_out_eth_value", "log_total_eth_value",
    "log_weighted_degree",
]

GRAPH_STRUCTURAL_FEATURES = [
    "in_degree", "out_degree", "degree",
    "pagerank", "pagerank_weighted",
    "clustering_coeff", "has_closed_triangle", "wcc_size",
    "log_degree", "log_pagerank", "log_pagerank_weighted",
]

feature_groups = {
    "transaction_behaviour": TRANSACTION_BEHAVIOUR_FEATURES,
    "graph_structural": GRAPH_STRUCTURAL_FEATURES,
}

import json
with open("feature_groups.json", "w") as f:
    json.dump(feature_groups, f, indent=2)
print(f"\nSaved feature group assignment ({len(TRANSACTION_BEHAVIOUR_FEATURES)} behaviour, "
      f"{len(GRAPH_STRUCTURAL_FEATURES)} structural) to feature_groups.json")
