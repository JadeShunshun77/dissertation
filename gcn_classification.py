import sys
import os

# See progress_summary.md Section 9 / graphsage_classification.py: same Windows
# torch DLL workaround needed here.
if not sys.exec_prefix.endswith(os.sep):
    sys.exec_prefix += os.sep
if not sys.base_exec_prefix.endswith(os.sep):
    sys.base_exec_prefix += os.sep

import json
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, balanced_accuracy_score
from scipy import stats

# =========================
# 0. Setup
# =========================
# Second GNN baseline requested alongside GraphSAGE (advisor feedback,
# 2026-07-31): does the most standard spectral/message-passing GNN (Kipf &
# Welling's GCN) tell a different story than GraphSAGE on the same
# reconstructed graph? Mirrors graphsage_classification.py's design exactly
# (same balanced 153-seed subset, same structural-only node features, same
# capacity/training protocol, same folds) so GCN-vs-GraphSAGE is an
# apples-to-apples comparison and not confounded by a different setup.

RANDOM_STATE = 42
N_SPLITS = 5
LABEL_COL = "label"
LABEL_ORDER = ["CEX", "DeFi", "Malicious_MEV"]
HIDDEN_DIM = 16
DROPOUT = 0.6
WEIGHT_DECAY = 1e-3
LR = 0.01
MAX_EPOCHS = 300
PATIENCE = 20
VAL_FRACTION = 0.2

# Optional CLI args: edges_file balanced_subset_dir graphsage_dir out_subdir
# e.g. `python gcn_classification.py raw_edges_reconstructed.csv balanced_subset_outputs_reconstructed graphsage_outputs_reconstructed gcn_outputs_reconstructed`
EDGES_FILE = sys.argv[1] if len(sys.argv) > 1 else "raw_edges.csv"
BALANCED_DIR = sys.argv[2] if len(sys.argv) > 2 else "balanced_subset_outputs"
GRAPHSAGE_DIR = sys.argv[3] if len(sys.argv) > 3 else "graphsage_outputs"
OUT_SUBDIR = sys.argv[4] if len(sys.argv) > 4 else "gcn_outputs"

OUT_DIR = Path(OUT_SUBDIR)
OUT_DIR.mkdir(exist_ok=True)

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": "#0b0b0b",
    "text.color": "#0b0b0b",
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
    "grid.color": "#e1e0d9",
    "font.size": 10,
})


def clean_address(x):
    if pd.isna(x):
        return np.nan
    return str(x).strip().lower()


# =========================
# 1. Build the full graph (same construction as graphsage_classification.py)
# =========================

edges = pd.read_csv(EDGES_FILE)
edges["source"] = edges["source"].apply(clean_address)
edges["target"] = edges["target"].apply(clean_address)
edges = edges.dropna(subset=["source", "target"])

G_directed = nx.from_pandas_edgelist(edges, source="source", target="target", edge_attr=["tx_count"], create_using=nx.DiGraph())
G_undirected = G_directed.to_undirected()
for _, _, d in G_undirected.edges(data=True):
    d["weight"] = d.get("tx_count", 1)

print(f"Graph: {G_undirected.number_of_nodes()} nodes, {G_undirected.number_of_edges()} edges (undirected, message-passing view)")

# =========================
# 2. Structural node features for ALL nodes (identical to graphsage_classification.py)
# =========================

in_degree = dict(G_directed.in_degree())
out_degree = dict(G_directed.out_degree())
pagerank_rank = pd.Series(nx.pagerank(G_directed)).rank(pct=True)
pagerank_weighted_rank = pd.Series(nx.pagerank(G_directed, weight="tx_count")).rank(pct=True)
clustering = nx.clustering(G_undirected)
wcc_size = {}
for component in nx.weakly_connected_components(G_directed):
    size = len(component)
    for node in component:
        wcc_size[node] = size

nodes = list(G_undirected.nodes())
struct_df = pd.DataFrame({
    "in_degree": pd.Series(in_degree),
    "out_degree": pd.Series(out_degree),
    "pagerank_rank": pagerank_rank,
    "pagerank_weighted_rank": pagerank_weighted_rank,
    "clustering_coeff": pd.Series(clustering),
    "wcc_size": pd.Series(wcc_size),
}).reindex(nodes).fillna(0.0)
struct_df["degree"] = struct_df["in_degree"] + struct_df["out_degree"]
struct_df["log_degree"] = np.log1p(struct_df["degree"])
struct_df["has_closed_triangle"] = (struct_df["clustering_coeff"] > 0).astype(float)

feature_cols = ["in_degree", "out_degree", "degree", "pagerank_rank", "pagerank_weighted_rank",
                "clustering_coeff", "has_closed_triangle", "wcc_size", "log_degree"]
struct_df = struct_df[feature_cols]

X_all = StandardScaler().fit_transform(struct_df.to_numpy(dtype=float))
print(f"Node feature matrix: {X_all.shape} (structural-only, {len(feature_cols)} features, standardised over all {len(nodes)} nodes)")

# =========================
# 3. PyG graph tensors
# =========================

node_to_idx = {addr: i for i, addr in enumerate(nodes)}
edges_list = list(G_undirected.edges())
src = [node_to_idx[u] for u, v in edges_list] + [node_to_idx[v] for u, v in edges_list]
dst = [node_to_idx[v] for u, v in edges_list] + [node_to_idx[u] for u, v in edges_list]
edge_index = torch.tensor([src, dst], dtype=torch.long)
x = torch.tensor(X_all, dtype=torch.float)

# =========================
# 4. Load the SAME balanced subset used for GraphSAGE
# =========================

balanced = pd.read_csv(f"{BALANCED_DIR}/balanced_subset_seeds.csv")
assert balanced["address"].isin(node_to_idx).all(), "balanced subset contains an address not in the graph"
balanced_idx = balanced["address"].map(node_to_idx).to_numpy()
label_encoder = LabelEncoder().fit(LABEL_ORDER)
y_bal = label_encoder.transform(balanced[LABEL_COL].to_numpy())
print(f"Balanced subset: {len(balanced)} labelled seeds ({dict(balanced[LABEL_COL].value_counts())}), "
      f"transductive setting: all {len(nodes)} graph nodes participate in message passing, "
      f"only these node indices ever contribute to the loss/metrics")


# =========================
# 5. Model: 2-layer GCN, same capacity/training protocol as GraphSAGE
# =========================

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        return h


def train_one_fold(train_pos_idx, val_pos_idx, test_pos_idx, node_idx_array, y_array):
    train_nodes = torch.tensor(node_idx_array[train_pos_idx], dtype=torch.long)
    val_nodes = torch.tensor(node_idx_array[val_pos_idx], dtype=torch.long)
    test_nodes = torch.tensor(node_idx_array[test_pos_idx], dtype=torch.long)
    y_train = torch.tensor(y_array[train_pos_idx], dtype=torch.long)
    y_val = torch.tensor(y_array[val_pos_idx], dtype=torch.long)
    y_test_np = y_array[test_pos_idx]

    model = GCN(in_dim=x.shape[1], hidden_dim=HIDDEN_DIM, out_dim=len(LABEL_ORDER), dropout=DROPOUT)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_f1 = -1.0
    best_state = None
    epochs_since_improve = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.cross_entropy(out[train_nodes], y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out_eval = model(x, edge_index)
            val_pred = out_eval[val_nodes].argmax(dim=1).numpy()
            val_f1 = f1_score(y_val.numpy(), val_pred, average="macro", zero_division=0)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out_final = model(x, edge_index)
        test_pred = out_final[test_nodes].argmax(dim=1).numpy()

    return {
        "macro_f1": f1_score(y_test_np, test_pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_test_np, test_pred),
        "best_epoch_val_f1": best_val_f1,
        "n_epochs_trained": epoch + 1,
    }


# =========================
# 6. 5-fold stratified CV over the balanced-subset nodes (same folds as GraphSAGE)
# =========================

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
fold_rows = []
rng = np.random.default_rng(RANDOM_STATE)

for fold_i, (train_pos, test_pos) in enumerate(skf.split(balanced_idx, y_bal)):
    train_pos = rng.permutation(train_pos)
    n_val = max(len(LABEL_ORDER), int(len(train_pos) * VAL_FRACTION))
    val_pos, fit_pos = train_pos[:n_val], train_pos[n_val:]

    result = train_one_fold(fit_pos, val_pos, test_pos, balanced_idx, y_bal)
    result["fold"] = fold_i
    fold_rows.append(result)
    print(f"Fold {fold_i}: macro_f1={result['macro_f1']:.4f}  balanced_acc={result['balanced_accuracy']:.4f}  "
          f"(val_f1={result['best_epoch_val_f1']:.4f}, stopped at epoch {result['n_epochs_trained']})")

fold_df = pd.DataFrame(fold_rows)[["fold", "macro_f1", "balanced_accuracy", "best_epoch_val_f1", "n_epochs_trained"]]
fold_df.to_csv(OUT_DIR / "gcn_cv_fold_scores.csv", index=False)
print(f"\n=== GCN 5-fold CV (balanced subset, n={len(balanced)}) ===")
print(f"Macro-F1: {fold_df['macro_f1'].mean():.4f} +/- {fold_df['macro_f1'].std():.4f}")
print(f"Balanced accuracy: {fold_df['balanced_accuracy'].mean():.4f} +/- {fold_df['balanced_accuracy'].std():.4f}")
print(f"Saved {OUT_DIR / 'gcn_cv_fold_scores.csv'}")

# =========================
# 7. Compare against the same balanced-subset baselines used for GraphSAGE,
#    AND against GraphSAGE itself (same folds throughout)
# =========================

baseline_folds = pd.read_csv(f"{BALANCED_DIR}/balanced_cv_fold_scores.csv")
sage_folds = pd.read_csv(f"{GRAPHSAGE_DIR}/graphsage_cv_fold_scores.csv").sort_values("fold")["macro_f1"].values

comparisons = []
for group, fs_name, model_name, label in [
    ("week9_ablation", "graph_structural", "RandomForest", "graph_structural (RF, structural-only, apples-to-apples)"),
    ("week9_ablation", "graph_structural", "XGBoost", "graph_structural (XGBoost, structural-only, apples-to-apples)"),
    ("week8_embedding", "hand_crafted", "RandomForest", "hand_crafted (RF, includes transaction features -- NOT apples-to-apples)"),
    ("week8_embedding", "hand_crafted", "XGBoost", "hand_crafted (XGBoost, includes transaction features -- NOT apples-to-apples)"),
]:
    base = baseline_folds[
        (baseline_folds.group == group) & (baseline_folds.feature_set == fs_name) & (baseline_folds.model == model_name)
    ].sort_values("fold")["macro_f1"].values
    gcn = fold_df.sort_values("fold")["macro_f1"].values
    diff = gcn - base
    t_stat, p_val = stats.ttest_rel(gcn, base)
    comparisons.append({
        "baseline": label, "gcn_mean": gcn.mean(), "baseline_mean": base.mean(),
        "mean_diff": diff.mean(), "paired_t_stat": t_stat, "paired_p_value": p_val,
        "n_folds_gcn_better": int((diff > 0).sum()),
    })

# GCN vs GraphSAGE -- the core new-baseline comparison
gcn_vals = fold_df.sort_values("fold")["macro_f1"].values
diff_sage = gcn_vals - sage_folds
t_stat_sage, p_val_sage = stats.ttest_rel(gcn_vals, sage_folds)
comparisons.append({
    "baseline": "GraphSAGE (same structural input, same folds)", "gcn_mean": gcn_vals.mean(), "baseline_mean": sage_folds.mean(),
    "mean_diff": diff_sage.mean(), "paired_t_stat": t_stat_sage, "paired_p_value": p_val_sage,
    "n_folds_gcn_better": int((diff_sage > 0).sum()),
})

comparison_df = pd.DataFrame(comparisons).round(4)
comparison_df.to_csv(OUT_DIR / "gcn_vs_baseline_comparison.csv", index=False)
print(f"\n=== GCN vs balanced-subset baselines and GraphSAGE (same folds, paired) ===")
print(comparison_df.to_string(index=False))
print("(n=5 folds -- treat p-values as indicative, not confirmatory.)")
print(f"Saved {OUT_DIR / 'gcn_vs_baseline_comparison.csv'}")

# =========================
# 8. Bar chart
# =========================

fig, ax = plt.subplots(figsize=(9, 5))
labels_plot = ["GCN\n(structural, e2e)", "GraphSAGE\n(structural, e2e)", "graph_structural\n(RF)", "graph_structural\n(XGBoost)",
               "hand_crafted\n(RF)*", "hand_crafted\n(XGBoost)*"]
means_plot = [fold_df["macro_f1"].mean(), sage_folds.mean()] + list(comparison_df["baseline_mean"].iloc[:4])
stds_plot = [fold_df["macro_f1"].std(), sage_folds.std()] + [
    baseline_folds[(baseline_folds.group == g) & (baseline_folds.feature_set == fs) & (baseline_folds.model == m)]["macro_f1"].std()
    for g, fs, m in [("week9_ablation", "graph_structural", "RandomForest"), ("week9_ablation", "graph_structural", "XGBoost"),
                     ("week8_embedding", "hand_crafted", "RandomForest"), ("week8_embedding", "hand_crafted", "XGBoost")]
]
colors_plot = ["#c1440e", "#6b4fa0", "#1baf7a", "#1baf7a", "#c3c2b7", "#c3c2b7"]
ax.bar(labels_plot, means_plot, yerr=stds_plot, capsize=3, color=colors_plot)
ax.set_ylabel("Macro-F1 (5-fold CV mean ± std, balanced subset)")
ax.set_title("GCN vs GraphSAGE and hand-crafted baselines (*not apples-to-apples: hand_crafted has transaction features neither GNN has)")
ax.grid(axis="y", linewidth=0.6, alpha=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT_DIR / "gcn_vs_baseline.png", dpi=150)
plt.close(fig)
print(f"Saved {OUT_DIR / 'gcn_vs_baseline.png'}")

print("\n=== GCN baseline experiment complete ===")
