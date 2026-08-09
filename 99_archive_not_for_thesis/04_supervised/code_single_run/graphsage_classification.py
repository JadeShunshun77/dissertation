import sys
import os

# See progress_summary.md Section 9 / node2vec_clustering.py Section 4.2: this
# machine's Python is installed directly at a drive root (D:\), so
# sys.exec_prefix is the bare string "D:" (no trailing separator). torch's
# Windows DLL loader does os.path.join(sys.exec_prefix, "bin"), which Windows
# resolves to the malformed "D:bin" instead of "D:\bin", causing
# OSError: [WinError 87] on import. Padding the prefix with a trailing
# separator before importing torch avoids the bug without touching the
# system-wide install.
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
from torch_geometric.nn import SAGEConv

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, balanced_accuracy_score
from scipy import stats

# =========================
# 0. Setup
# =========================
# Week 10 (optional): does an end-to-end supervised GraphSAGE beat the
# structural-only hand-crafted baseline, and does it close the gap to the
# hand_crafted (transaction + structural) baseline? Trained and evaluated on
# the SAME 153-seed class-balanced subset as balanced_subset_robustness.py
# (see progress_summary.md Section 7), so the comparison starts from an
# already-defended, imbalance-free footing rather than needing a second
# balanced-subset check after the fact.
#
# Deliberately tiny model: 153 labelled nodes is very little supervision for
# a GNN. hidden_dim=16, 2 layers, dropout=0.6, weight_decay, and early
# stopping on a held-out validation split (carved out of each fold's training
# set, never the test fold) are all chosen to keep capacity below the
# overfitting cliff. An underfit negative result is trustworthy; an inflated
# positive result from an overparameterised model on 153 points is not.

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

# Optional CLI args: edges_file balanced_subset_dir out_subdir node_feat_file.
# node_feat_file (Week 12, §14.2): optional path to a full-graph transaction-
# behaviour feature table (address + the 19 transaction_behaviour columns for
# ALL graph nodes, not just seeds). When given, node input features become
# graph_structural + transaction_behaviour (28 cols) instead of
# graph_structural only (9 cols) -- lets message passing use neighbours'
# transaction behaviour, which RF/XGBoost structurally cannot access (they
# only see a node's own row). Omit to reproduce the original structural-only
# run exactly.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
FEATURE_DIR = PROJECT_ROOT / "02_feature_engineering" / "outputs" / "main"
EDGES_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "00_data_graph" / "data" / "main" / "raw_edges_reconstructed.csv"
BALANCED_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else PROJECT_ROOT / "06_robustness_audit" / "outputs" / "balanced_subset"
OUT_SUBDIR = Path(sys.argv[3]) if len(sys.argv) > 3 else PROJECT_ROOT / "99_archive_not_for_thesis" / "04_supervised" / "outputs" / "graphsage_single_run_rerun"
NODE_FEAT_FILE = Path(sys.argv[4]) if len(sys.argv) > 4 else FEATURE_DIR / "full_graph_transaction_behaviour_features.csv"
FEATURE_GROUPS_FILE = FEATURE_DIR / "feature_groups.json"

OUT_DIR = Path(OUT_SUBDIR)
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
# 1. Build the full 8,029-node graph (same construction as node2vec_clustering.py)
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
# 2. Structural node features for ALL 8,029 nodes
# =========================
# Identical definitions to feature_groups.json's graph_structural group (with
# the same pagerank -> pagerank_rank substitution used from classification.py
# onward), computed here for every graph node rather than just the 500 seeds,
# since GraphSAGE needs input features for every node it passes messages
# through. Transaction-behaviour features (tx counts, ETH value, gas fees)
# are NOT available for non-seed counterparties (they require per-address
# BigQuery lookups only run for the 500 labelled seeds), so GraphSAGE here
# only has access to the same information as the graph_structural hand-crafted
# feature set -- NOT the full hand_crafted set. This is an important asymmetry
# to keep in the results: any GraphSAGE-vs-hand_crafted comparison is not
# apples-to-apples, whereas GraphSAGE-vs-graph_structural is.

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

if NODE_FEAT_FILE:
    import json as _json
    with open(FEATURE_GROUPS_FILE) as f:
        _txn_cols = _json.load(f)["transaction_behaviour"]
    txn_df = pd.read_csv(NODE_FEAT_FILE).set_index("address")[_txn_cols].reindex(nodes).fillna(0.0)
    combined_df = pd.concat([struct_df, txn_df], axis=1)
    X_all = StandardScaler().fit_transform(combined_df.to_numpy(dtype=float))
    print(f"Node feature matrix: {X_all.shape} (structural + transaction_behaviour, "
          f"{len(feature_cols)}+{len(_txn_cols)} features, standardised over all {len(nodes)} nodes)")
else:
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
# 4. Load the SAME 153-seed balanced subset used in balanced_subset_robustness.py
# =========================

balanced = pd.read_csv(f"{BALANCED_DIR}/balanced_subset_seeds.csv")
assert balanced["address"].isin(node_to_idx).all(), "balanced subset contains an address not in the graph"
balanced_idx = balanced["address"].map(node_to_idx).to_numpy()
label_encoder = LabelEncoder().fit(LABEL_ORDER)
y_bal = label_encoder.transform(balanced[LABEL_COL].to_numpy())
print(f"Balanced subset: {len(balanced)} labelled seeds ({dict(balanced[LABEL_COL].value_counts())}), "
      f"transductive setting: all {len(nodes)} graph nodes participate in message passing, "
      f"only these 153 node indices ever contribute to the loss/metrics")


# =========================
# 5. Model: 2-layer GraphSAGE, deliberately small capacity
# =========================

class SAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        return h


def train_one_fold(train_pos_idx, val_pos_idx, test_pos_idx, node_idx_array, y_array):
    """train/val/test_pos_idx are positions INTO node_idx_array/y_array (the
    153-length balanced-subset arrays), not raw graph node indices."""
    train_nodes = torch.tensor(node_idx_array[train_pos_idx], dtype=torch.long)
    val_nodes = torch.tensor(node_idx_array[val_pos_idx], dtype=torch.long)
    test_nodes = torch.tensor(node_idx_array[test_pos_idx], dtype=torch.long)
    y_train = torch.tensor(y_array[train_pos_idx], dtype=torch.long)
    y_val = torch.tensor(y_array[val_pos_idx], dtype=torch.long)
    y_test_np = y_array[test_pos_idx]

    model = SAGE(in_dim=x.shape[1], hidden_dim=HIDDEN_DIM, out_dim=len(LABEL_ORDER), dropout=DROPOUT)
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
# 6. 5-fold stratified CV over the 153 balanced-subset nodes
# =========================
# Same StratifiedKFold parameters (n_splits=5, shuffle=True, random_state=42)
# over the same seed order as balanced_subset_robustness.py's balanced
# DataFrame, so fold membership matches and the comparison against
# graph_structural / hand_crafted on the balanced subset is a fair,
# same-folds comparison (checked below in Section 7).
#
# Leakage note: StratifiedKFold splits WHICH OF THE 153 LABELLED NODES'
# LABELS enter the training loss (train_nodes / y_train above), not which
# nodes participate in message passing -- x and edge_index cover the full
# 8,029-node graph in every fold and are never masked. This is the standard
# transductive node-classification setup (same convention as the original
# GraphSAGE/GCN papers): a test node's FEATURES and its position in the graph
# are visible throughout (needed for message passing to even reach it), but
# its LABEL never appears in the loss, the validation early-stopping signal,
# or anywhere gradients are computed until final evaluation.

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
fold_rows = []
rng = np.random.default_rng(RANDOM_STATE)

for fold_i, (train_pos, test_pos) in enumerate(skf.split(balanced_idx, y_bal)):
    # Carve a validation split out of this fold's training portion only
    # (never touches test_pos) for early stopping.
    train_pos = rng.permutation(train_pos)
    n_val = max(len(LABEL_ORDER), int(len(train_pos) * VAL_FRACTION))
    val_pos, fit_pos = train_pos[:n_val], train_pos[n_val:]

    result = train_one_fold(fit_pos, val_pos, test_pos, balanced_idx, y_bal)
    result["fold"] = fold_i
    fold_rows.append(result)
    print(f"Fold {fold_i}: macro_f1={result['macro_f1']:.4f}  balanced_acc={result['balanced_accuracy']:.4f}  "
          f"(val_f1={result['best_epoch_val_f1']:.4f}, stopped at epoch {result['n_epochs_trained']})")

fold_df = pd.DataFrame(fold_rows)[["fold", "macro_f1", "balanced_accuracy", "best_epoch_val_f1", "n_epochs_trained"]]
fold_df.to_csv(OUT_DIR / "graphsage_cv_fold_scores.csv", index=False)
print(f"\n=== GraphSAGE 5-fold CV (balanced subset, n=153) ===")
print(f"Macro-F1: {fold_df['macro_f1'].mean():.4f} +/- {fold_df['macro_f1'].std():.4f}")
print(f"Balanced accuracy: {fold_df['balanced_accuracy'].mean():.4f} +/- {fold_df['balanced_accuracy'].std():.4f}")
print(f"Saved {OUT_DIR / 'graphsage_cv_fold_scores.csv'}")

# =========================
# 7. Compare against Week 10 balanced-subset baselines (same folds)
# =========================

baseline_folds = pd.read_csv(f"{BALANCED_DIR}/balanced_cv_fold_scores.csv")
comparisons = []
_struct_note = "apples-to-apples" if not NODE_FEAT_FILE else "GraphSAGE now has MORE info (structural+transaction) than this structural-only baseline"
_hand_note = "includes transaction features -- NOT apples-to-apples" if not NODE_FEAT_FILE else "apples-to-apples (GraphSAGE now also has transaction_behaviour features, see NODE_FEAT_FILE)"
for group, fs_name, model_name, label in [
    ("week9_ablation", "graph_structural", "RandomForest", f"graph_structural (RF, structural-only, {_struct_note})"),
    ("week9_ablation", "graph_structural", "XGBoost", f"graph_structural (XGBoost, structural-only, {_struct_note})"),
    ("week8_embedding", "hand_crafted", "RandomForest", f"hand_crafted (RF, {_hand_note})"),
    ("week8_embedding", "hand_crafted", "XGBoost", f"hand_crafted (XGBoost, {_hand_note})"),
]:
    base = baseline_folds[
        (baseline_folds.group == group) & (baseline_folds.feature_set == fs_name) & (baseline_folds.model == model_name)
    ].sort_values("fold")["macro_f1"].values
    sage = fold_df.sort_values("fold")["macro_f1"].values
    diff = sage - base
    t_stat, p_val = stats.ttest_rel(sage, base)
    comparisons.append({
        "baseline": label, "graphsage_mean": sage.mean(), "baseline_mean": base.mean(),
        "mean_diff": diff.mean(), "paired_t_stat": t_stat, "paired_p_value": p_val,
        "n_folds_graphsage_better": int((diff > 0).sum()),
    })

comparison_df = pd.DataFrame(comparisons).round(4)
comparison_df.to_csv(OUT_DIR / "graphsage_vs_baseline_comparison.csv", index=False)
print(f"\n=== GraphSAGE vs Week 10 balanced-subset baselines (same 5 folds, paired) ===")
print(comparison_df.to_string(index=False))
print("(n=5 folds, 153 total labelled nodes -- treat p-values as indicative, not confirmatory.)")
print(f"Saved {OUT_DIR / 'graphsage_vs_baseline_comparison.csv'}")

# =========================
# 8. Bar chart
# =========================

fig, ax = plt.subplots(figsize=(8, 5))
labels_plot = ["GraphSAGE\n(structural, e2e)", "graph_structural\n(RF)", "graph_structural\n(XGBoost)",
               "hand_crafted\n(RF)*", "hand_crafted\n(XGBoost)*"]
means_plot = [fold_df["macro_f1"].mean()] + list(comparison_df["baseline_mean"])
stds_plot = [fold_df["macro_f1"].std()] + [
    baseline_folds[(baseline_folds.group == g) & (baseline_folds.feature_set == fs) & (baseline_folds.model == m)]["macro_f1"].std()
    for g, fs, m in [("week9_ablation", "graph_structural", "RandomForest"), ("week9_ablation", "graph_structural", "XGBoost"),
                     ("week8_embedding", "hand_crafted", "RandomForest"), ("week8_embedding", "hand_crafted", "XGBoost")]
]
colors_plot = ["#6b4fa0", "#1baf7a", "#1baf7a", "#c3c2b7", "#c3c2b7"]
ax.bar(labels_plot, means_plot, yerr=stds_plot, capsize=3, color=colors_plot)
ax.set_ylabel("Macro-F1 (5-fold CV mean ± std, balanced subset n=153)")
_title_suffix = "*not apples-to-apples: hand_crafted has transaction features GraphSAGE lacks" if not NODE_FEAT_FILE else "GraphSAGE now has transaction_behaviour features too -- all comparisons apples-to-apples"
ax.set_title(f"GraphSAGE vs hand-crafted baselines ({_title_suffix})")
ax.grid(axis="y", linewidth=0.6, alpha=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT_DIR / "graphsage_vs_baseline.png", dpi=150)
plt.close(fig)
print(f"Saved {OUT_DIR / 'graphsage_vs_baseline.png'}")

print("\n=== Week 10 GraphSAGE experiment complete ===")
