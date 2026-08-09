import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, normalized_mutual_info_score,
    adjusted_rand_score,
)
from sklearn.cluster import KMeans
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from scipy import stats

# =========================
# 0. Setup
# =========================
# Week 10 defensive check (done BEFORE GraphSAGE): every negative finding so
# far (Node2Vec embedding redundant, graph_structural redundant) was measured
# on the full 436-seed sample, which is 299/86/51 MEV/DeFi/CEX -- badly
# imbalanced. class_weight="balanced" / sample_weight / macro-F1 / stratified
# CV all defend against imbalance distorting the METRIC, but they do not rule
# out imbalance distorting the CONCLUSION itself: a model could still be
# leaning on transaction-volume features that happen to separate the 299-seed
# majority class well, silently deflating the apparent value of embedding /
# structural features that might matter more once classes are balanced. This
# script re-runs the core Week 8/9 comparisons, plus the Week 7 KMeans
# permutation test, on a class-balanced subset (51 seeds per class, matching
# the smallest class, CEX) to check whether the same negative conclusions hold.

RANDOM_STATE = 42
N_SPLITS = 5
LABEL_COL = "label"
LABEL_ORDER = ["CEX", "DeFi", "Malicious_MEV"]
MODEL_ORDER = ["LogisticRegression", "RandomForest", "XGBoost"]
MODEL_COLORS = {"LogisticRegression": "#2a78d6", "RandomForest": "#1baf7a", "XGBoost": "#eb6834"}

# Optional CLI args: feat_file edges_file emb_dir out_subdir classification_dir.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIR = PROJECT_ROOT / "02_feature_engineering" / "feature_engineering"
FEAT_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else FEATURE_DIR / "seed_feature_matrix_reconstructed.csv"
EDGES_FILE = Path(sys.argv[2]) if len(sys.argv) > 2 else PROJECT_ROOT / "00_data_graph" / "data" / "main" / "raw_edges_reconstructed.csv"
EMB_DIR = Path(sys.argv[3]) if len(sys.argv) > 3 else PROJECT_ROOT / "03_unsupervised" / "node2vec_results"
OUT_SUBDIR = Path(sys.argv[4]) if len(sys.argv) > 4 else PROJECT_ROOT / "06_robustness_audit" / "outputs" / "balanced_subset"
CLASSIFICATION_DIR = Path(sys.argv[5]) if len(sys.argv) > 5 else PROJECT_ROOT / "04_supervised" / "outputs" / "classification"
FEATURE_GROUPS_FILE = FEATURE_DIR / "feature_groups.json"

OUT_DIR = Path(OUT_SUBDIR)
OUT_DIR.mkdir(parents=True, exist_ok=True)

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


def permutation_significance(y_true, y_pred, n_perm=2000, random_state=RANDOM_STATE):
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    nmi_obs = normalized_mutual_info_score(y_true, y_pred)
    ari_obs = adjusted_rand_score(y_true, y_pred)

    nmi_null = np.empty(n_perm)
    ari_null = np.empty(n_perm)
    for i in range(n_perm):
        y_shuf = rng.permutation(y_true)
        nmi_null[i] = normalized_mutual_info_score(y_shuf, y_pred)
        ari_null[i] = adjusted_rand_score(y_shuf, y_pred)

    return {
        "nmi_observed": nmi_obs, "nmi_null_mean": nmi_null.mean(), "nmi_null_std": nmi_null.std(),
        "nmi_p_value": (nmi_null >= nmi_obs).mean(),
        "ari_observed": ari_obs, "ari_null_mean": ari_null.mean(), "ari_null_std": ari_null.std(),
        "ari_p_value": (ari_null >= ari_obs).mean(),
    }


def run_cv(feature_sets, y, group_label):
    """5-fold stratified CV for every (feature_set_name, model) pair. Returns
    a long-format fold-score DataFrame."""
    label_encoder = LabelEncoder().fit(LABEL_ORDER)
    rows = []
    for fs_name, X in feature_sets.items():
        for model_name in MODEL_ORDER:
            skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
            for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                if model_name == "LogisticRegression":
                    model = Pipeline([
                        ("scaler", StandardScaler()),
                        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
                    ])
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                elif model_name == "RandomForest":
                    model = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                else:
                    model = XGBClassifier(n_estimators=300, random_state=RANDOM_STATE, eval_metric="mlogloss", n_jobs=-1)
                    y_train_enc = label_encoder.transform(y_train)
                    sample_weight = compute_sample_weight("balanced", y_train_enc)
                    model.fit(X_train, y_train_enc, sample_weight=sample_weight)
                    y_pred = label_encoder.inverse_transform(model.predict(X_test))

                rows.append({
                    "group": group_label, "feature_set": fs_name, "model": model_name, "fold": fold_i,
                    "macro_f1": f1_score(y_test, y_pred, average="macro"),
                    "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
                })
    return pd.DataFrame(rows)


def paired_tests(fold_df, pairs, group_label):
    rows = []
    for a_name, b_name in pairs:
        for model_name in MODEL_ORDER:
            a = fold_df[(fold_df.group == group_label) & (fold_df.feature_set == a_name) & (fold_df.model == model_name)].sort_values("fold")["macro_f1"].values
            b = fold_df[(fold_df.group == group_label) & (fold_df.feature_set == b_name) & (fold_df.model == model_name)].sort_values("fold")["macro_f1"].values
            diff = a - b
            t_stat, p_val = stats.ttest_rel(a, b)
            rows.append({
                "group": group_label, "comparison": f"{a_name} vs {b_name}", "model": model_name,
                "mean_diff": diff.mean(), "paired_t_stat": t_stat, "paired_p_value": p_val,
                "n_folds_improved": int((diff > 0).sum()),
            })
    return pd.DataFrame(rows)


# =========================
# 1. Load hand-crafted features, feature groups, pagerank rank transform
# =========================
# Identical construction to 01_classification.ipynb / feature_ablation_shap.ipynb so
# the balanced-subset numbers are directly comparable to the full-sample ones.

feat = pd.read_csv(FEAT_FILE)
with open(FEATURE_GROUPS_FILE) as f:
    feature_groups = json.load(f)

txn_cols = list(feature_groups["transaction_behaviour"])
struct_cols = list(feature_groups["graph_structural"])

edges = pd.read_csv(EDGES_FILE)
edges["source"] = edges["source"].apply(clean_address)
edges["target"] = edges["target"].apply(clean_address)
edges = edges.dropna(subset=["source", "target"])

G_directed = nx.from_pandas_edgelist(edges, source="source", target="target", edge_attr=["tx_count"], create_using=nx.DiGraph())
pagerank_rank = pd.Series(nx.pagerank(G_directed)).rank(pct=True)
pagerank_weighted_rank = pd.Series(nx.pagerank(G_directed, weight="tx_count")).rank(pct=True)

feat["pagerank_rank"] = feat["address"].map(pagerank_rank).fillna(0.0)
feat["pagerank_weighted_rank"] = feat["address"].map(pagerank_weighted_rank).fillna(0.0)

for col in ["pagerank", "pagerank_weighted", "log_pagerank", "log_pagerank_weighted"]:
    if col in struct_cols:
        struct_cols.remove(col)
struct_cols += ["pagerank_rank", "pagerank_weighted_rank"]

handcrafted_cols = txn_cols + struct_cols

# =========================
# 2. Build the balanced subset: 51 seeds per class (CEX = smallest active class)
# =========================

active = feat[feat["is_active"]].copy()
emb = pd.read_csv(f"{EMB_DIR}/seed_node2vec_embeddings.csv")
emb_cols = [c for c in emb.columns if c.startswith("emb_")]
data_full = active.merge(emb[["address"] + emb_cols], on="address", how="inner")

n_per_class = data_full[LABEL_COL].value_counts().min()
print(f"Full active-labelled sample: {dict(data_full[LABEL_COL].value_counts())}")
print(f"Balanced subset target: {n_per_class} seeds per class (= smallest class size)")

balanced = (
    data_full.groupby(LABEL_COL, group_keys=False)
    .apply(lambda g: g.sample(n=n_per_class, random_state=RANDOM_STATE))
    .reset_index(drop=True)
)
balanced.to_csv(OUT_DIR / "balanced_subset_seeds.csv", index=False)
print(f"Balanced subset: {len(balanced)} seeds, {dict(balanced[LABEL_COL].value_counts())}")
print(f"Saved seed list to {OUT_DIR / 'balanced_subset_seeds.csv'}")

y_bal = balanced[LABEL_COL].to_numpy()
X_hand_bal = balanced[handcrafted_cols].to_numpy(dtype=float)
X_emb_bal = balanced[emb_cols].to_numpy(dtype=float)
X_combined_bal = np.hstack([X_hand_bal, X_emb_bal])
X_txn_bal = balanced[txn_cols].to_numpy(dtype=float)
X_struct_bal = balanced[struct_cols].to_numpy(dtype=float)

# =========================
# 3. Week 8 robustness: hand_crafted vs embedding vs combined, on balanced subset
# =========================

week8_sets = {"hand_crafted": X_hand_bal, "embedding": X_emb_bal, "combined": X_combined_bal}
fold_df_w8 = run_cv(week8_sets, y_bal, group_label="week8_embedding")

week8_summary = (
    fold_df_w8.groupby(["feature_set", "model"])[["macro_f1", "balanced_accuracy"]]
    .agg(["mean", "std"])
)
week8_summary.columns = ["_".join(c) for c in week8_summary.columns]
week8_summary = week8_summary.reset_index().round(4)
print(f"\n=== [Balanced subset] Week 8 comparison: hand_crafted / embedding / combined ===")
print(week8_summary.to_string(index=False))

week8_paired = paired_tests(fold_df_w8, [("combined", "hand_crafted")], "week8_embedding")
print(f"\n=== [Balanced subset] Paired test: combined vs hand_crafted ===")
print(week8_paired.round(4).to_string(index=False))

# =========================
# 4. Week 9 robustness: transaction_behaviour vs graph_structural vs combined
# =========================

week9_sets = {"transaction_behaviour": X_txn_bal, "graph_structural": X_struct_bal, "combined": X_hand_bal}
fold_df_w9 = run_cv(week9_sets, y_bal, group_label="week9_ablation")

week9_summary = (
    fold_df_w9.groupby(["feature_set", "model"])[["macro_f1", "balanced_accuracy"]]
    .agg(["mean", "std"])
)
week9_summary.columns = ["_".join(c) for c in week9_summary.columns]
week9_summary = week9_summary.reset_index().round(4)
print(f"\n=== [Balanced subset] Week 9 ablation: transaction_behaviour / graph_structural / combined ===")
print(week9_summary.to_string(index=False))

week9_paired = paired_tests(fold_df_w9, [("combined", "transaction_behaviour"), ("combined", "graph_structural")], "week9_ablation")
print(f"\n=== [Balanced subset] Paired tests: combined vs single group ===")
print(week9_paired.round(4).to_string(index=False))

# Save everything
fold_all = pd.concat([fold_df_w8, fold_df_w9], ignore_index=True)
fold_all.to_csv(OUT_DIR / "balanced_cv_fold_scores.csv", index=False)

summary_all = pd.concat([
    week8_summary.assign(group="week8_embedding"),
    week9_summary.assign(group="week9_ablation"),
], ignore_index=True)
summary_all.to_csv(OUT_DIR / "balanced_summary.csv", index=False)

paired_all = pd.concat([week8_paired, week9_paired], ignore_index=True)
paired_all.round(4).to_csv(OUT_DIR / "balanced_paired_test.csv", index=False)
print(f"\nSaved {OUT_DIR / 'balanced_cv_fold_scores.csv'}, 'balanced_summary.csv', 'balanced_paired_test.csv'")

# =========================
# 5. Side-by-side comparison chart: full sample vs balanced subset (combined feature set)
# =========================

full_summary = pd.read_csv(f"{CLASSIFICATION_DIR}/classification_summary.csv")
full_combined = full_summary[full_summary.feature_set == "combined"].set_index("model")["macro_f1_mean"]
bal_combined = week8_summary[week8_summary.feature_set == "combined"].set_index("model")["macro_f1_mean"]

fig, ax = plt.subplots(figsize=(7, 5))
x = np.arange(len(MODEL_ORDER))
width = 0.35
ax.bar(x - width / 2, [full_combined.get(m, np.nan) for m in MODEL_ORDER], width, color="#c3c2b7", label="Full sample (n=436, 299/86/51)")
ax.bar(x + width / 2, [bal_combined.get(m, np.nan) for m in MODEL_ORDER], width, color="#2a78d6", label=f"Balanced subset (n={len(balanced)}, {n_per_class}/{n_per_class}/{n_per_class})")
ax.set_xticks(x)
ax.set_xticklabels(MODEL_ORDER)
ax.set_ylabel("Macro-F1 (5-fold CV mean), combined feature set")
ax.set_title("Does class imbalance inflate the combined-feature-set result?")
ax.legend(frameon=False, fontsize=8)
ax.grid(axis="y", linewidth=0.6, alpha=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT_DIR / "full_vs_balanced_comparison.png", dpi=150)
plt.close(fig)
print(f"Saved {OUT_DIR / 'full_vs_balanced_comparison.png'}")

# =========================
# 6. Week 7 robustness: KMeans permutation test on the balanced subset
# =========================
# Validates (or refutes) the earlier diagnosis that KMeans' high NMI on the
# full sample was a class-imbalance artifact (NMI biased upward when cluster
# sizes coincidentally track class proportions), while ARI stayed
# non-significant. If NMI drops once classes are balanced, that confirms the
# artifact interpretation instead of merely asserting it.

k = len(LABEL_ORDER)
kmeans_bal = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
bal_clusters = kmeans_bal.fit_predict(X_emb_bal)

kmeans_bal_result = permutation_significance(y_bal, bal_clusters, n_perm=2000, random_state=RANDOM_STATE)
print(f"\n=== [Balanced subset] KMeans (k=3) permutation test ===")
print(f"NMI={kmeans_bal_result['nmi_observed']:.4f} (p={kmeans_bal_result['nmi_p_value']:.4f}), "
      f"ARI={kmeans_bal_result['ari_observed']:.4f} (p={kmeans_bal_result['ari_p_value']:.4f})")

# Full-sample KMeans clustering, restricted to the SAME balanced subset of
# seeds, to isolate "does balancing the evaluation set change the picture"
# from "does balancing the sample fed into KMeans change the picture".
full_emb_df = pd.read_csv(f"{EMB_DIR}/seed_node2vec_embeddings.csv")
full_kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
full_clusters = full_kmeans.fit_predict(full_emb_df[emb_cols].to_numpy(dtype=float))
full_emb_df["kmeans_cluster"] = full_clusters
bal_addr_set = set(balanced["address"])
mask = full_emb_df["address"].isin(bal_addr_set)
eval_only_result = permutation_significance(
    full_emb_df.loc[mask, LABEL_COL].values, full_emb_df.loc[mask, "kmeans_cluster"].values,
    n_perm=2000, random_state=RANDOM_STATE,
)
print(f"\n=== [Balanced EVALUATION SUBSET of the full-sample KMeans clustering] ===")
print(f"NMI={eval_only_result['nmi_observed']:.4f} (p={eval_only_result['nmi_p_value']:.4f}), "
      f"ARI={eval_only_result['ari_observed']:.4f} (p={eval_only_result['ari_p_value']:.4f})")

full_result = pd.read_csv(f"{EMB_DIR}/clustering_significance_test.csv")
full_kmeans_row = full_result[full_result.method == "KMeans"].iloc[0]
full_louvain_row = full_result[full_result.method == "Louvain"].iloc[0]

clustering_compare = pd.DataFrame([
    {"condition": "Full sample (n=436)", "method": "KMeans", "NMI": full_kmeans_row["nmi_observed"], "NMI_p": full_kmeans_row["nmi_p_value"],
     "ARI": full_kmeans_row["ari_observed"], "ARI_p": full_kmeans_row["ari_p_value"]},
    {"condition": "Balanced evaluation subset of full-sample clustering (n=153)", "method": "KMeans", "NMI": eval_only_result["nmi_observed"],
     "NMI_p": eval_only_result["nmi_p_value"], "ARI": eval_only_result["ari_observed"], "ARI_p": eval_only_result["ari_p_value"]},
    {"condition": "KMeans re-fit on balanced subset only (n=153)", "method": "KMeans", "NMI": kmeans_bal_result["nmi_observed"],
     "NMI_p": kmeans_bal_result["nmi_p_value"], "ARI": kmeans_bal_result["ari_observed"], "ARI_p": kmeans_bal_result["ari_p_value"]},
]).round(4)

# =========================
# 7. Week 7 robustness: Louvain, evaluated on the same balanced subset
# =========================
# Louvain partitions the WHOLE 8,029-node graph topology, not just labelled
# seeds, so "re-fitting Louvain on a balanced subset" is not meaningful (the
# induced subgraph on 153 isolated seeds would mostly have no edges left,
# since counterparties are almost never seeds themselves). The valid check is
# the same one already used for KMeans' "evaluation subset" row: keep the
# existing full-graph Louvain partition fixed, and see whether restricting
# the EVALUATION to a balanced set of labelled seeds changes NMI/ARI.

G_undirected = G_directed.to_undirected()
for _, _, d in G_undirected.edges(data=True):
    d["weight"] = d.get("tx_count", 1)

partition = community_louvain.best_partition(G_undirected, weight="weight", random_state=RANDOM_STATE)
data_full["louvain_cluster"] = data_full["address"].map(partition)
mask_louvain = data_full["address"].isin(bal_addr_set)

louvain_full_result = permutation_significance(
    data_full[LABEL_COL].values, data_full["louvain_cluster"].values, n_perm=2000, random_state=RANDOM_STATE,
)
louvain_bal_eval_result = permutation_significance(
    data_full.loc[mask_louvain, LABEL_COL].values, data_full.loc[mask_louvain, "louvain_cluster"].values,
    n_perm=2000, random_state=RANDOM_STATE,
)
print(f"\n=== [Louvain] Full sample (n=436, consistency check against node2vec_results/) ===")
print(f"NMI={louvain_full_result['nmi_observed']:.4f} (p={louvain_full_result['nmi_p_value']:.4f}), "
      f"ARI={louvain_full_result['ari_observed']:.4f} (p={louvain_full_result['ari_p_value']:.4f}) "
      f"[stored value: NMI={full_louvain_row['nmi_observed']:.4f}, ARI={full_louvain_row['ari_observed']:.4f}]")
print(f"\n=== [Louvain] Balanced evaluation subset of the same full-graph partition (n=153) ===")
print(f"NMI={louvain_bal_eval_result['nmi_observed']:.4f} (p={louvain_bal_eval_result['nmi_p_value']:.4f}), "
      f"ARI={louvain_bal_eval_result['ari_observed']:.4f} (p={louvain_bal_eval_result['ari_p_value']:.4f})")

clustering_compare = pd.concat([clustering_compare, pd.DataFrame([
    {"condition": "Full sample (n=436)", "method": "Louvain", "NMI": louvain_full_result["nmi_observed"], "NMI_p": louvain_full_result["nmi_p_value"],
     "ARI": louvain_full_result["ari_observed"], "ARI_p": louvain_full_result["ari_p_value"]},
    {"condition": "Balanced evaluation subset of full-sample clustering (n=153)", "method": "Louvain", "NMI": louvain_bal_eval_result["nmi_observed"],
     "NMI_p": louvain_bal_eval_result["nmi_p_value"], "ARI": louvain_bal_eval_result["ari_observed"], "ARI_p": louvain_bal_eval_result["ari_p_value"]},
])], ignore_index=True).round(4)

clustering_compare.to_csv(OUT_DIR / "balanced_kmeans_comparison.csv", index=False)
print(f"\n=== NMI/ARI across imbalance conditions (KMeans + Louvain) ===\n{clustering_compare.to_string(index=False)}")
print(f"Saved {OUT_DIR / 'balanced_kmeans_comparison.csv'}")

print("\n=== Week 10 balanced-subset robustness check complete ===")
