import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from scipy import stats

# =========================
# 0. Setup -- Week 12, progress_summary.md Section 14.6
# =========================
# Section 14.5's "vs RF/XGBoost baseline" comparison was asymmetric: the GNN
# side was averaged over 5 distinct training seeds, but the RF/XGBoost
# baseline was still a single run (balanced_subset_robustness.py's
# random_state=42 only). This requests RF/XGBoost across the SAME 5 training
# seeds used for GraphSAGE/GCN, holding the fold split fixed (random_state=42
# for the StratifiedKFold split itself, identical to every other script in
# this project) and only varying the MODEL's own random_state -- the tree-
# ensemble analogue of varying a GNN's weight-init/dropout seed (RF's
# bootstrap/feature subsampling, XGBoost's tree-building randomness). This is
# a conditional algorithmic-stability check on one fixed sample and split,
# not population inference. With the current XGBoost configuration, all five
# requested seeds produce identical scores.

RANDOM_STATE = 42  # ALWAYS the fold-split seed -- must stay fixed so folds match every other script
N_SPLITS = 5
TRAIN_SEEDS = [42, 43, 44, 45, 46]  # model's own random_state, varies per trial
LABEL_COL = "label"
LABEL_ORDER = ["CEX", "DeFi", "Malicious_MEV"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIR = PROJECT_ROOT / "02_feature_engineering" / "feature_engineering"
FEAT_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else FEATURE_DIR / "seed_feature_matrix_reconstructed.csv"
EDGES_FILE = Path(sys.argv[2]) if len(sys.argv) > 2 else PROJECT_ROOT / "00_data_graph" / "data" / "main" / "raw_edges_reconstructed.csv"
BALANCED_DIR = Path(sys.argv[3]) if len(sys.argv) > 3 else PROJECT_ROOT / "06_robustness_audit" / "outputs" / "balanced_subset"
OUT_SUBDIR = Path(sys.argv[4]) if len(sys.argv) > 4 else PROJECT_ROOT / "04_supervised" / "outputs" / "tree_baseline_multiseed"
GNN_DIR = Path(sys.argv[5]) if len(sys.argv) > 5 else PROJECT_ROOT / "04_supervised" / "outputs" / "gnn_multiseed"
FEATURE_GROUPS_FILE = FEATURE_DIR / "feature_groups.json"

OUT_DIR = Path(OUT_SUBDIR)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_address(x):
    if pd.isna(x):
        return np.nan
    return str(x).strip().lower()


# =========================
# 1. Build hand_crafted (28) and graph_structural (9) feature sets for the
#    SAME fixed 153-seed balanced subset (reused, not resampled)
# =========================

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

balanced = pd.read_csv(f"{BALANCED_DIR}/balanced_subset_seeds.csv")
merged = balanced[["address", LABEL_COL]].merge(feat, on="address", how="left", suffixes=("", "_feat"))
assert len(merged) == len(balanced), "balanced subset address not found in feature matrix"

y_bal = merged[LABEL_COL].to_numpy()
X_hand_bal = merged[handcrafted_cols].to_numpy(dtype=float)
X_struct_bal = merged[struct_cols].to_numpy(dtype=float)
print(f"Balanced subset: {len(merged)} seeds, {dict(pd.Series(y_bal).value_counts())}")

label_encoder = LabelEncoder().fit(LABEL_ORDER)
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
FOLD_SPLITS = list(skf.split(X_hand_bal, y_bal))  # SAME split as balanced_subset_robustness.py / GNN scripts
print(f"{N_SPLITS} folds (fixed, random_state={RANDOM_STATE}), {len(TRAIN_SEEDS)} model-training seeds per condition")

# =========================
# 2. Train RF and XGBoost across the SAME 5 training seeds, SAME fixed folds
# =========================

FEATURE_SETS = {"graph_structural": X_struct_bal, "hand_crafted": X_hand_bal}
fold_rows = []

for train_seed in TRAIN_SEEDS:
    for fs_name, X in FEATURE_SETS.items():
        for model_name in ["RandomForest", "XGBoost"]:
            for fold_i, (train_idx, test_idx) in enumerate(FOLD_SPLITS):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y_bal[train_idx], y_bal[test_idx]

                if model_name == "RandomForest":
                    model = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=train_seed, n_jobs=-1)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                else:
                    model = XGBClassifier(n_estimators=300, random_state=train_seed, eval_metric="mlogloss", n_jobs=-1)
                    y_train_enc = label_encoder.transform(y_train)
                    sample_weight = compute_sample_weight("balanced", y_train_enc)
                    model.fit(X_train, y_train_enc, sample_weight=sample_weight)
                    y_pred = label_encoder.inverse_transform(model.predict(X_test))

                fold_rows.append({
                    "condition": f"{model_name}_{fs_name}", "train_seed": train_seed, "fold": fold_i,
                    "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
                })
    print(f"train_seed={train_seed}: done")

fold_df = pd.DataFrame(fold_rows)
fold_df.to_csv(OUT_DIR / "baseline_multiseed_fold_scores.csv", index=False)
print(f"\nSaved {OUT_DIR / 'baseline_multiseed_fold_scores.csv'} ({len(fold_df)} rows)")

# =========================
# 3. Seed-level aggregation (mean over 5 folds, per condition x seed)
# =========================

seed_level = fold_df.groupby(["condition", "train_seed"])["macro_f1"].mean().reset_index()
seed_level_pivot = seed_level.pivot(index="train_seed", columns="condition", values="macro_f1")
seed_level_pivot.to_csv(OUT_DIR / "baseline_multiseed_seed_level.csv")
print(f"\n=== Seed-level mean macro-F1 (mean over {N_SPLITS} folds, per training seed) ===\n{seed_level_pivot.to_string()}")

summary_rows = []
for cond in seed_level_pivot.columns:
    vals = seed_level_pivot[cond].values
    summary_rows.append({"condition": cond, "n_seeds": len(vals), "macro_f1_mean": vals.mean(), "macro_f1_std": vals.std()})
summary_df = pd.DataFrame(summary_rows).round(4)
summary_df.to_csv(OUT_DIR / "baseline_multiseed_summary.csv", index=False)
print(f"\n=== Condition summary (aggregated across {len(TRAIN_SEEDS)} training seeds) ===\n{summary_df.to_string(index=False)}")

# =========================
# 4. Configuration-matched comparisons using the same five seed labels.
#    Resulting p-values are conditional diagnostics, not population inference.
# =========================

gnn_seed_level = pd.read_csv(f"{GNN_DIR}/multiseed_seed_level.csv", index_col="train_seed")


def paired_ci(a, b):
    diff = a - b
    n = len(diff)
    mean_diff = diff.mean()
    sem = diff.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    if n > 1:
        ci_lo = mean_diff - stats.t.ppf(0.975, n - 1) * sem
        ci_hi = mean_diff + stats.t.ppf(0.975, n - 1) * sem
        t_stat, p_val = stats.ttest_rel(a, b)
    else:
        ci_lo = ci_hi = t_stat = p_val = np.nan
    return mean_diff, ci_lo, ci_hi, t_stat, p_val, (diff > 0).mean()


comparisons = []
for gnn_cond in ["GraphSAGE", "GCN"]:
    gnn_vals = gnn_seed_level[gnn_cond].reindex(TRAIN_SEEDS).values
    for baseline_cond in ["RandomForest_graph_structural", "XGBoost_graph_structural",
                           "RandomForest_hand_crafted", "XGBoost_hand_crafted"]:
        base_vals = seed_level_pivot[baseline_cond].reindex(TRAIN_SEEDS).values
        mean_diff, ci_lo, ci_hi, t_stat, p_val, pct = paired_ci(gnn_vals, base_vals)
        comparisons.append({
            "gnn": gnn_cond, "baseline": baseline_cond, "n_seeds": len(gnn_vals),
            "gnn_mean": gnn_vals.mean(), "baseline_mean": base_vals.mean(),
            "mean_diff": mean_diff, "diff_95ci_lo": ci_lo, "diff_95ci_hi": ci_hi,
            "paired_t_stat": t_stat, "paired_p_value": p_val, "pct_seeds_gnn_better": pct,
        })

comparison_df = pd.DataFrame(comparisons).round(4)
comparison_df.to_csv(OUT_DIR / "symmetric_multiseed_comparison.csv", index=False)
print(f"\n=== Fully symmetric comparison: GNN vs RF/XGBoost, BOTH sides averaged over "
      f"{len(TRAIN_SEEDS)} seeds, paired at seed/trial level ===\n{comparison_df.to_string(index=False)}")
print(f"Saved {OUT_DIR / 'symmetric_multiseed_comparison.csv'}")

print("\n=== RF/XGBoost multi-seed baseline (Section 14.6) complete ===")
