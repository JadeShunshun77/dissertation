import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from scipy import stats

# =========================
# 0. Setup
# =========================
# P2: the Week 8/9/10 paired t-tests all rest on a single 5-fold split (n=5
# per comparison) -- low statistical power, and a single split's fold
# boundaries are themselves a source of noise the tests don't account for.
# RepeatedStratifiedKFold(n_splits=5, n_repeats=10) gives 50 fold-level
# macro-F1 values per (feature_set, model) condition, drawn from 10
# independent 5-fold partitions of the same data. This does not fix small n
# (still 436 or 314 underlying samples) but it separates "the comparison is
# noisy because of few folds" from "the comparison is noisy because the
# underlying effect is genuinely small" -- and lets us report a distribution
# instead of a single point estimate.

RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 10
LABEL_COL = "label"
LABEL_ORDER = ["CEX", "DeFi", "Malicious_MEV"]
MODEL_ORDER = ["LogisticRegression", "RandomForest", "XGBoost"]
PRACTICAL_SIGNIFICANCE_THRESHOLD = 0.01  # macro-F1 points

# Optional CLI args: feat_file edges_file emb_dir out_subdir contract_dir.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIR = PROJECT_ROOT / "02_feature_engineering" / "feature_engineering"
FEAT_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else FEATURE_DIR / "seed_feature_matrix_reconstructed.csv"
EDGES_FILE = Path(sys.argv[2]) if len(sys.argv) > 2 else PROJECT_ROOT / "00_data_graph" / "data" / "main" / "raw_edges_reconstructed.csv"
EMB_DIR = Path(sys.argv[3]) if len(sys.argv) > 3 else PROJECT_ROOT / "03_unsupervised" / "node2vec_outputs" / "node2vec_reconstructed"
OUT_SUBDIR = Path(sys.argv[4]) if len(sys.argv) > 4 else PROJECT_ROOT / "04_supervised" / "outputs" / "repeated_cv"
CONTRACT_DIR = Path(sys.argv[5]) if len(sys.argv) > 5 else PROJECT_ROOT / "06_robustness_audit" / "outputs" / "contract_confound"
FEATURE_GROUPS_FILE = FEATURE_DIR / "feature_groups.json"

OUT_DIR = Path(OUT_SUBDIR)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_address(x):
    if pd.isna(x):
        return np.nan
    return str(x).strip().lower()


def repeated_cv_scores(X, y, model_order=MODEL_ORDER, n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE):
    label_encoder = LabelEncoder().fit(sorted(set(y)))
    rows = []
    for model_name in model_order:
        rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        for run_i, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
            repeat_i, fold_i = divmod(run_i, n_splits)
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if model_name == "LogisticRegression":
                model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state))])
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            elif model_name == "RandomForest":
                model = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=random_state, n_jobs=-1)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            else:
                model = XGBClassifier(n_estimators=300, random_state=random_state, eval_metric="mlogloss", n_jobs=-1)
                y_train_enc = label_encoder.transform(y_train)
                sample_weight = compute_sample_weight("balanced", y_train_enc)
                model.fit(X_train, y_train_enc, sample_weight=sample_weight)
                y_pred = label_encoder.inverse_transform(model.predict(X_test))

            rows.append({
                "model": model_name, "repeat": repeat_i, "fold": fold_i,
                "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
                "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            })
    return pd.DataFrame(rows)


def summarize_and_compare(fold_df_a, fold_df_b, name_a, name_b):
    """Pairs run i of A against run i of B (same repeat/fold index -> same
    RepeatedStratifiedKFold random_state -> identical partitions), per model."""
    rows = []
    for model_name in MODEL_ORDER:
        a = fold_df_a[fold_df_a.model == model_name].sort_values(["repeat", "fold"])["macro_f1"].values
        b = fold_df_b[fold_df_b.model == model_name].sort_values(["repeat", "fold"])["macro_f1"].values
        diff = a - b
        n = len(diff)
        mean_diff = diff.mean()
        sem_diff = diff.std(ddof=1) / np.sqrt(n)
        ci_lo, ci_hi = mean_diff - 1.96 * sem_diff, mean_diff + 1.96 * sem_diff
        t_stat, p_val = stats.ttest_rel(a, b)
        pct_practical = (np.abs(diff) >= PRACTICAL_SIGNIFICANCE_THRESHOLD).mean()
        pct_improved = (diff > 0).mean()
        rows.append({
            "comparison": f"{name_a} vs {name_b}", "model": model_name, "n_runs": n,
            "a_mean": a.mean(), "a_std": a.std(), "b_mean": b.mean(), "b_std": b.std(),
            "mean_diff": mean_diff, "diff_95ci_lo": ci_lo, "diff_95ci_hi": ci_hi,
            "paired_t_stat": t_stat, "paired_p_value": p_val,
            "pct_runs_improved": pct_improved,
            f"pct_runs_diff_over_{PRACTICAL_SIGNIFICANCE_THRESHOLD}": pct_practical,
        })
    return pd.DataFrame(rows)


# =========================
# 1. Load data (same construction as prior scripts)
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
G = nx.from_pandas_edgelist(edges, source="source", target="target", edge_attr=["tx_count"], create_using=nx.DiGraph())
pagerank_rank = pd.Series(nx.pagerank(G)).rank(pct=True)
pagerank_weighted_rank = pd.Series(nx.pagerank(G, weight="tx_count")).rank(pct=True)
feat["pagerank_rank"] = feat["address"].map(pagerank_rank).fillna(0.0)
feat["pagerank_weighted_rank"] = feat["address"].map(pagerank_weighted_rank).fillna(0.0)
for col in ["pagerank", "pagerank_weighted", "log_pagerank", "log_pagerank_weighted"]:
    if col in struct_cols:
        struct_cols.remove(col)
struct_cols += ["pagerank_rank", "pagerank_weighted_rank"]
handcrafted_cols = txn_cols + struct_cols

contract_df = pd.read_csv(Path(CONTRACT_DIR) / "seed_is_contract.csv")
feat = feat.merge(contract_df[["address", "is_contract"]], on="address", how="left")
feat["is_contract"] = feat["is_contract"].fillna(False)

emb = pd.read_csv(f"{EMB_DIR}/seed_node2vec_embeddings.csv")
emb_cols = [c for c in emb.columns if c.startswith("emb_")]
active = feat[feat["is_active"]].copy()
data = active.merge(emb[["address"] + emb_cols], on="address", how="inner")

y_main = data[LABEL_COL].to_numpy()
X_coarse = data[["is_contract", "has_out_edge", "out_tx_count"]].astype(float).to_numpy()
X_txn = data[txn_cols].to_numpy(dtype=float)
X_hand = data[handcrafted_cols].to_numpy(dtype=float)
X_emb = data[emb_cols].to_numpy(dtype=float)
X_combined = np.hstack([X_hand, X_emb])

eoa = data[~data["is_contract"]].copy()
y_eoa = eoa[LABEL_COL].to_numpy()
X_eoa_txn = eoa[txn_cols].to_numpy(dtype=float)
X_eoa_hand = eoa[handcrafted_cols].to_numpy(dtype=float)
X_eoa_emb = eoa[emb_cols].to_numpy(dtype=float)
X_eoa_combined = np.hstack([X_eoa_hand, X_eoa_emb])

# =========================
# 2. Run RepeatedStratifiedKFold for each condition
# =========================

conditions = {
    "main_coarse": (X_coarse, y_main), "main_transaction": (X_txn, y_main),
    "main_hand_crafted": (X_hand, y_main), "main_hand_crafted_embedding": (X_combined, y_main),
    "eoa_transaction": (X_eoa_txn, y_eoa), "eoa_hand_crafted": (X_eoa_hand, y_eoa),
    "eoa_hand_crafted_embedding": (X_eoa_combined, y_eoa),
}

fold_results = {}
for name, (X, y) in conditions.items():
    print(f"Running repeated CV: {name} (n={len(y)}) ...")
    fold_results[name] = repeated_cv_scores(X, y)
    print(f"  done, {len(fold_results[name])} runs")

all_fold_df = pd.concat([df.assign(condition=name) for name, df in fold_results.items()], ignore_index=True)
all_fold_df.to_csv(OUT_DIR / "repeated_cv_fold_scores.csv", index=False)

summary = all_fold_df.groupby(["condition", "model"])[["macro_f1", "balanced_accuracy"]].agg(["mean", "std"])
summary.columns = ["_".join(c) for c in summary.columns]
summary = summary.reset_index().round(4)
summary.to_csv(OUT_DIR / "repeated_cv_summary.csv", index=False)
print(f"\n=== Repeated CV summary (10x5=50 runs per condition) ===\n{summary.to_string(index=False)}")

# =========================
# 3. Paired comparisons with 95% CI + practical-significance threshold
# =========================

comparisons = [
    ("main_hand_crafted", "main_transaction"),
    ("main_hand_crafted_embedding", "main_hand_crafted"),
    ("main_hand_crafted", "main_coarse"),
    ("eoa_hand_crafted_embedding", "eoa_hand_crafted"),
    ("eoa_hand_crafted", "eoa_transaction"),
]
diff_rows = []
for a_name, b_name in comparisons:
    diff_rows.append(summarize_and_compare(fold_results[a_name], fold_results[b_name], a_name, b_name))
diff_df = pd.concat(diff_rows, ignore_index=True).round(4)
diff_df.to_csv(OUT_DIR / "repeated_cv_difference_ci.csv", index=False)
print(f"\n=== Paired differences with 95% CI (n=50 runs each) ===\n{diff_df.to_string(index=False)}")

# =========================
# 4. Difference distribution plot
# =========================

plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#c3c2b7",
                      "axes.labelcolor": "#0b0b0b", "text.color": "#0b0b0b", "xtick.color": "#52514e",
                      "ytick.color": "#52514e", "grid.color": "#e1e0d9", "font.size": 10})

fig, axes = plt.subplots(1, len(comparisons), figsize=(4 * len(comparisons), 4.5), sharey=True)
for ax, (a_name, b_name) in zip(axes, comparisons):
    box_data = []
    for model_name in MODEL_ORDER:
        a = fold_results[a_name][fold_results[a_name].model == model_name].sort_values(["repeat", "fold"])["macro_f1"].values
        b = fold_results[b_name][fold_results[b_name].model == model_name].sort_values(["repeat", "fold"])["macro_f1"].values
        box_data.append(a - b)
    ax.boxplot(box_data, tick_labels=MODEL_ORDER)
    ax.axhline(0, color="#c3c2b7", linestyle="--", linewidth=1)
    ax.set_title(f"{a_name}\n− {b_name}", fontsize=8)
    ax.tick_params(axis="x", rotation=30, labelsize=7)
    ax.grid(axis="y", linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)
axes[0].set_ylabel("Macro-F1 difference (50 runs)")
fig.suptitle("Repeated CV: paired macro-F1 differences (10x5-fold)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(OUT_DIR / "repeated_cv_difference_distribution.png", dpi=150)
plt.close(fig)
print(f"Saved {OUT_DIR / 'repeated_cv_difference_distribution.png'}")

# =========================
# 5. Repeat-level correction (n=10 independent repeats, not n=50 correlated folds)
# =========================
# The 5 folds within one repeat share heavily overlapping training sets (each
# fold's training set is 4/5 of the same 436/314 samples), so they are not
# independent observations. Treating all 50 fold-level runs as independent in
# Section 3's paired t-test overstates precision. This aggregates to one
# mean-macro-F1 per repeat first (n=10 genuinely independent observations
# per model), then re-runs the paired comparison -- the statistically
# defensible version of the Section 3 table.

repeat_level_rows = []
for a_name, b_name in comparisons:
    for model_name in MODEL_ORDER:
        a = fold_results[a_name][fold_results[a_name].model == model_name].groupby("repeat")["macro_f1"].mean().sort_index().values
        b = fold_results[b_name][fold_results[b_name].model == model_name].groupby("repeat")["macro_f1"].mean().sort_index().values
        diff = a - b
        n = len(diff)
        mean_diff = diff.mean()
        sem = diff.std(ddof=1) / np.sqrt(n)
        ci_lo, ci_hi = mean_diff - stats.t.ppf(0.975, n - 1) * sem, mean_diff + stats.t.ppf(0.975, n - 1) * sem
        t_stat, p_val = stats.ttest_rel(a, b)
        repeat_level_rows.append({
            "comparison": f"{a_name} vs {b_name}", "model": model_name, "n_independent_repeats": n,
            "mean_diff": mean_diff, "diff_95ci_lo": ci_lo, "diff_95ci_hi": ci_hi,
            "paired_t_stat": t_stat, "paired_p_value": p_val, "pct_repeats_improved": (diff > 0).mean(),
        })

repeat_level_df = pd.DataFrame(repeat_level_rows).round(4)
repeat_level_df.to_csv(OUT_DIR / "repeated_cv_repeat_level_ci.csv", index=False)
print(f"\n=== Repeat-level correction (n=10 independent repeats, avoids fold pseudo-replication) ===")
print(repeat_level_df.to_string(index=False))
print(f"Saved {OUT_DIR / 'repeated_cv_repeat_level_ci.csv'}")

print("\n=== P2 repeated CV complete ===")
