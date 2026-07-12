import json
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import StratifiedKFold
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
# Week 9: feature-group ablation (transaction_behaviour vs graph_structural vs
# combined hand-crafted) answers RQ1. SHAP answers "which specific features
# drive predictions" and extends the Week 8 Section 10 group-contribution
# diagnostic down to individual-feature level. Embeddings are deliberately
# excluded from this script -- Week 8 already established embeddings add no
# value for RF and only weak/non-robust value for LR/XGBoost, so this
# ablation isolates the question the Plan actually asks: within hand-crafted
# features, how much does each group contribute.

RANDOM_STATE = 42
N_SPLITS = 5
LABEL_COL = "label"
LABEL_ORDER = ["CEX", "DeFi", "Malicious_MEV"]
MODEL_ORDER = ["LogisticRegression", "RandomForest", "XGBoost"]
MODEL_COLORS = {"LogisticRegression": "#2a78d6", "RandomForest": "#1baf7a", "XGBoost": "#eb6834"}
GROUP_ORDER = ["transaction_behaviour", "graph_structural", "combined"]
GROUP_COLORS = {"transaction_behaviour": "#2a78d6", "graph_structural": "#1baf7a", "combined": "#6b4fa0"}

OUT_DIR = Path("ablation_shap_outputs")
OUT_DIR.mkdir(exist_ok=True)

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
# 1. Load hand-crafted features + feature groups
# =========================

feat = pd.read_csv("seed_feature_matrix.csv")
with open("feature_groups.json") as f:
    feature_groups = json.load(f)

txn_cols = list(feature_groups["transaction_behaviour"])
struct_cols = list(feature_groups["graph_structural"])

# =========================
# 2. Pagerank rank transform (same as classification.py Section 2)
# =========================
# Kept identical to the Week 8 script so the "combined" condition here is the
# exact same 28-column hand_crafted feature set already evaluated in Week 8
# (classification_outputs/classification_summary.csv), not a re-derivation
# that could silently drift.

edges = pd.read_csv("raw_edges.csv")
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

combined_cols = txn_cols + struct_cols
print(f"transaction_behaviour: {len(txn_cols)} features | graph_structural: {len(struct_cols)} features "
      f"| combined: {len(combined_cols)} features")

# =========================
# 3. Restrict to active seeds (same 436 as Week 8, embeddings not used here)
# =========================

data = feat[feat["is_active"]].copy()
print(f"Ablation sample: {len(data)} active labelled seeds")
print(data[LABEL_COL].value_counts().reindex(LABEL_ORDER))

X_txn = data[txn_cols].to_numpy(dtype=float)
X_struct = data[struct_cols].to_numpy(dtype=float)
X_combined = data[combined_cols].to_numpy(dtype=float)
y = data[LABEL_COL].to_numpy()

feature_sets = {"transaction_behaviour": X_txn, "graph_structural": X_struct, "combined": X_combined}

# =========================
# 4. Model builders (identical to classification.py Section 4)
# =========================

def build_lr():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])


def build_rf():
    return RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)


def build_xgb():
    return XGBClassifier(n_estimators=300, random_state=RANDOM_STATE, eval_metric="mlogloss", n_jobs=-1)


MODEL_BUILDERS = {"LogisticRegression": build_lr, "RandomForest": build_rf, "XGBoost": build_xgb}
label_encoder = LabelEncoder().fit(LABEL_ORDER)

# =========================
# 5. Stratified 5-fold CV for every (feature group, model) combination
# =========================
# Same random_state as classification.py, so fold membership over these 436
# seeds is identical to Week 8's folds -- the "combined" column here is
# directly paired-comparable against Week 8's hand_crafted results too.

fold_rows = []
for fs_name, X in feature_sets.items():
    for model_name, builder in MODEL_BUILDERS.items():
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = builder()
            if model_name == "XGBoost":
                y_train_enc = label_encoder.transform(y_train)
                sample_weight = compute_sample_weight("balanced", y_train_enc)
                model.fit(X_train, y_train_enc, sample_weight=sample_weight)
                y_pred = label_encoder.inverse_transform(model.predict(X_test))
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            fold_rows.append({
                "feature_group": fs_name, "model": model_name, "fold": fold_i,
                "macro_f1": f1_score(y_test, y_pred, average="macro"),
                "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            })
        print(f"Done: {fs_name} / {model_name}")

fold_df = pd.DataFrame(fold_rows)
fold_df.to_csv(OUT_DIR / "ablation_cv_fold_scores.csv", index=False)

# =========================
# 6. Summary table
# =========================

summary = (
    fold_df.groupby(["feature_group", "model"])[["macro_f1", "balanced_accuracy"]]
    .agg(["mean", "std"])
)
summary.columns = ["_".join(c) for c in summary.columns]
summary = summary.reset_index().round(4)
summary["feature_group"] = pd.Categorical(summary["feature_group"], GROUP_ORDER, ordered=True)
summary["model"] = pd.Categorical(summary["model"], MODEL_ORDER, ordered=True)
summary = summary.sort_values(["feature_group", "model"]).reset_index(drop=True)
summary.to_csv(OUT_DIR / "ablation_summary.csv", index=False)
print(f"\n=== Feature-group ablation: 5-fold CV summary ===\n{summary.to_string(index=False)}")

# =========================
# 7. Paired tests per model: does adding graph_structural to transaction_behaviour
# (i.e. combined vs transaction_behaviour alone) actually help, and same for
# combined vs graph_structural alone?
# =========================

pair_specs = [("combined", "transaction_behaviour"), ("combined", "graph_structural")]
sig_rows = []
for a_name, b_name in pair_specs:
    for model_name in MODEL_ORDER:
        a = fold_df[(fold_df.feature_group == a_name) & (fold_df.model == model_name)].sort_values("fold")["macro_f1"].values
        b = fold_df[(fold_df.feature_group == b_name) & (fold_df.model == model_name)].sort_values("fold")["macro_f1"].values
        diff = a - b
        t_stat, p_val = stats.ttest_rel(a, b)
        sig_rows.append({
            "comparison": f"{a_name} vs {b_name}", "model": model_name,
            "mean_diff": diff.mean(), "paired_t_stat": t_stat, "paired_p_value": p_val,
            "n_folds_improved": int((diff > 0).sum()),
        })

paired_df = pd.DataFrame(sig_rows).round(4)
paired_df.to_csv(OUT_DIR / "ablation_paired_test.csv", index=False)
print(f"\n=== Paired tests (n=5 folds, low power -- indicative only) ===\n{paired_df.to_string(index=False)}")

# =========================
# 8. Bar chart: macro-F1 by feature group x model
# =========================

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(GROUP_ORDER))
width = 0.25
for i, model_name in enumerate(MODEL_ORDER):
    sub = summary[summary["model"] == model_name].set_index("feature_group").reindex(GROUP_ORDER)
    ax.bar(x + (i - 1) * width, sub["macro_f1_mean"], width,
           yerr=sub["macro_f1_std"], capsize=3,
           color=MODEL_COLORS[model_name], label=model_name)

ax.set_xticks(x)
ax.set_xticklabels(GROUP_ORDER)
ax.set_ylabel("Macro-F1 (5-fold CV mean ± std)")
ax.set_title("Feature-group ablation: transaction vs structural vs combined")
ax.legend(title="Model", frameon=False)
ax.grid(axis="y", linewidth=0.6, alpha=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT_DIR / "ablation_macro_f1.png", dpi=150)
plt.close(fig)
print(f"Saved {OUT_DIR / 'ablation_macro_f1.png'}")

# =========================
# 9. SHAP: which individual features drive predictions, and does the
# per-feature picture agree with the group-level ablation above?
# =========================
# RF TreeExplainer on the full 436-sample combined (hand_crafted) feature set.
# This is an interpretability diagnostic fit on all available labelled data
# (same convention as classification.py Section 10), not a held-out
# performance claim -- CV in Sections 5-7 above already covers generalisation.

rf_full = build_rf()
rf_full.fit(X_combined, y)

explainer = shap.TreeExplainer(rf_full)
shap_values = explainer.shap_values(X_combined)

# shap>=0.45 TreeExplainer multiclass output is (n_samples, n_features, n_classes)
shap_values = np.asarray(shap_values)
if shap_values.ndim == 3 and shap_values.shape[-1] == len(LABEL_ORDER):
    per_class_mean_abs = np.abs(shap_values).mean(axis=0)  # (n_features, n_classes)
else:
    # older API: list of (n_samples, n_features) arrays, one per class
    per_class_mean_abs = np.stack([np.abs(sv).mean(axis=0) for sv in shap_values], axis=1)

mean_abs_shap_overall = per_class_mean_abs.mean(axis=1)  # averaged across classes

group_lookup = {c: "transaction_behaviour" for c in txn_cols}
group_lookup.update({c: "graph_structural" for c in struct_cols})

shap_summary = pd.DataFrame({
    "feature": combined_cols,
    "group": [group_lookup[c] for c in combined_cols],
    "mean_abs_shap": mean_abs_shap_overall,
})
for i, label in enumerate(LABEL_ORDER):
    shap_summary[f"mean_abs_shap_{label}"] = per_class_mean_abs[:, i]

shap_summary = shap_summary.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
shap_summary.round(5).to_csv(OUT_DIR / "shap_feature_importance.csv", index=False)
print(f"\n=== Top 10 features by mean |SHAP| (RF, combined hand-crafted set, all 436 seeds) ===")
print(shap_summary.head(10)[["feature", "group", "mean_abs_shap"]].to_string(index=False))

shap_group_mass = shap_summary.groupby("group")["mean_abs_shap"].sum()
shap_group_share = (shap_group_mass / shap_group_mass.sum()).round(4)
print(f"\nSHAP mass by group: {shap_group_share.to_dict()}")
print(f"(feature-count share: transaction_behaviour={len(txn_cols)/len(combined_cols):.3f}, "
      f"graph_structural={len(struct_cols)/len(combined_cols):.3f})")

fig, ax = plt.subplots(figsize=(7, 6))
top_n = 15
top_feats = shap_summary.head(top_n).iloc[::-1]
colors = [GROUP_COLORS[g] for g in top_feats["group"]]
ax.barh(top_feats["feature"], top_feats["mean_abs_shap"], color=colors)
ax.set_xlabel("Mean |SHAP value| (averaged across 3 classes)")
ax.set_title(f"Top {top_n} features by SHAP importance (RF, hand-crafted set)")
ax.grid(axis="x", linewidth=0.6, alpha=0.6)
ax.set_axisbelow(True)
handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS[g]) for g in GROUP_ORDER[:2]]
ax.legend(handles, GROUP_ORDER[:2], loc="lower right", frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(OUT_DIR / "shap_top_features.png", dpi=150)
plt.close(fig)
print(f"Saved {OUT_DIR / 'shap_top_features.png'}")

print("\n=== Week 9 feature-group ablation + SHAP complete ===")
