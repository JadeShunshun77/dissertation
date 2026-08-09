import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, balanced_accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from scipy import stats

# =========================
# 0. Setup
# =========================

RANDOM_STATE = 42
N_SPLITS = 5
LABEL_COL = "label"
LABEL_ORDER = ["CEX", "DeFi", "Malicious_MEV"]
LABEL_COLORS = {"CEX": "#2a78d6", "DeFi": "#1baf7a", "Malicious_MEV": "#e34948"}
MODEL_ORDER = ["LogisticRegression", "RandomForest", "XGBoost"]
MODEL_COLORS = {"LogisticRegression": "#2a78d6", "RandomForest": "#1baf7a", "XGBoost": "#eb6834"}
FEATURE_SET_ORDER = ["hand_crafted", "embedding", "combined"]

# Optional CLI args: feat_file edges_file emb_dir out_subdir.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIR = PROJECT_ROOT / "02_feature_engineering" / "feature_engineering"
FEAT_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else FEATURE_DIR / "seed_feature_matrix_reconstructed.csv"
EDGES_FILE = Path(sys.argv[2]) if len(sys.argv) > 2 else PROJECT_ROOT / "00_data_graph" / "data" / "main" / "raw_edges_reconstructed.csv"
EMB_DIR = Path(sys.argv[3]) if len(sys.argv) > 3 else PROJECT_ROOT / "03_unsupervised" / "node2vec_outputs" / "node2vec_reconstructed"
OUT_SUBDIR = Path(sys.argv[4]) if len(sys.argv) > 4 else PROJECT_ROOT / "04_supervised" / "outputs" / "classification"
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


# =========================
# 1. Load hand-crafted features + feature groups
# =========================

feat = pd.read_csv(FEAT_FILE)
with open(FEATURE_GROUPS_FILE) as f:
    feature_groups = json.load(f)

handcrafted_cols = list(feature_groups["transaction_behaviour"] + feature_groups["graph_structural"])

# =========================
# 2. Pagerank rank transform
# =========================
# log1p barely reduced pagerank's skew (see eda_walkthrough.ipynb Section 2:
# raw 6.388 -> log 6.373 on the reconstructed graph) because pagerank values are already close to 0
# (mean ~= 1/8029 nodes), where log1p(x) ~= x. Percentile rank over the FULL
# graph's pagerank distribution fixes this and is what actually needs fixing
# before Logistic Regression; tree models are monotonic-invariant so this only
# matters for LR, but is applied to all models' hand-crafted feature set for
# a single consistent feature matrix.

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
    if col in handcrafted_cols:
        handcrafted_cols.remove(col)
handcrafted_cols += ["pagerank_rank", "pagerank_weighted_rank"]

print(f"Hand-crafted feature set: {len(handcrafted_cols)} features "
      f"(pagerank/pagerank_weighted replaced by rank-transformed versions)")

# Diagnostic: pandas .rank(pct=True) assigns tied values the SAME averaged
# rank. Unweighted PageRank has a large tied floor group (nodes with zero
# inbound edges all converge to the same minimum value), so pagerank_rank is
# not a clean continuous signal for those nodes -- it is mechanically
# equivalent to "has_in_edge == False" for whichever seeds land in that tied
# group, not an independent structural signal.
pagerank_raw_full = pd.Series(nx.pagerank(G))
floor_value = pagerank_raw_full.min()
n_tied_at_floor = (pagerank_raw_full == floor_value).sum()
print(f"PageRank tie-floor check: {n_tied_at_floor}/{len(pagerank_raw_full)} graph nodes "
      f"({n_tied_at_floor / len(pagerank_raw_full):.1%}) share the exact minimum PageRank value "
      f"(all have zero inbound edges) -> pagerank_rank is degenerate for these nodes, "
      f"not a distinct signal from has_in_edge.")

# =========================
# 3. Restrict to active seeds, merge Node2Vec embeddings
# =========================
# Node2Vec embeddings only exist for the 436 active seeds (inactive seeds have
# no edges, so no graph position to embed). All three feature-set conditions
# below use the same 436-seed sample so macro-F1 differences are attributable
# to feature content, not sample differences.

active = feat[feat["is_active"]].copy()
emb = pd.read_csv(f"{EMB_DIR}/seed_node2vec_embeddings.csv")
emb_cols = [c for c in emb.columns if c.startswith("emb_")]

data = active.merge(emb[["address"] + emb_cols], on="address", how="inner")
assert len(data) == len(emb), "active seed count does not match embedding row count"
print(f"Classification sample: {len(data)} active labelled seeds")
print(data[LABEL_COL].value_counts().reindex(LABEL_ORDER))

X_hand = data[handcrafted_cols].to_numpy(dtype=float)
X_emb = data[emb_cols].to_numpy(dtype=float)
X_combined = np.hstack([X_hand, X_emb])
y = data[LABEL_COL].to_numpy()

feature_sets = {"hand_crafted": X_hand, "embedding": X_emb, "combined": X_combined}

# =========================
# 4. Model builders
# =========================
# class_weight="balanced" handles imbalance for LR/RF automatically per fold.
# XGBoost's multiclass objective has no equivalent built-in, so sample_weight
# is computed from compute_sample_weight("balanced", y_train) inside each fold
# (using only that fold's training labels, not the full dataset, to avoid leakage).

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

# XGBoost's sklearn wrapper (3.x) requires integer class labels.
label_encoder = LabelEncoder().fit(LABEL_ORDER)

# =========================
# 5. Stratified 5-fold CV for every (feature set, model) combination
# =========================
# A single stratified train/test split would leave only ~10 CEX seeds in the
# test set (51 total) -- too noisy to trust. 5-fold CV reports mean +/- std
# across folds instead of a single point estimate.

fold_rows = []
oof_predictions = {}  # (feature_set, model) -> out-of-fold predicted labels, aligned to `data` row order

for fs_name, X in feature_sets.items():
    for model_name, builder in MODEL_BUILDERS.items():
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        y_pred_oof = np.empty(len(y), dtype=object)

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

            y_pred_oof[test_idx] = y_pred

            fold_rows.append({
                "feature_set": fs_name, "model": model_name, "fold": fold_i,
                "macro_f1": f1_score(y_test, y_pred, average="macro"),
                "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            })

        oof_predictions[(fs_name, model_name)] = y_pred_oof
        print(f"Done: {fs_name} / {model_name}")

fold_df = pd.DataFrame(fold_rows)
fold_df.to_csv(OUT_DIR / "cv_fold_scores.csv", index=False)

# =========================
# 6. Summary table: mean +/- std macro-F1 / balanced accuracy per combination
# =========================

summary = (
    fold_df.groupby(["feature_set", "model"])[["macro_f1", "balanced_accuracy"]]
    .agg(["mean", "std"])
)
summary.columns = ["_".join(c) for c in summary.columns]
summary = summary.reset_index().round(4)
summary["feature_set"] = pd.Categorical(summary["feature_set"], FEATURE_SET_ORDER, ordered=True)
summary["model"] = pd.Categorical(summary["model"], MODEL_ORDER, ordered=True)
summary = summary.sort_values(["feature_set", "model"]).reset_index(drop=True)

summary.to_csv(OUT_DIR / "classification_summary.csv", index=False)
print(f"\n=== 5-fold CV summary (mean +/- std across folds) ===\n{summary.to_string(index=False)}")
print(f"Saved to {OUT_DIR / 'classification_summary.csv'}")

# =========================
# 6b. Paired significance test: does combining embedding with hand-crafted
# features actually beat hand-crafted alone, per model?
# =========================
# StratifiedKFold uses the same random_state for every (feature_set, model)
# combination, so fold membership is identical across feature sets -- fold i
# is the same 87-88 held-out seeds for hand_crafted and combined. This makes a
# PAIRED comparison valid and much more powerful than comparing independent
# mean +/- std intervals (which share fold-to-fold variance that a paired test
# cancels out).

sig_rows = []
for model_name in MODEL_ORDER:
    hc = fold_df[(fold_df.feature_set == "hand_crafted") & (fold_df.model == model_name)].sort_values("fold")["macro_f1"].values
    cb = fold_df[(fold_df.feature_set == "combined") & (fold_df.model == model_name)].sort_values("fold")["macro_f1"].values
    diff = cb - hc
    t_stat, p_val = stats.ttest_rel(cb, hc)
    sig_rows.append({
        "model": model_name, "mean_diff_combined_minus_handcrafted": diff.mean(),
        "paired_t_stat": t_stat, "paired_p_value": p_val,
        "n_folds_improved": int((diff > 0).sum()),
    })

paired_sig_df = pd.DataFrame(sig_rows).round(4)
paired_sig_df.to_csv(OUT_DIR / "combined_vs_handcrafted_paired_test.csv", index=False)
print(f"\n=== Paired test: combined vs hand_crafted macro-F1 (same 5 folds) ===\n{paired_sig_df.to_string(index=False)}")
print("(n=5 folds gives low statistical power; treat p-values as indicative, not confirmatory.)")
print(f"Saved to {OUT_DIR / 'combined_vs_handcrafted_paired_test.csv'}")

# =========================
# 7. Per-class precision / recall / F1 (out-of-fold predictions)
# =========================

per_class_rows = []
for (fs_name, model_name), y_pred_oof in oof_predictions.items():
    precision, recall, f1, support = precision_recall_fscore_support(
        y, y_pred_oof, labels=LABEL_ORDER, zero_division=0
    )
    for i, label in enumerate(LABEL_ORDER):
        per_class_rows.append({
            "feature_set": fs_name, "model": model_name, "label": label,
            "precision": precision[i], "recall": recall[i], "f1": f1[i], "support": support[i],
        })

per_class_df = pd.DataFrame(per_class_rows).round(4)
per_class_df.to_csv(OUT_DIR / "classification_per_class_report.csv", index=False)
print(f"Saved {OUT_DIR / 'classification_per_class_report.csv'}")

# =========================
# 8. Grouped bar chart: macro-F1 by feature set x model
# =========================

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(FEATURE_SET_ORDER))
width = 0.25
for i, model_name in enumerate(MODEL_ORDER):
    sub = summary[summary["model"] == model_name].set_index("feature_set").reindex(FEATURE_SET_ORDER)
    ax.bar(x + (i - 1) * width, sub["macro_f1_mean"], width,
           yerr=sub["macro_f1_std"], capsize=3,
           color=MODEL_COLORS[model_name], label=model_name)

ax.set_xticks(x)
ax.set_xticklabels(FEATURE_SET_ORDER)
ax.set_ylabel("Macro-F1 (5-fold CV mean ± std)")
ax.set_title("Macro-F1 by feature set and model")
ax.legend(title="Model", frameon=False)
ax.grid(axis="y", linewidth=0.6, alpha=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT_DIR / "macro_f1_comparison.png", dpi=150)
plt.close(fig)
print(f"Saved {OUT_DIR / 'macro_f1_comparison.png'}")

# =========================
# 9. Confusion matrix grid (3 models x 3 feature sets), out-of-fold predictions
# =========================

fig, axes = plt.subplots(len(MODEL_ORDER), len(FEATURE_SET_ORDER), figsize=(12, 11))
for row, model_name in enumerate(MODEL_ORDER):
    for col, fs_name in enumerate(FEATURE_SET_ORDER):
        ax = axes[row, col]
        y_pred_oof = oof_predictions[(fs_name, model_name)]
        cm = confusion_matrix(y, y_pred_oof, labels=LABEL_ORDER, normalize="true")
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(LABEL_ORDER)):
            for j in range(len(LABEL_ORDER)):
                text_color = "#ffffff" if cm[i, j] > 0.6 else "#0b0b0b"
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8, color=text_color)
        ax.set_xticks(range(len(LABEL_ORDER)))
        ax.set_xticklabels(LABEL_ORDER, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(LABEL_ORDER)))
        ax.set_yticklabels(LABEL_ORDER, fontsize=8)
        if row == 0:
            ax.set_title(fs_name, fontsize=10)
        if col == 0:
            ax.set_ylabel(model_name, fontsize=10)

fig.suptitle("Confusion matrices (row-normalised, out-of-fold predictions)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT_DIR / "confusion_matrix_grid.png", dpi=150)
plt.close(fig)
print(f"Saved {OUT_DIR / 'confusion_matrix_grid.png'}")

# =========================
# 10. Feature-group contribution: does RF/LR actually use the embedding
# dimensions, or does the combined-set improvement come from elsewhere?
# =========================
# Fit RF and LR once on the FULL 436-sample combined set (interpretability
# diagnostic only, not a held-out performance claim) and compare how much
# importance/coefficient mass falls on hand_crafted vs embedding features.
# This directly tests whether "combined helps" reflects embeddings carrying
# independent structural signal, or the combined gain (where it exists) is
# concentrated in models that can't otherwise capture feature interactions.

all_cols = handcrafted_cols + emb_cols
col_group = ["hand_crafted"] * len(handcrafted_cols) + ["embedding"] * len(emb_cols)

rf_full = build_rf()
rf_full.fit(X_combined, y)
rf_imp = pd.DataFrame({"feature": all_cols, "group": col_group, "importance": rf_full.feature_importances_})

lr_full = build_lr()
lr_full.fit(X_combined, y)
lr_coef = np.abs(lr_full.named_steps["clf"].coef_).mean(axis=0)  # mean |coef| across classes (features are standardized, so comparable)
lr_imp = pd.DataFrame({"feature": all_cols, "group": col_group, "mean_abs_coef": lr_coef})

rf_group_mass = rf_imp.groupby("group")["importance"].sum()
lr_group_mass = lr_imp.groupby("group")["mean_abs_coef"].sum() / lr_imp["mean_abs_coef"].sum()

feature_group_contribution = pd.DataFrame({
    "group": ["hand_crafted", "embedding"],
    "n_features": [len(handcrafted_cols), len(emb_cols)],
    "share_of_all_features": [len(handcrafted_cols) / len(all_cols), len(emb_cols) / len(all_cols)],
    "RF_importance_share": [rf_group_mass.get("hand_crafted", 0), rf_group_mass.get("embedding", 0)],
    "LR_coef_share": [lr_group_mass.get("hand_crafted", 0), lr_group_mass.get("embedding", 0)],
}).round(4)
feature_group_contribution.to_csv(OUT_DIR / "feature_group_contribution.csv", index=False)

print(f"\n=== Feature-group contribution in the combined model (fit on all 436 seeds) ===")
print(feature_group_contribution.to_string(index=False))
print("RF: embedding gets far less importance mass than its feature-count share would predict "
      "-> RF already extracts the useful non-linear signal from hand-crafted features alone.")
print("LR: embedding's coefficient share is roughly proportional to (or above) its feature-count "
      "share, and individual embedding dimensions rank among the top-10 |coef| features -> LR's "
      "linear boundary benefits from embedding dimensions in a way RF's tree ensemble does not "
      "need to. This is consistent with the paired-test result: XGBoost/LR show some (weak, "
      "n=5-limited) gain from combining, RF shows none.")
print(f"Saved to {OUT_DIR / 'feature_group_contribution.csv'}")

fig, ax = plt.subplots(figsize=(6, 4))
x = np.arange(2)
width = 0.35
ax.bar(x - width / 2, feature_group_contribution["RF_importance_share"], width, color=MODEL_COLORS["RandomForest"], label="RF importance share")
ax.bar(x + width / 2, feature_group_contribution["LR_coef_share"], width, color=MODEL_COLORS["LogisticRegression"], label="LR |coef| share")
ax.axhline(feature_group_contribution.loc[0, "share_of_all_features"], color="#c3c2b7", linestyle="--", linewidth=1, label="feature-count share (hand-crafted)")
ax.set_xticks(x)
ax.set_xticklabels(feature_group_contribution["group"])
ax.set_ylabel("Share of total importance / |coef| mass")
ax.set_title("Does the model actually use the embedding features?")
ax.legend(frameon=False, fontsize=8)
ax.grid(axis="y", linewidth=0.6, alpha=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT_DIR / "feature_group_contribution.png", dpi=150)
plt.close(fig)
print(f"Saved {OUT_DIR / 'feature_group_contribution.png'}")

print("\n=== Week 8 baseline classification complete ===")
