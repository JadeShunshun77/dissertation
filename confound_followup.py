import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, precision_recall_fscore_support, confusion_matrix,
)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# =========================
# 0. Setup
# =========================
# P1 follow-up to contract_confound_check.py's headline finding (DeFi is
# 100% contracts, 3.9% for CEX, 11.4% for Malicious_MEV). Three sub-experiments:
#   10.8a) EOA-only feature-group ablation -- does the Week 9 "graph features
#          add nothing" conclusion survive once the contract/EOA confound is
#          fully removed?
#   10.8b) Contract-only DeFi vs Malicious_MEV -- among addresses that are ALL
#          contracts, is there still a class-specific signal, or does
#          performance collapse to near-chance once account-type is held fixed?
#   10.8c) Per-class breakdown of the outbound-feature-removal experiment --
#          which class's F1 actually moves when out_* features are dropped?

RANDOM_STATE = 42
N_SPLITS = 5
LABEL_COL = "label"
LABEL_ORDER = ["CEX", "DeFi", "Malicious_MEV"]
MODEL_ORDER = ["LogisticRegression", "RandomForest", "XGBoost"]

OUT_DIR = Path("contract_confound_outputs")
OUT_DIR.mkdir(exist_ok=True)


def clean_address(x):
    if pd.isna(x):
        return np.nan
    return str(x).strip().lower()


def build_cv_fold_scores(X, y, model_order=MODEL_ORDER, n_splits=N_SPLITS, random_state=RANDOM_STATE):
    """Returns (fold_df, oof_pred) -- fold_df has per-fold macro_f1/balanced_accuracy
    per model; oof_pred[model_name] is an out-of-fold prediction array aligned to y."""
    label_encoder = LabelEncoder().fit(sorted(set(y)))
    rows = []
    oof_pred = {}
    for model_name in model_order:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        y_pred_oof = np.empty(len(y), dtype=object)
        for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
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

            y_pred_oof[test_idx] = y_pred
            rows.append({
                "model": model_name, "fold": fold_i,
                "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
                "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            })
        oof_pred[model_name] = y_pred_oof
    return pd.DataFrame(rows), oof_pred


# =========================
# 1. Load data (same construction as classification.py / feature_ablation_shap.py)
# =========================

feat = pd.read_csv("seed_feature_matrix.csv")
with open("feature_groups.json") as f:
    feature_groups = json.load(f)
txn_cols = list(feature_groups["transaction_behaviour"])
struct_cols = list(feature_groups["graph_structural"])

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
handcrafted_cols = txn_cols + struct_cols

contract_df = pd.read_csv(OUT_DIR / "seed_is_contract.csv")
feat = feat.merge(contract_df[["address", "is_contract", "bytecode_length", "rpc_ok"]], on="address", how="left")
feat["is_contract"] = feat["is_contract"].fillna(False)

emb = pd.read_csv("embedding_outputs/seed_node2vec_embeddings.csv")
emb_cols = [c for c in emb.columns if c.startswith("emb_")]

active = feat[feat["is_active"]].copy()
data = active.merge(emb[["address"] + emb_cols], on="address", how="inner")

# =========================
# 2. RPC query log (P1 technical requirement)
# =========================

query_log = {
    "rpc_endpoint": "https://ethereum-rpc.publicnode.com",
    "method": "eth_getCode",
    "empty_bytecode_rule": 'result == "0x" (or length <= 2) => EOA; otherwise => contract',
    "query_run_recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    "n_addresses_queried": int(len(contract_df)),
    "n_rpc_success": int(contract_df["rpc_ok"].sum()),
    "n_rpc_failed_after_retries": int((~contract_df["rpc_ok"]).sum()),
    "retry_policy": "up to 3 attempts per address, 1.5s * attempt_number backoff",
    "coverage_pct": round(100 * contract_df["rpc_ok"].mean(), 2),
}
with open(OUT_DIR / "contract_query_log.json", "w") as f:
    json.dump(query_log, f, indent=2)
print(f"=== RPC query log ===\n{json.dumps(query_log, indent=2)}")
print(f"Saved {OUT_DIR / 'contract_query_log.json'}")

# =========================
# 3. EOA-only feature-group ablation (10.8a)
# =========================

eoa = data[~data["is_contract"]].copy()
print(f"\nEOA-only sample (active, embeddings available): {len(eoa)} seeds, {dict(eoa[LABEL_COL].value_counts())}")

X_txn = eoa[txn_cols].to_numpy(dtype=float)
X_struct = eoa[struct_cols].to_numpy(dtype=float)
X_hand = eoa[handcrafted_cols].to_numpy(dtype=float)
X_emb = eoa[emb_cols].to_numpy(dtype=float)
X_combined = np.hstack([X_hand, X_emb])
y_eoa = eoa[LABEL_COL].to_numpy()

eoa_feature_sets = {
    "transaction_behaviour": X_txn, "graph_structural": X_struct, "hand_crafted": X_hand,
    "embedding": X_emb, "hand_crafted_embedding": X_combined,
}

eoa_fold_rows = []
for fs_name, X in eoa_feature_sets.items():
    fold_df, _ = build_cv_fold_scores(X, y_eoa)
    fold_df["feature_set"] = fs_name
    eoa_fold_rows.append(fold_df)
    print(f"EOA-only / {fs_name}: done")

eoa_fold_df = pd.concat(eoa_fold_rows, ignore_index=True)
eoa_fold_df.to_csv(OUT_DIR / "eoa_only_cv_scores.csv", index=False)

eoa_summary = eoa_fold_df.groupby(["feature_set", "model"])[["macro_f1", "balanced_accuracy"]].agg(["mean", "std"])
eoa_summary.columns = ["_".join(c) for c in eoa_summary.columns]
eoa_summary = eoa_summary.reset_index().round(4)
eoa_summary.to_csv(OUT_DIR / "eoa_only_summary.csv", index=False)
print(f"\n=== EOA-only feature-group ablation (n={len(eoa)}) ===\n{eoa_summary.to_string(index=False)}")

from scipy import stats
eoa_paired_rows = []
for a_name, b_name in [("hand_crafted", "transaction_behaviour"), ("hand_crafted_embedding", "hand_crafted")]:
    for model_name in MODEL_ORDER:
        a = eoa_fold_df[(eoa_fold_df.feature_set == a_name) & (eoa_fold_df.model == model_name)].sort_values("fold")["macro_f1"].values
        b = eoa_fold_df[(eoa_fold_df.feature_set == b_name) & (eoa_fold_df.model == model_name)].sort_values("fold")["macro_f1"].values
        diff = a - b
        t_stat, p_val = stats.ttest_rel(a, b)
        eoa_paired_rows.append({"comparison": f"{a_name} vs {b_name}", "model": model_name, "mean_diff": diff.mean(),
                                 "paired_t_stat": t_stat, "paired_p_value": p_val, "n_folds_improved": int((diff > 0).sum())})
eoa_paired_df = pd.DataFrame(eoa_paired_rows).round(4)
eoa_paired_df.to_csv(OUT_DIR / "eoa_only_paired_differences.csv", index=False)
print(f"\n=== EOA-only paired differences ===\n{eoa_paired_df.to_string(index=False)}")

import matplotlib.pyplot as plt
plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#c3c2b7",
                      "axes.labelcolor": "#0b0b0b", "text.color": "#0b0b0b", "xtick.color": "#52514e",
                      "ytick.color": "#52514e", "grid.color": "#e1e0d9", "font.size": 10})
FS_ORDER = ["transaction_behaviour", "graph_structural", "hand_crafted", "embedding", "hand_crafted_embedding"]
MODEL_COLORS = {"LogisticRegression": "#2a78d6", "RandomForest": "#1baf7a", "XGBoost": "#eb6834"}
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(FS_ORDER))
width = 0.25
for i, model_name in enumerate(MODEL_ORDER):
    sub = eoa_summary[eoa_summary.model == model_name].set_index("feature_set").reindex(FS_ORDER)
    ax.bar(x + (i - 1) * width, sub["macro_f1_mean"], width, yerr=sub["macro_f1_std"], capsize=3, color=MODEL_COLORS[model_name], label=model_name)
ax.set_xticks(x)
ax.set_xticklabels(FS_ORDER, rotation=15, ha="right")
ax.set_ylabel("Macro-F1 (5-fold CV, EOA-only n={})".format(len(eoa)))
ax.set_title("EOA-only feature-group ablation (contract/EOA confound removed)")
ax.legend(frameon=False)
ax.grid(axis="y", linewidth=0.6, alpha=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT_DIR / "eoa_only_feature_ablation.png", dpi=150)
plt.close(fig)
print(f"Saved {OUT_DIR / 'eoa_only_feature_ablation.png'}")

# =========================
# 4. Contract-only DeFi vs Malicious_MEV, 50x repeated balanced subsampling (10.8b)
# =========================
# Pool: active DeFi contracts (all 86 active DeFi are contracts) + active
# Malicious_MEV contracts (34, all active since MEV is 100% active). Balanced
# to n_mev=34 per class per repeat so all 6 feature sets share the same
# sampled pairs within a repeat (paired comparison across feature sets).

defi_contract_pool = data[(data[LABEL_COL] == "DeFi") & (data["is_contract"])].copy()
mev_contract_pool = data[(data[LABEL_COL] == "Malicious_MEV") & (data["is_contract"])].copy()
n_per_class = min(len(defi_contract_pool), len(mev_contract_pool))
print(f"\nContract-only pool: DeFi contracts (active)={len(defi_contract_pool)}, "
      f"Malicious_MEV contracts (active)={len(mev_contract_pool)}, balanced n={n_per_class}/class")

coarse_cols = ["has_out_edge", "out_tx_count", "total_tx_count"]
contract_feature_sets = {
    "coarse": coarse_cols, "transaction_behaviour": txn_cols, "graph_structural": struct_cols,
    "hand_crafted": handcrafted_cols, "embedding": emb_cols, "hand_crafted_embedding": handcrafted_cols + emb_cols,
}

N_REPEATS = 50
contract_rows = []
rng_master = np.random.default_rng(RANDOM_STATE)
repeat_seeds = rng_master.integers(0, 1_000_000, size=N_REPEATS)

for rep_i, seed in enumerate(repeat_seeds):
    defi_sample = defi_contract_pool.sample(n=n_per_class, random_state=int(seed))
    balanced = pd.concat([defi_sample, mev_contract_pool], ignore_index=True)
    y_bal = balanced[LABEL_COL].to_numpy()

    for fs_name, cols in contract_feature_sets.items():
        X = balanced[cols].to_numpy(dtype=float)
        fold_df, _ = build_cv_fold_scores(X, y_bal, random_state=int(seed))
        for model_name in MODEL_ORDER:
            sub = fold_df[fold_df.model == model_name]
            contract_rows.append({
                "repeat": rep_i, "feature_set": fs_name, "model": model_name,
                "macro_f1_mean": sub["macro_f1"].mean(), "balanced_accuracy_mean": sub["balanced_accuracy"].mean(),
            })
    if (rep_i + 1) % 10 == 0:
        print(f"  contract-only repeat {rep_i + 1}/{N_REPEATS} done")

contract_runs_df = pd.DataFrame(contract_rows)
contract_runs_df.to_csv(OUT_DIR / "contract_only_balanced_runs.csv", index=False)

contract_summary = contract_runs_df.groupby(["feature_set", "model"])["macro_f1_mean"].agg(
    ["mean", "std", lambda s: s.quantile(0.05), lambda s: s.quantile(0.95)]
)
contract_summary.columns = ["macro_f1_mean_of_means", "macro_f1_std_across_repeats", "p5", "p95"]
contract_summary = contract_summary.reset_index().round(4)
contract_summary.to_csv(OUT_DIR / "contract_only_summary.csv", index=False)
print(f"\n=== Contract-only DeFi vs Malicious_MEV, {N_REPEATS}x repeated balanced (n={2*n_per_class}) ===\n{contract_summary.to_string(index=False)}")

fig, ax = plt.subplots(figsize=(10, 5))
fs_order_c = list(contract_feature_sets.keys())
data_for_box = [contract_runs_df[(contract_runs_df.feature_set == fs) & (contract_runs_df.model == "RandomForest")]["macro_f1_mean"].values for fs in fs_order_c]
ax.boxplot(data_for_box, tick_labels=fs_order_c)
ax.axhline(0.5, color="#c3c2b7", linestyle="--", linewidth=1, label="chance (2-class)")
ax.set_ylabel(f"Macro-F1 across {N_REPEATS} repeats (RandomForest)")
ax.set_title(f"Contract-only DeFi vs Malicious_MEV: distribution across {N_REPEATS} balanced resamples")
ax.tick_params(axis="x", rotation=15)
ax.legend(frameon=False)
ax.grid(axis="y", linewidth=0.6, alpha=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT_DIR / "contract_only_macro_f1_distribution.png", dpi=150)
plt.close(fig)
print(f"Saved {OUT_DIR / 'contract_only_macro_f1_distribution.png'}")

# =========================
# 5. Per-class breakdown of the outbound-feature-removal experiment (10.8c)
# =========================

OUTBOUND_COLS = [c for c in handcrafted_cols if c.startswith("out_") or c.startswith("log_out_") or c == "weighted_out_degree" or c == "out_degree" or c == "has_out_edge"]
no_outbound_cols = [c for c in handcrafted_cols if c not in OUTBOUND_COLS]

X_full = data[handcrafted_cols].to_numpy(dtype=float)
X_no_out = data[no_outbound_cols].to_numpy(dtype=float)
y_full = data[LABEL_COL].to_numpy()

_, oof_full = build_cv_fold_scores(X_full, y_full)
_, oof_no_out = build_cv_fold_scores(X_no_out, y_full)

per_class_rows = []
for condition, oof_dict in [("full_28_features", oof_full), ("no_outbound_21_features", oof_no_out)]:
    for model_name in MODEL_ORDER:
        y_pred = oof_dict[model_name]
        precision, recall, f1, support = precision_recall_fscore_support(y_full, y_pred, labels=LABEL_ORDER, zero_division=0)
        for i, label in enumerate(LABEL_ORDER):
            per_class_rows.append({"condition": condition, "model": model_name, "label": label,
                                    "precision": precision[i], "recall": recall[i], "f1": f1[i], "support": support[i]})
per_class_df = pd.DataFrame(per_class_rows).round(4)
per_class_df.to_csv(OUT_DIR / "remove_out_features_per_class.csv", index=False)
print(f"\n=== Per-class F1: full vs no-outbound (out-of-fold predictions) ===\n{per_class_df.to_string(index=False)}")

fig, axes = plt.subplots(2, len(MODEL_ORDER), figsize=(13, 8))
for row, (condition, oof_dict) in enumerate([("full_28_features", oof_full), ("no_outbound_21_features", oof_no_out)]):
    for col, model_name in enumerate(MODEL_ORDER):
        ax = axes[row, col]
        cm = confusion_matrix(y_full, oof_dict[model_name], labels=LABEL_ORDER, normalize="true")
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(LABEL_ORDER)):
            for j in range(len(LABEL_ORDER)):
                text_color = "#ffffff" if cm[i, j] > 0.6 else "#0b0b0b"
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8, color=text_color)
        ax.set_xticks(range(len(LABEL_ORDER))); ax.set_xticklabels(LABEL_ORDER, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(LABEL_ORDER))); ax.set_yticklabels(LABEL_ORDER, fontsize=8)
        if row == 0:
            ax.set_title(model_name, fontsize=10)
        if col == 0:
            ax.set_ylabel(condition, fontsize=9)
fig.suptitle("Confusion matrices: full hand_crafted vs outbound features removed (out-of-fold)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT_DIR / "remove_out_features_confusion_matrix.png", dpi=150)
plt.close(fig)
print(f"Saved {OUT_DIR / 'remove_out_features_confusion_matrix.png'}")

print("\n=== P1 confound follow-up experiments complete ===")
