import json
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_recall_fscore_support

# =========================
# 0. Setup
# =========================
# The contract-only DeFi-vs-Malicious_MEV result (progress_summary.md
# Section 10.8b) reached macro-F1 ~0.97-1.00 with hand_crafted+embedding,
# using ordinary per-address stratified CV. sub_label composition shows
# severe entity concentration: 64/86 (74%) of DeFi contracts are "uniswap",
# and 33/34 (97%) of Malicious_MEV contracts share the sub_label "Arbitrage
# Trader". Random per-address splitting can put addresses from the same
# protocol/operator in both train and test, letting the model recognise a
# near-duplicate entity fingerprint rather than a genuine class boundary.
# This script checks whether the near-perfect result survives when the
# split is done at the ENTITY level instead of the address level.
#
# MEV's near-total homogeneity (97% one sub_label, and this experiment
# always uses the SAME 34 MEV contracts -- there is no larger MEV-contract
# pool to resample from) means a clean group-based train/test split is not
# fully achievable on the MEV side: holding out "Arbitrage Trader" as a group
# would remove essentially the entire MEV class from one side of the split.
# This script reports that limitation transparently rather than concealing
# it, alongside what CAN be tested (protocol-level generalisation on the
# DeFi side, which has real sub_label diversity).

RANDOM_STATE = 42
LABEL_COL = "label"
MODEL_COLORS = {"LogisticRegression": "#2a78d6", "RandomForest": "#1baf7a"}

OUT_DIR = Path("contract_confound_outputs")
OUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#c3c2b7",
                      "axes.labelcolor": "#0b0b0b", "text.color": "#0b0b0b", "xtick.color": "#52514e",
                      "ytick.color": "#52514e", "grid.color": "#e1e0d9", "font.size": 10})


def clean_address(x):
    if pd.isna(x):
        return np.nan
    return str(x).strip().lower()


def fit_predict(model_name, X_train, y_train, X_test, random_state=RANDOM_STATE):
    if model_name == "LogisticRegression":
        model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state))])
    else:
        model = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    return model.predict(X_test)


# =========================
# 1. Load data (same construction as confound_followup.py)
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
feat = feat.merge(contract_df[["address", "is_contract"]], on="address", how="left")
feat["is_contract"] = feat["is_contract"].fillna(False)

emb = pd.read_csv("embedding_outputs/seed_node2vec_embeddings.csv")
emb_cols = [c for c in emb.columns if c.startswith("emb_")]
active = feat[feat["is_active"]].copy()
data = active.merge(emb[["address"] + emb_cols], on="address", how="inner")

defi_pool = data[(data[LABEL_COL] == "DeFi") & (data["is_contract"])].copy()
mev_pool = data[(data[LABEL_COL] == "Malicious_MEV") & (data["is_contract"])].copy()

print("=== sub_label composition (contract-only pools) ===")
print("DeFi contracts:\n", defi_pool["sub_label"].value_counts().to_string())
print("\nMalicious_MEV contracts:\n", mev_pool["sub_label"].value_counts().to_string())

composition = pd.concat([
    defi_pool["sub_label"].value_counts().rename("n").reset_index().rename(columns={"index": "sub_label"}).assign(label="DeFi"),
    mev_pool["sub_label"].value_counts().rename("n").reset_index().rename(columns={"index": "sub_label"}).assign(label="Malicious_MEV"),
], ignore_index=True)
composition.to_csv(OUT_DIR / "contract_only_entity_composition.csv", index=False)
print(f"\nSaved {OUT_DIR / 'contract_only_entity_composition.csv'}")

pool = pd.concat([defi_pool, mev_pool], ignore_index=True)
X_combined_cols = handcrafted_cols + emb_cols
X_all = pool[X_combined_cols].to_numpy(dtype=float)
y_all = pool[LABEL_COL].to_numpy()
groups = pool["sub_label"].to_numpy()

n_unique_groups = len(set(groups))
print(f"\nFull contract-only pool (no rebalancing): n={len(pool)}, "
      f"{dict(pool[LABEL_COL].value_counts())}, unique sub_label groups={n_unique_groups}")

# =========================
# 2. GroupKFold by sub_label (entity-level split) -- report transparently,
# including any fold-composition degeneracy caused by MEV's near-monolithic group.
# =========================

N_SPLITS = 5
gkf = GroupKFold(n_splits=N_SPLITS)
group_fold_rows = []
fold_composition_rows = []

for model_name in ["LogisticRegression", "RandomForest"]:
    for fold_i, (train_idx, test_idx) in enumerate(gkf.split(X_all, y_all, groups=groups)):
        y_train, y_test = y_all[train_idx], y_all[test_idx]
        test_groups = set(groups[test_idx])

        if len(set(y_test)) < 2 or len(set(y_train)) < 2:
            # Degenerate fold: one class entirely absent from train or test
            # because its dominant group landed on one side. Record it as
            # such rather than silently computing a meaningless macro-F1.
            group_fold_rows.append({"model": model_name, "fold": fold_i, "macro_f1": np.nan,
                                     "degenerate": True, "test_groups": ";".join(sorted(test_groups)),
                                     "n_train": len(train_idx), "n_test": len(test_idx)})
            continue

        y_pred = fit_predict(model_name, X_all[train_idx], y_train, X_all[test_idx])
        group_fold_rows.append({
            "model": model_name, "fold": fold_i, "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "degenerate": False, "test_groups": ";".join(sorted(test_groups)),
            "n_train": len(train_idx), "n_test": len(test_idx),
        })

    if model_name == "LogisticRegression":  # composition is model-independent, record once
        for fold_i, (train_idx, test_idx) in enumerate(gkf.split(X_all, y_all, groups=groups)):
            test_label_counts = pd.Series(y_all[test_idx]).value_counts().to_dict()
            train_label_counts = pd.Series(y_all[train_idx]).value_counts().to_dict()
            fold_composition_rows.append({"fold": fold_i, "test_groups": ";".join(sorted(set(groups[test_idx]))),
                                           "test_label_counts": test_label_counts, "train_label_counts": train_label_counts})

group_fold_df = pd.DataFrame(group_fold_rows)
group_fold_df.to_csv(OUT_DIR / "contract_only_grouped_cv_scores.csv", index=False)
print(f"\n=== GroupKFold(sub_label) on contract-only DeFi vs Malicious_MEV, full pool (n={len(pool)}) ===")
print(group_fold_df.to_string(index=False))

fold_comp_df = pd.DataFrame(fold_composition_rows)
fold_comp_df.to_csv(OUT_DIR / "contract_only_grouped_fold_composition.csv", index=False)
print(f"\n=== Fold composition (train/test label counts per fold) ===\n{fold_comp_df.to_string(index=False)}")

n_degenerate = group_fold_df["degenerate"].sum()
print(f"\n{n_degenerate}/{len(group_fold_df)} fold x model combinations were degenerate "
      f"(one class entirely missing from train or test after entity-level grouping).")

# =========================
# 3. Leave-one-DeFi-protocol-out (isolates DeFi-side generalisation; MEV
# side still reuses "Arbitrage Trader" in both train and test since there is
# no second MEV entity type with enough addresses to hold out instead).
# =========================

defi_protocols_testable = defi_pool["sub_label"].value_counts()
defi_protocols_testable = defi_protocols_testable[defi_protocols_testable >= 2].index.tolist()
print(f"\nDeFi protocols with >=2 contract addresses (testable as held-out groups): {defi_protocols_testable}")

lopo_rows = []
for protocol in defi_protocols_testable:
    test_defi = defi_pool[defi_pool["sub_label"] == protocol]
    train_defi = defi_pool[defi_pool["sub_label"] != protocol]
    # Balance train_defi against MEV's size for a fair training distribution,
    # same convention as the 50x repeated experiment (34 = size of MEV pool).
    n_take = min(len(train_defi), len(mev_pool))
    train_defi_bal = train_defi.sample(n=n_take, random_state=RANDOM_STATE)

    train_pool = pd.concat([train_defi_bal, mev_pool], ignore_index=True)
    # Held-out test set: the unseen protocol's DeFi addresses vs a held-out
    # slice of MEV (MEV has no second entity type, so this reuses MEV
    # addresses that WERE in training -- flagged explicitly, not hidden).
    test_pool = pd.concat([test_defi, mev_pool.sample(n=min(len(test_defi), len(mev_pool)), random_state=RANDOM_STATE)], ignore_index=True)

    X_train = train_pool[X_combined_cols].to_numpy(dtype=float)
    y_train = train_pool[LABEL_COL].to_numpy()
    X_test = test_pool[X_combined_cols].to_numpy(dtype=float)
    y_test = test_pool[LABEL_COL].to_numpy()

    for model_name in ["LogisticRegression", "RandomForest"]:
        y_pred = fit_predict(model_name, X_train, y_train, X_test)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, labels=["DeFi", "Malicious_MEV"], zero_division=0)
        lopo_rows.append({
            "held_out_defi_protocol": protocol, "model": model_name, "n_test_defi": len(test_defi),
            "defi_recall": recall[0], "defi_f1": f1[0], "mev_recall": recall[1], "mev_f1": f1[1],
            "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        })

lopo_df = pd.DataFrame(lopo_rows).round(4)
lopo_df.to_csv(OUT_DIR / "contract_only_leave_protocol_out.csv", index=False)
print(f"\n=== Leave-one-DeFi-protocol-out (MEV side still homogeneous -- see caveat) ===\n{lopo_df.to_string(index=False)}")

# =========================
# 4. Plot: entity-grouped vs address-level result
# =========================

original = pd.read_csv(OUT_DIR / "contract_only_summary.csv")
orig_rf = original[(original.feature_set == "hand_crafted_embedding") & (original.model == "RandomForest")]["macro_f1_mean_of_means"].values[0]

fig, ax = plt.subplots(figsize=(8, 5))
categories = ["Address-level\nstratified CV\n(original)", "Entity-grouped\nGroupKFold\n(sub_label)"] + [f"LOPO:\nheld-out\n{p}" for p in defi_protocols_testable]
rf_vals = [orig_rf, group_fold_df[(group_fold_df.model == "RandomForest") & (~group_fold_df.degenerate)]["macro_f1"].mean()] + \
          [lopo_df[(lopo_df.held_out_defi_protocol == p) & (lopo_df.model == "RandomForest")]["macro_f1"].values[0] for p in defi_protocols_testable]
ax.bar(categories, rf_vals, color="#1baf7a")
ax.axhline(0.5, color="#c3c2b7", linestyle="--", linewidth=1, label="chance (2-class)")
ax.set_ylabel("Macro-F1 (RandomForest, hand_crafted+embedding)")
ax.set_title("Does the contract-only DeFi-vs-MEV result survive entity-level grouping?")
ax.tick_params(axis="x", rotation=20, labelsize=8)
ax.legend(frameon=False)
ax.grid(axis="y", linewidth=0.6, alpha=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT_DIR / "contract_only_entity_grouped_comparison.png", dpi=150)
plt.close(fig)
print(f"Saved {OUT_DIR / 'contract_only_entity_grouped_comparison.png'}")

print("\n=== Entity-grouped robustness check complete ===")
