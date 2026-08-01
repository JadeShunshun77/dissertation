import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# =========================
# 0. Setup
# =========================
# Confound check: active DeFi seeds have out_edge_count == 100% zero (never
# initiate transactions), while CEX/Malicious_MEV mostly do. Since DeFi labels
# are almost all smart contracts (protocols, pools) and CEX/MEV labels are a
# mix of EOA wallets and bot contracts, the classifier may largely be learning
# "is this a contract" (which correlates near-perfectly with is DeFi) rather
# than genuine economic behaviour differences between the three categories.
# This script gets ground-truth is_contract via eth_getCode (bytecode length
# > 0 => contract) for all 500 seeds and re-runs the core classification
# comparisons with that confound controlled for.

RANDOM_STATE = 42
N_SPLITS = 5
LABEL_COL = "label"
LABEL_ORDER = ["CEX", "DeFi", "Malicious_MEV"]
MODEL_ORDER = ["LogisticRegression", "RandomForest", "XGBoost"]

# Optional CLI args: feat_file edges_file out_subdir contract_cache_dir --
# reruns the topology-dependent parts (outbound_ablation) on the Plan-B
# reconstructed graph. is_contract is an on-chain bytecode property,
# independent of edge-sampling topology, so its cache is always read from
# contract_cache_dir (defaults to the original contract_confound_outputs/) to
# avoid re-querying the RPC.
FEAT_FILE = sys.argv[1] if len(sys.argv) > 1 else "seed_feature_matrix_reconstructed.csv"
EDGES_FILE = sys.argv[2] if len(sys.argv) > 2 else "raw_edges_reconstructed.csv"
OUT_SUBDIR = sys.argv[3] if len(sys.argv) > 3 else "contract_confound_outputs_reconstructed"
CONTRACT_CACHE_DIR = sys.argv[4] if len(sys.argv) > 4 else "contract_confound_outputs"

RPC_URL = "https://ethereum-rpc.publicnode.com"
OUT_DIR = Path(OUT_SUBDIR)
OUT_DIR.mkdir(exist_ok=True)


def get_code(address, retries=3):
    payload = {"jsonrpc": "2.0", "method": "eth_getCode", "params": [address, "latest"], "id": 1}
    for attempt in range(retries):
        try:
            r = requests.post(RPC_URL, json=payload, timeout=15)
            data = r.json()
            if "result" in data:
                return data["result"]
        except Exception as e:
            pass
        time.sleep(1.5 * (attempt + 1))
    return None


def build_cv_results(X, y, group_label, feature_set_name):
    label_encoder = LabelEncoder().fit(sorted(set(y)))
    rows = []
    for model_name in MODEL_ORDER:
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if model_name == "LogisticRegression":
                model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE))])
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
                "group": group_label, "feature_set": feature_set_name, "model": model_name, "fold": fold_i,
                "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
                "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            })
    return pd.DataFrame(rows)


def clean_address(x):
    if pd.isna(x):
        return np.nan
    return str(x).strip().lower()


# =========================
# 1. Query is_contract for all 500 seeds
# =========================

feat = pd.read_csv(FEAT_FILE)
cache_path = Path(CONTRACT_CACHE_DIR) / "seed_is_contract.csv"

if cache_path.exists():
    contract_df = pd.read_csv(cache_path)
    print(f"Loaded cached is_contract results for {len(contract_df)} seeds from {cache_path}")
else:
    print(f"Querying eth_getCode for {len(feat)} seeds via {RPC_URL} ...")
    results = []
    for i, addr in enumerate(feat["address"]):
        code = get_code(addr)
        is_contract = (code is not None) and (code != "0x") and (len(code) > 2)
        results.append({"address": addr, "bytecode_length": (len(code) - 2) // 2 if code and code != "0x" else 0, "is_contract": is_contract, "rpc_ok": code is not None})
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(feat)} queried")
    contract_df = pd.DataFrame(results)
    contract_df.to_csv(cache_path, index=False)
    print(f"Saved {cache_path}")

failed = (~contract_df["rpc_ok"]).sum()
if failed > 0:
    print(f"WARNING: {failed} addresses failed RPC lookup (treated as EOA by default -- check seed_is_contract.csv)")

feat = feat.merge(contract_df[["address", "is_contract", "bytecode_length"]], on="address", how="left")
feat["is_contract"] = feat["is_contract"].fillna(False)

# =========================
# 2. Contract/EOA ratio per class
# =========================

ratio_table = feat.groupby(LABEL_COL)["is_contract"].agg(["sum", "count", "mean"]).rename(
    columns={"sum": "n_contract", "count": "n_total", "mean": "contract_ratio"}
).reindex(LABEL_ORDER).round(4)
ratio_table.to_csv(OUT_DIR / "contract_ratio_by_label.csv")
print(f"\n=== Contract/EOA ratio by label (all 500 seeds) ===\n{ratio_table.to_string()}")

active = feat[feat["is_active"]].copy()
ratio_table_active = active.groupby(LABEL_COL)["is_contract"].agg(["sum", "count", "mean"]).rename(
    columns={"sum": "n_contract", "count": "n_total", "mean": "contract_ratio"}
).reindex(LABEL_ORDER).round(4)
ratio_table_active.to_csv(OUT_DIR / "contract_ratio_by_label_active.csv")
print(f"\n=== Contract/EOA ratio by label (436 active seeds only) ===\n{ratio_table_active.to_string()}")

# =========================
# 3. Minimal baseline: is_contract + has_out_edge + out_tx_count only
# =========================

minimal_cols = ["is_contract", "has_out_edge", "out_tx_count"]
X_min = active[minimal_cols].astype(float).to_numpy()
y = active[LABEL_COL].to_numpy()

minimal_fold_df = build_cv_results(X_min, y, "minimal_baseline", "is_contract+has_out_edge+out_tx_count")
minimal_summary = minimal_fold_df.groupby("model")[["macro_f1", "balanced_accuracy"]].agg(["mean", "std"])
minimal_summary.columns = ["_".join(c) for c in minimal_summary.columns]
minimal_summary = minimal_summary.reset_index().round(4)
minimal_summary.to_csv(OUT_DIR / "minimal_baseline_summary.csv", index=False)
print(f"\n=== Minimal 3-feature baseline (is_contract + has_out_edge + out_tx_count) ===\n{minimal_summary.to_string(index=False)}")

# =========================
# 4. Full hand_crafted set with ALL outbound-direction features removed
# =========================

with open("feature_groups.json") as f:
    feature_groups = json.load(f)
handcrafted_cols = list(feature_groups["transaction_behaviour"] + feature_groups["graph_structural"])

edges = pd.read_csv(EDGES_FILE)
edges["source"] = edges["source"].apply(clean_address)
edges["target"] = edges["target"].apply(clean_address)
edges = edges.dropna(subset=["source", "target"])
import networkx as nx
G = nx.from_pandas_edgelist(edges, source="source", target="target", edge_attr=["tx_count"], create_using=nx.DiGraph())
pagerank_rank = pd.Series(nx.pagerank(G)).rank(pct=True)
pagerank_weighted_rank = pd.Series(nx.pagerank(G, weight="tx_count")).rank(pct=True)
feat["pagerank_rank"] = feat["address"].map(pagerank_rank).fillna(0.0)
feat["pagerank_weighted_rank"] = feat["address"].map(pagerank_weighted_rank).fillna(0.0)
for col in ["pagerank", "pagerank_weighted", "log_pagerank", "log_pagerank_weighted"]:
    if col in handcrafted_cols:
        handcrafted_cols.remove(col)
handcrafted_cols += ["pagerank_rank", "pagerank_weighted_rank"]

active = feat[feat["is_active"]].copy()  # refresh with pagerank_rank columns

OUTBOUND_COLS = [c for c in handcrafted_cols if c.startswith("out_") or c.startswith("log_out_") or c == "weighted_out_degree" or c == "out_degree" or c == "has_out_edge"]
no_outbound_cols = [c for c in handcrafted_cols if c not in OUTBOUND_COLS]
print(f"\nRemoving {len(OUTBOUND_COLS)} outbound-direction features: {OUTBOUND_COLS}")
print(f"Remaining {len(no_outbound_cols)} features: {no_outbound_cols}")

X_full = active[handcrafted_cols].astype(float).to_numpy()
X_no_out = active[no_outbound_cols].astype(float).to_numpy()
y = active[LABEL_COL].to_numpy()

full_fold_df = build_cv_results(X_full, y, "outbound_ablation", "hand_crafted_full")
no_out_fold_df = build_cv_results(X_no_out, y, "outbound_ablation", "hand_crafted_no_outbound")

outbound_fold_df = pd.concat([full_fold_df, no_out_fold_df], ignore_index=True)
outbound_summary = outbound_fold_df.groupby(["feature_set", "model"])[["macro_f1", "balanced_accuracy"]].agg(["mean", "std"])
outbound_summary.columns = ["_".join(c) for c in outbound_summary.columns]
outbound_summary = outbound_summary.reset_index().round(4)
outbound_summary.to_csv(OUT_DIR / "outbound_ablation_summary.csv", index=False)
outbound_fold_df.to_csv(OUT_DIR / "outbound_ablation_fold_scores.csv", index=False)
print(f"\n=== hand_crafted_full vs hand_crafted_no_outbound (all outbound-direction features removed) ===\n{outbound_summary.to_string(index=False)}")

from scipy import stats
paired_rows = []
for model_name in MODEL_ORDER:
    full = full_fold_df[full_fold_df.model == model_name].sort_values("fold")["macro_f1"].values
    no_out = no_out_fold_df[no_out_fold_df.model == model_name].sort_values("fold")["macro_f1"].values
    diff = full - no_out
    t_stat, p_val = stats.ttest_rel(full, no_out)
    paired_rows.append({"model": model_name, "full_mean": full.mean(), "no_outbound_mean": no_out.mean(),
                         "mean_diff": diff.mean(), "paired_t_stat": t_stat, "paired_p_value": p_val})
paired_df = pd.DataFrame(paired_rows).round(4)
paired_df.to_csv(OUT_DIR / "outbound_ablation_paired_test.csv", index=False)
print(f"\n=== Paired test: full vs no-outbound (same 5 folds) ===\n{paired_df.to_string(index=False)}")

# =========================
# 5. Sub-experiment: CEX vs Malicious_MEV only, restricted to EOA addresses (is_contract == False)
# =========================

eoa_only = active[(~active["is_contract"]) & (active[LABEL_COL].isin(["CEX", "Malicious_MEV"]))].copy()
print(f"\nEOA-only CEX vs Malicious_MEV subset: {len(eoa_only)} seeds, {dict(eoa_only[LABEL_COL].value_counts())}")

X_eoa = eoa_only[handcrafted_cols].astype(float).to_numpy()
y_eoa = eoa_only[LABEL_COL].to_numpy()

eoa_fold_df = build_cv_results(X_eoa, y_eoa, "eoa_only_binary", "hand_crafted_full")
eoa_summary = eoa_fold_df.groupby("model")[["macro_f1", "balanced_accuracy"]].agg(["mean", "std"])
eoa_summary.columns = ["_".join(c) for c in eoa_summary.columns]
eoa_summary = eoa_summary.reset_index().round(4)
eoa_summary.to_csv(OUT_DIR / "eoa_only_cex_vs_mev_summary.csv", index=False)
print(f"\n=== CEX vs Malicious_MEV, EOA-only subset (contract-type confound removed) ===\n{eoa_summary.to_string(index=False)}")

print("\n=== Contract/EOA confound check complete ===")
