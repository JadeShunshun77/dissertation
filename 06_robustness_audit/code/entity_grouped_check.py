"""Audit entity-level identifiability and run defensible one-sided holdouts.

The stored ``sub_label`` is an entity/protocol identifier for CEX and DeFi
samples, but ``Arbitrage Trader`` is a role label rather than an MEV entity
identifier. A fully entity-disjoint three-class split is therefore not
identifiable from the current labels. This script records that limitation and
tests only the two entity sides that the data can support:

* EOA-only: unseen CEX entity versus address-disjoint MEV-related EOAs.
* Contract-only: unseen DeFi protocol versus address-disjoint MEV contracts.

The MEV controls are address-disjoint, but cannot be claimed to be
entity-disjoint. Outputs must be interpreted as one-sided entity holdouts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
N_REPEATS = 50
MAX_TEST_PER_CLASS = 10
MIN_GROUP_SIZE = 2
LABEL_COL = "label"
MEV_LABEL = "Malicious_MEV"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIR = PROJECT_ROOT / "02_feature_engineering" / "feature_engineering"
DEFAULT_CONTRACT_DIR = (
    PROJECT_ROOT / "06_robustness_audit" / "outputs" / "contract_confound"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "06_robustness_audit" / "outputs" / "entity_generalisation"
)

# Optional CLI args: feat_file edges_file emb_dir out_dir contract_cache_dir.
FEAT_FILE = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else FEATURE_DIR / "seed_feature_matrix_reconstructed.csv"
)
EDGES_FILE = (
    Path(sys.argv[2])
    if len(sys.argv) > 2
    else PROJECT_ROOT / "00_data_graph" / "data" / "main" / "raw_edges_reconstructed.csv"
)
EMB_DIR = (
    Path(sys.argv[3])
    if len(sys.argv) > 3
    else PROJECT_ROOT / "03_unsupervised" / "node2vec_results"
)
OUT_DIR = Path(sys.argv[4]) if len(sys.argv) > 4 else DEFAULT_OUTPUT_DIR
CONTRACT_CACHE_DIR = Path(sys.argv[5]) if len(sys.argv) > 5 else DEFAULT_CONTRACT_DIR
FEATURE_GROUPS_FILE = FEATURE_DIR / "feature_groups.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_address(value: object) -> str:
    return str(value).strip().lower()


def normalise_entity_group(label: str, sub_label: str) -> str | None:
    """Return a defensible entity/protocol group, or None when unavailable."""
    name = str(sub_label).strip().lower()
    if label == "CEX":
        if name.startswith("coinbase"):
            return "coinbase"
        if name.startswith("binance"):
            return "binance"
        return name
    if label == "DeFi":
        return name
    # MEV sub-labels describe roles, not independently identified entities.
    return None


def build_model(model_name: str, random_state: int):
    if model_name == "LogisticRegression":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        )
    return RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )


def load_analysis_data() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    feat = pd.read_csv(FEAT_FILE)
    feat["address"] = feat["address"].map(clean_address)

    with FEATURE_GROUPS_FILE.open(encoding="utf-8") as handle:
        feature_groups = json.load(handle)
    transaction_cols = list(feature_groups["transaction_behaviour"])
    structural_cols = list(feature_groups["graph_structural"])

    edges = pd.read_csv(EDGES_FILE)
    edges["source"] = edges["source"].map(clean_address)
    edges["target"] = edges["target"].map(clean_address)
    graph = nx.from_pandas_edgelist(
        edges,
        source="source",
        target="target",
        edge_attr=["tx_count"],
        create_using=nx.DiGraph(),
    )
    pagerank_rank = pd.Series(nx.pagerank(graph)).rank(pct=True)
    weighted_rank = pd.Series(nx.pagerank(graph, weight="tx_count")).rank(pct=True)
    feat["pagerank_rank"] = feat["address"].map(pagerank_rank).fillna(0.0)
    feat["pagerank_weighted_rank"] = feat["address"].map(weighted_rank).fillna(0.0)

    for column in [
        "pagerank",
        "pagerank_weighted",
        "log_pagerank",
        "log_pagerank_weighted",
    ]:
        if column in structural_cols:
            structural_cols.remove(column)
    structural_cols += ["pagerank_rank", "pagerank_weighted_rank"]

    contract = pd.read_csv(CONTRACT_CACHE_DIR / "seed_is_contract.csv")
    contract["address"] = contract["address"].map(clean_address)
    feat = feat.merge(contract[["address", "is_contract"]], on="address", how="left")
    feat["is_contract"] = feat["is_contract"].fillna(False).astype(bool)

    embeddings = pd.read_csv(EMB_DIR / "seed_node2vec_embeddings.csv")
    embeddings["address"] = embeddings["address"].map(clean_address)
    embedding_cols = [column for column in embeddings if column.startswith("emb_")]

    active = feat[feat["is_active"].astype(bool)].copy()
    data = active.merge(
        embeddings[["address"] + embedding_cols], on="address", how="inner"
    )
    data["entity_group"] = [
        normalise_entity_group(label, sub_label)
        for label, sub_label in zip(data[LABEL_COL], data["sub_label"])
    ]

    columns = {
        "hand_crafted": transaction_cols + structural_cols,
        "hand_crafted_embedding": transaction_cols + structural_cols + embedding_cols,
    }
    return data, columns


def write_identifiability_audit(data: pd.DataFrame) -> None:
    composition_rows = []
    for (label, sub_label), group in data.groupby([LABEL_COL, "sub_label"], sort=True):
        entity_group = normalise_entity_group(label, sub_label)
        composition_rows.append(
            {
                "label": label,
                "sub_label": sub_label,
                "entity_group": entity_group if entity_group is not None else "UNAVAILABLE",
                "group_semantics": (
                    "entity_or_protocol" if entity_group is not None else "role_not_entity"
                ),
                "n_active": len(group),
                "n_contract": int(group["is_contract"].sum()),
                "testable_holdout": bool(entity_group is not None and len(group) >= MIN_GROUP_SIZE),
            }
        )
    composition = pd.DataFrame(composition_rows)
    composition.to_csv(OUT_DIR / "entity_group_composition_active.csv", index=False)

    audit = pd.DataFrame(
        [
            {
                "evaluation": "full_three_class_entity_disjoint",
                "identifiable": False,
                "reason": (
                    "MEV sub_labels are role labels and provide no independent entity groups"
                ),
            },
            {
                "evaluation": "cex_side_unseen_entity_eoa_only",
                "identifiable": True,
                "reason": (
                    "CEX entity groups are available; MEV controls can be address-disjoint only"
                ),
            },
            {
                "evaluation": "defi_side_unseen_protocol_contract_only",
                "identifiable": True,
                "reason": (
                    "DeFi protocol groups are available; MEV controls can be address-disjoint only"
                ),
            },
            {
                "evaluation": "mev_side_unseen_entity",
                "identifiable": False,
                "reason": "MEV entity ownership/group metadata is absent",
            },
            {
                "evaluation": "temporal_holdout_from_current_edges",
                "identifiable": False,
                "reason": "active edge tables are aggregated and contain no timestamps",
            },
        ]
    )
    audit.to_csv(OUT_DIR / "entity_split_identifiability.csv", index=False)


def run_one_sided_holdouts(
    data: pd.DataFrame,
    feature_sets: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scenarios = [
        {
            "scenario": "cex_unseen_entity_eoa_only",
            "role_label": "CEX",
            "role_pool": data[(data[LABEL_COL] == "CEX") & (~data["is_contract"])],
            "control_pool": data[(data[LABEL_COL] == MEV_LABEL) & (~data["is_contract"])],
        },
        {
            "scenario": "defi_unseen_protocol_contract_only",
            "role_label": "DeFi",
            "role_pool": data[(data[LABEL_COL] == "DeFi") & (data["is_contract"])],
            "control_pool": data[(data[LABEL_COL] == MEV_LABEL) & (data["is_contract"])],
        },
    ]

    rows: list[dict[str, object]] = []
    for scenario in scenarios:
        role_pool = scenario["role_pool"].copy()
        control_pool = scenario["control_pool"].copy()
        group_counts = role_pool["entity_group"].value_counts()
        testable_groups = group_counts[group_counts >= MIN_GROUP_SIZE].index.tolist()

        for held_out_group in testable_groups:
            held_out = role_pool[role_pool["entity_group"] == held_out_group]
            role_train_pool = role_pool[role_pool["entity_group"] != held_out_group]

            for repeat in range(N_REPEATS):
                seed = RANDOM_STATE + repeat
                rng = np.random.default_rng(seed)
                n_test = min(MAX_TEST_PER_CLASS, len(held_out))
                n_train = min(len(role_train_pool), len(control_pool) - n_test)
                if n_test < MIN_GROUP_SIZE or n_train < MIN_GROUP_SIZE:
                    continue

                held_test = held_out.iloc[
                    rng.choice(len(held_out), size=n_test, replace=False)
                ]
                role_train = role_train_pool.iloc[
                    rng.choice(len(role_train_pool), size=n_train, replace=False)
                ]
                control_order = rng.permutation(len(control_pool))
                control_test = control_pool.iloc[control_order[:n_test]]
                control_train = control_pool.iloc[
                    control_order[n_test : n_test + n_train]
                ]

                train = pd.concat([role_train, control_train], ignore_index=True)
                test = pd.concat([held_test, control_test], ignore_index=True)
                overlap = set(train["address"]) & set(test["address"])
                if overlap:
                    raise AssertionError(
                        f"Address leakage in {scenario['scenario']} / {held_out_group}: {overlap}"
                    )
                if held_out_group in set(role_train["entity_group"]):
                    raise AssertionError("Held-out role group leaked into training")

                y_train = np.where(
                    train[LABEL_COL] == scenario["role_label"],
                    scenario["role_label"],
                    "MEV_related",
                )
                y_test = np.where(
                    test[LABEL_COL] == scenario["role_label"],
                    scenario["role_label"],
                    "MEV_related",
                )

                for feature_name, columns in feature_sets.items():
                    x_train = train[columns].to_numpy(dtype=float)
                    x_test = test[columns].to_numpy(dtype=float)
                    for model_name in ["LogisticRegression", "RandomForest"]:
                        model = build_model(model_name, seed)
                        model.fit(x_train, y_train)
                        prediction = model.predict(x_test)
                        role_mask = y_test == scenario["role_label"]
                        control_mask = ~role_mask
                        rows.append(
                            {
                                "scenario": scenario["scenario"],
                                "held_out_group": held_out_group,
                                "repeat": repeat,
                                "seed": seed,
                                "feature_set": feature_name,
                                "model": model_name,
                                "n_train_per_class": n_train,
                                "n_test_per_class": n_test,
                                "train_test_address_overlap": 0,
                                "macro_f1": f1_score(
                                    y_test, prediction, average="macro", zero_division=0
                                ),
                                "balanced_accuracy": balanced_accuracy_score(
                                    y_test, prediction
                                ),
                                "heldout_role_recall": float(
                                    np.mean(prediction[role_mask] == y_test[role_mask])
                                ),
                                "mev_control_recall": float(
                                    np.mean(prediction[control_mask] == y_test[control_mask])
                                ),
                            }
                        )

    runs = pd.DataFrame(rows)
    runs.to_csv(OUT_DIR / "one_sided_entity_holdout_runs.csv", index=False)

    summary = (
        runs.groupby(["scenario", "held_out_group", "feature_set", "model"])
        .agg(
            n_repeats=("repeat", "nunique"),
            n_test_per_class=("n_test_per_class", "max"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            macro_f1_p05=("macro_f1", lambda values: values.quantile(0.05)),
            macro_f1_p95=("macro_f1", lambda values: values.quantile(0.95)),
            heldout_role_recall_mean=("heldout_role_recall", "mean"),
            mev_control_recall_mean=("mev_control_recall", "mean"),
            max_address_overlap=("train_test_address_overlap", "max"),
        )
        .reset_index()
    )
    summary.to_csv(OUT_DIR / "one_sided_entity_holdout_summary.csv", index=False)

    group_equal = (
        summary.groupby(["scenario", "feature_set", "model"])
        .agg(
            n_held_out_groups=("held_out_group", "nunique"),
            group_equal_macro_f1_mean=("macro_f1_mean", "mean"),
            worst_group_macro_f1=("macro_f1_mean", "min"),
            best_group_macro_f1=("macro_f1_mean", "max"),
            group_equal_heldout_recall_mean=("heldout_role_recall_mean", "mean"),
        )
        .reset_index()
    )
    group_equal.to_csv(OUT_DIR / "one_sided_entity_holdout_group_equal.csv", index=False)
    return runs, summary, group_equal


def main() -> int:
    data, feature_sets = load_analysis_data()
    write_identifiability_audit(data)
    runs, summary, group_equal = run_one_sided_holdouts(data, feature_sets)

    if runs.empty:
        raise RuntimeError("No valid one-sided entity holdout was generated")
    if runs["train_test_address_overlap"].max() != 0:
        raise AssertionError("At least one holdout contains address overlap")

    assessment = {
        "n_active_labels": int(len(data)),
        "n_repeats": N_REPEATS,
        "max_test_per_class": MAX_TEST_PER_CLASS,
        "full_three_class_entity_disjoint_identifiable": False,
        "mev_entity_groups_available": False,
        "one_sided_holdout_runs": int(len(runs)),
        "max_train_test_address_overlap": 0,
        "interpretation": (
            "One-sided unseen CEX-entity and DeFi-protocol tests with "
            "address-disjoint MEV controls; not full unseen-entity generalisation"
        ),
    }
    with (OUT_DIR / "m02_entity_generalisation_assessment.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(assessment, handle, indent=2)

    print("=== Entity split identifiability ===")
    print(pd.read_csv(OUT_DIR / "entity_split_identifiability.csv").to_string(index=False))
    print("\n=== One-sided entity holdout: group-equal summary ===")
    print(group_equal.round(4).to_string(index=False))
    print(f"\nRuns: {len(runs)}; maximum train/test address overlap: 0")
    print(f"Outputs: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
