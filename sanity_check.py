import sys
import pandas as pd
import numpy as np

# =========================
# 0. Read data
# =========================
# Optional CLI arg: edges_file -- defaults to the Plan-B reconstructed graph
# (progress_summary.md §13), the project's main-line dataset from Week 11 on.
EDGES_FILE = sys.argv[1] if len(sys.argv) > 1 else "raw_edges_reconstructed.csv"

seeds = pd.read_csv("final_seed_address500.csv")
edges = pd.read_csv(EDGES_FILE)

print("Seeds columns:", seeds.columns.tolist())
print("Edges columns:", edges.columns.tolist())
print("\n")

# =========================
# 1. Basic column settings
# =========================

SEED_ADDRESS_COL = "address"
LABEL_COL = "label"
SOURCE_COL = "source"
TARGET_COL = "target"

required_seed_cols = [SEED_ADDRESS_COL, LABEL_COL]
required_edge_cols = [SOURCE_COL, TARGET_COL]

for col in required_seed_cols:
    if col not in seeds.columns:
        raise ValueError(f"Missing column in seeds file: {col}")

for col in required_edge_cols:
    if col not in edges.columns:
        raise ValueError(f"Missing column in edges file: {col}")

# =========================
# 2. Clean address format
# =========================

def clean_address(x):
    if pd.isna(x):
        return np.nan
    return str(x).strip().lower()

seeds[SEED_ADDRESS_COL] = seeds[SEED_ADDRESS_COL].apply(clean_address)
edges[SOURCE_COL] = edges[SOURCE_COL].apply(clean_address)
edges[TARGET_COL] = edges[TARGET_COL].apply(clean_address)

# Remove rows with missing source / target
before_edges = len(edges)
edges = edges.dropna(subset=[SOURCE_COL, TARGET_COL])
after_edges = len(edges)

print("=== Basic File Summary ===")
print(f"Seeds rows: {len(seeds)}")
print(f"Edges rows before dropping missing source/target: {before_edges}")
print(f"Edges rows after dropping missing source/target: {after_edges}")
print("\n")

# =========================
# 3. Seed label distribution
# =========================

print("=== Check 1: Seed Label Distribution ===")
label_counts = seeds[LABEL_COL].value_counts()
print(label_counts)
print("\n")

# =========================
# 4. Duplicate seed addresses and label conflicts
# =========================

print("=== Check 2: Duplicate Seeds and Label Conflicts ===")

duplicate_seed_rows = seeds[seeds.duplicated(subset=[SEED_ADDRESS_COL], keep=False)]
print(f"Duplicate seed rows: {len(duplicate_seed_rows)}")

label_per_address = seeds.groupby(SEED_ADDRESS_COL)[LABEL_COL].nunique()
conflict_addresses = label_per_address[label_per_address > 1]

print(f"Addresses with conflicting labels: {len(conflict_addresses)}")

if len(conflict_addresses) > 0:
    print("Example conflicting addresses:")
    print(seeds[seeds[SEED_ADDRESS_COL].isin(conflict_addresses.index)].sort_values(SEED_ADDRESS_COL).head(20))

print("\n")

# Optional: remove exact duplicate address-label rows
seeds = seeds.drop_duplicates(subset=[SEED_ADDRESS_COL, LABEL_COL])

# If an address has multiple labels, you should resolve manually or remove it
if len(conflict_addresses) > 0:
    seeds = seeds[~seeds[SEED_ADDRESS_COL].isin(conflict_addresses.index)]
    print(f"After removing conflicted addresses, seeds rows: {len(seeds)}")
    print("\n")

# =========================
# 5. Edge quality checks
# =========================

print("=== Check 3: Edge Quality ===")

self_loops = (edges[SOURCE_COL] == edges[TARGET_COL]).sum()
print(f"Self-loops: {self_loops}")

duplicate_edges = edges.duplicated(subset=[SOURCE_COL, TARGET_COL]).sum()
print(f"Duplicated source-target rows: {duplicate_edges}")

unique_nodes = set(edges[SOURCE_COL]).union(set(edges[TARGET_COL]))
print(f"Unique graph nodes: {len(unique_nodes)}")
print(f"Raw edge rows: {len(edges)}")
print(f"Unique directed source-target pairs: {edges[[SOURCE_COL, TARGET_COL]].drop_duplicates().shape[0]}")
print("\n")

# =========================
# 6. Seed activity in past 90 days
# =========================

print("=== Check 4: Seed Activity Rate ===")

active_nodes = unique_nodes
seeds["is_active"] = seeds[SEED_ADDRESS_COL].isin(active_nodes)

activity_stats = (
    seeds.groupby(LABEL_COL)["is_active"]
    .agg(["sum", "count", "mean"])
    .rename(columns={"sum": "active_seeds", "count": "total_seeds", "mean": "active_rate"})
)

activity_stats["inactive_seeds"] = activity_stats["total_seeds"] - activity_stats["active_seeds"]
activity_stats["active_rate"] = activity_stats["active_rate"].round(4)

print(activity_stats)
print("\n")

# =========================
# 7. Count transactions / edges per seed
# =========================

print("=== Check 5: Edge / Transaction Count per Seed ===")

# Judge whether raw_edges already has a tx_count column
if "tx_count" in edges.columns:
    print("Detected 'tx_count' column. Using summed tx_count instead of row count.")
    source_counts = edges.groupby(SOURCE_COL)["tx_count"].sum()
    target_counts = edges.groupby(TARGET_COL)["tx_count"].sum()
else:
    print("No 'tx_count' column detected. Using row count as transaction count.")
    source_counts = edges.groupby(SOURCE_COL).size()
    target_counts = edges.groupby(TARGET_COL).size()

all_counts = source_counts.add(target_counts, fill_value=0)

seeds["tx_count"] = seeds[SEED_ADDRESS_COL].map(all_counts).fillna(0)

edge_stats = seeds.groupby(LABEL_COL)["tx_count"].agg(
    total_tx_or_edges="sum",
    mean_tx_or_edges="mean",
    median_tx_or_edges="median",
    max_tx_or_edges="max"
).round(2)

print(edge_stats)
print("\n")

# =========================
# 8. Directional activity: incoming vs outgoing
# =========================

print("=== Check 6: Incoming vs Outgoing Activity by Label ===")

if "tx_count" in edges.columns:
    out_counts = edges.groupby(SOURCE_COL)["tx_count"].sum()
    in_counts = edges.groupby(TARGET_COL)["tx_count"].sum()
else:
    out_counts = edges.groupby(SOURCE_COL).size()
    in_counts = edges.groupby(TARGET_COL).size()

seeds["out_tx_count"] = seeds[SEED_ADDRESS_COL].map(out_counts).fillna(0)
seeds["in_tx_count"] = seeds[SEED_ADDRESS_COL].map(in_counts).fillna(0)

direction_stats = seeds.groupby(LABEL_COL)[["in_tx_count", "out_tx_count"]].agg(["sum", "mean", "median", "max"]).round(2)
print(direction_stats)
print("\n")

# =========================
# 9. Top high-degree / high-activity seeds
# =========================

print("=== Check 7: Top 20 Most Active Seed Addresses ===")

top_seeds = seeds.sort_values("tx_count", ascending=False)[
    [SEED_ADDRESS_COL, LABEL_COL, "tx_count", "in_tx_count", "out_tx_count"]
].head(20)

print(top_seeds)
print("\n")

# How much do top seeds dominate?
total_seed_tx = seeds["tx_count"].sum()
top_5_share = seeds.sort_values("tx_count", ascending=False)["tx_count"].head(5).sum() / total_seed_tx
top_10_share = seeds.sort_values("tx_count", ascending=False)["tx_count"].head(10).sum() / total_seed_tx

print(f"Top 5 seeds share of all seed transactions/edges: {top_5_share:.4f}")
print(f"Top 10 seeds share of all seed transactions/edges: {top_10_share:.4f}")
print("\n")

if top_5_share > 0.5:
    print("WARNING: Top 5 seeds contribute more than 50% of seed activity. Consider top-K neighbour filtering.")
elif top_10_share > 0.5:
    print("NOTE: Top 10 seeds contribute more than 50% of seed activity. Check whether CEX nodes dominate the graph.")
else:
    print("No extreme domination detected from top seeds based on this threshold.")

print("\n")

# =========================
# 10. Edge contribution by class
# =========================

print("=== Check 8: Class Contribution to Total Seed Activity ===")

class_activity = seeds.groupby(LABEL_COL)["tx_count"].sum()
class_activity_share = (class_activity / class_activity.sum()).sort_values(ascending=False)

class_activity_report = pd.DataFrame({
    "total_tx_or_edges": class_activity,
    "share": class_activity_share
}).sort_values("share", ascending=False)

print(class_activity_report.round(4))
print("\n")

# =========================
# 11. Optional: transaction value checks
# =========================

print("=== Check 9: Transaction Value Checks ===")

value_cols = [
    c for c in edges.columns
    if c.lower() in [
        "value", "value_eth", "amount", "amount_eth",
        "total_value", "total_value_eth", "total_eth_value"
    ]
]

if len(value_cols) == 0:
    print("No obvious value column detected. Skipping value checks.")
else:
    value_col = value_cols[0]
    print(f"Detected value column: {value_col}")

    edges[value_col] = pd.to_numeric(edges[value_col], errors="coerce")
    print(edges[value_col].describe())

    zero_value = (edges[value_col] == 0).sum()
    negative_value = (edges[value_col] < 0).sum()
    missing_value = edges[value_col].isna().sum()

    print(f"Zero-value rows: {zero_value}")
    print(f"Negative-value rows: {negative_value}")
    print(f"Missing-value rows: {missing_value}")

print("\n")

# =========================
# 12. Save checked seed summary
# =========================

seeds.to_csv("seed_sanity_check_summary.csv", index=False)
print("Saved seed-level sanity check summary to seed_sanity_check_summary.csv")