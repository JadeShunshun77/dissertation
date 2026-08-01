import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# =========================
# 0. Setup
# =========================
# Optional CLI arg: feat_file -- defaults to the Plan-B reconstructed graph's
# feature matrix (progress_summary.md §13), since the reconstructed graph is
# the project's main-line dataset from Week 11 onward.
FEAT_FILE = sys.argv[1] if len(sys.argv) > 1 else "seed_feature_matrix_reconstructed.csv"

feat = pd.read_csv(FEAT_FILE)

OUT_DIR = Path("eda_outputs")
OUT_DIR.mkdir(exist_ok=True)

LABEL_COL = "label"
LABEL_ORDER = ["CEX", "DeFi", "Malicious_MEV"]
LABEL_COLORS = {"CEX": "#2a78d6", "DeFi": "#1baf7a", "Malicious_MEV": "#e34948"}

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

print(f"Loaded feature matrix: {feat.shape[0]} rows, {feat.shape[1]} columns")

# =========================
# 1. Sanity check: feature_engineering.py already handles gas fee NaNs
# =========================
# (fillna + has_out_edge/has_in_edge flags live in feature_engineering.py so the
# feature matrix stays canonical there; EDA only reads/analyses it.)

assert feat["out_gas_fee_mean"].isna().sum() == 0, "run feature_engineering.py first"
assert feat["in_gas_fee_mean"].isna().sum() == 0, "run feature_engineering.py first"

# =========================
# 2. Activity breakdown by label
# =========================

activity = feat.groupby(LABEL_COL)["is_active"].agg(
    active="sum", total="count", active_rate="mean"
).reindex(LABEL_ORDER)
activity["active_rate"] = activity["active_rate"].round(4)

print("\n=== Activity by label ===")
print(activity)
activity.to_csv(OUT_DIR / "activity_by_label.csv")

active = feat[feat["is_active"]].copy()
print(f"\nActive subset used for distribution/skew/correlation analysis: {len(active)} of {len(feat)} seeds "
      "(inactive seeds have all-zero transaction/graph features and would flatten the plots).")

# =========================
# 3. Skewness: raw vs log-transformed
# =========================

skew_pairs = [
    ("total_tx_count", "log_total_tx_count"),
    ("total_eth_value", "log_total_eth_value"),
    ("degree", "log_degree"),
    ("weighted_degree", "log_weighted_degree"),
    ("pagerank", "log_pagerank"),
]

skew_rows = []
for raw_col, log_col in skew_pairs:
    skew_rows.append({
        "feature": raw_col,
        "raw_skew": active[raw_col].skew(),
        "log_skew": active[log_col].skew(),
    })
skew_df = pd.DataFrame(skew_rows).round(3)

print("\n=== Skewness: raw vs log1p (active seeds only) ===")
print(skew_df.to_string(index=False))
skew_df.to_csv(OUT_DIR / "skewness_raw_vs_log.csv", index=False)

# =========================
# 4. Zero-inflation by label (full dataset, including inactive seeds)
# =========================

zero_cols = ["total_tx_count", "total_eth_value", "out_edge_count", "in_edge_count",
             "clustering_coeff", "weighted_degree"]

zero_rows = []
for label in LABEL_ORDER:
    sub = feat[feat[LABEL_COL] == label]
    row = {"label": label, "n": len(sub)}
    for col in zero_cols:
        row[f"pct_zero_{col}"] = round((sub[col] == 0).mean(), 4)
    zero_rows.append(row)
zero_df = pd.DataFrame(zero_rows)

print("\n=== Zero-inflation by label (all 500 seeds, includes inactive) ===")
print(zero_df.to_string(index=False))
zero_df.to_csv(OUT_DIR / "zero_inflation_by_label.csv", index=False)

# =========================
# 5. Distribution plots by label (active seeds only, log-scaled where skewed)
# =========================

plot_cols = [
    ("log_total_tx_count", "log1p(total tx count)"),
    ("log_total_eth_value", "log1p(total ETH value)"),
    ("log_degree", "log1p(degree)"),
    ("log_weighted_degree", "log1p(weighted degree)"),
    ("log_pagerank", "log1p(pagerank)"),
    ("clustering_coeff", "clustering coefficient"),
    ("zero_value_tx_ratio", "zero-value tx ratio"),
]

n_cols = 4
n_rows = int(np.ceil(len(plot_cols) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.6 * n_rows))
axes = axes.flatten()

for ax, (col, title) in zip(axes, plot_cols):
    sns.boxplot(
        data=active, x=LABEL_COL, y=col, order=LABEL_ORDER,
        hue=LABEL_COL, palette=LABEL_COLORS, legend=False,
        width=0.55, linewidth=1, fliersize=2, ax=ax,
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)

for ax in axes[len(plot_cols):]:
    ax.axis("off")

fig.suptitle("Feature distributions by address label (active seeds only, n=%d)" % len(active), fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT_DIR / "feature_distributions_by_label.png", dpi=150)
plt.close(fig)
print(f"\nSaved {OUT_DIR / 'feature_distributions_by_label.png'}")

# =========================
# 6. Correlation heatmap among engineered features (active seeds only)
# =========================

corr_cols = [
    "log_total_tx_count", "log_total_eth_value", "zero_value_tx_ratio",
    "log_degree", "log_weighted_degree", "log_pagerank", "clustering_coeff", "wcc_size",
]
corr = active[corr_cols].corr()

diverging_cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#2a78d6", "#f0efec", "#e34948"]
)

fig, ax = plt.subplots(figsize=(7.5, 6.5))
im = ax.imshow(corr, cmap=diverging_cmap, vmin=-1, vmax=1)
ax.set_xticks(range(len(corr_cols)))
ax.set_xticklabels(corr_cols, rotation=45, ha="right")
ax.set_yticks(range(len(corr_cols)))
ax.set_yticklabels(corr_cols)
for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        val = corr.iloc[i, j]
        text_color = "#ffffff" if abs(val) > 0.6 else "#0b0b0b"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=text_color)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Pearson correlation")
ax.set_title("Engineered feature correlation (active seeds only)", fontsize=12)
fig.tight_layout()
fig.savefig(OUT_DIR / "feature_correlation_heatmap.png", dpi=150)
plt.close(fig)
print(f"Saved {OUT_DIR / 'feature_correlation_heatmap.png'}")

# =========================
# 7. Summary
# =========================

print("\n=== EDA summary ===")
print(f"- Activity table, skew table, zero-inflation table saved as CSV in {OUT_DIR}/")
print(f"- Distribution plot and correlation heatmap saved as PNG in {OUT_DIR}/")
