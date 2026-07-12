from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

# =========================
# 0. Setup
# =========================
# P2: balanced_subset_robustness.py (Week 10, progress_summary.md Section 7)
# only drew ONE random_state=42 balanced subset (51/51/51) and found KMeans/
# Louvain ARI jump from ~0 (full sample) to ~0.25-0.31 (that one subset).
# A single draw cannot establish whether that jump is a stable property of
# the embedding space or an artifact of which 51 CEX/DeFi seeds happened to
# be sampled. This repeats the balanced-subset evaluation 50 times with
# independent random seeds and reports the distribution of ARI, both for (a)
# evaluating the FIXED full-sample KMeans/Louvain clustering on each balanced
# subset, and (b) refitting KMeans fresh on each balanced subset.

RANDOM_STATE = 42
N_REPEATS = 50
LABEL_COL = "label"
LABEL_ORDER = ["CEX", "DeFi", "Malicious_MEV"]
N_PERM = 500  # reduced from 2000 (Week 7/10 default) since this runs 50x -- still ample for a 0.05 alpha threshold

OUT_DIR = Path("embedding_outputs")
OUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#c3c2b7",
                      "axes.labelcolor": "#0b0b0b", "text.color": "#0b0b0b", "xtick.color": "#52514e",
                      "ytick.color": "#52514e", "grid.color": "#e1e0d9", "font.size": 10})


def permutation_p(y_true, y_pred, n_perm=N_PERM, random_state=RANDOM_STATE):
    rng = np.random.default_rng(random_state)
    ari_obs = adjusted_rand_score(y_true, y_pred)
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = adjusted_rand_score(rng.permutation(y_true), y_pred)
    return ari_obs, (null >= ari_obs).mean()


# =========================
# 1. Load embeddings + labels, and the FIXED full-sample KMeans clustering
# =========================

seed_emb = pd.read_csv("embedding_outputs/seed_node2vec_embeddings.csv")
emb_cols = [c for c in seed_emb.columns if c.startswith("emb_")]
X_full = seed_emb[emb_cols].to_numpy(dtype=float)
y_full = seed_emb[LABEL_COL].to_numpy()

fixed_kmeans = KMeans(n_clusters=len(LABEL_ORDER), random_state=RANDOM_STATE, n_init=10)
fixed_clusters = fixed_kmeans.fit_predict(X_full)
seed_emb["kmeans_cluster_fixed"] = fixed_clusters

# =========================
# 2. Repeat balanced subsampling 50x
# =========================

n_per_class = seed_emb[LABEL_COL].value_counts().min()  # CEX = 51
rng_master = np.random.default_rng(RANDOM_STATE)
repeat_seeds = rng_master.integers(0, 1_000_000, size=N_REPEATS)

rows = []
for rep_i, seed in enumerate(repeat_seeds):
    balanced = (
        seed_emb.groupby(LABEL_COL, group_keys=False)
        .apply(lambda g: g.sample(n=n_per_class, random_state=int(seed)), include_groups=True)
    )

    # (a) evaluate the FIXED full-sample clustering on this balanced subset
    ari_fixed_eval, p_fixed_eval = permutation_p(balanced[LABEL_COL].values, balanced["kmeans_cluster_fixed"].values, random_state=int(seed))

    # (b) refit KMeans fresh on this balanced subset's embeddings
    X_bal = balanced[emb_cols].to_numpy(dtype=float)
    km_refit = KMeans(n_clusters=len(LABEL_ORDER), random_state=int(seed), n_init=10)
    refit_clusters = km_refit.fit_predict(X_bal)
    ari_refit, p_refit = permutation_p(balanced[LABEL_COL].values, refit_clusters, random_state=int(seed))

    rows.append({
        "repeat": rep_i, "seed": int(seed),
        "ari_fixed_clustering_eval": ari_fixed_eval, "p_fixed_clustering_eval": p_fixed_eval,
        "ari_refit_on_subset": ari_refit, "p_refit_on_subset": p_refit,
    })
    if (rep_i + 1) % 10 == 0:
        print(f"  repeat {rep_i + 1}/{N_REPEATS} done")

runs_df = pd.DataFrame(rows)
runs_df.to_csv(OUT_DIR / "balanced_ari_repeated_runs.csv", index=False)

# =========================
# 3. Summary: mean, std, percentiles, % significant
# =========================

summary_rows = []
for col_ari, col_p, label in [
    ("ari_fixed_clustering_eval", "p_fixed_clustering_eval", "KMeans (fixed full-sample clustering, balanced eval)"),
    ("ari_refit_on_subset", "p_refit_on_subset", "KMeans (refit fresh on each balanced subset)"),
]:
    ari_vals = runs_df[col_ari]
    p_vals = runs_df[col_p]
    summary_rows.append({
        "method": label, "ari_mean": ari_vals.mean(), "ari_std": ari_vals.std(),
        "ari_p5": ari_vals.quantile(0.05), "ari_p50": ari_vals.quantile(0.50), "ari_p95": ari_vals.quantile(0.95),
        "ari_min": ari_vals.min(), "ari_max": ari_vals.max(),
        "pct_significant_at_0.05": (p_vals < 0.05).mean(),
    })
summary_df = pd.DataFrame(summary_rows).round(4)
summary_df.to_csv(OUT_DIR / "balanced_ari_summary.csv", index=False)
print(f"\n=== ARI stability across {N_REPEATS} independent balanced-subset draws ===\n{summary_df.to_string(index=False)}")

# =========================
# 4. Distribution plot
# =========================

fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(runs_df["ari_fixed_clustering_eval"], bins=15, alpha=0.6, color="#2a78d6", label="Fixed clustering, balanced eval")
ax.hist(runs_df["ari_refit_on_subset"], bins=15, alpha=0.6, color="#1baf7a", label="Refit on each subset")
ax.axvline(0.0166, color="#c3c2b7", linestyle="--", linewidth=1, label="Full-sample ARI (0.017)")
ax.set_xlabel("KMeans ARI")
ax.set_ylabel(f"Count (of {N_REPEATS} repeats)")
ax.set_title(f"KMeans ARI distribution across {N_REPEATS} independent balanced (51/51/51) resamples")
ax.legend(frameon=False, fontsize=8)
ax.grid(axis="y", linewidth=0.6, alpha=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT_DIR / "balanced_ari_distribution.png", dpi=150)
plt.close(fig)
print(f"Saved {OUT_DIR / 'balanced_ari_distribution.png'}")

print("\n=== P2 repeated balanced ARI check complete ===")
