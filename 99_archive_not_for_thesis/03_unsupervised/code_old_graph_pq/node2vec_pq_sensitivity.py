import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import sys

from node2vec import Node2Vec
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

# =========================
# 0. Setup
# =========================
# Week 10 sensitivity analysis (Plan Risk 5 mitigation, Week 10 milestone
# "sensitivity analysis" -- flagged but deferred in node2vec_clustering.py's
# own comment at training time). Node2Vec's p/q parameters control the random
# walk's exploration bias: p=q=1 (used throughout Week 7/8) is an unbiased
# walk, equivalent to DeepWalk. This script checks whether the Week 7/8/10
# conclusions about embedding structure (moderate-but-real, see progress_
# summary.md Section 4.4/7) depend on that specific, arbitrary choice, or
# hold up under biased walks that explore the graph differently:
#   - BFS-like (p small, q large): stays close to the start node, favours
#     capturing local/structural-equivalence roles
#   - DFS-like (p large, q small): wanders further from the start node,
#     favours capturing homophily/community structure
# in [(0.25, 4), (4, 0.25)] are the standard extreme grid points from Grover &
# Leskovec (2016)'s own p/q sensitivity grid.

RANDOM_STATE = 42
EMBED_DIM = 64
N_BALANCED_RESAMPLES = 30
LABEL_COL = "label"
LABEL_ORDER = ["CEX", "DeFi", "Malicious_MEV"]
PQ_SETTINGS = [
    ("baseline (unbiased, DeepWalk-equivalent)", 1, 1),
    ("BFS-like (local/structural)", 0.25, 4),
    ("DFS-like (outward/community)", 4, 0.25),
]

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EDGES_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "00_data_graph" / "data" / "reconstruction_reference" / "raw_edges.csv"
FEAT_FILE = Path(sys.argv[2]) if len(sys.argv) > 2 else PROJECT_ROOT / "99_archive_not_for_thesis" / "02_feature_engineering" / "seed_feature_matrix_pre_reconstruction.csv"
OUT_DIR = Path(sys.argv[3]) if len(sys.argv) > 3 else PROJECT_ROOT / "99_archive_not_for_thesis" / "03_unsupervised" / "outputs" / "embedding_outputs_pre_reconstruction"
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


def permutation_significance(y_true, y_pred, n_perm=2000, random_state=RANDOM_STATE):
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    nmi_obs = normalized_mutual_info_score(y_true, y_pred)
    ari_obs = adjusted_rand_score(y_true, y_pred)

    nmi_null = np.empty(n_perm)
    ari_null = np.empty(n_perm)
    for i in range(n_perm):
        y_shuf = rng.permutation(y_true)
        nmi_null[i] = normalized_mutual_info_score(y_shuf, y_pred)
        ari_null[i] = adjusted_rand_score(y_shuf, y_pred)

    return {
        "nmi_observed": nmi_obs, "nmi_p_value": (nmi_null >= nmi_obs).mean(),
        "ari_observed": ari_obs, "ari_null_mean": ari_null.mean(),
        "ari_p_value": (ari_null >= ari_obs).mean(),
    }


# =========================
# 1. Build graph (identical construction to node2vec_clustering.py)
# =========================

edges = pd.read_csv(EDGES_FILE)
edges["source"] = edges["source"].apply(clean_address)
edges["target"] = edges["target"].apply(clean_address)
edges = edges.dropna(subset=["source", "target"])

G_directed = nx.from_pandas_edgelist(edges, source="source", target="target", edge_attr=["tx_count"], create_using=nx.DiGraph())
G = G_directed.to_undirected()
for _, _, d in G.edges(data=True):
    d["weight"] = d.get("tx_count", 1)

feat = pd.read_csv(FEAT_FILE)
seed_labels = feat[["address", LABEL_COL, "is_active"]].copy()

# =========================
# 2. Train Node2Vec at each (p, q), evaluate KMeans clustering on labelled seeds
# =========================
# Baseline (1, 1) reuses the exact numbers already in
# embedding_outputs/clustering_significance_test.csv from node2vec_clustering.py
# (re-trained here too, for a same-script, apples-to-apples comparison table --
# Node2Vec's walk generation is stochastic even at a fixed random_state across
# library versions/call sites, so re-deriving it here rather than reusing the
# old CSV keeps the comparison internally consistent).

results = []
for label, p, q in PQ_SETTINGS:
    print(f"\n=== Training Node2Vec: {label} (p={p}, q={q}) ===")
    np.random.seed(RANDOM_STATE)
    node2vec = Node2Vec(G, dimensions=EMBED_DIM, walk_length=30, num_walks=10, p=p, q=q, workers=1, quiet=False)
    model = Word2Vec(node2vec.walks, vector_size=EMBED_DIM, window=10, min_count=1, sg=1, workers=1, seed=RANDOM_STATE)

    emb_df = pd.DataFrame({node: model.wv[str(node)] for node in G.nodes()}).T
    emb_df.columns = [f"emb_{i}" for i in range(EMBED_DIM)]
    emb_df.index.name = "address"

    seed_emb = seed_labels.merge(emb_df, left_on="address", right_index=True, how="left")
    seed_emb = seed_emb[~seed_emb[emb_df.columns].isna().any(axis=1)].reset_index(drop=True)
    X = seed_emb[emb_df.columns].values
    y_true = seed_emb[LABEL_COL].values

    kmeans = KMeans(n_clusters=len(LABEL_ORDER), random_state=RANDOM_STATE, n_init=10)
    clusters = kmeans.fit_predict(X)

    sig = permutation_significance(y_true, clusters, n_perm=2000, random_state=RANDOM_STATE)
    print(f"KMeans: NMI={sig['nmi_observed']:.4f} (p={sig['nmi_p_value']:.4f}), "
          f"ARI={sig['ari_observed']:.4f} (p={sig['ari_p_value']:.4f})")

    # ---- balanced-subset re-evaluation (fixed clustering, 30 independent
    # balanced 51/51/51 resamples) -- full-sample ARI is known (Sec 4.4/7.3)
    # to be a weak/underpowered evaluation condition under this 299/86/51
    # label imbalance, so comparing p/q settings only on full-sample ARI
    # (as this script originally did) evaluates them at a condition already
    # known to attenuate the signal. Re-score each p/q setting's SAME fixed
    # clustering under the balanced condition used in balanced_subset_
    # robustness.py / repeated_balanced_ari.py for a fair comparison.
    df_eval = seed_emb[[LABEL_COL]].copy()
    df_eval["cluster"] = clusters
    n_per_class = df_eval[LABEL_COL].value_counts().min()
    rng_master = np.random.default_rng(RANDOM_STATE)
    resample_seeds = rng_master.integers(0, 1_000_000, size=N_BALANCED_RESAMPLES)
    balanced_aris, balanced_ps = [], []
    for rs in resample_seeds:
        bal = (
            df_eval.groupby(LABEL_COL, group_keys=False)
            .apply(lambda g: g.sample(n=n_per_class, random_state=int(rs)), include_groups=True)
        )
        ari_b, p_b = adjusted_rand_score(bal[LABEL_COL].values, bal["cluster"].values), None
        rng = np.random.default_rng(int(rs))
        null = np.array([adjusted_rand_score(rng.permutation(bal[LABEL_COL].values), bal["cluster"].values) for _ in range(500)])
        p_b = (null >= ari_b).mean()
        balanced_aris.append(ari_b)
        balanced_ps.append(p_b)
    balanced_aris = np.array(balanced_aris)
    balanced_ps = np.array(balanced_ps)
    print(f"Balanced resample ARI ({N_BALANCED_RESAMPLES}x, fixed clustering): "
          f"mean={balanced_aris.mean():.4f}, std={balanced_aris.std():.4f}, "
          f"pct_significant={100 * (balanced_ps < 0.05).mean():.0f}%")

    results.append({
        "setting": label, "p": p, "q": q, "n_seeds": len(seed_emb),
        "kmeans_nmi": sig["nmi_observed"], "kmeans_nmi_p": sig["nmi_p_value"],
        "kmeans_ari": sig["ari_observed"], "kmeans_ari_p": sig["ari_p_value"],
        "balanced_ari_mean": balanced_aris.mean(), "balanced_ari_std": balanced_aris.std(),
        "balanced_ari_pct_significant": (balanced_ps < 0.05).mean(),
    })

results_df = pd.DataFrame(results).round(4)
results_df.to_csv(OUT_DIR / "pq_sensitivity_results.csv", index=False)
print(f"\n=== p/q sensitivity summary ===\n{results_df.to_string(index=False)}")
print(f"Saved {OUT_DIR / 'pq_sensitivity_results.csv'}")

# =========================
# 3. Bar chart
# =========================

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(results_df))
bars = ax.bar(x, results_df["kmeans_ari"], color="#2a78d6")
for i, row in results_df.iterrows():
    marker = "significant" if row["kmeans_ari_p"] < 0.05 else "not significant"
    ax.text(i, row["kmeans_ari"] + 0.005, f"p={row['kmeans_ari_p']:.3f}\n({marker})", ha="center", fontsize=8)
ax.axhline(0, color="#c3c2b7", linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(results_df["setting"], rotation=15, ha="right", fontsize=8)
ax.set_ylabel("KMeans ARI (full sample, n=436)")
ax.set_title("Node2Vec p/q sensitivity: does walk bias change the clustering result?")
ax.grid(axis="y", linewidth=0.6, alpha=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT_DIR / "pq_sensitivity.png", dpi=150)
plt.close(fig)
print(f"Saved {OUT_DIR / 'pq_sensitivity.png'}")

print("\n=== Week 10 Node2Vec p/q sensitivity check complete ===")
