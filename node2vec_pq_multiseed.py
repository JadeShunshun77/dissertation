import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

from node2vec import Node2Vec
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# =========================
# 0. Setup
# =========================
# Follow-up to node2vec_pq_sensitivity.py: that script trained Node2Vec ONCE
# per (p, q) setting and then resampled 30 balanced evaluation subsets from
# that single fixed embedding/clustering. That tests robustness of the
# balanced-ARI *evaluation* to which subset is drawn, but not robustness to
# Node2Vec's own training randomness (walk generation + Word2Vec SGD are
# stochastic even at fixed random_state, per the caveat already noted in
# progress_summary.md Sec 7.5). This script retrains each of the 3 p/q
# settings from node2vec_pq_sensitivity.py across 5 independent training
# seeds (15 total Node2Vec/Word2Vec trainings) and reports how much the
# balanced-subset ARI varies across training seeds, not just across
# evaluation subsets -- the missing piece flagged in review before treating
# BFS-like (p=0.25, q=4) as anything more than an exploratory finding.

EMBED_DIM = 64
N_TRAIN_SEEDS = 5
N_BALANCED_RESAMPLES_PER_SEED = 10  # reduced from 30 (Sec 7.5) since cost is now x5 (train seeds)
TRAIN_SEEDS = [42, 43, 44, 45, 46]
LABEL_COL = "label"
LABEL_ORDER = ["CEX", "DeFi", "Malicious_MEV"]
PQ_SETTINGS = [
    ("baseline (unbiased, DeepWalk-equivalent)", 1, 1),
    ("BFS-like (local/structural)", 0.25, 4),
    ("DFS-like (outward/community)", 4, 0.25),
]

OUT_DIR = Path("embedding_outputs")
OUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": "#0b0b0b", "text.color": "#0b0b0b", "xtick.color": "#52514e",
    "ytick.color": "#52514e", "grid.color": "#e1e0d9", "font.size": 10,
})


def clean_address(x):
    if pd.isna(x):
        return np.nan
    return str(x).strip().lower()


# =========================
# 1. Build graph (identical construction to node2vec_clustering.py)
# =========================

edges = pd.read_csv("raw_edges.csv")
edges["source"] = edges["source"].apply(clean_address)
edges["target"] = edges["target"].apply(clean_address)
edges = edges.dropna(subset=["source", "target"])

G_directed = nx.from_pandas_edgelist(edges, source="source", target="target", edge_attr=["tx_count"], create_using=nx.DiGraph())
G = G_directed.to_undirected()
for _, _, d in G.edges(data=True):
    d["weight"] = d.get("tx_count", 1)

feat = pd.read_csv("seed_feature_matrix.csv")
seed_labels = feat[["address", LABEL_COL, "is_active"]].copy()

# =========================
# 2. For each (p, q), retrain across N_TRAIN_SEEDS independent seeds
# =========================

per_seed_rows = []
for label, p, q in PQ_SETTINGS:
    for train_seed in TRAIN_SEEDS:
        print(f"\n=== {label} (p={p}, q={q}), train_seed={train_seed} ===")
        np.random.seed(train_seed)
        node2vec = Node2Vec(G, dimensions=EMBED_DIM, walk_length=30, num_walks=10, p=p, q=q, workers=1, quiet=True)
        model = Word2Vec(node2vec.walks, vector_size=EMBED_DIM, window=10, min_count=1, sg=1, workers=1, seed=train_seed)

        emb_df = pd.DataFrame({node: model.wv[str(node)] for node in G.nodes()}).T
        emb_df.columns = [f"emb_{i}" for i in range(EMBED_DIM)]
        emb_df.index.name = "address"

        seed_emb = seed_labels.merge(emb_df, left_on="address", right_index=True, how="left")
        seed_emb = seed_emb[~seed_emb[emb_df.columns].isna().any(axis=1)].reset_index(drop=True)
        X = seed_emb[emb_df.columns].values

        kmeans = KMeans(n_clusters=len(LABEL_ORDER), random_state=train_seed, n_init=10)
        clusters = kmeans.fit_predict(X)

        df_eval = seed_emb[[LABEL_COL]].copy()
        df_eval["cluster"] = clusters
        n_per_class = df_eval[LABEL_COL].value_counts().min()
        rng_master = np.random.default_rng(train_seed)
        resample_seeds = rng_master.integers(0, 1_000_000, size=N_BALANCED_RESAMPLES_PER_SEED)
        balanced_aris = []
        for rs in resample_seeds:
            bal = (
                df_eval.groupby(LABEL_COL, group_keys=False)
                .apply(lambda g: g.sample(n=n_per_class, random_state=int(rs)), include_groups=True)
            )
            balanced_aris.append(adjusted_rand_score(bal[LABEL_COL].values, bal["cluster"].values))
        balanced_aris = np.array(balanced_aris)

        print(f"  balanced ARI ({N_BALANCED_RESAMPLES_PER_SEED}x resamples): mean={balanced_aris.mean():.4f}, std={balanced_aris.std():.4f}")

        per_seed_rows.append({
            "setting": label, "p": p, "q": q, "train_seed": train_seed,
            "balanced_ari_mean_over_resamples": balanced_aris.mean(),
            "balanced_ari_std_over_resamples": balanced_aris.std(),
        })

per_seed_df = pd.DataFrame(per_seed_rows).round(4)
per_seed_df.to_csv(OUT_DIR / "pq_multiseed_per_seed.csv", index=False)
print(f"\nSaved {OUT_DIR / 'pq_multiseed_per_seed.csv'}")

# =========================
# 3. Aggregate across training seeds per (p, q) setting
# =========================

summary_rows = []
for label, p, q in PQ_SETTINGS:
    sub = per_seed_df[per_seed_df["setting"] == label]
    vals = sub["balanced_ari_mean_over_resamples"].values
    summary_rows.append({
        "setting": label, "p": p, "q": q, "n_train_seeds": len(vals),
        "ari_mean_across_seeds": vals.mean(), "ari_std_across_seeds": vals.std(),
        "ari_min_across_seeds": vals.min(), "ari_max_across_seeds": vals.max(),
    })
summary_df = pd.DataFrame(summary_rows).round(4)
summary_df.to_csv(OUT_DIR / "pq_multiseed_summary.csv", index=False)
print(f"\n=== Aggregated across {N_TRAIN_SEEDS} training seeds ===\n{summary_df.to_string(index=False)}")
print(f"Saved {OUT_DIR / 'pq_multiseed_summary.csv'}")

# =========================
# 4. Plot: per-seed balanced ARI, grouped by setting
# =========================

fig, ax = plt.subplots(figsize=(8, 5))
colors = {"baseline (unbiased, DeepWalk-equivalent)": "#2a78d6",
          "BFS-like (local/structural)": "#1baf7a",
          "DFS-like (outward/community)": "#d65f2a"}
for i, (label, p, q) in enumerate(PQ_SETTINGS):
    sub = per_seed_df[per_seed_df["setting"] == label]
    xs = np.full(len(sub), i) + np.random.default_rng(0).uniform(-0.1, 0.1, size=len(sub))
    ax.scatter(xs, sub["balanced_ari_mean_over_resamples"], color=colors[label], s=40, alpha=0.8)
    ax.scatter([i], [sub["balanced_ari_mean_over_resamples"].mean()], color=colors[label], s=140, marker="D", edgecolor="black", linewidth=1, zorder=5)
ax.set_xticks(range(len(PQ_SETTINGS)))
ax.set_xticklabels([l for l, _, _ in PQ_SETTINGS], rotation=15, ha="right", fontsize=8)
ax.set_ylabel(f"Balanced-subset ARI (mean over {N_BALANCED_RESAMPLES_PER_SEED} resamples per training seed)")
ax.set_title(f"Node2Vec p/q x training-seed robustness ({N_TRAIN_SEEDS} training seeds each; diamond = mean)")
ax.grid(axis="y", linewidth=0.6, alpha=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT_DIR / "pq_multiseed_distribution.png", dpi=150)
plt.close(fig)
print(f"Saved {OUT_DIR / 'pq_multiseed_distribution.png'}")

print("\n=== p/q x training-seed robustness check complete ===")
