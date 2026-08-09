import sys
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

from node2vec import Node2Vec
import community as community_louvain
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score

from node2vec_reproducibility import (
    build_compatibility_metadata,
    seed_everything,
    stable_hash,
    write_json,
)

# =========================
# 0. Setup
# =========================
# Optional CLI args: edges_file output_subdir feature_file.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
EDGES_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "00_data_graph" / "data" / "main" / "raw_edges_reconstructed.csv"
OUT_SUBDIR = Path(sys.argv[2]) if len(sys.argv) > 2 else PROJECT_ROOT / "03_unsupervised" / "node2vec_outputs" / "node2vec_reconstructed"
FEAT_FILE = Path(sys.argv[3]) if len(sys.argv) > 3 else PROJECT_ROOT / "02_feature_engineering" / "feature_engineering" / "seed_feature_matrix_reconstructed.csv"

edges = pd.read_csv(EDGES_FILE)
feat = pd.read_csv(FEAT_FILE)

OUT_DIR = Path(OUT_SUBDIR)
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_COL, TARGET_COL, TX_COUNT_COL = "source", "target", "tx_count"
LABEL_COL = "label"
LABEL_ORDER = ["CEX", "DeFi", "Malicious_MEV"]
LABEL_COLORS = {"CEX": "#2a78d6", "DeFi": "#1baf7a", "Malicious_MEV": "#e34948"}
EMBED_DIM = 64
RANDOM_STATE = 42

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
    """Compare observed NMI/ARI against a null distribution built by shuffling
    y_true n_perm times (class proportions preserved, y_pred held fixed).

    NMI alone is misleading on imbalanced classes: it can look "high" purely
    because cluster sizes happen to match class proportions, with no real
    correspondence. ARI is chance-corrected (expected 0 under random labelling)
    and is the metric that should be trusted; NMI is reported alongside it only
    for reference. p-value = P(null statistic >= observed), one-sided.
    """
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
        "nmi_observed": nmi_obs, "nmi_null_mean": nmi_null.mean(), "nmi_null_std": nmi_null.std(),
        "nmi_p_value": (nmi_null >= nmi_obs).mean(),
        "ari_observed": ari_obs, "ari_null_mean": ari_null.mean(), "ari_null_std": ari_null.std(),
        "ari_p_value": (ari_null >= ari_obs).mean(),
    }


edges[SOURCE_COL] = edges[SOURCE_COL].apply(clean_address)
edges[TARGET_COL] = edges[TARGET_COL].apply(clean_address)
edges = edges.dropna(subset=[SOURCE_COL, TARGET_COL])

# =========================
# 1. Build graph for embedding
# =========================
# The sampled ego-network is directed, but most non-seed counterparties are
# leaves with only one recorded direction (top-100-per-seed sampling did not
# crawl their own neighbours), so directed random walks dead-end almost
# immediately for most nodes. Node2Vec is therefore trained on the undirected
# view, same as clustering_coeff in feature_engineering.py.

G_directed = nx.from_pandas_edgelist(
    edges, source=SOURCE_COL, target=TARGET_COL, edge_attr=[TX_COUNT_COL],
    create_using=nx.DiGraph(),
)
G = G_directed.to_undirected()
for _, _, d in G.edges(data=True):
    d["weight"] = d.get(TX_COUNT_COL, 1)

print(f"Graph for embedding: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# =========================
# 2. Train Node2Vec
# =========================
# p = q = 1 (unbiased walk, DeepWalk-equivalent) as a baseline; revisit as a
# sensitivity check later per Plan Risk 5 if time allows.

# workers=1 avoids joblib's multiprocessing resource_tracker, which crashes here
# because the Windows user profile path contains non-ASCII characters.
seed_everything(RANDOM_STATE)
node2vec = Node2Vec(
    G, dimensions=EMBED_DIM, walk_length=30, num_walks=10,
    p=1, q=1, weight_key="weight", workers=1, quiet=False,
)

# node2vec 0.3.2's own .fit() hardcodes gensim's pre-4.0 kwarg name 'size',
# which the installed gensim 4.4.0 no longer accepts (renamed to
# 'vector_size'). Call gensim directly on the walks it already generated
# instead of going through the incompatible wrapper.
from gensim.models import Word2Vec
model = Word2Vec(
    node2vec.walks, vector_size=EMBED_DIM, window=10, min_count=1,
    sg=1, workers=1, seed=RANDOM_STATE, hashfxn=stable_hash,
)

emb_df = pd.DataFrame(
    {node: model.wv[str(node)] for node in G.nodes()}
).T
emb_df.columns = [f"emb_{i}" for i in range(EMBED_DIM)]
emb_df.index.name = "address"
emb_df.to_csv(OUT_DIR / "node2vec_embeddings.csv")
print(f"Saved embeddings for {len(emb_df)} nodes to {OUT_DIR / 'node2vec_embeddings.csv'}")

# =========================
# 3. Join embeddings to labelled seeds
# =========================

seed_labels = feat[["address", LABEL_COL, "is_active"]].copy()
seed_emb = seed_labels.merge(emb_df, left_on="address", right_index=True, how="left")

emb_cols = list(emb_df.columns)
missing = seed_emb[emb_cols].isna().any(axis=1)
print(f"Labelled seeds without an embedding (not in sampled graph): {missing.sum()} / {len(seed_emb)}")
seed_emb = seed_emb[~missing].reset_index(drop=True)
seed_emb.to_csv(OUT_DIR / "seed_node2vec_embeddings.csv", index=False)

X = seed_emb[emb_cols].values
y_true = seed_emb[LABEL_COL].values

# =========================
# 4. t-SNE visualisation of labelled seeds
# =========================
# UMAP was skipped: umap-learn's package __init__ unconditionally imports a
# ParametricUMAP path that pulls in a broken system-wide torch install on this
# machine (OSError on DLL load, unrelated to this project). t-SNE satisfies
# the same Plan requirement ("UMAP/t-SNE visualisations") without that dependency.

tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30, init="pca")
X_tsne = tsne.fit_transform(X)
seed_emb["tsne_x"], seed_emb["tsne_y"] = X_tsne[:, 0], X_tsne[:, 1]

fig, ax = plt.subplots(figsize=(7, 6))
for label in LABEL_ORDER:
    sub = seed_emb[seed_emb[LABEL_COL] == label]
    ax.scatter(sub["tsne_x"], sub["tsne_y"], s=20, alpha=0.75,
               color=LABEL_COLORS[label], label=label, edgecolors="none")
ax.set_title(f"t-SNE of Node2Vec embeddings (labelled seeds, n={len(seed_emb)})")
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.legend(title="Label", frameon=False)
ax.grid(linewidth=0.6, alpha=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT_DIR / "tsne_node2vec_seeds.png", dpi=150)
plt.close(fig)
print(f"Saved {OUT_DIR / 'tsne_node2vec_seeds.png'}")

# =========================
# 5. K-Means clustering (on embeddings) + evaluation
# =========================

k = len(LABEL_ORDER)
kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
seed_emb["kmeans_cluster"] = kmeans.fit_predict(X)

nmi_kmeans = normalized_mutual_info_score(y_true, seed_emb["kmeans_cluster"])
ari_kmeans = adjusted_rand_score(y_true, seed_emb["kmeans_cluster"])
sil_kmeans = silhouette_score(X, seed_emb["kmeans_cluster"])

print(f"\nKMeans (k={k}): NMI={nmi_kmeans:.3f}  ARI={ari_kmeans:.3f}  Silhouette={sil_kmeans:.3f}")

# =========================
# 6. Louvain community detection (on graph topology) + evaluation
# =========================
# Louvain runs on the full 8,029-node graph (not just the embeddings), then
# results are read off for the labelled seeds. It will likely find more than
# 3 communities since it optimises modularity over the whole sampled network,
# not our 3-class taxonomy -- NMI/ARI still compare cluster agreement without
# needing matching cluster counts or labels.

partition = community_louvain.best_partition(G, weight="weight", random_state=RANDOM_STATE)
seed_emb["louvain_cluster"] = seed_emb["address"].map(partition)

n_louvain_clusters = seed_emb["louvain_cluster"].nunique()
nmi_louvain = normalized_mutual_info_score(y_true, seed_emb["louvain_cluster"])
ari_louvain = adjusted_rand_score(y_true, seed_emb["louvain_cluster"])
sil_louvain = (
    silhouette_score(X, seed_emb["louvain_cluster"])
    if n_louvain_clusters > 1 else np.nan
)

print(f"Louvain ({n_louvain_clusters} communities among labelled seeds): "
      f"NMI={nmi_louvain:.3f}  ARI={ari_louvain:.3f}  Silhouette={sil_louvain:.3f}")
print("(Louvain Silhouette is measured in Node2Vec embedding space as an approximation, "
      "since Louvain itself partitions on graph topology, not embedding distance.)")

# =========================
# 7. Save clustering evaluation summary
# =========================

summary = pd.DataFrame([
    {"method": "KMeans", "n_clusters": k, "NMI": nmi_kmeans, "ARI": ari_kmeans, "Silhouette": sil_kmeans},
    {"method": "Louvain", "n_clusters": n_louvain_clusters, "NMI": nmi_louvain, "ARI": ari_louvain, "Silhouette": sil_louvain},
]).round(4)

summary.to_csv(OUT_DIR / "clustering_evaluation_summary.csv", index=False)
print(f"\n=== Clustering evaluation summary ===\n{summary.to_string(index=False)}")
print(f"\nSaved to {OUT_DIR / 'clustering_evaluation_summary.csv'}")

# =========================
# 8. Permutation significance test (KMeans, Louvain)
# =========================
# Raw NMI/ARI can look deceptively "good" purely from class imbalance
# (299/86/51 MEV/DeFi/CEX). This compares each clustering's NMI/ARI against a
# null distribution from 2000 random label permutations, holding the
# clustering fixed, to check whether the agreement is actually above chance.

print("\n=== Permutation significance test (2000 shuffles, holding cluster assignment fixed) ===")

sig_rows = []
for method_name, pred_col in [("KMeans", "kmeans_cluster"), ("Louvain", "louvain_cluster")]:
    result = permutation_significance(y_true, seed_emb[pred_col].values, n_perm=2000, random_state=RANDOM_STATE)
    result["method"] = method_name
    sig_rows.append(result)
    verdict = "SIGNIFICANT" if result["ari_p_value"] < 0.05 else "NOT significant"
    print(f"{method_name}: ARI={result['ari_observed']:.4f} (null mean={result['ari_null_mean']:.4f}, "
          f"p={result['ari_p_value']:.4f}) -> {verdict} at alpha=0.05  "
          f"[NMI={result['nmi_observed']:.4f}, NMI p={result['nmi_p_value']:.4f} -- "
          f"NMI is reported for reference only, ARI is the trustworthy metric here]")

sig_df = pd.DataFrame(sig_rows)[[
    "method", "nmi_observed", "nmi_null_mean", "nmi_null_std", "nmi_p_value",
    "ari_observed", "ari_null_mean", "ari_null_std", "ari_p_value",
]].round(4)
sig_df.to_csv(OUT_DIR / "clustering_significance_test.csv", index=False)
print(f"Saved to {OUT_DIR / 'clustering_significance_test.csv'}")

# =========================
# 9. Silhouette / NMI / ARI scan across k=2..8
# =========================
# Answers "is k=3 the wrong choice?" directly: if the curve stays flat/low
# across all k, the weak KMeans result is a property of the embedding space,
# not a bad k choice.

k_range = range(2, 9)
k_rows = []
for k_try in k_range:
    km_k = KMeans(n_clusters=k_try, random_state=RANDOM_STATE, n_init=10)
    labels_k = km_k.fit_predict(X)
    k_rows.append({
        "k": k_try,
        "silhouette": silhouette_score(X, labels_k),
        "NMI": normalized_mutual_info_score(y_true, labels_k),
        "ARI": adjusted_rand_score(y_true, labels_k),
    })
k_scan_df = pd.DataFrame(k_rows).round(4)
k_scan_df.to_csv(OUT_DIR / "kmeans_k_scan.csv", index=False)
print(f"\n=== KMeans k=2..8 scan ===\n{k_scan_df.to_string(index=False)}")
print(f"Saved to {OUT_DIR / 'kmeans_k_scan.csv'}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(k_scan_df["k"], k_scan_df["silhouette"], marker="o", color="#2a78d6", linewidth=2)
axes[0].set_title("Silhouette score vs k")
axes[0].set_xlabel("k")
axes[0].set_ylabel("Silhouette")
axes[0].axvline(3, color="#c3c2b7", linestyle="--", linewidth=1)
axes[0].grid(linewidth=0.6, alpha=0.6)
axes[0].set_axisbelow(True)

axes[1].plot(k_scan_df["k"], k_scan_df["ARI"], marker="o", color="#2a78d6", linewidth=2)
axes[1].set_title("ARI vs k (against true label)")
axes[1].set_xlabel("k")
axes[1].set_ylabel("ARI")
axes[1].axvline(3, color="#c3c2b7", linestyle="--", linewidth=1)
axes[1].axhline(0, color="#c3c2b7", linestyle="-", linewidth=1)
axes[1].grid(linewidth=0.6, alpha=0.6)
axes[1].set_axisbelow(True)

fig.suptitle("KMeans robustness scan: is k=3 the reason the result is weak?", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT_DIR / "kmeans_k_scan.png", dpi=150)
plt.close(fig)
print(f"Saved {OUT_DIR / 'kmeans_k_scan.png'}")

# =========================
# 10. DBSCAN robustness check
# =========================
# Density-based clustering as a third, independent method. eps candidates are
# picked from the empirical distribution of 5th-nearest-neighbour distances
# (standard k-distance elbow heuristic) rather than guessed constants.

X_scaled = StandardScaler().fit_transform(X)
MIN_SAMPLES = 5

nn = NearestNeighbors(n_neighbors=MIN_SAMPLES).fit(X_scaled)
kth_distances = np.sort(nn.kneighbors(X_scaled)[0][:, -1])

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(np.arange(len(kth_distances)), kth_distances, color="#2a78d6", linewidth=2)
ax.set_title(f"{MIN_SAMPLES}-NN distance curve (DBSCAN eps selection)")
ax.set_xlabel("points sorted by distance")
ax.set_ylabel(f"distance to {MIN_SAMPLES}th nearest neighbour")
ax.grid(linewidth=0.6, alpha=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT_DIR / "dbscan_k_distance.png", dpi=150)
plt.close(fig)

eps_candidates = np.percentile(kth_distances, [30, 50, 70, 80, 90, 95])
dbscan_rows = []
for eps in eps_candidates:
    labels_db = DBSCAN(eps=eps, min_samples=MIN_SAMPLES).fit_predict(X_scaled)
    n_clusters_db = len(set(labels_db)) - (1 if -1 in labels_db else 0)
    noise_ratio = (labels_db == -1).mean()
    row = {"eps": eps, "n_clusters": n_clusters_db, "noise_ratio": noise_ratio}
    if n_clusters_db >= 2:
        row["NMI"] = normalized_mutual_info_score(y_true, labels_db)
        row["ARI"] = adjusted_rand_score(y_true, labels_db)
    else:
        row["NMI"] = np.nan
        row["ARI"] = np.nan
    dbscan_rows.append(row)

dbscan_df = pd.DataFrame(dbscan_rows).round(4)
dbscan_df.to_csv(OUT_DIR / "dbscan_eps_scan.csv", index=False)
print(f"\n=== DBSCAN eps scan (min_samples={MIN_SAMPLES}) ===\n{dbscan_df.to_string(index=False)}")
print(f"Saved {OUT_DIR / 'dbscan_eps_scan.csv'} and {OUT_DIR / 'dbscan_k_distance.png'}")

if (dbscan_df["noise_ratio"] > 0.5).all():
    print("\nAt every eps tried, DBSCAN labels the majority of points as noise: "
          "no density-based cluster structure in this embedding space, consistent "
          "with the flat silhouette scan and the non-significant KMeans result.")
print(f"\nSaved to {OUT_DIR / 'clustering_evaluation_summary.csv'}")

run_metadata = build_compatibility_metadata(
    graph_file=EDGES_FILE,
    feature_file=FEAT_FILE,
    parameters={
        "dimensions": EMBED_DIM,
        "walk_length": 30,
        "num_walks": 10,
        "p": 1,
        "q": 1,
        "weight_key": "weight",
        "window": 10,
        "min_count": 1,
        "sg": 1,
        "seed": RANDOM_STATE,
    },
    code_files=[Path(__file__), Path(__file__).with_name("node2vec_reproducibility.py")],
)
run_metadata["status"] = "complete"
write_json(OUT_DIR / "node2vec_run_metadata.json", run_metadata)
print(f"Saved {OUT_DIR / 'node2vec_run_metadata.json'}")
