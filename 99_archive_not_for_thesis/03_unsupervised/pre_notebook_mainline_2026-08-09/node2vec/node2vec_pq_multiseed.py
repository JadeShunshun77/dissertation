import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import sys

from node2vec import Node2Vec
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from node2vec_reproducibility import (
    build_compatibility_metadata,
    read_json,
    seed_everything,
    sha256_file,
    stable_hash,
    write_json,
)

# =========================
# 0. Setup
# =========================
# Reconstructed-graph follow-up to node2vec_pq_sensitivity.py. The archived
# copy preserves the original pre-reconstruction inputs and outputs.
#
# Follow-up to node2vec_pq_sensitivity.py: that script trained Node2Vec ONCE
# per (p, q) setting and then resampled 30 balanced evaluation subsets from
# that single fixed embedding/clustering. That tests robustness of the
# balanced-ARI *evaluation* to which subset is drawn, but not robustness to
# Node2Vec's own training randomness comes from walk generation and Word2Vec
# SGD. M07 now seeds Python and NumPy explicitly and uses a stable gensim hash
# so a training seed identifies a reproducible run in the recorded environment.
# This script retrains each of the 3 p/q
# settings from node2vec_pq_sensitivity.py across 5 independent training
# seeds (15 total Node2Vec/Word2Vec trainings) and reports how much the
# balanced-subset ARI varies across training seeds, not just across
# evaluation subsets -- the missing piece flagged in review before treating
# BFS-like (p=0.25, q=4) as anything more than an exploratory finding.

EMBED_DIM = 64
WALK_LENGTH = 30
NUM_WALKS = 10
WINDOW = 10
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EDGES_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "00_data_graph" / "data" / "main" / "raw_edges_reconstructed.csv"
FEAT_FILE = Path(sys.argv[2]) if len(sys.argv) > 2 else PROJECT_ROOT / "02_feature_engineering" / "feature_engineering" / "seed_feature_matrix_reconstructed.csv"
OUT_DIR = Path(sys.argv[3]) if len(sys.argv) > 3 else PROJECT_ROOT / "03_unsupervised" / "node2vec_outputs" / "node2vec_reconstructed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PER_SEED_FILE = OUT_DIR / "pq_multiseed_per_seed.csv"
CHECKPOINT_METADATA_FILE = OUT_DIR / "pq_multiseed_checkpoint_metadata.json"

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

EXPERIMENT_PARAMETERS = {
    "dimensions": EMBED_DIM,
    "walk_length": WALK_LENGTH,
    "num_walks": NUM_WALKS,
    "pq_settings": PQ_SETTINGS,
    "weight_key": "weight",
    "window": WINDOW,
    "min_count": 1,
    "sg": 1,
    "workers": 1,
    "train_seeds": TRAIN_SEEDS,
    "balanced_resamples_per_seed": N_BALANCED_RESAMPLES_PER_SEED,
    "label_order": LABEL_ORDER,
}
COMPATIBILITY = build_compatibility_metadata(
    graph_file=EDGES_FILE,
    feature_file=FEAT_FILE,
    parameters=EXPERIMENT_PARAMETERS,
    code_files=[Path(__file__), Path(__file__).with_name("node2vec_reproducibility.py")],
)
EXPECTED_KEYS = {
    (label, seed) for label, _, _ in PQ_SETTINGS for seed in TRAIN_SEEDS
}


def validate_checkpoint_rows(frame):
    required_cols = {
        "setting", "p", "q", "train_seed",
        "balanced_ari_mean_over_resamples",
        "balanced_ari_std_over_resamples",
    }
    if not required_cols.issubset(frame.columns):
        raise ValueError(f"Cannot resume: {PER_SEED_FILE} has unexpected columns")
    keys = [(row.setting, int(row.train_seed)) for row in frame.itertuples()]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Cannot resume: {PER_SEED_FILE} contains duplicate run keys")
    if not set(keys).issubset(EXPECTED_KEYS):
        raise ValueError(f"Cannot resume: {PER_SEED_FILE} contains unknown run keys")
    expected_pq = {label: (float(p), float(q)) for label, p, q in PQ_SETTINGS}
    for row in frame.itertuples():
        if (float(row.p), float(row.q)) != expected_pq[row.setting]:
            raise ValueError(
                f"Cannot resume: p/q values do not match setting {row.setting!r}"
            )
    return set(keys)


def strict_checkpoint_payload(frame):
    return {
        "mode": "strict",
        "compatibility": COMPATIBILITY,
        "checkpoint": {
            "row_count": len(frame),
            "completed_keys": sorted(
                [[row.setting, int(row.train_seed)] for row in frame.itertuples()]
            ),
            "csv_sha256": sha256_file(PER_SEED_FILE) if PER_SEED_FILE.exists() else None,
        },
    }


def save_strict_checkpoint(rows):
    frame = pd.DataFrame(rows).round(4)
    frame.to_csv(PER_SEED_FILE, index=False)
    write_json(CHECKPOINT_METADATA_FILE, strict_checkpoint_payload(frame))

# =========================
# 2. For each (p, q), retrain across N_TRAIN_SEEDS independent seeds
# =========================

legacy_read_only = False
if PER_SEED_FILE.exists():
    existing_df = pd.read_csv(PER_SEED_FILE)
    existing_keys = validate_checkpoint_rows(existing_df)
    if not CHECKPOINT_METADATA_FILE.exists():
        raise ValueError(
            f"Cannot resume: {CHECKPOINT_METADATA_FILE} is missing. "
            "Use a new output directory rather than mixing an unverified checkpoint."
        )
    checkpoint_metadata = read_json(CHECKPOINT_METADATA_FILE)
    mode = checkpoint_metadata.get("mode")
    if mode == "legacy_complete_read_only":
        if checkpoint_metadata.get("input_hashes") != COMPATIBILITY["input_hashes"]:
            raise ValueError("Cannot read legacy checkpoint: input hashes have changed")
        if checkpoint_metadata.get("parameters") != COMPATIBILITY["parameters"]:
            raise ValueError("Cannot read legacy checkpoint: experiment parameters have changed")
        checkpoint = checkpoint_metadata.get("checkpoint", {})
        if checkpoint.get("csv_sha256") != sha256_file(PER_SEED_FILE):
            raise ValueError("Cannot read legacy checkpoint: CSV hash has changed")
        if existing_keys != EXPECTED_KEYS:
            raise ValueError(
                "Cannot resume a partial legacy checkpoint. Use a new output directory."
            )
        legacy_read_only = True
    elif mode == "strict":
        if checkpoint_metadata.get("compatibility") != COMPATIBILITY:
            raise ValueError(
                "Cannot resume: graph, features, parameters, packages, or code differ "
                "from the checkpoint metadata. Use a new output directory."
            )
        checkpoint = checkpoint_metadata.get("checkpoint", {})
        if checkpoint.get("csv_sha256") != sha256_file(PER_SEED_FILE):
            raise ValueError("Cannot resume: checkpoint CSV hash does not match metadata")
    else:
        raise ValueError(f"Cannot resume: unsupported checkpoint mode {mode!r}")
    per_seed_rows = existing_df.to_dict("records")
    print(f"Resuming from {len(per_seed_rows)} completed runs in {PER_SEED_FILE}")
else:
    if CHECKPOINT_METADATA_FILE.exists():
        checkpoint_metadata = read_json(CHECKPOINT_METADATA_FILE)
        if (
            checkpoint_metadata.get("mode") != "strict"
            or checkpoint_metadata.get("compatibility") != COMPATIBILITY
            or checkpoint_metadata.get("checkpoint", {}).get("row_count") != 0
            or checkpoint_metadata.get("checkpoint", {}).get("csv_sha256") is not None
        ):
            raise ValueError(
                f"Cannot start: incompatible stale metadata exists at "
                f"{CHECKPOINT_METADATA_FILE}"
            )
    per_seed_rows = []
    write_json(CHECKPOINT_METADATA_FILE, strict_checkpoint_payload(pd.DataFrame()))

completed = {(row["setting"], int(row["train_seed"])) for row in per_seed_rows}
for label, p, q in PQ_SETTINGS:
    for train_seed in TRAIN_SEEDS:
        if (label, train_seed) in completed:
            print(f"Skipping completed run: {label}, train_seed={train_seed}")
            continue

        print(f"\n=== {label} (p={p}, q={q}), train_seed={train_seed} ===")
        seed_everything(train_seed)
        node2vec = Node2Vec(G, dimensions=EMBED_DIM, walk_length=WALK_LENGTH, num_walks=NUM_WALKS, p=p, q=q, weight_key="weight", workers=1, quiet=True)
        model = Word2Vec(node2vec.walks, vector_size=EMBED_DIM, window=WINDOW, min_count=1, sg=1, workers=1, seed=train_seed, hashfxn=stable_hash)

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

        # A full reconstructed-graph training takes several minutes. Persist
        # each completed condition so an interrupted run can resume exactly
        # from the next (p, q, seed) combination.
        save_strict_checkpoint(per_seed_rows)
        completed.add((label, train_seed))
        print(f"  checkpoint: {len(per_seed_rows)}/{len(PQ_SETTINGS) * len(TRAIN_SEEDS)} runs saved")

per_seed_df = pd.DataFrame(per_seed_rows).round(4)
if legacy_read_only:
    print(f"\nVerified legacy checkpoint without rewriting {PER_SEED_FILE}")
else:
    save_strict_checkpoint(per_seed_rows)
    print(f"\nSaved {PER_SEED_FILE} and {CHECKPOINT_METADATA_FILE}")

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
if not legacy_read_only:
    summary_df.to_csv(OUT_DIR / "pq_multiseed_summary.csv", index=False)
print(f"\n=== Aggregated across {N_TRAIN_SEEDS} training seeds ===\n{summary_df.to_string(index=False)}")
if legacy_read_only:
    print("Verified frozen summary in memory; no active result was rewritten")
else:
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
if not legacy_read_only:
    fig.savefig(OUT_DIR / "pq_multiseed_distribution.png", dpi=150)
plt.close(fig)
if not legacy_read_only:
    print(f"Saved {OUT_DIR / 'pq_multiseed_distribution.png'}")

print("\n=== Reconstructed-graph p/q x training-seed robustness check complete ===")
