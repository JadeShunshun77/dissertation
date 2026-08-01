import pandas as pd
import networkx as nx
from pathlib import Path

# Records topology metrics for raw_edges.csv BEFORE the Plan-B reconstruction
# (filling in missing counterparty-counterparty edges). A matching "after"
# run on the reconstructed graph gives a before/after comparison that is
# itself evidence in Limitations/Discussion that the star-like topology was
# measurably improved, not just asserted.

OUT_DIR = Path("topology_outputs")
OUT_DIR.mkdir(exist_ok=True)


def clean_address(x):
    if pd.isna(x):
        return None
    return str(x).strip().lower()


def compute_metrics(edges_path, seeds_path, label):
    edges = pd.read_csv(edges_path)
    edges["source"] = edges["source"].apply(clean_address)
    edges["target"] = edges["target"].apply(clean_address)
    edges = edges.dropna(subset=["source", "target"])

    seeds = pd.read_csv(seeds_path)
    seeds["address"] = seeds["address"].apply(clean_address)
    seed_set = set(seeds["address"])

    G = nx.from_pandas_edgelist(edges, source="source", target="target", create_using=nx.DiGraph())
    Gu = G.to_undirected()

    n_nodes = Gu.number_of_nodes()
    n_edges = Gu.number_of_edges()
    degrees = [d for _, d in Gu.degree()]
    avg_degree = sum(degrees) / n_nodes

    clustering = nx.clustering(Gu)
    clustering_vals = list(clustering.values())
    mean_clustering = sum(clustering_vals) / len(clustering_vals)
    pct_zero_clustering = sum(1 for v in clustering_vals if v == 0) / len(clustering_vals)

    n_triangles = sum(nx.triangles(Gu).values()) // 3

    components = list(nx.connected_components(Gu))
    component_sizes = sorted((len(c) for c in components), reverse=True)
    n_isolates = sum(1 for s in component_sizes if s == 1)
    isolate_ratio = n_isolates / n_nodes
    largest_cc_size = component_sizes[0] if component_sizes else 0
    largest_cc_ratio = largest_cc_size / n_nodes

    # seed-only clustering (this is what actually feeds the classifiers' graph
    # structural features -- whole-graph clustering can look fine while every
    # seed itself sits in a zero-clustering star)
    seed_clustering_vals = [clustering[a] for a in seed_set if a in clustering]
    seed_mean_clustering = sum(seed_clustering_vals) / len(seed_clustering_vals)
    seed_pct_zero_clustering = sum(1 for v in seed_clustering_vals if v == 0) / len(seed_clustering_vals)

    return {
        "stage": label,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "avg_degree": round(avg_degree, 4),
        "mean_clustering_coeff": round(mean_clustering, 6),
        "pct_zero_clustering": round(pct_zero_clustering, 4),
        "seed_mean_clustering_coeff": round(seed_mean_clustering, 6),
        "seed_pct_zero_clustering": round(seed_pct_zero_clustering, 4),
        "n_triangles": n_triangles,
        "n_connected_components": len(components),
        "n_isolates": n_isolates,
        "isolate_ratio": round(isolate_ratio, 4),
        "largest_cc_size": largest_cc_size,
        "largest_cc_ratio": round(largest_cc_ratio, 4),
    }


if __name__ == "__main__":
    import sys

    out_path = OUT_DIR / "topology_comparison.csv"

    results = []
    before = compute_metrics("raw_edges.csv", "final_seed_address500.csv", "before_reconstruction")
    results.append(before)

    if len(sys.argv) > 1 and sys.argv[1] == "--after":
        after = compute_metrics("raw_edges_reconstructed.csv", "final_seed_address500.csv", "after_reconstruction")
        results.append(after)

    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)
    print(f"Saved topology metrics to {out_path}")
    print(df.to_string(index=False))
