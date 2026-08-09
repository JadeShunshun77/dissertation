# Data And Graph

This stage contains the frozen labelled-address input and the reconstructed
transaction graph used by every graph-dependent thesis experiment.

## Active inputs

- `data/main/final_seed_address500.csv`: 500 public address-role labels
  (`CEX=51`, `DeFi=150`, stored technical label `Malicious_MEV=299`).
- `data/main/raw_edges_reconstructed.csv`: 47,197 directed aggregated pairs
  over 8,029 nodes. This is the only thesis-facing graph input.
- `data/reconstruction_reference/raw_edges.csv`: the original 12,861-edge
  label-seeded Top-100 graph, retained only as reconstruction evidence.
- `data/reconstruction_reference/nonseed_to_nonseed_augmentation_edges.csv`:
  34,336 additional nonseed-to-nonseed pairs over the fixed original node set.

The graph records successful top-level transactions from
`bigquery-public-data.crypto_ethereum.transactions` in the window
2026-03-25 through 2026-06-23. It is a label-seeded, fixed-node, partially
reconstructed transaction subgraph, not a complete Ethereum induced subgraph.

## Reproducibility boundary

The exact historical BigQuery SQL and Job ID for `raw_edges.csv` were not
retained. `sql/documented_original_top100_query.sql` is an explicitly marked
method reconstruction, not a claim to be the original query text. The
nonseed-to-nonseed augmentation logic was recovered from the historical
project log and is stored in
`sql/nonseed_to_nonseed_augmentation_query.sql`.

The CEX and DeFi snapshots were exported from Dune, but their query IDs,
retrieval dates and licence notes were not retained. The upstream platform for
the MEV-related snapshot was also not retained. These gaps are recorded in
`provenance/label_source_registry.csv`; unknown metadata is never inferred.

The active seed file's complete address-label mapping is auditable against the
restored snapshots. Its historical post-merge row-ordering step is not
recoverable, so the frozen file and its SHA-256 are authoritative. The
reconstructed graph is reproducible byte-for-byte from its two frozen edge
components.

## Local reconstruction checks

The ordinary checks make no network, BigQuery, Dune, RPC or database call and
do not overwrite active files:

```powershell
python 00_data_graph/code/build_seed_labels.py
python 00_data_graph/code/build_reconstructed_graph.py
```

To regenerate row-level provenance into a new path, use:

```powershell
python 00_data_graph/code/build_seed_labels.py --provenance-output <new-path>
```

Both builders refuse to overwrite an existing output. The SQL files are
preserved methodology records and are not executed by any local pipeline
command.

## Sampling scope

The frozen scope outputs show that the augmentation preserves the original
8,029-node set and all seed-incident edge rows while adding only
nonseed-to-nonseed pairs. They are `graph_sampling_scope.csv`,
`label_graph_coverage.csv` and `graph_sampling_scope.json` in `provenance/`.

The reconstruction supports a controlled structural sensitivity comparison:
graph-derived quantities may be compared before and after adding observed
neighbour-to-neighbour edges while the labelled-address boundary is held
fixed. It is not a causal design and does not turn the sample into a complete
Ethereum induced subgraph. Top-100 applies to candidate selection per seed and
direction; the union of incoming and outgoing selections is not a hard final
degree cap.
