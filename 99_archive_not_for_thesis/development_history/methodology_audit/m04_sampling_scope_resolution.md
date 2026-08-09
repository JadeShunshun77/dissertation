# M04 Sampling Scope Resolution Record

**Date:** 2026-08-06  
**Scope:** Definition and audit of the reconstructed-graph sampling boundary

## Changes

- Added `00_data_graph/code/audit_graph_sampling_scope.py`.
- Added deterministic graph-component and label-coverage outputs under
  `00_data_graph/provenance/`.
- Added the precise graph definition and interpretation boundary to the data
  README and thesis evidence summary.
- Replaced causal language with controlled structural sensitivity language.

## Validation

The audit confirmed:

- 12,861 original directed rows, all touching at least one seed;
- 34,336 augmentation rows, all nonseed-to-nonseed;
- 47,197 active directed rows and 46,953 undirected pairs;
- the same 8,029-node set before and after reconstruction;
- no component overlap, self-loop, new node or augmentation edge touching a
  seed;
- unchanged seed-incident pairs and edge aggregates;
- graph coverage of 436/500 labels: 51 CEX, 86 DeFi and 299 MEV-related.

No graph, feature matrix, embedding, model output or database was changed.

## Interpretation

The design supports a structural sensitivity comparison conditional on the
frozen label-seeded sample. It does not support a causal reconstruction claim,
complete induced-subgraph coverage, unbiased Ethereum-network inference or
complete address-degree interpretation.
