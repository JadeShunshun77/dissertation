# M06 Graph Representation Resolution Record

**Date:** 2026-08-06  
**Scope:** Direction, weight, time and interaction-type representation

## Changes

- Added `06_robustness_audit/code/audit_graph_representation.py` and three
  formal outputs under `06_robustness_audit/outputs/graph_representation/`.
- Made Node2Vec's existing `weight_key="weight"` behaviour explicit in all
  three active Node2Vec scripts.
- Added comments specifying that GraphSAGE/GCN and P3 propagation is
  undirected and unweighted.
- Added the binding representation boundary to the thesis evidence summary.

## Validated Scope

- Hand-crafted and full-feature GNN node inputs preserve directed and weighted
  aggregates.
- Node2Vec uses undirected `tx_count`-weighted walks.
- GNN message passing is undirected and unweighted.
- The graph contains 47,197 directed rows and 46,953 undirected pairs, with
  244 reciprocal dyads.
- Reciprocal collapse removes 0.11% of aggregate transaction-count weight from
  the exact NetworkX view used by Node2Vec.
- Seed-node in/out degree correlation is approximately -0.036.
- The top 1% of directed pairs carry approximately 80.0% of transaction count.
- Per-transaction time, token transfers, internal traces, events and decoded
  protocol interactions are absent.

## Sensitivity Decision

A direction-aware and weighted GNN comparison was not added. It would require
architecture/relation changes, a common weighting design for GraphSAGE and
GCN, and multiseed retraining; it is not a small controlled ablation. It also
cannot recover interaction fields that were never collected. The existing
models remain static baselines with restricted interpretation.

No data, graph, feature matrix, embedding, model score or database changed.
