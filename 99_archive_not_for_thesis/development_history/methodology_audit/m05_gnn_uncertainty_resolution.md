# M05 GNN Uncertainty Resolution Record

**Date:** 2026-08-06  
**Scope:** Full-feature GNN, symmetric tree comparison and P3 ablation

## Changes

- Added `06_robustness_audit/code/audit_gnn_uncertainty.py` and three formal
  outputs under `06_robustness_audit/outputs/gnn_uncertainty/`.
- Corrected explanatory comments in the three multiseed training scripts
  without changing training or output logic.
- Replaced population-significance language in the thesis evidence summary
  with conditional means, ranges and direction consistency.
- Retained the symmetric multiseed table as the final numerical GNN-vs-tree
  source while binding its interpretation to the uncertainty audit.

## Validation

- Fixed design: one balanced 153-address subset, one five-fold assignment and
  one reconstructed graph snapshot.
- Training seed labels: 42--46 for every condition.
- GNN fold rows: 50; tree-baseline fold rows: 100; P3 fold rows: 75.
- GraphSAGE-vs-GCN direction is mixed (3/5 GraphSAGE higher).
- Both GNNs exceed structural RF in 5/5 seed labels and fall below both
  complete hand-crafted baselines in 5/5.
- Structural-XGBoost differences have mixed direction.
- Both XGBoost conditions are identical across all five requested seeds.
- Real-edge GraphSAGE exceeds rewired-edge GraphSAGE in 5/5 paired labels, but
  P3 remains conditional on the frozen sample and architecture.

No graph, feature matrix, model score, prediction or database was changed.
