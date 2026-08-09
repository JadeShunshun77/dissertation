# M07 Node2Vec Reproducibility Resolution Record

**Date:** 2026-08-06  
**Scope:** Random-state control, Word2Vec initialisation and p/q checkpoint safety

## Defects Confirmed

- `node2vec 0.3.2` uses both Python `random.shuffle` and NumPy sampling when
  generating walks; the active scripts previously seeded only NumPy.
- Gensim Word2Vec received a seed and one worker but used Python's built-in
  process-dependent hash function by default.
- `pq_multiseed_per_seed.csv` was resumed from only `(setting, train_seed)`;
  graph, feature, parameter, package and code changes were not detected.

## Changes

- Added `node2vec_reproducibility.py` and applied its Python/NumPy seeding and
  stable SHA-256-derived gensim hash to all three active Node2Vec scripts.
- Added complete run metadata for new baseline and p/q sensitivity runs.
- Added strict checkpoint metadata and fail-closed compatibility checks for
  graph, features, parameters, packages and code.
- Bound the existing 15-row checkpoint to its exact CSV/input hashes as
  `legacy_complete_read_only`; its historical reproducibility is not claimed.
- Added an independent full-parameter validation under the robustness audit.

## Validation

Two clean runs used the full 8,029-node reconstructed graph, 64 dimensions,
walk length 30, 10 walks per node and `p=q=1`. The processes used different
`PYTHONHASHSEED` values (101 and 202). Walk SHA-256, embedding SHA-256,
assignment SHA-256, NMI, ARI and Silhouette were identical.

Checkpoint tests passed for read-only legacy access, missing-metadata
rejection, graph-hash mismatch rejection and code-hash mismatch rejection.
All active-output hashes were unchanged by the read-only test.

The controlled scores were NMI 0.17168, ARI 0.04294 and Silhouette 0.17914.
Their absolute differences from the frozen pre-M07 values were +0.00848,
+0.01644 and +0.00134. The exact realisation changed, but the weak unsupervised
clustering conclusion did not. Frozen thesis results were not replaced.

No data, graph, feature matrix, frozen result or database was changed.
