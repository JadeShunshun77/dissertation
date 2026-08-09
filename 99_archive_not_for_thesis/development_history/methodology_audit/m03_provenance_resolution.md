# M03 Provenance Resolution Record

**Date:** 2026-08-06  
**Scope:** Data provenance and offline graph reconstruction only

## Changes

- Restored four label-export snapshots from Git history into
  `00_data_graph/provenance/label_snapshots/`.
- Added a source registry and row-level provenance for all 500 active labels.
- Added read-only BigQuery SQL records for the documented original Top-100
  method and the recovered fixed-node augmentation logic.
- Added local scripts for label audit, byte-exact graph reconstruction and
  deterministic lineage verification.
- Added the new CSV/JSON provenance artifacts to the frozen artifact manifest.

## Validation

Commands executed:

```powershell
python -m py_compile 00_data_graph/code/build_seed_labels.py 00_data_graph/code/build_reconstructed_graph.py 00_data_graph/code/verify_data_lineage.py
python 00_data_graph/code/build_seed_labels.py --provenance-output 00_data_graph/provenance/seed_label_provenance.csv
python 00_data_graph/code/build_reconstructed_graph.py
python 00_data_graph/code/verify_data_lineage.py
python 07_thesis_synthesis/verify_artifacts.py
```

Validated facts:

- All 500 active address-label mappings match the restored snapshots.
- One duplicated MEV address contributes two merged sub-labels.
- The graph rebuild contains 47,197 rows and 8,029 nodes and is byte-identical
  to the active graph.
- Plan-B adds 34,336 nonseed-to-nonseed rows, no nodes and no seed-touching
  rows; the components have no pair overlap or self-loops.
- No BigQuery, Dune or RPC query was executed and no local database was found.
- No pre-existing data or experiment artifact changed.

## Accepted Historical Limits

The exact original Top-100 SQL and BigQuery Job ID, Dune query references,
retrieval dates, CEX active-selection procedure, upstream MEV-label platform
and final seed-row ordering step were not retained. The project records these
gaps explicitly and does not fabricate exact source regeneration.
