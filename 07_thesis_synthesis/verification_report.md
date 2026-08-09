# Workspace Reorganisation Verification

**Migration verification date:** 2026-08-04  
**Reconstructed-mainline rerun:** 2026-08-05

## Outcome

- The active pipeline is organised under directories `00` through `07`.
- Superseded and pre-reconstruction material is isolated under
  `99_archive_not_for_thesis`.
- All 190 pre-existing data, result, image, log and reference artifacts were
  moved byte-for-byte. The migration-before and migration-after SHA-256
  multisets had zero differences.
- After the methodology audit and cleanup, the persistent artifact manifest
  contains 226 unique frozen files. The final independent hash check found
  zero missing or changed artifacts.
- No local database file (`.db`, `.sqlite`, `.sqlite3`, `.duckdb`) existed
  before or after the reorganisation.
- On 2026-08-06, the active code path was reviewed after the methodology
  audit. Reproducible pipeline scripts and the audit scripts needed to defend
  the thesis claims were retained; superseded one-off validation material was
  kept under `99_archive_not_for_thesis` rather than cited as active evidence.
  A direct post-audit check found all 226 manifest artifacts unchanged, all
  README Python entry points present and all retained Python sources
  syntactically valid.

## Runtime verification

All Python files compiled successfully. The EDA notebook executed successfully
from its new location. The following local experiments were rerun into a
temporary directory, without overwriting the frozen outputs:

| Experiment | Result comparison |
|---|---|
| Sanity check | exact SHA-256 match |
| Topology before/after | exact SHA-256 match |
| Feature engineering and feature groups | exact SHA-256 match |
| Main classification | 5/5 CSV files exact |
| Feature ablation and SHAP | 4/4 CSV files exact |
| Balanced-subset robustness | 5/5 CSV files exact |
| Repeated CV | 4/4 CSV files exact |
| Repeated balanced ARI | 2/2 CSV files exact |
| GraphSAGE/GCN full-feature multiseed | 5/5 CSV files exact |
| Symmetric RF/XGBoost multiseed comparison | 4/4 CSV files exact |
| P3 message-passing ablation | 4/4 CSV files exact |
| Contract confound check | 7/7 CSV files exact |
| Confound follow-up and entity-grouped checks | 10/10 CSV files exact |

In total, 55 regenerated core artifacts matched the frozen versions exactly
(54 CSV files plus `feature_groups.json`).

The two archived single-run GraphSAGE/GCN scripts were also executed from their
new locations with temporary outputs and completed successfully. At the time
of the migration check, the archived old-graph p/q sweeps were intentionally
not repeated.

## Reconstructed-mainline rerun

On 2026-08-05 the active Node2Vec pipeline was deliberately retrained on
`raw_edges_reconstructed.csv`, and all direct embedding-dependent experiments
were refreshed. The p/q sensitivity and five-training-seed sweeps used the
reconstructed inputs and completed in full; their producing scripts are now
retained in the dated Module 3 archive.

- Before rerun: 84 active output artifacts.
- After rerun: 89 active output artifacts; 43 unchanged, 41 refreshed, five
  new p/q artifacts, zero missing.
- All 66 active CSV files parsed successfully and were non-empty.
- Key row counts: 8,029 full-graph embeddings, 436 labelled-seed embeddings,
  15 p/q multiseed rows, 50 GNN fold rows and 100 tree-baseline fold rows.
- The final GraphSAGE/GCN and symmetric RF/XGBoost outputs reproduced their
  prior values. GraphSAGE remains `0.8274 +/- 0.0217`, GCN remains
  `0.8255 +/- 0.0238`. Their paired seed-level difference is retained only as
  a conditional stability diagnostic, not as population-level evidence.
- Reconstructed p/q multiseed balanced ARI is `0.1394 +/- 0.0082` for the
  unbiased baseline, `0.1117 +/- 0.0082` for BFS-like walks and
  `0.1046 +/- 0.0074` for DFS-like walks.

## Stochastic boundary

`node2vec_clustering.py` was stochastic at the random-walk stage before the
M07 reproducibility fix. The 2026-08-05 approved rerun intentionally refreshed
the active embedding artifacts: KMeans ARI moved from `0.0357` to `0.0265`,
and repeated balanced-evaluation ARI moved from `0.1436` to `0.1303`.
Repeat-level embedding increments remained positive for all three classifiers:
LR `+0.0211`, RF `+0.0118`, XGBoost `+0.0161`. These increments are
transductive observed-graph evidence and must not be presented as cold-start
or unseen-entity generalisation. Therefore exact Node2Vec numbers changed
within the known stochastic boundary, but the thesis-level conclusions did
not reverse.

## Unsupervised notebook migration

On 2026-08-09 the five active Node2Vec scripts and the reading-only walkthrough
were consolidated into the executed `03_unsupervised/node2vec.ipynb`.

- All nine code cells executed successfully with zero errors and five inline figures.
- Twelve active CSV/JSON evidence artifacts were preserved byte-for-byte.
- Both embedding matrices retain their approved SHA-256 hashes:
  `20E174AC...` for all graph nodes and `98734F6D...` for labelled nodes.
- The former scripts, walkthrough and 18 original output artifacts are retained
  under `99_archive_not_for_thesis/03_unsupervised/pre_notebook_mainline_2026-08-09/`.
- Default notebook execution validates frozen hashes. Optional stochastic reruns
  write to `node2vec_rerun_validation/` and cannot overwrite active evidence.

## Supervised notebook migration

On 2026-08-09 the five active supervised experiment scripts were converted
into executed, thesis-facing notebooks and a sixth synthesis notebook was
added. The notebooks were run from clean kernels in dependency order.

- All 57 code cells across the six notebooks executed successfully.
- The 14 retained core CSV artifacts reproduce the archived script outputs
  byte-for-byte, including all fold-, repeat-, seed- and comparison-level
  tables used by the thesis conclusions.
- GraphSAGE remains `0.8274`, GCN remains `0.8255`, and every repeated-CV,
  P3 and matched-tree comparison is unchanged.
- The former scripts and 28 complete output artifacts are preserved under
  `99_archive_not_for_thesis/04_supervised/pre_notebook_mainline_2026-08-09/`.
- Active output paths, the artifact manifest and the M08 inference registry
  were updated to the notebook-centric structure.

## Interpretability notebook migration

On 2026-08-09 the active feature-ablation and SHAP script was converted into
the executed `05_interpretability/feature_ablation_shap.ipynb` workflow.
All 12 code cells completed successfully, both figures are embedded inline,
and the three retained core CSVs reproduce the archived outputs byte-for-byte.
The original script and all six prior artifacts remain under
`99_archive_not_for_thesis/05_interpretability/pre_notebook_mainline_2026-08-09/`.

## Database and network boundary

- `query_nonseed_tx_behaviour.py` was not executed. Its BigQuery statement is
  read-only (`SELECT`/CTEs) and contains no table mutation statement.
- `verify_contract_reproducibility.py` was not executed because it intentionally
  performs live Ethereum RPC calls.
- `contract_confound_check.py` and the follow-up scripts were run with the
  existing local `seed_is_contract.csv` cache. The log confirmed the cache was
  loaded; no `eth_getCode` query branch was entered.
- No database schema, table or local database file was created, updated or
  deleted during this work.

## Environment notes

The active scripts ran with pandas 2.3.3, NumPy 2.3.3, NetworkX 3.6.1,
scikit-learn 1.7.2, XGBoost 3.3.0, SHAP 0.52.0, node2vec 0.3.2 and
torch-geometric from the current environment. Direct bare `import torch` on
this Windows installation still encounters the existing `sys.exec_prefix`
path issue; the GNN scripts' existing compatibility patch was exercised by
the full P3 and multiseed runs and worked correctly.
