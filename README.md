# Dissertation Experiment Workspace

This workspace contains the code, data and frozen outputs used for the
dissertation experiments on graph-based clustering and classification of
publicly labelled Ethereum addresses.

The active project uses a label-seeded, fixed-node, partially reconstructed
90-day Ethereum transaction subgraph. The main labelled sample contains 436
graph-active addresses from 500 public seed labels.

## Recommended Reading Order

Open the folders in this order if you want to understand or reproduce the
project:

1. `00_data_graph`
2. `01_sanity_eda`
3. `02_feature_engineering`
4. `03_unsupervised`
5. `04_supervised/01_traditional_ml`
6. `05_interpretability`
7. `04_supervised/02_gnn_models`
8. `04_supervised/03_gnn_evaluation`
9. `06_robustness_audit`

This order follows the dissertation logic: data construction, graph checking,
feature construction, Node2Vec, traditional classification, SHAP
interpretation, GNN evaluation and robustness checks.

## Folder Guide

| Folder | Role | Main files or outputs |
|---|---|---|
| `00_data_graph` | Stores the final seed-label file, reconstructed graph and graph provenance checks. | `data/main/final_seed_address500.csv`, `data/main/raw_edges_reconstructed.csv`, `code/build_seed_labels.py`, `code/build_reconstructed_graph.py` |
| `01_sanity_eda` | Checks graph quality and compares topology before and after reconstruction. | `sanity_check/sanity_summary.csv`, `sanity_check/topology_comparison.csv`, `eda_walkthrough.ipynb` |
| `02_feature_engineering` | Builds transaction-behaviour and graph-structural features. | `feature_engineering/seed_feature_matrix_reconstructed.csv`, `feature_engineering/feature_groups.json`, `feature_engineering/feature_engineering.py` |
| `03_unsupervised` | Runs Node2Vec, clustering, p/q sensitivity and balanced ARI checks. | `node2vec.ipynb`, `node2vec_results/` |
| `04_supervised/01_traditional_ml` | Runs Logistic Regression, Random Forest, XGBoost and repeated cross-validation. | `01_classification.ipynb`, `02_repeated_cv.ipynb`, `classification_cv_results/` |
| `04_supervised/02_gnn_models` | Runs the final multi-seed GraphSAGE and GCN experiments. | `03_graphsage_gcn.ipynb`, `graphsage_gcn_multiseed_results/` |
| `04_supervised/03_gnn_evaluation` | Compares GNNs with matched RF/XGBoost baselines and message-passing diagnostics. | `04_message_passing_ablation.ipynb`, `05_rf_xgb_matched_baselines.ipynb`, `06_gnn_evaluation.ipynb`, `message_passing_and_baseline_results/` |
| `05_interpretability` | Runs feature-group ablation and SHAP interpretation. | `feature_ablation_shap.ipynb`, `feature_ablation_shap_results/` |
| `06_robustness_audit` | Stores robustness checks for balance, contract status, entity/protocol holdout and methodology evidence. | `01_balanced_subset/`, `02_contract_confound/`, `03_entity_generalisation/`, `04_methodology_evidence/` |

## Key Data Files

- `00_data_graph/data/main/final_seed_address500.csv`: 500 public seed labels
  (`CEX=51`, `DeFi=150`, internal stored label `Malicious_MEV=299`).
- `00_data_graph/data/main/raw_edges_reconstructed.csv`: final reconstructed
  graph with 47,197 directed aggregated transaction rows over 8,029 nodes.
- `00_data_graph/data/reconstruction_reference/raw_edges.csv`: original
  12,861-row label-seeded Top-100 graph, retained as reconstruction evidence.
- `00_data_graph/data/reconstruction_reference/nonseed_to_nonseed_augmentation_edges.csv`:
  34,336 neighbour-to-neighbour rows added during fixed-node reconstruction.

The dissertation text should refer to the internal `Malicious_MEV` label as
`MEV-related` or `MEV/arbitrage`, because the category mainly represents
arbitrage-labelled addresses rather than independently verified malicious
actors.

## Main Execution Order

The project already contains frozen outputs. Re-running notebooks may overwrite
some result files, so use this order only when intentionally reproducing the
pipeline.

```powershell
python 00_data_graph/code/build_seed_labels.py
python 00_data_graph/code/build_reconstructed_graph.py
python 01_sanity_eda/sanity_check/sanity_check.py
python 01_sanity_eda/sanity_check/topology_baseline.py
python 02_feature_engineering/feature_engineering/feature_engineering.py
jupyter nbconvert --to notebook --execute --inplace 03_unsupervised/node2vec.ipynb
jupyter nbconvert --to notebook --execute --inplace 04_supervised/01_traditional_ml/01_classification.ipynb
jupyter nbconvert --to notebook --execute --inplace 04_supervised/01_traditional_ml/02_repeated_cv.ipynb
jupyter nbconvert --to notebook --execute --inplace 05_interpretability/feature_ablation_shap.ipynb
jupyter nbconvert --to notebook --execute --inplace 04_supervised/02_gnn_models/03_graphsage_gcn.ipynb
jupyter nbconvert --to notebook --execute --inplace 04_supervised/03_gnn_evaluation/04_message_passing_ablation.ipynb
jupyter nbconvert --to notebook --execute --inplace 04_supervised/03_gnn_evaluation/05_rf_xgb_matched_baselines.ipynb
jupyter nbconvert --to notebook --execute --inplace 04_supervised/03_gnn_evaluation/06_gnn_evaluation.ipynb
jupyter nbconvert --to notebook --execute --inplace 06_robustness_audit/01_balanced_subset/balanced_subset_robustness.ipynb
jupyter nbconvert --to notebook --execute --inplace 06_robustness_audit/02_contract_confound/contract_confound_robustness.ipynb
jupyter nbconvert --to notebook --execute --inplace 06_robustness_audit/03_entity_generalisation/entity_generalisation.ipynb
```

## Thesis Claim Boundary

The project classifies public address-role labels within one observed graph.
It should not be described as a general malicious-address detector, a complete
Ethereum network analysis, a real-world identity attribution system or an
inductive cold-start classifier for arbitrary unseen addresses.

The main conclusion is bounded: transaction-behaviour features are the most
stable predictive source in this sample, while graph-derived methods provide
limited or conditional additional value.

## Notes On External Calls

Most scripts and notebooks use stored local files. Two areas may require
external services if deliberately re-run:

- `02_feature_engineering/nonseed_transaction_query/query_nonseed_tx_behaviour.py`
  performs a BigQuery query.
- `06_robustness_audit/02_contract_confound/query_contract_status.py` and
  related checks may use Ethereum RPC calls.

The frozen outputs needed for the dissertation are already present, so normal
inspection of the workspace does not require network access.

## Git Ignore

The `.gitignore` file tells Git to ignore temporary caches, notebook
checkpoints and rerun-validation outputs. It does not affect code execution or
the dissertation PDF.
