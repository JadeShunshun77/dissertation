# Thesis Evidence Summary

**Project:** Graph-Based Clustering and Classification of Publicly Labelled
Ethereum Addresses Using Public On-Chain Data

**Current evidence freeze:** 2026-08-06

## Terminology and target construct

`Malicious_MEV` is retained only as the internal label stored in frozen data
and result files. Because 297 of its 299 addresses are sub-labelled
`Arbitrage Trader`, thesis prose, figures and tables must call this class
`MEV-related` or `MEV/arbitrage`; the class is not ground truth for malicious
or unlawful conduct. The prediction target is a set of public address-role
labels under the observed sampling design, not real-world identity or intent.

The binding rationale and approved Results/Discussion wording are recorded in
`07_thesis_synthesis/m01_label_confound_resolution.md`.

## Thesis-facing conclusion boundary

The dissertation should present the work as graph-based clustering and
classification of publicly labelled Ethereum address roles inside the observed
sample. The defensible positive claim is:

> Public address-role labels are predictable within the label-seeded,
> fixed-node, partially reconstructed 90-day transaction subgraph. Transaction
> behaviour features dominate; Node2Vec gives a small transductive increment in
> supervised classifiers; GraphSAGE and GCN are useful static graph baselines
> but do not outperform complete hand-crafted RF/XGBoost models under the
> fixed balanced design.

The dissertation should not claim a general malicious-address detector,
real-world identity attribution, complete Ethereum network coverage,
cold-start classification of arbitrary new addresses, future-period
prediction, or entity-disjoint three-class generalisation. These restrictions
follow directly from the MEV label semantics, the contract-status shortcut,
the label-seeded sampling design, the transductive embedding/GNN setup and
the absence of MEV entity groups.

## M01 resolved: label semantics and contract confounding

**Decision:** Resolved for the experiment pipeline on 2026-08-05. No label
file, model or result was changed, and no experiment rerun is required. The
contract-status association is an accepted design limitation rather than a
numerical failure of the experiments.

### Label composition used by the experiments

| Internal label | Total seeds | Active seeds | Active contracts | Thesis-facing name |
|---|---:|---:|---:|---|
| `CEX` | 51 | 51 | 2 | CEX-associated addresses |
| `DeFi` | 150 | 86 | 86 | DeFi protocol contract addresses |
| `Malicious_MEV` | 299 | 299 | 34 | MEV-related/arbitrage addresses |

The 299 stored `Malicious_MEV` samples comprise 297 `Arbitrage Trader`
sub-labels, one `Arbitrage Trader; Sandwich Attacker`, and one
`mevrevolutionx.eth`. Arbitrage alone is not evidence of malicious or unlawful
conduct. The internal name is retained only to preserve artifact continuity.

### What the robustness evidence supports

- Contract status is strongly associated with class membership. All 86 active
  DeFi samples are contracts, compared with 2/51 CEX and 34/299 MEV-related
  samples.
- A Random Forest using only `is_contract`, `has_out_edge` and `out_tx_count`
  reaches 0.9169 mean macro-F1. Account type and its activity proxies are
  therefore a strong shortcut for the observed labels.
- `is_contract` is not included in the main hand-crafted feature matrix. The
  minimal baseline diagnoses the dataset design; it does not show that the
  main models directly use bytecode status.
- Removing outbound features does not collapse classification performance;
  none of the three fixed-split diagnostic tests has raw `p<0.05`. This is
  exploratory robustness evidence, not a separate confirmatory family, and
  shows that the main score is not attributable to `out_tx_count` alone.
- EOA-only CEX-versus-MEV-related classification retains approximately
  0.92--0.93 macro-F1. Contract status does not explain every label-associated
  difference.
- Contract-only DeFi-versus-MEV-related models also score highly. The one-sided
  protocol holdout retains RF group-equal macro-F1 of 0.9270
  with hand-crafted features, but MEV-side entity generalisation remains
  unidentifiable.

### Binding interpretation rules

The following conclusions may be used:

- The models distinguish the observed public address-role labels within the
  label-seeded, 90-day reconstructed graph.
- Transaction-behaviour features dominate prediction in this dataset.
- CEX and MEV-related EOAs retain category-associated differences after
  contract addresses are removed, conditional on address-level sampling.
- Relative Node2Vec, tree-model and GNN comparisons remain numerically valid
  for this dataset.

The following conclusions must not be used:

- The system detects malicious Ethereum addresses in general.
- All arbitrage traders are malicious, abusive or unlawful.
- The three-class score isolates DeFi economic behaviour from generic contract
  characteristics.
- Contract confounding has been eliminated.
- The score demonstrates generalisation to unseen entities, protocols or
  future periods.

### Thesis-ready wording

**Methods:** The stored dataset used the technical label `Malicious_MEV`;
however, 297 of its 299 addresses were sub-labelled as arbitrage traders. The
class is therefore reported as MEV-related/arbitrage rather than as a
ground-truth malicious-address category.

**Results:** Contract status was strongly nested within the label construction:
all 86 active DeFi addresses were contracts, compared with 2 of 51 CEX and 34
of 299 MEV-related addresses. A three-feature contract-status baseline reached
0.9169 macro-F1, demonstrating substantial account-type confounding.

**Discussion:** The three-class results demonstrate predictability of the
observed public role labels, not independent recovery of economic intent.
EOA-only CEX-versus-MEV-related classification remained strong, so account
type did not explain every difference; nevertheless, DeFi-specific behaviour
cannot be separated completely from generic contract characteristics in the
current sample.

The exact historical extraction query and row-level label-source metadata are
not fully preserved. That reproducibility boundary is documented under M03
and must not be described as resolved by M01.

## Entity-level robustness and evaluation scope

The main address-level CV measures interpolation among sampled addresses; it
does not by itself establish unseen-entity generalisation. A fully
entity-disjoint three-class split is not identifiable from the current labels
because MEV entity ownership/group metadata is absent.

The entity-level robustness evaluation therefore uses two one-sided tests:

1. EOA-only unseen CEX entity versus address-disjoint MEV-related EOAs.
2. Contract-only unseen DeFi protocol versus address-disjoint MEV contracts.

Coinbase and Coinbase Prime are treated as one entity; Binance and Binance US
are treated as one entity. The testable CEX groups are Coinbase, OKX and
Binance. The testable DeFi groups are Uniswap, Curve, 0x API and Dodo. Each
group has 50 deterministic repeats with balanced train/test classes and at
most ten test addresses per class. Across 1,400 model fits, train/test address
overlap is zero.

| One-sided holdout | Features | RF group-equal macro-F1 | Worst group | Best group |
|---|---|---:|---:|---:|
| Unseen CEX entity, EOA-only | Hand-crafted | 0.8784 | 0.8278 | 0.9200 |
| Unseen CEX entity, EOA-only | Hand-crafted + Node2Vec | 0.8332 | 0.7705 | 0.9145 |
| Unseen DeFi protocol, contract-only | Hand-crafted | 0.9270 | 0.8881 | 0.9627 |
| Unseen DeFi protocol, contract-only | Hand-crafted + Node2Vec | 0.9324 | 0.8987 | 0.9680 |

These results support limited transfer to held-out CEX entities and DeFi
protocols. They do not establish transfer to unseen MEV entities or arbitrary
Ethereum addresses. The 0x API and Dodo groups contain only two addresses each
and are low-power supplements. Fifty repeats quantify address/model variation
within the fixed groups; they do not create 50 independent entities. Node2Vec
embeddings are also transductive, so the address-level Node2Vec increment must
not be promoted to an entity-level or inductive claim.

**Thesis-ready wording:** A fully entity-disjoint three-class evaluation was
not identifiable because the MEV-related labels did not contain entity
ownership groups. One-sided address-disjoint holdouts produced Random Forest
group-equal macro-F1 of 0.878 for unseen CEX entities and 0.927 for unseen DeFi
protocols using hand-crafted features. This supports limited transfer across
the identifiable CEX and DeFi groups, but not generalisation to unseen MEV
entities or arbitrary Ethereum addresses.

Primary sources:

- `06_robustness_audit/outputs/entity_generalisation/`

This is the thesis-facing project log. It contains only the current evidence
chain. The full chronological development log is preserved at
`99_archive_not_for_thesis/development_history/progress_summary_full_2026-08-04.md`
for audit purposes and is not the source for thesis result selection.

## Data provenance and reconstruction

The active seed input contains 500 unique public addresses: 51 CEX, 150 DeFi
and 299 addresses under the stored technical label `Malicious_MEV`. All 500
address-label-sub-label mappings are auditable against frozen source snapshots
in `00_data_graph/provenance/label_snapshots/`. The MEV-related source contains
501 rows before address deduplication because one address has both arbitrage
and sandwich sub-labels; the active file combines those annotations. The exact
Dune query IDs, retrieval dates, CEX active-selection procedure and upstream
MEV-label platform were not retained. The historical final-row ordering step
was also not retained, so `final_seed_address500.csv` and its SHA-256 remain
the authoritative frozen input rather than being presented as a byte-exact
source re-export.

Graph edges come from successful top-level transactions in
`bigquery-public-data.crypto_ethereum.transactions` between 2026-03-25 and
2026-06-23. The exact historical SQL and BigQuery Job ID for the original
Top-100 extraction were not retained; the repository therefore stores an
explicitly marked equivalent method query, not purported original SQL. The
fixed-node nonseed-to-nonseed augmentation query was recovered from the
historical log and is stored as
`00_data_graph/sql/nonseed_to_nonseed_augmentation_query.sql`. Offline
reconstruction combines 12,861 original rows and 34,336 augmentation rows from
`00_data_graph/data/reconstruction_reference/nonseed_to_nonseed_augmentation_edges.csv`
into the active 47,197-row, 8,029-node graph. It
introduces no nodes, seed-touching augmentation edges, self-loops or duplicate
component pairs and reproduces `raw_edges_reconstructed.csv` byte-for-byte
with SHA-256
`59CECEDA94AF4792C1B3397A9EA5F5E44A3B15C2A59BB14FA27AE01B05BB1DAB`.

The reproducibility claim is therefore bounded: frozen local inputs and the
complete downstream graph assembly are independently auditable, while exact
re-extraction of the original Top-100 table and every label export is not
claimed. The local verification performs no BigQuery, Dune, RPC or database
operation and found no local database files.

Primary sources:

- `00_data_graph/README.md`
- `00_data_graph/provenance/data_lineage_verification.json`
- `00_data_graph/provenance/seed_label_provenance.csv`
- `00_data_graph/sql/`

## Graph sampling scope and interpretation

The active graph must be described as a **label-seeded, fixed-node, partially
reconstructed 90-day transaction subgraph**, not as a complete induced
subgraph of Ethereum. The original graph was assembled from bidirectional
Top-100 counterparty candidates for labelled seeds. Top-100 applies per seed
and direction during candidate selection; after incoming and outgoing results
are combined, it is not a hard cap on a seed's final degree or incident rows.

| Component | Directed rows | Nodes | Seed-seed | Seed-nonseed | Nonseed-nonseed |
|---|---:|---:|---:|---:|---:|
| Original Top-100 graph | 12,861 | 8,029 | 158 | 12,703 | 0 |
| Fixed-node augmentation | 34,336 | 5,314 | 0 | 0 | 34,336 |
| Active reconstructed graph | 47,197 | 8,029 | 158 | 12,703 | 34,336 |

The augmentation introduces no node and no seed-touching pair. The complete
seed-incident pair set and its transaction aggregates are unchanged, while
the nonseed neighbourhood gains observed internal connections. Consequently,
the before/after reconstruction analysis is a controlled **structural
sensitivity comparison** within the frozen sample: it tests whether topology,
embeddings and graph-model conclusions change when neighbour-to-neighbour
edges are added while seed-incident transaction rows and the node set remain
fixed. It is not a randomised or causal comparison.

Of the 500 labelled addresses, 436 occur in the graph: 51/51 CEX, 86/150 DeFi
and 299/299 MEV-related. Graph-derived features, Node2Vec and GNN results
therefore use graph-observed labelled addresses rather than all 500 labels.
The missing 64 labels are all DeFi addresses in this sample.

Permitted claims are restricted to the sampled graph. The results do not show
complete transaction-history coverage, unbiased representation of Ethereum,
complete on-chain degree for labelled addresses, or generalisation to the
Ethereum address population. Reconstruction improves internal connectivity;
it does not remove the original label-seeded sampling boundary.

Primary sources:

- `00_data_graph/provenance/graph_sampling_scope.csv`
- `00_data_graph/provenance/label_graph_coverage.csv`
- `00_data_graph/provenance/graph_sampling_scope.json`

## GNN uncertainty and comparison scope

The final GraphSAGE, GCN, RF and XGBoost comparison uses one fixed balanced
subset of 153 graph-observed addresses (51 per class), one fixed five-fold
assignment and one reconstructed graph snapshot. Seeds 42--46 vary GNN
initialisation, dropout and validation permutation, and vary Random Forest
bootstrap/feature subsampling. They do not create five independent address
samples, entities, fold assignments, periods or graph snapshots. Under the
current XGBoost configuration, all five requested seed labels produce the
same scores, so they are not repeated XGBoost realisations.

Seed-level t intervals and p-values in the frozen raw output tables are
retained for audit continuity but are conditional diagnostics. They must not
be used as population-level significance tests. Thesis-facing comparisons
report means, across-seed ranges and direction consistency:

| Conditional comparison | Mean macro-F1 difference | Across-seed range | Direction |
|---|---:|---:|---:|
| GraphSAGE - GCN | +0.0019 | -0.0166 to +0.0218 | GraphSAGE higher in 3/5 |
| GraphSAGE - structural RF | +0.0414 | +0.0189 to +0.0787 | 5/5 |
| GCN - structural RF | +0.0395 | +0.0166 to +0.0743 | 5/5 |
| GraphSAGE - structural XGBoost | +0.0128 | -0.0071 to +0.0526 | 3/5 |
| GCN - structural XGBoost | +0.0108 | -0.0121 to +0.0482 | 2/5 |
| GraphSAGE - hand-crafted RF | -0.0810 | -0.1108 to -0.0514 | 0/5 |
| GCN - hand-crafted RF | -0.0829 | -0.1055 to -0.0558 | 0/5 |
| GraphSAGE - hand-crafted XGBoost | -0.0569 | -0.0767 to -0.0170 | 0/5 |
| GCN - hand-crafted XGBoost | -0.0588 | -0.0817 to -0.0214 | 0/5 |

GraphSAGE and GCN are numerically similar under the frozen design. Both are
consistently above structural-only RF across the five seed labels, have no
consistent advantage over structural XGBoost, and are consistently below
RF/XGBoost using the complete hand-crafted feature set. These are within-design
directional statements, not population significance claims.

For P3, real-edge GraphSAGE exceeds the no-edge MLP by +0.0339 on average but
the seed range crosses zero (-0.0171 to +0.0797; 4/5 positive). Real-edge
GraphSAGE exceeds degree-preserving rewired GraphSAGE by +0.1455, with all five
paired differences positive (+0.0306 to +0.2928). The latter supports
conditional sensitivity to observed adjacency under this architecture, while
the rewired condition combines optimisation and graph-rewiring randomness.

Because seed-level p-values are excluded from confirmatory claims, no
significance family is asserted for these GNN comparisons and multiplicity
correction is not used to rehabilitate them. Generalisation uncertainty is
instead bounded by the address-, entity- and graph-scope limitations recorded
elsewhere in this log.

Primary sources:

- `06_robustness_audit/outputs/gnn_uncertainty/model_seed_variability.csv`
- `06_robustness_audit/outputs/gnn_uncertainty/conditional_stability_comparisons.csv`
- `06_robustness_audit/outputs/gnn_uncertainty/gnn_uncertainty_design.json`

## Graph representation boundary

The stored graph consists of directed source-target pairs aggregated over the
single 90-day window. Each pair contains transaction count, total native ETH
value and mean gas fee. Direction and weights are not uniformly discarded;
their use differs by pipeline component:

| Component | Adjacency/structure | Edge propagation | Direction and weights retained elsewhere |
|---|---|---|---|
| Hand-crafted models | Directed | Not applicable | In/out degree, in/out counts, ETH, gas and weighted PageRank |
| Node2Vec | Undirected | `tx_count`-weighted walks | No separate direction, ETH or gas in embedding |
| Full-feature GraphSAGE/GCN | Undirected | Unweighted | Directed/weighted structural and transaction node features |
| P3 GraphSAGE/MLP | Undirected or no edges | Unweighted | In/out degree and weighted PageRank node features |

The active graph has 47,197 directed rows and 46,953 undirected pairs. Only
244 dyads are observed in both directions. NetworkX undirected conversion
retains one directed weight for those reciprocal dyads rather than summing
both; this removes 12,278 of 11,089,956 recorded transaction counts (0.11%)
from the Node2Vec walk-weight representation. This small aggregate fraction
does not mean direction is generally unimportant: seed-node in-degree and
out-degree have Spearman correlation `-0.036`. Transaction weights are also
highly concentrated, with the top 1% of directed pairs carrying approximately
80.0% of transaction count. Weighted and unweighted PageRank ranks remain
strongly correlated among graph-observed seeds (`rho=0.968`) but are not
identical.

The representation is static and interaction-limited. It does not store
per-transaction timestamps or ordering, ERC-20/ERC-721 transfers, internal
call traces, event logs, decoded protocol actions, calldata/function
signatures or failed transactions. Zero-native-ETH top-level pairs are
retained, but they do not reconstruct token flows or internal contract state
changes.

A new directed/weighted GNN sensitivity run is not added to the frozen
experiment matrix. A valid comparison would require direction-aware relation
channels or a directed architecture, a common edge-weight design for
GraphSAGE and GCN, and multiseed retraining under the M05 standard. That is an
architecture extension rather than a small one-parameter ablation, and it
cannot recover absent temporal or interaction-type data. The present models
are therefore retained as documented static baselines; representation loss
is an accepted limitation, not evidence that the omitted information lacks
value.

Permitted claims concern prediction and clustering on the sampled 90-day
top-level transaction graph. The results must not be described as complete
DeFi activity modelling, direction-aware or amount-aware GNN propagation,
temporal prediction, or coverage of token/internal/protocol interactions.

Primary sources:

- `06_robustness_audit/outputs/graph_representation/representation_usage_matrix.csv`
- `06_robustness_audit/outputs/graph_representation/representation_loss_metrics.csv`
- `06_robustness_audit/outputs/graph_representation/graph_representation_audit.json`

## Node2Vec reproducibility and checkpoint integrity

M07 is resolved without replacing the frozen Node2Vec evidence. The prior
scripts seeded NumPy and gensim but did not seed Python's `random`, which
`node2vec 0.3.2` uses to shuffle walk start nodes. Gensim also defaulted to the
process-dependent built-in hash for vocabulary initialisation. Active code now
seeds Python and NumPy, supplies a stable SHA-256-derived Word2Vec hash
function, and retains `workers=1`.

Two clean baseline runs used the full reconstructed graph and the thesis
parameters (`64` dimensions, walk length `30`, `10` walks per node, `p=q=1`)
in separate Python processes with `PYTHONHASHSEED=101` and `202`. Their walk
corpora, embedding arrays and labelled-node assignments had identical SHA-256
values; NMI, ARI and Silhouette were also exactly identical. The controlled
scores were `0.17168`, `0.04294` and `0.17914`, respectively. Against the
frozen pre-M07 values (`0.1632`, `0.0265`, `0.1778`), the absolute changes are
`+0.00848`, `+0.01644` and `+0.00134`. This changes the exact realisation but
not the substantive finding that unsupervised label-cluster agreement is weak.

New p/q multiseed checkpoints bind the graph and feature SHA-256 values,
complete model/evaluation parameters, package versions and code hashes. A
resume with missing metadata, a different graph hash or a different code hash
is rejected. The existing 15-row pre-M07 checkpoint is marked
`legacy_complete_read_only`: it can be verified and summarised but cannot be
extended. Opening it in the audit changed no active-output hash.

The frozen embedding, clustering and p/q tables remain the thesis evidence;
the controlled M07 runs are reproducibility evidence only. No database,
network query or active result replacement was performed.

Primary sources:

- `06_robustness_audit/outputs/node2vec_reproducibility/determinism_comparison.csv`
- `06_robustness_audit/outputs/node2vec_reproducibility/frozen_baseline_impact.csv`
- `06_robustness_audit/outputs/node2vec_reproducibility/checkpoint_safety_tests.csv`
- `06_robustness_audit/outputs/node2vec_reproducibility/m07_reproducibility_assessment.json`

## SHAP interpretation and multiple-testing policy

M08 is resolved without changing the frozen SHAP values or model scores. The
stored SHAP table comes from one Random Forest fitted to all 436 graph-active
labelled addresses. It is therefore **model-specific descriptive attribution
on the observed sample**, not causal contribution, a recovered economic
mechanism, held-out predictive evidence or a population-stable feature rank.

Correlation is material. Of the 28 hand-crafted predictors, 39 pairs have
absolute Spearman correlation of at least `0.8`; 23 features fall into six
connected correlation groups. The largest group contains seven outbound and
total transaction-count/weighted-degree raw or log features and receives
57.69% of total mean absolute SHAP mass. Its highest-ranked member,
`weighted_out_degree`, receives only 21.07% of that group's mass. Individual
members may therefore exchange attribution, and correlated groups must be
interpreted before individual rank. The aggregate split remains 84.85%
transaction-behaviour versus 15.15% graph-structural attribution, but this is
an RF attribution share rather than evidence that transactions causally
contribute 84.85% of class identity.

The thesis-primary conditional family contains six two-sided, repeat-level
main-sample comparisons: hand-crafted versus transaction-only features and
hand-crafted plus Node2Vec versus hand-crafted features, each across LR, RF
and XGBoost. Holm step-down correction controls family-wise error at 0.05.
Adding structural features has adjusted p-values `0.054620` (LR), `0.000536`
(RF) and `0.856562` (XGBoost); only the RF decrease is retained. The three
Node2Vec increments remain after correction, with adjusted p-values
`0.000733`, `0.007579` and `0.002265`, respectively.

Nine coarse-baseline and EOA comparisons form a separate Holm-adjusted
secondary family and remain exploratory scope checks regardless of adjusted
p-value. Single-split five-fold ablation/classification p-values, balanced-
subset tests and clustering permutation tests are exploratory diagnostics.
GNN seed-level p-values retain the conditional-only M05 status. Because all
repeated splits reuse one fixed labelled sample, even Holm-adjusted primary
p-values describe conditional split stability and do not establish population
or unseen-entity generalisation.

Primary sources:

- `06_robustness_audit/outputs/interpretability_multiplicity/correlated_feature_groups.csv`
- `06_robustness_audit/outputs/interpretability_multiplicity/multiplicity_adjusted_tests.csv`
- `06_robustness_audit/outputs/interpretability_multiplicity/m08_interpretability_policy.json`

## Frozen main line

1. All graph-dependent thesis results use
   `00_data_graph/data/main/raw_edges_reconstructed.csv` and downstream
   reconstructed products.
2. Pre-reconstruction outputs and superseded single-run experiments are not
   thesis evidence. They are isolated under `99_archive_not_for_thesis/`.
3. Final GNN performance is taken from
   `04_supervised/02_gnn_models/graphsage_gcn_multiseed_results/`.
4. Final GNN-vs-tree comparisons are taken from
   `04_supervised/03_gnn_evaluation/message_passing_and_baseline_results/symmetric_multiseed_comparison.csv`;
   inferential wording is governed by
   `06_robustness_audit/outputs/gnn_uncertainty/`.
5. The archived single-run full-feature GraphSAGE/GCN outputs must not be used
   in Results, Discussion, Abstract, Conclusion or the defence.
6. Entity-generalisation evidence comes only from
   `06_robustness_audit/outputs/entity_generalisation/`.
7. No result table should be interpreted outside the thesis-facing conclusion
   boundary stated near the top of this file.

## Pipeline and ownership

| Module | Code | Thesis-facing results |
|---|---|---|
| 0 Data and graph | Data files in `00_data_graph/data/` | Reconstructed graph; before graph only for topology validation |
| 1 Sanity and EDA | `01_sanity_eda/sanity_check/`, `01_sanity_eda/eda_walkthrough.ipynb` | sanity and topology checks in `01_sanity_eda/sanity_check/` |
| 2 Feature engineering | `02_feature_engineering/feature_engineering/`; nonseed query in `02_feature_engineering/nonseed_transaction_query/` | feature matrices and feature groups in `02_feature_engineering/feature_engineering/` |
| 3 Unsupervised | Executed `03_unsupervised/node2vec.ipynb` | `03_unsupervised/node2vec_results/` |
| 4 Supervised | Executed notebooks in `04_supervised/01_traditional_ml/`, `04_supervised/02_gnn_models/` and `04_supervised/03_gnn_evaluation/` | Classification CV, GNN multiseed, P3 and matched-baseline core CSVs beside their notebooks |
| 5 Interpretability | `05_interpretability/feature_ablation_shap.ipynb` | Core ablation and SHAP CSVs in `05_interpretability/feature_ablation_shap_results/` |
| 6 Robustness audit | `06_robustness_audit/code/` | Balance, contract confound and corrected entity-generalisation checks |
| 7 Synthesis | This file | Final evidence map |

`04_supervised/01_traditional_ml/02_repeated_cv.ipynb` belongs primarily to Module 4 because it supplies the final
statistical inference for supervised feature-set comparisons. Its EOA-only and
feature-ablation contrasts may also be cited in Modules 5 and 6 without
duplicating the script.

## RQ1: feature evidence

- Transaction-behaviour features are the dominant predictive source.
- The fitted Random Forest assigns approximately 84.85% of its mean absolute
  SHAP attribution mass to transaction-behaviour features and 15.15% to graph
  structural features. This is descriptive and model-specific; 23/28 features
  belong to strongly correlated components, so individual ranks are not
  interpreted as independent or causal effects.
- Adding hand-crafted structural features to transaction behaviour does not
  provide stable positive increment. In reconstructed repeated CV, the mean
  hand-crafted-minus-transaction difference is `+0.0096` for LR, `-0.0192`
  for RF and `-0.0006` for XGBoost. Under the six-test primary Holm family,
  only the RF decrease is retained (`p_adj=0.000536`); LR has
  `p_adj=0.054620` and XGBoost `p_adj=0.856562`.

Primary sources:

- `05_interpretability/feature_ablation_shap_results/`
- `04_supervised/01_traditional_ml/classification_cv_results/repeated_cv_repeat_level_ci.csv`
- `06_robustness_audit/outputs/interpretability_multiplicity/`

## RQ2: graph representation evidence

- Reconstructed Node2Vec embeddings add predictive information on the main
  active sample: mean macro-F1 increments are `+0.0211` (LR), `+0.0118` (RF)
  and `+0.0161` (XGBoost). All three repeat-level 95% confidence intervals
  exclude zero and all 10/10 repeat-level directions are positive. This is an
  address-level transductive result computed on the observed graph: corrected
  M02 holdouts do not establish a general Node2Vec gain for unseen entities or
  a cold-start prediction capability for arbitrary new addresses.
  Holm-adjusted p-values across the six primary tests are `0.000733` (LR),
  `0.007579` (RF) and `0.002265` (XGBoost); these remain conditional on the
  fixed labelled sample.
- Reconstructed-graph p/q robustness favours the unbiased baseline. Across
  five Node2Vec training seeds, balanced ARI is `0.1394 +/- 0.0082` for
  `p=q=1`, `0.1117 +/- 0.0082` for BFS-like `(p=0.25, q=4)`, and
  `0.1046 +/- 0.0074` for DFS-like `(p=4, q=0.25)`.
- Full-feature mean macro-F1 is `0.8274` for GraphSAGE (seed range
  `0.8076--0.8673`) and `0.8255` for GCN (`0.8026--0.8629`). Their mean
  difference is only `+0.0019`, with mixed direction across seeds. This should
  be reported as near parity, not as GraphSAGE superiority.
- Both GNNs are above graph-structural RF in all five paired seed labels, but
  neither has a directionally consistent advantage over graph-structural
  XGBoost.
- Against RF/XGBoost trained on the same complete hand-crafted information,
  both GNNs are lower in every paired seed label. This is the final
  apples-to-apples GNN conclusion within the fixed 153-address design; the
  thesis should not frame the GNNs as the best-performing models.

Primary sources:

- `03_unsupervised/node2vec_results/`
- `04_supervised/01_traditional_ml/classification_cv_results/repeated_cv_repeat_level_ci.csv`
- `04_supervised/02_gnn_models/graphsage_gcn_multiseed_results/multiseed_summary.csv`
- `04_supervised/02_gnn_models/graphsage_gcn_multiseed_results/multiseed_graphsage_vs_gcn.csv`
- `04_supervised/03_gnn_evaluation/message_passing_and_baseline_results/symmetric_multiseed_comparison.csv`

## Required limitation

All 150 DeFi seeds are contract addresses. DeFi identity is therefore nested
inside contract identity in this sample, so the full three-class experiment
cannot completely separate DeFi-specific economic behaviour from generic
contract-address characteristics. Corrected address-disjoint, one-sided
protocol holdouts show partial transfer across the identifiable DeFi groups,
but do not eliminate this limitation or establish unseen-MEV-entity
generalisation.

The minimal three-feature baseline (`is_contract`, `has_out_edge`,
`out_tx_count`) reaches 0.9169 Random Forest macro-F1, so account type is a
major shortcut in the observed label construction. This does not invalidate
the numerical model comparisons, but it prevents interpreting the three-class
score as independent recovery of DeFi-specific economic behaviour. EOA-only
CEX-versus-MEV-related performance remains approximately 0.92--0.93, showing
that contract status does not explain every label-associated difference.

Primary source: `06_robustness_audit/outputs/contract_confound/`.

## Page-budget rule

The 25-page dissertation should present one reconstructed-graph narrative.
Do not include full old-vs-new result tables. Use only a short topology
validation statement to explain that reconstruction increased average degree
from 3.16 to 11.70, mean clustering from 0.0065 to 0.1306 and triangles from
525 to 14,847, while the main conclusions remained stable.
