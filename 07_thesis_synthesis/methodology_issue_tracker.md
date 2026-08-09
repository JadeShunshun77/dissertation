# Methodology Issue Resolution Tracker

**Created:** 2026-08-05  
**Scope:** Active reconstructed-graph dissertation pipeline  
**Baseline protection:** The frozen data and result artifacts must not be
overwritten while an issue is being investigated. Validation runs must use a
separate output directory until their effect has been reviewed.

This tracker turns the methodology review into eight bounded tasks. An issue
is closed only when its acceptance checks are recorded here. A limitation can
be closed as `ACCEPTED LIMITATION` when it cannot be removed with the available
data, provided the thesis claim is narrowed accordingly.

## Status Summary

| ID | Issue | Severity | Current status | Existing results invalidated? |
|---|---|---:|---|---|
| M01 | Label semantics and contract-status confounding | Critical | RESOLVED / ACCEPTED LIMITATION | No; interpretation is restricted |
| M02 | Address-level entity/protocol leakage | Critical | RESOLVED / PARTIALLY IDENTIFIABLE | No; generalisation claim is restricted |
| M03 | Missing executable graph construction and label provenance | Critical | RESOLVED / HISTORICAL LIMITATION | No; provenance boundary is explicit |
| M04 | Asymmetric reconstructed-graph sampling | High | RESOLVED / SCOPE RESTRICTED | No; graph scope is explicit |
| M05 | GNN uncertainty and significance inference | High | RESOLVED / CONDITIONAL INFERENCE | No; numerical results are unchanged |
| M06 | Direction, time, edge type and weight information loss | Medium | RESOLVED / DOCUMENTED BASELINE | No; representation scope is explicit |
| M07 | Node2Vec random-state and checkpoint integrity | Medium | RESOLVED / FROZEN BASELINE PRESERVED | No; deterministic audit is separate |
| M08 | SHAP interpretation, correlation and multiple testing | Medium | RESOLVED / CONDITIONAL INTERPRETATION | No; wording and inference policy changed |

## Post-Review Claim-Control Note

The later teacher-style audit did not require new experiments, new active
result files or model reruns. Its required corrections are documentation-level
claim controls: the thesis must report `Malicious_MEV` as MEV-related/arbitrage
rather than malicious ground truth, treat contract-status confounding as a
central limitation, describe the graph as a label-seeded 90-day subgraph, frame
Node2Vec and GNN evidence as transductive observed-graph evidence, and report
GraphSAGE/GCN as static baselines that do not outperform complete
hand-crafted RF/XGBoost models under the fixed balanced design.

## M01: Label Semantics And Contract Confounding

**Resolution:** Closed for the experiment pipeline on 2026-08-05. The stored
technical label remains unchanged for artifact continuity, while the thesis
uses `MEV-related/arbitrage`. Contract status is retained as an explicit design
limitation. Full rationale and validated wording are recorded in
`07_thesis_synthesis/m01_label_confound_resolution.md`. M03 still owns the
documented label-provenance boundary, and M02 owns unseen-entity
generalisation.

**Evidence:** Active DeFi addresses are all contracts, while CEX and
`Malicious_MEV` are mostly EOAs. The three-feature baseline
`is_contract + has_out_edge + out_tx_count` reaches approximately 0.92
macro-F1. Most `Malicious_MEV` addresses are labelled `Arbitrage Trader`.

**Resolution actions:**

- Audit and document the source and meaning of every top-level label.
- Decide whether the thesis-facing class name should be `MEV/arbitrage` rather
  than implying that every sample is malicious.
- Treat the contract baseline as a central result, not a footnote.
- Define the estimand as public address-role classification under the observed
  label construction.

**Acceptance checks:** PASS at the evidence-synthesis layer. Label wording is
evidence-supported; account-type confounding has mandatory Results and
Limitations wording; no permitted claim equates all arbitrage with malicious
conduct. Final thesis text must be checked against the resolution document.

## M02: Address-Level Entity And Protocol Leakage

**Resolution:** Closed for the identifiable scope on 2026-08-05. The old
GroupKFold/LOPO outputs reused MEV addresses or treated a role label as an
entity and are archived as non-thesis evidence. The corrected script performs
one-sided unseen CEX-entity and DeFi-protocol holdouts with zero train/test
address overlap. Full three-class unseen-entity generalisation remains
unidentifiable because MEV entity groups are absent. See
`07_thesis_synthesis/m02_entity_generalisation_resolution.md`.

**Evidence:** Main CV splits addresses randomly. Major `sub_label` groups can
therefore occur in both train and test folds. Existing GroupKFold folds are
often class-degenerate, and the leave-one-protocol-out check reuses MEV
addresses because only one usable MEV entity group is available.

**Resolution actions:**

- Produce a final table of active counts by label and `sub_label`.
- Establish which unseen-entity tests are statistically feasible.
- If a valid three-class grouped test is impossible, report that fact and
  restrict the thesis to address-level interpolation.
- Prefer one defensible grouped or temporal holdout over adding another model.

**Acceptance checks:** PASS. No unseen-entity claim relies on random address
CV; the corrected 1,400 runs have zero address overlap; unsupported full
three-class entity generalisation is explicitly prohibited; old leaky outputs
are isolated from active thesis evidence.

## M03: Graph Construction And Label Provenance

**Resolution:** Closed on 2026-08-06 for independent audit and downstream
reconstruction. The active graph is reproducible byte-for-byte from its two
frozen components, and every active address-label mapping is linked to a
frozen source snapshot. Missing historical source-query metadata is retained
as an accepted limitation rather than reconstructed speculatively.

**Evidence:** `00_data_graph/code/` retains read-only label and graph rebuilds;
`00_data_graph/sql/` distinguishes the recovered nonseed-to-nonseed
augmentation query from the documented equivalent original Top-100 query; and
`00_data_graph/provenance/` contains the frozen source registry, row-level
label provenance and verification report.

**Resolution actions:**

- Restore or reconstruct the exact read-only BigQuery SQL and assembly steps.
- Record source table, date/block window, filters, aggregation and sampling.
- Create a label provenance table with source, retrieval date, evidence and
  any licence or usage constraints.
- Add checksums and row-count assertions for generated graph inputs.

**Acceptance checks:** PASS within the preserved evidence boundary. The
47,197-row graph rebuild is byte-identical to the frozen active graph; all 500
active label mappings match restored snapshots; file hashes and row counts are
machine-readable; SQL records are read-only; and no database file or live
query is involved. Exact re-extraction of the original Top-100 result, Dune
exports and historical seed-row order is not claimed because their query IDs,
Job ID or assembly detail were not retained.

## M04: Reconstructed-Graph Sampling Scope

**Resolution:** Closed on 2026-08-06 as a documented sampling design with
restricted claims. The active graph remains unchanged. A deterministic audit
now quantifies every edge category, verifies the fixed-node and seed-edge
invariants, and records label coverage.

**Evidence:** Reconstruction retains the fixed 8,029-node set and adds 34,336
nonseed-to-nonseed directed rows. All 12,861 original rows touch at least one
seed; the augmentation touches none. The active graph contains 436/500 labels
(51/51 CEX, 86/150 DeFi and 299/299 MEV-related). Edges touching seeds remain
the union of bidirectional Top-100 candidate selections.

**Resolution actions:**

- Describe the graph as a label-seeded, fixed-node, partially reconstructed
  transaction subgraph, not a complete Ethereum induced subgraph.
- Document why seed-touching edges were held fixed and which controlled
  structural sensitivity comparison this design supports.
- Quantify original edges, added edges, node coverage and active-label coverage.

**Acceptance checks:** PASS. The frozen scope outputs establish that the active
pair set equals the disjoint union of the two components, the node set and
seed-incident pairs are fixed, component values are unchanged, and the
augmentation introduces no node or seed edge. Thesis-facing wording prohibits
complete-network, unbiased-sample, complete-degree and causal claims.

## M05: GNN Uncertainty And Significance

**Resolution:** Closed on 2026-08-06 by reclassifying all GNN seed-level tests
as descriptive conditional-stability analyses. Frozen model scores and raw
p-values remain unchanged for audit continuity, but thesis-facing inference
uses means, ranges and seed-direction consistency rather than significance
language.

**Evidence:** GraphSAGE and GCN use one fixed 153-address subset, one fixed
five-fold assignment and one graph snapshot. Five model-training seed labels
measure optimisation variability, not five independent address samples or
graph snapshots. The P3 rewired condition additionally varies graph rewiring.
Both XGBoost conditions produce one unique score across all requested seeds.

**Resolution actions:**

- Reclassify seed-level tests as conditional stability analyses.
- Remove or soften population-level `significant` language based on n=5 model
  seeds.
- Where feasible, estimate uncertainty over valid data splits or groups and
  apply a stated multiple-comparison policy.

**Acceptance checks:** PASS. The formal audit records one address sample, one
fold assignment and one graph snapshot; all comparison rows set
`population_inference_supported=False`; raw seed-level p-values are labelled
conditional diagnostics; and the thesis log contains no GNN population-level
`significant`/`not significant` claim. GNN p-values are excluded from the
confirmatory family rather than subjected to a correction that cannot repair
the sampling-unit mismatch.

## M06: Simplified Graph Representation

**Resolution:** Closed on 2026-08-06 as a documented static baseline with
restricted claims. No new model was added. The audit distinguishes edge
propagation from node features and corrects the earlier overbroad statement
that Node2Vec discards all weights.

**Evidence:** Node2Vec uses an undirected `tx_count`-weighted graph. Final GNN
message passing is undirected and unweighted, but its node features retain
directed in/out statistics, transaction count, ETH value, gas fee and weighted
PageRank. The edge table has no per-transaction time, token-transfer, internal
trace, event or decoded protocol fields. Seed-node in/out degree correlation
is approximately -0.036 and the top 1% of pairs carry 80.0% of transaction
count, so direction and weight information is not treated as redundant.

**Resolution actions:**

- Justify the static undirected representation as the dissertation baseline.
- State which Ethereum interactions are absent and how this affects contracts.
- Decide whether one small directed/weighted sensitivity test is feasible; do
  not expand scope unless it changes the research question materially.

**Acceptance checks:** PASS. A machine-readable usage matrix and loss audit
record each component's direction and weight handling, absent interaction
types and permitted claims. Node2Vec's weight key is explicit in active code.
A directed/weighted GNN run is not misrepresented as a small sensitivity: it
would require architecture changes and multiseed retraining, while absent
temporal and interaction-type data cannot be recovered by rerunning models.
No claim treats the graph as a complete record of DeFi activity.

## M07: Node2Vec Reproducibility And Resume Safety

**Resolution:** Closed on 2026-08-06. Active code now seeds Python's
`random` and NumPy, supplies a process-independent SHA-256 hash function to
gensim Word2Vec, uses one worker, and writes input/configuration/package/code
metadata. New multiseed checkpoints reject missing or incompatible metadata.
The completed pre-M07 checkpoint is explicitly legacy, complete and read-only.

**Evidence:** NumPy is seeded, but Python's `random` module is not. The p/q
checkpoint key records setting and training seed but not the graph hash or full
configuration.

**Resolution actions:**

- Seed every random-number source used by walk generation and Word2Vec.
- Store graph SHA256, parameters, package versions and code version beside the
  checkpoint; reject incompatible resumes.
- Run controlled before/after validations in new output directories.

**Acceptance checks:** PASS. Two clean, full-parameter runs used different
`PYTHONHASHSEED` values and produced identical walk, embedding and assignment
SHA-256 values and identical NMI/ARI/Silhouette. Missing metadata, mismatched
graph hashes and mismatched code hashes were all rejected. Opening the legacy
checkpoint changed no active-output hash. The controlled ARI was 0.04294
versus the frozen 0.0265 (absolute delta +0.01644); the qualitative weak-
clustering conclusion is unchanged, and no frozen result was replaced.

## M08: SHAP And Multiple-Testing Interpretation

**Resolution:** Closed on 2026-08-06. SHAP is now governed as in-sample,
model-specific descriptive attribution rather than causal contribution or
held-out evidence. A Spearman audit groups strongly associated predictors,
and a single Holm policy separates six primary main-sample comparisons from
nine secondary exploratory scope checks. Single-split and GNN seed p-values
remain diagnostics rather than confirmatory evidence.

**Evidence:** SHAP is calculated from a model fitted on the full labelled
sample. Raw/log and related count/degree features are correlated, and several
paired tests are reported without a single family-wise correction policy.

**Resolution actions:**

- Describe SHAP mass as model-specific diagnostic attribution, not causal
  contribution.
- Report correlated feature groups and avoid overinterpreting individual rank.
- Define primary hypotheses and apply Holm correction to secondary families,
  or label them exploratory.

**Acceptance checks:** PASS. Among 28 hand-crafted features, 39 pairs have
`|rho| >= 0.8`; 23 features form six correlated components. The 84.85% versus
15.15% SHAP split is retained only as RF attribution mass. Holm correction
over the six primary repeat-level tests retains the RF structural-feature
decrease and all three Node2Vec increments; the LR structural increment moves
from raw `p=0.02731` to adjusted `p=0.05462` and is not rejected. Secondary
tests are separately adjusted but remain exploratory. Every adjusted test is
explicitly marked as not supporting population inference from the fixed
sample. No frozen result was replaced.

## Resolution Order

Work through `M01` to `M08` in order. After each issue, record the files
changed, commands run, validation evidence, result impact and final status.
Do not mark several issues resolved together without their individual checks.
