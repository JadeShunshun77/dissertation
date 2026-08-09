# M08 Interpretability And Multiplicity Resolution Record

**Date:** 2026-08-06  
**Scope:** SHAP interpretation, correlated predictors and thesis-facing tests

## Defects Confirmed

- SHAP was computed from a Random Forest fitted on all 436 active labelled
  addresses, so it was not held-out interpretability evidence.
- Raw/log, total/directional and count/degree predictors were strongly
  correlated, making individual SHAP rank unstable as an independent claim.
- The project exposed p-values from single five-fold splits, repeated folds,
  repeat-level summaries and model seeds without one binding family policy.

## Changes

- Added `audit_interpretability_multiplicity.py` and four formal audit outputs
  under `06_robustness_audit/outputs/interpretability_multiplicity/`.
- Restricted SHAP wording to model-specific descriptive attribution and made
  correlated groups the preferred individual-feature interpretation level.
- Defined six main-sample repeat-level tests as one thesis-primary conditional
  Holm family and nine coarse/EOA tests as a separate exploratory Holm family.
- Classified single-split, balanced-subset, clustering and GNN seed p-values
  as exploratory or conditional diagnostics.

## Results

At `|Spearman rho| >= 0.8`, 39 feature pairs form six components containing
23 of 28 predictors. The largest seven-feature transaction count/degree group
contains 57.69% of SHAP mass. Transaction and structural groups receive
84.85% and 15.15%, respectively, but these are attribution shares, not causal
contributions.

Four of six primary comparisons remain after Holm correction. The RF
hand-crafted-minus-transaction decrease remains (`p_adj=0.000536`), while the
LR increment does not (`p_adj=0.054620`) and XGBoost remains unsupported
(`p_adj=0.856562`). Node2Vec increments remain for LR, RF and XGBoost with
adjusted p-values `0.000733`, `0.007579` and `0.002265`.

All tests remain conditional on one fixed labelled sample. No population,
causal, unseen-entity or stable individual-feature claim is authorised.
No frozen SHAP value, model score, graph, feature matrix or database changed.
