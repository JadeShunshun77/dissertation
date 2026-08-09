# M02 Resolution: Entity And Protocol Generalisation

**Resolution date:** 2026-08-05  
**Status:** Resolved for the identifiable evaluation scope; full three-class
entity generalisation accepted as unidentifiable from the current labels  
**Main experiment rerun required:** No  
**New robustness run:** Yes, 1,400 address-disjoint model fits

## Defect In The Previous Check

The previous `entity_grouped_check.py` treated every `sub_label` as an entity.
That was not valid for the MEV-related class: `Arbitrage Trader` is a role, not
evidence that 297 addresses belong to one independently identified entity. Its
leave-one-DeFi-protocol-out test also reused sampled MEV addresses in training
and testing. Those five outputs are preserved under
`99_archive_not_for_thesis/06_robustness_audit/outputs/entity_grouped_leaky_pre_m02/`
and must not be cited.

## Identifiability Decision

- A fully entity-disjoint three-class evaluation is not identifiable because
  the dataset has no MEV entity ownership/group metadata.
- CEX entity groups are identifiable after combining `Coinbase Prime` with
  `Coinbase` and `Binance US` with `Binance`.
- DeFi protocol groups are identifiable directly from their sub-labels.
- The active aggregated edge table has no timestamp, so it cannot support a
  post-hoc temporal holdout.

The corrected evaluation therefore measures one-sided transfer only:

1. EOA-only unseen CEX entity versus address-disjoint MEV-related EOAs.
2. Contract-only unseen DeFi protocol versus address-disjoint MEV contracts.

The MEV controls are address-disjoint but cannot be claimed to be
entity-disjoint.

## Corrected Design

- Held-out groups: Coinbase, OKX and Binance; Uniswap, Curve, 0x API and Dodo.
- Fifty deterministic repeats per held-out group.
- At most ten test addresses per class in each repeat.
- Balanced train and test classes within each repeat.
- Zero train/test address overlap, enforced by runtime assertions.
- Hand-crafted and hand-crafted-plus-Node2Vec feature sets.
- Logistic Regression and Random Forest models.
- All Node2Vec embeddings remain transductive and label-free.

The 0x API and Dodo holdouts contain only two addresses each and are low-power
supplements. Repeated resampling estimates address/model variability within a
fixed small set of groups; it does not create 50 independent entities.

## Primary Descriptive Results

| One-sided holdout | Features | RF group-equal macro-F1 | Worst group | Best group |
|---|---|---:|---:|---:|
| Unseen CEX entity, EOA-only | Hand-crafted | 0.8784 | 0.8278 | 0.9200 |
| Unseen CEX entity, EOA-only | Hand-crafted + Node2Vec | 0.8332 | 0.7705 | 0.9145 |
| Unseen DeFi protocol, contract-only | Hand-crafted | 0.9270 | 0.8881 | 0.9627 |
| Unseen DeFi protocol, contract-only | Hand-crafted + Node2Vec | 0.9324 | 0.8987 | 0.9680 |

Across all 1,400 fits, maximum train/test address overlap was zero. The six
active output files reproduced byte-for-byte in two independent runs.

## Interpretation

The corrected tests provide evidence that some label-associated patterns
transfer to held-out CEX entities and DeFi protocols. They do not establish
full unseen-entity generalisation for the three-class task. In particular,
the main address-level CV and its p-values remain interpolation evidence, and
the Node2Vec increment observed there must not be promoted to an entity-level
claim. Node2Vec did not improve the RF CEX holdout average and made little
difference to the RF DeFi holdout average.

## Thesis-Ready Wording

> A fully entity-disjoint three-class evaluation was not identifiable because
> the MEV-related labels did not contain entity ownership groups. We therefore
> performed one-sided group holdouts with address-disjoint MEV controls. Across
> 50 repeats, Random Forest achieved group-equal macro-F1 of 0.878 for unseen
> CEX entities in the EOA-only comparison and 0.927 for unseen DeFi protocols
> in the contract-only comparison. These results support limited transfer
> across the identifiable CEX and DeFi groups, but not generalisation to unseen
> MEV entities or arbitrary Ethereum addresses.

## Validation Record

- Corrected script: `06_robustness_audit/code/entity_grouped_check.py`.
- Active evidence: `06_robustness_audit/outputs/entity_generalisation/`.
- 1,400 expected rows, no duplicate run keys and zero address overlap.
- Validation and active outputs have identical SHA256 hashes.
- Previous leaky outputs were moved byte-for-byte to the archive.
