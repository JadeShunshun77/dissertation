# M01 Resolution: Label Semantics And Contract Confounding

**Resolution date:** 2026-08-05  
**Status:** Resolved for the experiment pipeline; contract confounding accepted
as a study limitation  
**Experiment rerun required:** No

## Decision

The stored label `Malicious_MEV` remains unchanged in code, CSV files and
frozen outputs. Changing it would be a presentation-only relabelling and would
create unnecessary artifact churn without changing any metric.

In the dissertation, this class must instead be described as:

> **MEV-related addresses, predominantly labelled as arbitrage traders**

The short form `MEV/arbitrage` may be used in figures and tables. The term
`malicious` must not be applied to the whole class. Of the 299 addresses in the
stored class, 297 have the sub-label `Arbitrage Trader`, one has
`Arbitrage Trader; Sandwich Attacker`, and one has `mevrevolutionx.eth`.
Arbitrage activity is not, by itself, evidence of malicious or unlawful
conduct.

## Label Composition

| Stored label | Total seeds | Active seeds | Active contracts | Thesis-facing meaning |
|---|---:|---:|---:|---|
| `CEX` | 51 | 51 | 2 | Publicly labelled centralised-exchange-associated addresses |
| `DeFi` | 150 | 86 | 86 | Publicly labelled DeFi protocol contract addresses |
| `Malicious_MEV` | 299 | 299 | 34 | MEV-related addresses, predominantly arbitrage-trader labels |

The exact historical extraction query and row-level source metadata were not
preserved. CEX and DeFi candidate files were named as Dune exports, while the
MEV source file did not record its source. Reconstructing that provenance is
tracked separately under M03 and must not be inferred here.

## What The Confound Tests Establish

1. Contract status is strongly associated with the three-class label. All
   active DeFi samples are contracts, compared with 2/51 CEX and 34/299
   MEV-related samples.
2. A Random Forest using only `is_contract`, `has_out_edge` and
   `out_tx_count` reaches 0.9169 mean macro-F1. Account type and its activity
   proxies therefore provide a strong shortcut for the observed labels.
3. `is_contract` is not part of the main hand-crafted feature set. The minimal
   baseline diagnoses the dataset construction; it does not prove that the
   main classifier directly used bytecode status.
4. Removing outbound features does not collapse performance: none of the
   paired five-fold differences is significant. The result is therefore not
   attributable to `out_tx_count` alone.
5. In the EOA-only CEX-versus-MEV-related task, hand-crafted models retain
   approximately 0.92--0.93 macro-F1. There is label-associated signal that
   cannot be explained solely by the contract/EOA distinction.
6. Contract-only DeFi-versus-MEV-related performance is also high, but entity
   concentration and grouped-fold degeneracy limit that evidence. M02 owns
   this remaining generalisation question.

## Claims Retained

- The models distinguish the three public address-role labels within the
  observed, label-seeded, 90-day reconstructed graph.
- Transaction-behaviour features dominate predictive performance in this
  dataset.
- CEX and MEV-related EOAs remain distinguishable after removing contract
  addresses, conditional on the address-level sampling design.
- The relative Node2Vec, tree-model and GNN comparisons remain numerically
  valid for this dataset.

## Claims Prohibited

- The system detects malicious Ethereum addresses in general.
- Every arbitrage trader is malicious, abusive or unlawful.
- The three-class score isolates DeFi-specific economic behaviour from generic
  smart-contract characteristics.
- Contract confounding has been eliminated.
- The observed score establishes generalisation to unseen entities, protocols
  or future time periods.

## Thesis-Ready Wording

**Methods:**

> The stored dataset used the technical label `Malicious_MEV`; however, 297 of
> its 299 addresses were sub-labelled as arbitrage traders. The dissertation
> therefore reports this class as MEV-related/arbitrage rather than treating it
> as a ground-truth malicious-address category.

**Results:**

> Contract status was strongly nested within the label construction: all 86
> active DeFi addresses were contracts, compared with 2 of 51 CEX and 34 of
> 299 MEV-related addresses. A three-feature contract-status baseline achieved
> 0.9169 macro-F1, demonstrating substantial account-type confounding.

**Discussion:**

> The three-class results show predictability of the observed public role
> labels, not independent recovery of economic intent. EOA-only CEX versus
> MEV-related classification remained strong, so account type did not explain
> every label-associated difference; nevertheless, DeFi-specific behaviour
> cannot be separated completely from generic contract characteristics in the
> current sample.

## Validation Record

- Audited all 500 seed labels and all 436 active labelled addresses.
- Cross-checked active labels against cached `eth_getCode` contract status.
- Reviewed minimal-baseline, outbound-ablation, EOA-only and contract-only
  outputs.
- No data, model code, embeddings, predictions or experiment outputs changed.
- Frozen artifact verification remains the required post-edit integrity check.

Academic terminology support: the MEV literature treats arbitrage, sandwich
attacks and other extraction mechanisms as distinct manifestations rather than
making `arbitrage` synonymous with `malicious`:
https://doi.org/10.1007/s12525-024-00727-x
