# GOLDmicro Batch 20260916_003948 — Zero-Shortlist Review

## Scope

Operator-provided local evidence from the prefix-causal GOLDmicro research run at Git SHA `cb281cfca5231f8d2d5e1621a3856f74ded15a5f`.

This note records the research outcome only. It does not authorize live trading, model activation, credential changes, or promotion.

## Observed run result

- 24 challenger configurations
- 5 chronological samples per configuration
- 120/120 training jobs completed successfully
- causal research SMC/HMM/V2 path enabled
- multi-sample AUC/generalization screen: 0 stable passes
- shortlist: 0 configurations
- therefore PF/DD/cost strategy OOS was not reached
- live model unchanged
- promotion disabled

The launcher then exited with code 1 only because the strategy-OOS runner treated an empty queue as an exception. An empty predictive shortlist is a valid research result and must not be classified as an infrastructure failure.

## AUC distribution from the 120 completed causal jobs

Parsed from the operator run log:

- mean test AUC: approximately 0.5177
- median test AUC: approximately 0.5193
- minimum test AUC: approximately 0.4908
- maximum test AUC: approximately 0.5337
- 95th percentile test AUC: approximately 0.5298
- required first-stage test AUC gate: 0.55

No individual causal job reached the 0.55 AUC threshold, so the zero-shortlist result is not a borderline failure caused by the 4/5 stability rule. The entire tested grid sat below the current predictive-quality gate.

## Strongest configuration families by median causal test AUC

These are diagnostic leaders only and are NOT promotion-eligible:

1. `ch-013`: 20k bars, seed 29, balanced XGB, HMM 400, confidence 0.62, core_plus_v2, normal-cost label — median AUC approximately 0.5251.
2. `ch-018`: 15k bars, seed 47, balanced XGB, HMM 500, confidence 0.60, core, conservative-cost label — median AUC approximately 0.5235.
3. `ch-016`: 10k bars, seed 29, conservative XGB, HMM 750, confidence 0.62, core_plus_v2, normal-cost label — median AUC approximately 0.5230.

The responsive families also showed some larger train-test gaps, so simply increasing model flexibility is not supported by this run.

## Interpretation

The causal corrections appear to have removed a large amount of apparent predictive strength. That is a positive validation outcome even though it eliminates the current shortlist: the research pipeline is now refusing to manufacture a candidate from weak evidence.

The current binary target is still `future close > current close` with `lookahead=1` M15 bar. This is potentially misaligned with the hybrid trading strategy, whose SMC setup and ATR/TP/timeout logic can hold for multiple bars. A one-bar directional target can be dominated by short-horizon noise even when a multi-bar setup has useful expectancy.

## Decision

Do NOT lower the 0.55 AUC gate merely to force candidates into PF/DD testing.

Before another full 24 x 5 optimization run, perform a focused non-live target-alignment study. Candidate directions should include multi-bar horizons (for example 4, 8, and 16 M15 bars) and/or a volatility-aware target that distinguishes economically meaningful movement from near-zero noise. Any such experiment must preserve causal feature construction, chronological splits, the 50-bar leakage gap, broker-correct GOLDmicro cost modeling, and the existing Human Gate.

## Pipeline correction

The research scripts are being corrected so an empty shortlist writes `NO_ELIGIBLE_CONFIGURATIONS`, produces an empty shadow queue, leaves promotion disabled, and exits successfully as a valid research conclusion rather than throwing an exception.
