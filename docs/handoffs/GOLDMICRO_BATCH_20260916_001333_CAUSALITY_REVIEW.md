# GOLDmicro Batch 20260916_001333 — Causality Review

Date: 2026-09-16 (Asia/Bangkok)
Scope: non-live research only
Source batch Git SHA: `e5f0138a3615f38210a8373e0bf1bf1d1bcc3ea0`

## Observed batch result

The local 24 x 5 chronological matrix completed all 120 training jobs. The AUC/generalization pre-screen reported 24/24 stable passes and emitted a six-configuration shortlist. Active/live model paths were not overwritten and promotion remained disabled.

These results are useful execution evidence for the research pipeline, but they are **not canonical model-selection evidence** until the causality issue below is removed and the matrix is rerun.

## Causality finding

`src/smc_polars.py` detects an order-block origin candle only after a later confirmation bar is observed, then writes the `ob`, `ob_top`, and `ob_bottom` marker back onto the earlier origin row. That is useful for retrospective chart annotation but is not prefix-causal when a full historical frame is reused as ML training data.

The GOLDmicro candidate feature policy includes `ob` in the core feature set and derives V2 continuous SMC features from order-block fields. Therefore the 20260916_001333 candidate artifacts may contain future-confirmed order-block information on earlier feature rows.

The shared/live SMC implementation has **not** been changed. A research-only wrapper, `src/goldmicro_causal_smc.py`, emits an order-block event on its confirmation bar while retaining the historical origin-zone prices. Prefix-invariance tests were added.

## Chronological sample interpretation

The five samples are separated by 1,000 M15 bars. Candidate OOS partitions are materially longer than 1,000 bars for the larger training windows, so these five samples are overlapping temporal stability probes, not five statistically independent OOS folds.

Accordingly, a 4/5 pass rule is useful as a stability filter but must not be interpreted as five independent confirmations. Independent forward/shadow evidence remains mandatory before any future Human-Gated activation decision.

## Next research gate

The one-click pipeline now routes candidate training through the research-only causal SMC wrapper, then performs the multi-sample AUC pre-screen and continues automatically into strategy OOS validation.

The strategy OOS stage applies:

- GOLDmicro 0.10-lot strategy ladder with downward-only quantization;
- calibrated THB cash conversion (`33.29 THB` per +1.00 GOLD move per 1.00 lot for the inspected account);
- normal execution cost: 50-point spread + 1 point adverse slippage per side;
- conservative execution cost: 55-point spread + 12 points adverse slippage per side;
- PF >= 1.30;
- max DD <= 10%;
- risk-sizing skip <= 20%;
- minimum 30 realized trades per sample;
- positive expectancy;
- at least 4/5 sample passes per cost profile;
- any >10% sample DD is a hard configuration rejection;
- both normal and conservative cost profiles must pass before a configuration can enter the non-executing shadow queue.

The strategy OOS simulator is a conservative research proxy, not an exact tick replay of legacy #24. If TP and SL are both touched inside the same M15 bar, it assumes the stop was hit first.

## Governance

No live order path, `main_live.py`, active model path, production credential, or live AutoTrainer behavior is changed by this work. Model promotion remains disabled. Independent review, forward shadow evidence, and explicit Human Gate are still required before any future activation work.
