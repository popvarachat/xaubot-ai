# GOLDmicro Event-Target V2 — Claude Opus Independent Audit Handoff

## Role

Act as an independent senior quantitative-ML / trading-research reviewer. Do not assume the ChatGPT PM conclusion is correct. Challenge the target design before any new large local training run.

## Repository / governance

- Repository: `popvarachat/xaubot-ai`
- Branch: `feat/goldmicro-broker-profile-v1`
- PR: #1 (must remain Draft)
- Human Gate: Pop
- Scope: research only
- No live trading, active-model overwrite, credential/auth mutation, production deployment, or model promotion

Check root `AGENTS.md` first. If absent, state that explicitly.

## START HERE

Read:

1. `docs/handoffs/GOLDMICRO_TARGET_ALIGNMENT_20260916_064904_RESULT.md`
2. `src/goldmicro_strategy_oos.py`
3. `src/goldmicro_causal_smc.py`
4. `src/goldmicro_candidate_trainer.py`
5. `src/goldmicro_causal_hmm.py`
6. `src/goldmicro_causal_v2.py`
7. `backtests/backtest_24_final_combined.py`
8. `src/feature_eng.py`
9. `backtests/ml_v2/ml_v2_model.py`
10. `src/multisample_screening.py`
11. `src/candidate_screening.py`

## Evidence

The completed target-horizon study tested 24 configurations x 5 temporal probes x four fixed future-close horizons = 480 jobs.

Observed summary:

- t+1: 0/24 stable, median AUC 0.5190, best config median 0.5248, max single 0.5395
- t+4: 0/24 stable, median AUC 0.5077, best config median 0.5140, max single 0.5340
- t+8: 0/24 stable, median AUC 0.5106, best config median 0.5332, max single 0.5521
- t+16: 0/24 stable, median AUC 0.5026, best config median 0.5231, max single 0.5390
- shortlist 0; PF/DD/cost skipped; live model unchanged; promotion disabled

The hard AUC gate was not relaxed.

## PM hypothesis for next experiment

The fixed future-close direction target is misaligned with the intended XGBoost role. SMC creates candidate entries; XGBoost should confirm/block or help manage them. Therefore the next target should be conditional on a causal SMC setup event rather than on every market bar.

Proposed binary label:

- event row = exact bar where `GoldmicroCausalSMCAnalyzer.generate_signal(prefix)` returns a setup
- use only features available at that event bar
- label 1 = the setup's own SMC take-profit is touched before its own SMC stop-loss within 32 M15 bars
- label 0 = stop-loss first, same-bar TP+SL ambiguity (adverse ordering), or TP does not occur first within 32 bars
- no XGBoost prediction is used to construct the label
- no transaction-cost result is used to construct the label
- strategy costs remain a later PF/DD gate

## Questions you must resolve before recommending a run

1. Is this event-conditioned target statistically and architecturally better aligned than unconditional future-close direction?
2. Is using the SMC setup's own TP/SL in the label circular in any problematic sense, or is it legitimate because those levels are known at decision time?
3. Is `32` bars the right horizon given the current strategy proxy and legacy exit logic? Recommend a different single predeclared horizon only if justified.
4. Should timeout/no-TP-first events be label 0, censored/null, or a third class? Explain bias tradeoffs.
5. Confirm same-bar TP+SL should be treated adversely because OHLC does not reveal touch order.
6. Define the correct train/OOS split and embargo in **raw bar time** for event labels. Do not rely on `TradingModelV2.fit()`'s 50-row gap if rows have been filtered to events.
7. Is it safe to precompute causal features on the full frozen frame if prefix-invariance tests pass, or should event rows be generated strictly prefix-by-prefix?
8. Can causal SMC event generation be vectorized/cached without changing semantics?
9. What minimum train/test event counts should be predeclared based on statistical precision rather than desired pass/fail outcomes?
10. Should AUC >= 0.55 remain the hard first-stage gate? If you recommend another primary metric, justify it **before** seeing event-target results. PR-AUC, Brier, calibration, and top-quantile lift may be added as diagnostics.
11. How should class imbalance be handled without leakage or post-hoc threshold tuning?
12. Does the HMM/feature pipeline remain causal when restricted to event rows?
13. The current `TradingModelV2.fit()` splits after `drop_nulls()` and applies a gap in cleaned-row units. Explain whether a dedicated event trainer is required.
14. The strategy evaluator currently interprets XGBoost as BUY/SELL direction/reversal. If the new model predicts **setup success probability**, specify a research-only evaluator contract that does not accidentally reuse directional semantics.
15. Identify any multiple-testing risk from the 24 predictive configurations and recommend how many configurations to retain.
16. Are the five temporal probes sufficiently separated? They may overlap and must not be called independent if they are not.
17. Identify any remaining leakage in SMC, H1 alignment, HMM inference, V2 features, target construction, or model evaluation.
18. Identify any path that could touch live model files, orders, secrets, or production behavior.

## Required output

Return exactly one review containing:

- `Verdict`: PASS / PASS_WITH_CHANGES / HOLD
- `Critical findings`
- `High findings`
- `Medium findings`
- `Target-definition decision`
- `Raw-time split / embargo specification`
- `Minimum event-count requirements`
- `Primary and secondary metrics`
- `Research-only inference contract`
- `Code patch recommendations`
- `Safe-to-run matrix`, only if appropriate
- `Do not do`

For every finding, cite exact file/function names. Do not promote a model. Do not modify live execution paths. If you create patches, keep them on the feature branch and leave PR #1 Draft.
