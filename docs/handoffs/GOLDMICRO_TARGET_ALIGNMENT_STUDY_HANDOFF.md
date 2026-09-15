# GOLDmicro Target Alignment Study — Claude Opus Independent Audit Handoff

## Purpose

Act as an independent senior ML/research reviewer. Do not assume the ChatGPT design is correct. The immediate question is whether the previous one-bar target was misaligned with the actual multi-bar SMC/ATR trading objective, and whether the new target-alignment study can answer that question without reintroducing leakage or p-hacking.

## Repository / scope

- Repository: `popvarachat/xaubot-ai`
- Working branch: `feat/goldmicro-broker-profile-v1`
- PR: #1
- Research only. No live trading, no model activation, no credential/auth changes.
- Human Gate: Pop.

## START HERE

Read these files first:

1. `src/goldmicro_target_trainer.py`
2. `src/target_alignment.py`
3. `scripts/run_goldmicro_target_alignment_study.py`
4. `src/goldmicro_candidate_trainer.py`
5. `src/goldmicro_causal_smc.py`
6. `src/goldmicro_causal_hmm.py`
7. `src/goldmicro_causal_v2.py`
8. `src/multisample_screening.py`
9. `src/candidate_screening.py`
10. `src/goldmicro_strategy_oos.py`
11. `scripts/run_goldmicro_strategy_oos.py`
12. `docs/handoffs/GOLDMICRO_BATCH_20260916_003948_ZERO_SHORTLIST_REVIEW.md`

If root `AGENTS.md` exists, read it before making any recommendation. If it does not exist, state that explicitly.

## Evidence that motivated this study

The causal 24 x 5 batch completed 120/120 training jobs, but no configuration passed the stable AUC screen. The observed one-bar target distribution was approximately:

- mean test AUC around 0.518
- median around 0.519
- maximum single test AUC around 0.534
- existing hard screen remains AUC >= 0.55 on at least 4/5 chronological samples

Do not recommend lowering the hard gate merely to create survivors.

## New study design

Default one-run matrix:

- 24 predictive configurations
- 5 chronological samples/configuration
- target horizons: 1, 4, 8, 16 M15 bars
- total: 24 x 5 x 4 = 480 training/evaluation jobs
- one frozen M15/H1 master snapshot
- same causal SMC/HMM/V2 feature path
- target is future close direction at the selected horizon
- OOS leakage gap remains 50 M15 bars, larger than every tested target horizon
- training cost profile fixed to `normal` because execution cost does not change predictive model fitting
- confidence threshold fixed at 0.60 because it does not change training AUC
- normal and conservative costs are still tested later at strategy OOS
- AUC >= 0.55, train-test gap <= 0.12, test samples >= 500, stable pass >= 80%
- no gate auto-relaxation

## Audit questions

Review specifically for:

1. Target leakage at train/OOS boundaries for multi-bar labels.
2. Any remaining non-causal SMC, HMM, H1, or V2 feature behavior.
3. Whether overwriting the one-bar target after shared feature preparation is safe.
4. Whether a 50-bar leakage gap is sufficient for horizons 1/4/8/16 and current feature lookbacks.
5. Whether overlapping labels at longer horizons materially invalidate AUC interpretation.
6. Whether the 5 chronological samples overlap too much to call them independent; recommend precise wording.
7. Multiple-testing / selection bias from 24 configs x 4 horizons.
8. Whether keeping confidence threshold fixed at 0.60 and cost profile fixed at normal during predictive screening is methodologically correct.
9. Whether strategy OOS is using the horizon-trained model correctly without accidentally reading target labels.
10. Whether target class balance should be incorporated into the report/gate.
11. Whether AUC is still the right cheap first-stage metric for this objective, and what secondary diagnostics should be added without changing the hard gate post hoc.
12. Whether horizons 1/4/8/16 are reasonable for the observed strategy holding behavior; if not, propose alternatives and why.
13. Whether an ATR/volatility-aware target, triple-barrier/event target, or conditional target would be a better *next experiment* if all four fixed horizons fail.
14. Any duplicated computation that can be safely cached without changing statistical semantics.
15. Any code path that could touch active models, order execution, credentials, or live trading despite the intended research-only scope.

## Required output

Return one report with:

- `Verdict`: PASS / PASS_WITH_CHANGES / HOLD
- `Critical findings`
- `High findings`
- `Medium findings`
- `Statistical interpretation warnings`
- `Code patch recommendations`
- `Experiment design recommendations`
- `Safe-to-run command`, only if the study is safe to run
- `Do not do` list

For each finding, cite the exact file/function and explain the failure mode. Do not modify production/live execution paths. Do not promote a model. Do not change security/auth/secrets. If you make code changes, keep them on the feature branch and leave PR #1 Draft.
