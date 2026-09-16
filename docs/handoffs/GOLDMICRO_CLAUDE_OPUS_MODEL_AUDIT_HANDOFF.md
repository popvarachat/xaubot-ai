# GOLDmicro / Claude Opus Deep Model Audit Handoff

## Purpose

This is an independent second-opinion audit task for Claude Opus. ChatGPT remains PM / Control Plane; GitHub remains canonical evidence; Pop remains Human Gate.

The objective is **not** to maximize backtest profit. The objective is to determine whether the current GOLDmicro multi-challenger research and model-lifecycle design can produce trustworthy PF/DD evidence without leakage, hidden model-selection bias, or unsafe promotion behavior.

## Repository and canonical state

- Repository: `popvarachat/xaubot-ai`
- Working branch: `feat/goldmicro-broker-profile-v1`
- Pull Request: `#1`
- Scope: non-live research only
- Root `AGENTS.md`: not present when this handoff was created. Re-check before beginning in case it has since appeared.

**START HERE:** re-read current GitHub state. Do not trust the SHA written in this handoff if the branch has moved. Record the exact branch head SHA, PR state, workflow state, changed files, and any applicable repository instructions before auditing.

## Roles

- **Pop:** Human Gate. Only Pop may approve future live model activation / production trading changes.
- **ChatGPT:** PM / Control Plane. Owns scope, acceptance criteria, reconciliation of findings, and decision sequencing.
- **Claude Opus:** independent Senior Quant/ML/Code Reviewer. Deep-audit assumptions, evidence quality, leakage risk, statistical validity, implementation risk, and lifecycle design.
- **GitHub:** canonical source for code, branch/commit/PR/CI evidence.

Claude must not treat prior chat summaries as canonical when GitHub evidence is available.

## Hard scope and prohibitions

This audit is **research-only and non-live**.

Do not:

- place or simulate a real broker order through an order-send path;
- enable live trading;
- overwrite `models/xgboost_model.pkl` or `models/hmm_regime.pkl`;
- mutate `models/active/*`;
- auto-promote a Challenger;
- bypass Human Gate;
- change credentials, secrets, OAuth, MT5 account configuration, or broker settings;
- direct-write or force-push `main`;
- weaken CI, tests, PF/DD gates, cost gates, or review requirements merely to make the branch pass;
- claim profitability, live readiness, or statistical significance without supporting evidence.

Code patches are allowed only on the existing research branch or a new feature/fix branch, and only if they remain non-live/reversible. Prefer an audit report first; do not make broad refactors before identifying concrete findings.

## Fixed GOLDmicro research constraints

Treat the following as intentional project requirements unless a finding proves they are internally inconsistent:

1. Symbol: `GOLDmicro`.
2. Strategy lot ladder: `0.10, 0.20, 0.30, ...` only.
3. Broker metadata may expose a finer `0.01` volume step; the **research strategy step remains 0.10**.
4. Raw risk sizing must be floored, never rounded upward.
5. If `0.10` lot exceeds the risk budget, the trade must be skipped rather than forced.
6. Research risk cap baseline: 1% unless a validation matrix explicitly tests alternatives.
7. PF hard gate currently: 1.30.
8. DD hard gate currently: 10%.
9. Execution-cost target/hard envelope currently around 85/90 points as a provisional research policy; this is empirical, not universal.
10. Training may be automatic. Promotion may not be automatic.
11. Champion / Challenger / Shadow are distinct lifecycle states.
12. Future activation requires Human Gate, flat-position boundary, next newly closed execution candle, and rollback reference.

## Current research architecture to audit

At minimum inspect these areas and all directly imported dependencies:

### Model / feature pipeline

- `src/goldmicro_candidate_trainer.py`
- `src/challenger_batch.py`
- `src/candidate_screening.py`
- `backtests/ml_v2/ml_v2_model.py`
- `backtests/ml_v2/ml_v2_feature_eng.py`
- `src/feature_eng.py`
- `src/smc_polars.py`
- `src/regime_detector.py`
- `src/ml_model.py`

### Lifecycle / governance

- `src/model_lifecycle.py`
- `src/model_registry.py`
- `src/shadow_evaluator.py`
- `config/goldmicro_model_lifecycle.json`
- `docs/GOLDMICRO_MODEL_LIFECYCLE.md`

### Batch runners

- `scripts/train_goldmicro_challenger_batch.py`
- `scripts/score_goldmicro_challenger_batch.py`
- `scripts/run_goldmicro_challenger_research.py`
- `scripts/run_goldmicro_lifecycle_batch.py`
- `scripts/run_goldmicro_model_review_batch.py`
- `scripts/goldmicro_research_preflight.py`
- `scripts/setup_goldmicro_research_env.ps1`
- `requirements-goldmicro-research.txt`

### GOLDmicro economics / replay

- `src/broker_profile.py`
- `src/goldmicro_risk.py`
- `backtests/goldmicro_cost_model.py`
- `backtests/goldmicro_replay.py`
- `scripts/run_goldmicro_model_matrix.py`
- `scripts/run_goldmicro_cost_surface.py`
- `scripts/run_goldmicro_full_batch.py`

### Legacy strategy path used as reference

- `backtests/backtest_24_final_combined.py`

Audit whether results derived from legacy trade paths are being labelled correctly as post-trade/replay research rather than true predictive walk-forward evidence.

## Deep audit questions

### A. Temporal and target leakage

Determine whether any feature, target, HMM fit, scaler, H1 join, SMC structure, target generation, candidate screening metric, or train/OOS boundary can see information that would not have existed at decision time.

Specifically inspect:

- whether target creation happens before or after splitting and whether it leaks across the split boundary;
- whether rolling indicators use future-centered windows;
- whether H1 `join_asof` uses only already-closed H1 information or can attach the currently forming H1 bar to M15 rows;
- whether HMM/scaler fit occurs only on training data;
- whether regime labels or smoothed states use future observations;
- whether feature engineering is fitted globally when it should be train-only;
- whether the 50-bar gap is sufficient for the target horizon and feature lookbacks;
- whether the test/OOS data influences early stopping or hyperparameter/model selection;
- whether repeated challenger selection turns the nominal OOS set into a de-facto validation set.

### B. Candidate diversity and search bias

Audit whether a 24-candidate batch is genuinely diverse or still structurally correlated.

Check:

- sampling across train windows, random seeds, XGBoost profiles, HMM lookbacks, confidence thresholds, feature profiles, and cost profiles;
- whether `cost_profile` actually changes training/evaluation behavior or is merely metadata;
- whether confidence threshold changes are reflected in strategy-level trade decisions;
- whether HMM lookback changes have real effect;
- whether random seed changes all relevant sources of randomness;
- whether repeated screening creates multiple-testing / winner's-curse risk;
- whether the shortlist should be selected with a diversity/Pareto rule rather than AUC alone.

### C. Metric correctness

Audit every metric used to retain/reject/promote models:

- AUC and train/test gap;
- PF;
- maximum DD;
- expectancy;
- trade count;
- skip rate;
- execution-cost sensitivity;
- Sharpe-like metrics;
- stable-window count;
- shadow trade/day counts.

Check denominators, equity ordering, skipped-trade handling, no-loss/infinite-PF handling, sample-size minimums, and whether each metric is in the correct account currency.

### D. GOLDmicro position sizing and costs

Verify that research sizing obeys the 0.10 strategy ladder and never increases risk through rounding.

Audit whether:

- account-currency risk is used consistently;
- historical FX conversion can distort PF/DD under dynamic sizing;
- spread is applied to correct BUY/SELL sides;
- slippage semantics are consistent (per side versus round trip);
- commission/swap assumptions are explicit;
- cost scenarios are not double-counted;
- post-trade repricing is clearly distinguished from tick-accurate execution replay;
- spread 100 / high-cost HOLD conclusions are supported only within their tested assumptions.

### E. Walk-forward design

Design the **next validation layer** carefully. Do not call fixed-log filtering "true ML walk-forward".

Recommend a concrete fold structure including:

- train window length(s);
- purge/embargo gap based on target horizon and lookback;
- validation window;
- final untouched OOS segment;
- minimum trades per fold;
- how to aggregate PF/DD across folds;
- how to rank candidates without overfitting to PF;
- whether nested selection is needed;
- how to treat regime imbalance;
- whether Monte Carlo/permutation/parameter perturbation is useful.

### F. Champion / Challenger / Shadow lifecycle

Audit whether the current lifecycle can accidentally:

- overwrite active artifacts;
- promote an unvalidated candidate;
- use stale runtime policy;
- switch while positions are open;
- lose rollback traceability;
- compare Champion and Challenger on non-equivalent market observations;
- reuse shadow results after material model/config changes.

Recommend exact promotion evidence requirements. Promotion must remain Human-Gated.

### G. Operational architecture

Review the proposed split:

- GitHub = canonical code/config/model-report registry;
- local PC/VPS = heavy ML compute;
- n8n = scheduler/orchestrator/notification layer;
- Cloudflare = lightweight runtime policy/state, not training compute.

Identify what should and should not be placed in n8n/Cloudflare. Consider idempotency, stale-state detection, job retries, duplicate training, audit trail, and failure modes. Do not introduce infrastructure merely because it is available.

## Required independent challenge

Do not optimize for agreement with the existing design. Explicitly try to falsify these assumptions:

1. `M15` is the correct execution/model timeframe.
2. AUC is useful enough to be a first-stage screen.
3. XGBoost binary + HMM is the correct model family for the next stage.
4. 24 challengers / top 6 is a reasonable compute allocation.
5. PF >= 1.30 and DD < 10% are sufficient lifecycle thresholds.
6. 30 trades is enough for rolling PF monitoring.
7. 100 validation trades and 50 shadow trades are enough for promotion evidence.
8. Weekly retraining is an appropriate default cadence.
9. The 85/90-point execution-cost thresholds should be fixed rather than regime-dependent.
10. The legacy SMC-entry / smart-exit path is a suitable base for model comparison.

For each assumption, return `SUPPORTED`, `WEAK`, or `REJECT`, and explain the evidence.

## Required output

Create one audit report, preferably:

`docs/reviews/GOLDMICRO_CLAUDE_OPUS_MODEL_AUDIT_<YYYY-MM-DD>.md`

The report must contain:

1. **Executive verdict** — maximum one page.
2. **Canonical evidence** — branch, exact SHA, PR, CI status reviewed.
3. **Finding table** with severity `CRITICAL / HIGH / MEDIUM / LOW`.
4. **Leakage map** — every suspected train/test/time leakage path.
5. **Metric audit** — PF/DD/AUC/expectancy/skip/cost/shadow.
6. **Candidate search audit** — diversity and multiple-testing risk.
7. **Walk-forward specification** detailed enough for another engineer to implement.
8. **Model lifecycle verdict** — what is safe now, what remains blocker.
9. **Architecture verdict** for GitHub/n8n/Cloudflare/local compute.
10. **Assumption challenge table** with `SUPPORTED / WEAK / REJECT`.
11. **Patch plan** ordered by dependency, with exact files/functions.
12. **Do-not-change list** identifying validated components that should remain stable.
13. **Go/No-Go** for running the 24-model batch and separately for trusting its results.

For each finding include:

- evidence path/function/line or code behavior;
- why it matters to PF/DD trustworthiness;
- whether it invalidates existing historical results or only future claims;
- recommended fix;
- verification test.

## Acceptance criteria

The task is complete only when:

- current GitHub state was revalidated before analysis;
- all relevant source was actually inspected, not inferred from filenames;
- leakage risks were traced end-to-end;
- the distinction between model AUC and strategy PF/DD was preserved;
- 0.10-lot GOLDmicro sizing constraints were preserved;
- no active/live model was changed;
- no automatic promotion path was introduced;
- every CRITICAL/HIGH finding has a concrete fix/test plan;
- the report explicitly states whether the current 24-model batch may be run and whether its output may be used for promotion decisions.

## Return protocol

Do not merge. Do not promote. Commit only the audit report and tightly scoped non-live tests/patches if needed. Return:

- exact branch and head SHA;
- audit report path;
- changed files;
- CI/test results;
- CRITICAL/HIGH findings summary;
- blockers requiring ChatGPT PM or Pop Human Gate.

ChatGPT will reconcile the audit against current repo evidence before any next phase is approved.
