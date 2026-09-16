# GOLDmicro Model Lifecycle V1

## Purpose

This layer keeps a profitable research model from being destabilized by over-frequent retraining. It separates **fast trading decisions** from **slow model-learning decisions** and treats PF, drawdown, execution cost, sample size, stability and shadow evidence as promotion gates.

The design is intentionally non-live in this PR. Nothing here should replace an active model automatically.

## Current repo finding

The existing `main_live.py` already separates fast runtime work from training conceptually: the main loop targets sub-second execution, checks positions frequently, loads XGBoost/HMM models, and instantiates `AutoTrainer`. The current `AutoTrainer`, however, can retrain on a daily/time/AUC trigger and writes directly to `models/xgboost_model.pkl` and `models/hmm_regime.pkl`. That behavior is acceptable for experimentation but is **not acceptable for the GOLDmicro production lifecycle** because a newly trained candidate can replace the active model before PF/DD/cost/shadow gates are proven.

Therefore GOLDmicro V1 introduces this rule:

> Training may be automatic; promotion may not be automatic.

A training job must write to a candidate/versioned path, validate, shadow, and request Human Gate before activation.

## Two clocks: trading clock vs learning clock

### Trading clock — fast, deterministic, no retraining

1. **Open-position management:** approximately every 5 seconds. This is for stop/trailing/exit management only.
2. **Execution-cost gate:** every candidate order using current spread/slippage/economic state.
3. **Signal inference:** once per newly closed execution candle. Do not repeatedly infer on every tick when the underlying feature bar has not changed.
4. **H1 context:** refresh when a new H1 candle closes rather than recomputing heavy context on every execution loop.
5. **No model training inside the signal/position loop.** Heavy learning work is isolated from order latency.

This gives the strategy enough reaction speed while avoiding model churn and duplicated decisions.

### Learning clock — delayed and evidence-based

The learning clock has six levels:

| Level | Trigger | Action | Can change active model? |
|---|---|---|---|
| L0 Trade | each closed trade | append metrics/evidence | No |
| L1 Micro | every ~20 closed trades | rolling PF/DD/expectancy/cost/skip review | No |
| L2 Session | London/NY session close | session drift report | No |
| L3 Daily | market/session close | health review and candidate-training decision | No |
| L4 Weekly | weekend/maintenance window | train many challengers + walk-forward/OOS/cost validation | No |
| L5 Event | PF/DD/cost/regime abnormality | accelerated challenger review; HOLD if hard gate | No; only Human Gate can promote |

The delay is deliberate. PF on 5-10 trades is too noisy to justify retraining. A small sample can warn, but it cannot decide promotion.

## Champion / Challenger / Shadow model

### Champion
The only model allowed to influence the trading strategy. Its exact model hash, config hash, training-data window and validation report should be registered as canonical evidence.

### Challengers
Every scheduled training cycle should train **multiple candidates in one batch**, not one model at a time. A challenger may vary training window, feature subset, XGBoost seed/hyperparameters, HMM window/regime assumptions, confidence threshold or strategy filter. All challengers must be scored on the same evaluation windows and execution-cost assumptions.

### Shadow
The strongest challengers run beside Champion but do not place orders. They receive the same market observations and proposed trades, and their hypothetical outcomes are logged. This is necessary because a candidate can look strong in historical OOS and still fail under the current spread/regime/execution environment.

Minimum policy before requesting promotion:

- at least 100 validation/OOS trades,
- at least 3 stable validation windows,
- at least 50 shadow trades,
- at least 3 shadow trading days,
- PF >= 1.30,
- DD < 10%,
- skip <= 20%,
- execution cost <= 90 points,
- positive expectancy,
- PF not materially worse than Champion,
- DD not materially worse than Champion,
- Human Gate approval.

These are minimum evidence gates, not profitability guarantees.

## PF/DD health state machine

### HEALTHY
- rolling sample >= 30 trades,
- PF >= 1.40,
- DD < 8%,
- execution cost <= 85 points,
- skip <= 20%.

Action: Champion stays active. No retrain required solely from these metrics.

### WATCH
Examples: PF 1.30-1.40, cost 85-90 points, or elevated skip rate.

Action: keep Champion, increase review frequency, prepare/refresh challengers. Do not switch models.

### PROTECT
Examples: PF < 1.30 with sufficient sample or DD >= 8% but < 10%.

Action: Champion may continue under risk/entry restrictions defined by the trading-risk layer, while challenger training and review become urgent. This lifecycle layer itself does not mutate live risk.

### HOLD
- DD >= 10%, or
- execution cost > 90 points.

Action: no new entries through lifecycle policy until conditions recover or a Human-Gated recovery decision is made. Open-position safety management continues separately.

### INSUFFICIENT_SAMPLE
Fewer than 30 closed trades. Action: collect evidence. Hard DD/cost gates can still HOLD, but PF alone should not cause model replacement.

## Promotion hysteresis

Model selection must have memory. A challenger should not replace Champion after one lucky test window.

The proposed hysteresis is:

- 3 stable windows minimum,
- candidate PF must be at least the hard PF threshold and no more than 0.03 below Champion PF,
- candidate DD may not exceed Champion DD by more than 0.50 percentage points and still must stay below 10%,
- shadow requirements must pass,
- activation occurs only when no position is open,
- after Human Gate, activate at the next newly closed execution candle boundary.

This reduces flip-flopping between models.

## Where GitHub, n8n and Cloudflare fit

### GitHub — Canonical evidence
Use GitHub for source code, lifecycle policy, model manifests, validation reports, candidate metadata, hashes, PR review and Human Gate evidence. GitHub is not the low-latency runtime state store and should not be queried synchronously before each trade.

Recommended model manifest fields:

- model_id and semantic version,
- git SHA,
- training-data start/end,
- features/config hash,
- random seed,
- training metrics,
- walk-forward/OOS metrics,
- cost-surface metrics,
- shadow metrics,
- model file hash,
- Champion/Challenger/Rejected status,
- approval reference,
- rollback model id.

### n8n — Orchestration
Use n8n to schedule daily/weekly review jobs, request local/VPS computation, collect result JSON, notify on WATCH/PROTECT/HOLD, and optionally create/update GitHub issues/PR evidence. n8n is not canonical model state and must not decide production promotion by itself.

Suggested workflows:

1. `GOLD-REVIEW-DAILY`: fetch metrics -> evaluate lifecycle -> notify only on state change.
2. `GOLD-CHALLENGER-WEEKLY`: start batch candidate training -> validation -> ranking -> GitHub report.
3. `GOLD-DRIFT-EVENT`: react to PF/DD/cost/regime breach -> launch accelerated review.
4. `GOLD-SHADOW-REPORT`: aggregate shadow trades -> evaluate promotion readiness.

### Cloudflare — Runtime policy/state
Use Cloudflare Worker + KV/D1 only for lightweight state such as current Champion model id, lifecycle state, latest metrics timestamp, HOLD/ALLOW, current policy version and review-job status. It can expose a small read endpoint to the bot or dashboard.

Do not use Worker CPU for XGBoost/HMM training, walk-forward or Monte Carlo.

A proposed state object:

```json
{
  "symbol": "GOLDmicro",
  "champion_model_id": "gold-v1-...",
  "state": "HEALTHY",
  "allow_new_entries": true,
  "policy_version": "1.0",
  "last_review_at": "...",
  "pf": 1.52,
  "dd_pct": 6.4,
  "execution_cost_points": 78,
  "review_seq": 123
}
```

The Worker should reject stale state and fail conservatively if runtime policy cannot be verified.

## Compute placement

Heavy training belongs on the Windows trading machine, a dedicated VPS, or another controlled compute node because it requires Python/MT5/data/scikit/xgboost/hmmlearn and may take minutes. A recommended approach is to keep the trading process and training process separate even when they run on the same PC.

Training should run when market/order latency is least sensitive. Daily health evaluation is cheap and can happen after sessions. Full challenger training belongs in a market-closed maintenance window unless an event requires an accelerated candidate build.

## Candidate batch design

A future weekly batch should produce many candidates in one run. For example:

- multiple rolling training windows,
- 2-4 XGBoost regularization/complexity profiles,
- several random seeds,
- HMM window variants,
- confidence/filter variants,
- the fixed GOLDmicro 0.10-lot ladder,
- operational and conservative cost regimes.

Candidates are ranked with a Pareto-style view rather than maximum PF alone. Primary objectives are PF, DD, expectancy, stability, sample count, skip rate and cost robustness.

A candidate with PF 1.70 / DD 9.8% may be inferior operationally to PF 1.50 / DD 5.0% if the latter is much more stable across windows.

## Required migration before any live GOLDmicro use

The current AutoTrainer writes newly trained models directly to the active paths. Before a live GOLDmicro rollout this must be migrated to:

`active Champion -> train candidate path -> validate -> shadow -> Human Gate -> atomic activation -> rollback pointer`

No automatic `retrain()` path should overwrite active Champion files.

This is a **live-readiness blocker**, not a cosmetic enhancement.

## Files introduced by this layer

- `config/goldmicro_model_lifecycle.json` — auditable cadence/gate policy.
- `src/model_lifecycle.py` — pure Champion/Challenger policy evaluator.
- `scripts/run_goldmicro_model_review_batch.py` — evaluates many challengers in one command.
- `tests/test_model_lifecycle.py` — guardrail tests.

The implementation remains read-only and does not touch `main_live.py` or `AutoTrainer` behavior in this PR.
