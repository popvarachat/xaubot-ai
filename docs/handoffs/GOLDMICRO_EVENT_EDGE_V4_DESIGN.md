# GOLDmicro Event Economic-Edge V4 Design

## Status

Research-only design. No live model activation, no order placement, no credential change, no promotion.

## Evidence that motivates V4

The calibrated Event Target V3 predictive screen produced stable survivors, but the downstream untouched-OOS strategy gate rejected all 6 shortlisted configurations under both normal and conservative GOLDmicro cost profiles.

Failure-diagnostic summary supplied from the canonical V3 batch `event_target_20260916_124321`:

- 60 sample-cost results, 0 strategy passes
- MIN_TRADES: 57/60
- PROFIT_FACTOR: 55/60
- EXPECTANCY: 46/60
- DRAWDOWN: 2/60
- RISK_SKIP: 0/60
- NO_ACCEPTED_EVENTS: 7/60

Economic-tail calibration also showed systematic overstatement in the selected tail. Across all 12 configuration/cost aggregates, observed TP-before-SL rates were below the setup-level cost break-even probabilities. The gap `observed - break-even` ranged roughly from -0.035 to -0.172, while average calibrated predictions in the selected tail remained materially above observed outcomes.

This means the next research question is not "which probability threshold should be lowered?" The target itself must align more directly with realized economic payoff.

## Core problem in V3

V3 predicts a binary event:

`P(SMC setup reaches TP before SL within 32 raw M15 bars)`

But strategy PnL is not binary. A negative target includes both:

1. stop-loss first, and
2. timeout/no-TP-first.

At strategy evaluation time a timeout exits at the actual close after the holding horizon, not necessarily at the stop price. Therefore a binary TP-before-SL probability plus a two-outcome TP/SL break-even formula is only an approximation of realized expected value.

V4 removes this semantic mismatch by predicting realized payoff in risk units instead of predicting a binary TP-first event.

## V4 target semantics

Prediction unit remains one causal SMC setup event.

For each event define the decision-time risk denominator from the setup's own entry and stop:

`risk_distance = abs(entry_mid - stop_mid)`

The future path is resolved using exactly the existing research execution contract:

- scan from the next raw M15 bar
- maximum holding period = 32 raw M15 bars
- TP hit first -> exit at setup TP
- SL hit first -> exit at setup SL
- same-bar TP/SL ambiguity -> adverse, SL first
- neither hit by horizon -> exit at horizon close

Define gross realized R before broker cost:

`gross_R = signed(exit_mid - entry_mid) / risk_distance`

with sign adjusted for BUY/SELL direction.

Because a timeout has not touched either boundary, its gross_R naturally lies between the stop and TP boundaries. This preserves its actual economic magnitude instead of forcing every non-TP event into the same binary loss class.

## Cost treatment

Do not train separate models for normal and conservative costs.

The model predicts `gross_R`, which is market-path dependent and cost-agnostic. At evaluation time deterministic broker-cost drag is converted to R units for each setup and each cost profile:

`predicted_net_R = predicted_gross_R - setup_cost_R`

where `setup_cost_R` is calculated from the existing broker-correct GOLDmicro cost model using the setup entry/exit economics and risk distance.

This preserves one model semantics across cost profiles while keeping execution assumptions explicit.

## Model objective

Primary V4 learner:

- XGBoost regression
- objective: `reg:squarederror`
- target: realized `gross_R`
- no OOS-driven early stopping
- fixed predeclared boosting rounds by profile, matching the governance approach used in V3

The prediction semantics become:

`E[gross_R | causal SMC setup features]`

not BUY/SELL and not probability.

SMC remains the sole source of direction, entry, SL and TP.

## Chronological separation

Retain the V3 chronology discipline:

`FIT -> raw-bar embargo -> CALIBRATION -> raw-bar embargo -> UNTOUCHED OOS`

The 32-bar label horizon remains a 32-raw-bar embargo around each boundary.

HMM is fit only on the fit partition and inferred causally forward.

The untouched OOS partition must never be used for training, model selection, target transformation, calibration, or threshold selection.

## Regression calibration

The pre-OOS calibration partition may be used only for a frozen affine bias calibration of regression output:

`calibrated_R = a * raw_pred_R + b`

Fit `(a, b)` on the calibration partition only. If this calibration is unstable or degenerate, fail closed rather than falling back to OOS tuning.

Diagnostics must report at minimum:

- OOS MAE
- OOS RMSE
- rank correlation between predicted R and realized R
- mean predicted R vs mean realized R
- calibration slope/intercept
- selected-tail observed vs predicted R

## Frozen economic gate

The strategy gate is predeclared and return-independent:

`calibrated_predicted_net_R > 0`

No probability threshold exists in V4.

No post-hoc margin such as +0.05R may be introduced after seeing PF/DD. If a safety margin is later studied, it must be a separately declared experiment with fresh untouched OOS evidence.

## Strategy OOS gates

Keep the existing gates unchanged:

- initial capital = 20,000 THB
- risk per trade = 1.00%
- PF >= 1.30
- max DD <= 10.00%
- risk-skip <= 20%
- minimum trades >= 30 per chronological sample
- multi-sample stability >= 4/5
- both normal and conservative cost profiles must pass

Do not lower these gates because V3 failed.

## Predictive pre-screen for regression

AUC is no longer applicable to the primary target.

Before PF/DD evaluation, require a regression-quality pre-screen that is frozen before the first V4 run. Recommended initial screen:

- OOS rank correlation > 0 on at least 4/5 probes
- OOS predicted-vs-realized calibration slope > 0 on at least 4/5 probes
- no catastrophic mean-bias sign reversal in the selected positive-net-R tail
- minimum OOS event count remains >= 100

Exact numeric magnitude gates beyond sign/stability should not be invented from V3 realized returns. They must be declared before the first V4 batch is run.

## Research outputs

V4 should produce isolated artifacts only, for example:

- `event_edge_xgb.json`
- `event_edge_calibration.json`
- `event_edge_data.parquet`
- `event_edge_report.json`
- `event_edge_strategy_oos_queue.json`

No artifact may overwrite or alias an active/live model path.

## Safety and promotion

V4 is research-only.

Even if a configuration later clears predictive and PF/DD/cost gates, promotion remains disabled until all of the following are satisfied:

1. forward shadow evidence,
2. perturbation/sensitivity review,
3. independent code/research audit,
4. explicit Human Gate approval.

Live model remains unchanged throughout this phase.
