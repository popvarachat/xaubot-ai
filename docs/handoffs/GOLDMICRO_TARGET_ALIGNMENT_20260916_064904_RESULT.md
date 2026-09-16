# GOLDmicro Target Alignment Result — 2026-09-16 06:49:04

## Scope

Research-only result from `RUN-GOLDMICRO-TARGET-STUDY.bat` on branch `feat/goldmicro-broker-profile-v1`.

Root `AGENTS.md` was checked before this PM review and was not present on the branch.

No live model activation, order execution, credential change, or production write is part of this result.

## Completed matrix

- 24 predictive configurations
- 5 chronological temporal probes per configuration
- fixed future-close target horizons: +1 / +4 / +8 / +16 M15 bars
- 24 x 5 x 4 = 480 training/evaluation jobs
- existing hard AUC gate retained at 0.55
- stable requirement retained at >= 4/5 temporal probes

## Observed summary

| Horizon | Stable pass | Median AUC | Best config median AUC | Best single-probe AUC | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| t+1 | 0/24 | 0.5190 | 0.5248 | 0.5395 | REJECT |
| t+4 | 0/24 | 0.5077 | 0.5140 | 0.5340 | REJECT |
| t+8 | 0/24 | 0.5106 | 0.5332 | 0.5521 | REJECT |
| t+16 | 0/24 | 0.5026 | 0.5231 | 0.5390 | REJECT |

Shortlist: **0**.

Strategy OOS therefore executed zero jobs and closed normally with `NO_ELIGIBLE_CONFIGURATIONS`. PF/DD/cost validation was intentionally skipped. Live model remained unchanged and promotion remained disabled.

## PM interpretation

The result rejects the hypothesis that the main problem is merely choosing the wrong fixed future-close horizon between 1 and 16 M15 bars. One isolated t+8 probe exceeded AUC 0.55, but no t+8 configuration was stable across >=4/5 temporal probes; it must not be cherry-picked.

Do **not** lower the AUC gate after observing this result. Do **not** expand the hyperparameter search around these fixed-close targets simply to manufacture survivors.

The next research question should change the ML objective, not the threshold. The current bot architecture creates entries from SMC setups while XGBoost is intended as confirmation/reversal assistance. Training XGBoost on unconditional every-bar future-close direction is therefore likely misaligned with its intended decision role.

## Next experiment — setup-conditioned event outcome

Preferred next target is a causal SMC-setup outcome classifier:

> At the exact decision bar where a causal SMC signal exists, does that setup reach its own SMC take-profit before its own SMC stop-loss within the predeclared maximum holding window?

Research target definition proposed for audit:

- sample unit: **causal SMC setup event**, not every M15 bar
- decision features: only values available at the setup decision bar
- label `1`: SMC take-profit is reached before stop-loss within 32 bars
- label `0`: stop-loss is reached first, both barriers occur in the same OHLC bar (adverse ordering), or TP is not reached first by 32 bars
- no cost or XGBoost-dependent exit is allowed in the target label, avoiding circular labels
- target horizon: 32 M15 bars, matching the research strategy proxy hard maximum holding window
- train/test separation must be defined in **raw bar time**, with label horizon + embargo respected before OOS; do not rely on a gap measured only in filtered event rows
- HMM remains fit on training history only
- candidate artifacts remain isolated under `models/candidates`
- no live path changes

This proposed target changes the ML question from "which direction will the next close move?" to "is this SMC setup likely to succeed?", which is closer to the intended confirmation/blocking role.

## Gates to preserve

- no post-result auto-relaxation
- AUC >= 0.55 remains the hard first-stage discrimination gate unless an independent audit justifies a metric change **before** seeing the new experiment results
- generalization gap <= 0.12
- stability requirement >= 4/5 temporal probes
- report event count and positive-class rate per probe
- add calibration/Brier and PR-AUC as diagnostics, not as post-hoc replacement gates
- strategy PF/DD/cost remains a later gate only after predictive stability passes
- Shadow + independent audit + Human Gate remain mandatory before any future promotion

## Safety state

- Live model: UNCHANGED
- Promotion: DISABLED
- PR #1: keep Draft
- Human Gate: Pop
