# GOLDmicro Batch 20260915_233805 — Causality Audit Handoff

Status: **INVALID FOR MODEL RANKING / PF-DD VALIDATION UNTIL RERUN**

Scope: research-only, non-live. No active model, production path, credential, or trading execution behavior is changed by this audit.

## 1. Why this audit was opened

Batch `20260915_233805` completed training for all 24 challengers and produced a 6-model AUC shortlist. The prior sparse-feature failure was fixed, so both `core` and `core_plus_v2` candidates trained successfully.

However, the `core_plus_v2` test AUC values increased materially relative to the core family (several V2 candidates reached roughly 0.75–0.86 while core candidates were mostly around 0.55–0.62). That gap was large enough to require an explicit leakage/causality review before any expensive strategy OOS, PF/DD, execution-cost, walk-forward, or shadow validation.

**Decision:** do not use the `20260915_233805` shortlist as promotion or PF/DD input. Treat it as diagnostic evidence only.

## 2. Confirmed causality defects

### A. H1 data could be visible before the H1 bar closed

The generic V2 feature engineer joined H1 rows to M15 using a backward as-of join on the raw bar timestamp. MT5 historical rate timestamps represent the bar timestamp/open boundary. Therefore an H1 bar stamped `10:00` contains final OHLC/features accumulated through that hour but could be joined to M15 rows at `10:15`, `10:30`, or `10:45`.

That exposes information that was not available at the M15 decision time.

### B. `consecutive_direction` used final group length

The generic V2 implementation used a full-group `count().over(group)` for candle-direction runs. Earlier rows in a bullish/bearish run therefore received the eventual final length of that run, which is future information.

### C. `regime_duration_bars` used final group length

The generic V2 implementation used the same full-group-count pattern for regime duration. If/when the regime column is present at feature-calculation time, earlier rows receive the eventual final duration of the regime run.

Even though the current challenger trainer computes V2 before candidate HMM prediction, this implementation is not prefix-causal and is unsafe for reuse.

## 3. Fix applied on feature branch

Branch: `feat/goldmicro-broker-profile-v1`

Causality changes:

- `053f30cc0075b9109726afc0bf8578cab9334d3b` — add `src/goldmicro_causal_v2.py`
  - delays H1 availability by one hour before the as-of join;
  - replaces regime-duration full-group count with prefix-causal cumulative run length;
  - replaces candle-direction full-group count with prefix-causal cumulative run length;
  - research-only wrapper; generic upstream V2 module remains untouched.
- `2e38b73ec6749e39e7d4ecffc66dc06b6f1c270d` — route GOLDmicro challenger training through the causal V2 wrapper.
- `a64ab360305c88dba97c58af2e98e5ac97f64ee2` — add causality regression tests.

Regression tests cover:

1. prefix invariance of `consecutive_direction`;
2. prefix invariance of `regime_duration_bars`;
3. H1 row stamped `10:00` is not visible to M15 rows before `11:00`.

GitHub Actions `GOLDmicro Unit Tests` run `129` completed successfully at head `a64ab360305c88dba97c58af2e98e5ac97f64ee2`.

## 4. What remains valid from batch 20260915_233805

Valid as operational evidence:

- Python 3.11 research environment worked;
- MT5 research connector attached read-only to the logged-in terminal;
- one frozen M15 snapshot and one frozen H1 snapshot were reused across the batch;
- 24/24 candidates trained after the sparse-feature bug was fixed;
- candidate isolation, report writing, screening, and no-auto-promotion controls operated.

Not valid as model-quality evidence:

- V2 AUC ranking;
- the 6-model shortlist;
- any conclusion that V2 is superior to core;
- any downstream PF/DD or lifecycle decision derived from that shortlist.

## 5. Required rerun

Rerun the same 24-candidate research batch from the latest feature-branch head. The rerun becomes the first candidate batch eligible for AUC screening **only if** causality checks remain green and no new suspicious performance discontinuity appears.

After rerun, compare:

- core vs core_plus_v2 test AUC distributions;
- train/test generalization gaps;
- candidate sample counts and time ranges;
- whether V2 performance remains unusually high after H1-close and prefix-causality fixes.

If V2 AUC remains abnormally high, stop again and inspect target alignment, feature timestamp semantics, HMM ordering, feature warm-up, and any other possible future-data path before PF/DD.

## 6. Independent Claude Opus audit request

Please independently review, without assuming the PM findings are correct:

1. Is the one-hour H1 availability shift correct for MT5 H1 rate timestamps used by this repo?
2. Are all GOLDmicro V2 features prefix-causal after the wrapper?
3. Does `FeatureEngineer.create_target(..., lookahead=1)` remain correctly separated from predictor features and train/test boundaries?
4. Is the 50-bar OOS gap sufficient for the current one-bar target and feature windows, or should it be changed?
5. Is HMM fit/predict ordering leakage-safe for the challenger workflow?
6. Are SMC swing/FVG/BOS/CHoCH implementations causal at the point where signals/features become available?
7. Are any feature calculations performed on a full frame in a way that changes historical feature values when future rows are appended?
8. Should the next valid batch be permitted to proceed to strategy OOS + GOLDmicro PF/DD/cost, or should another causality patch be required first?

Expected audit result: PASS / HOLD with file-and-line evidence and any required patch suggestions. Do **not** promote or modify live/active models.

## 7. Governance state

- PR remains Draft/non-live.
- Promotion is disabled.
- Active/live model remains unchanged.
- Human Gate is still required before any future activation.
- No merge is authorized by this handoff.
