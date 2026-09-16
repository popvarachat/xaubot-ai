# GOLDmicro Challenger Batch 20260915_233130 — Evidence Note

## Status

Research-only, non-live. No Champion activation or promotion occurred.

## Canonical context

- Repository: `popvarachat/xaubot-ai`
- Branch at local run start: `feat/goldmicro-broker-profile-v1`
- Git SHA reported by runner: `74a68cd5ac77a16cdec535a72bd7198f37b68927`
- Batch ID: `20260915_233130`
- Requested candidates: 24
- Requested shortlist: 6
- MT5 mode: read-only attach to an already logged-in terminal session
- Frozen snapshot: one M15 cut (20,000 bars) and one H1 cut (5,000 bars) shared by the batch
- Promotion: disabled
- Live model: unchanged

## Observed outcome

- 13 / 24 candidates trained successfully.
- 11 / 24 candidates failed before XGBoost training could proceed.
- All 13 successful candidates used the `core` feature profile.
- All 11 failed candidates used the `core_plus_v2` feature profile.
- The failed V2 candidates reported `Insufficient data for training: 1 samples` and then `XGBoost candidate training failed`.
- The pipeline still completed screening of the 13 successful candidates and produced a 6-model shortlist / shadow queue.

That shortlist is **not valid evidence for comparing `core` vs `core_plus_v2`**, because one whole feature family was structurally excluded by the failure. It must not be used for promotion or final model-family selection.

## Root cause found after the run

`src/goldmicro_candidate_trainer.py` previously selected every numeric column for `core_plus_v2` candidates. This unintentionally included sparse raw SMC event columns such as FVG / order-block price levels. `TradingModelV2.fit()` performs `drop_nulls()` across every selected feature, so the intersection of many sparse event columns collapsed the training set to approximately one usable row.

This was a feature-policy / null-handling defect, not evidence that V2 features are intrinsically bad.

## Corrective action

The candidate trainer was changed after this batch so that:

1. `core_plus_v2` uses canonical core features plus the explicit 23 engineered V2 features only.
2. Sparse raw SMC event-price columns are excluded from the XGBoost feature list.
3. Engineered V2 proximity / recency fields receive explicit neutral or far/old sentinel values when structurally missing.
4. A pre-training usable-row guard now fails early with a descriptive error if a feature policy leaves fewer than 500 usable rows.
5. Unit tests assert that sparse raw SMC columns cannot silently re-enter the V2 training feature set.

## Interpretation of successful core candidates

The successful candidates produced test AUC values roughly in the mid-0.55 to low-0.62 range. These values are only a cheap model pre-screen. They are **not** PF/DD evidence and must not be treated as promotion evidence.

The best observed AUC in this incomplete batch was approximately 0.618 from a responsive core candidate using 20,000 training bars. Several other responsive core candidates were around 0.611–0.616. These rankings are provisional because the V2 family was missing.

## Required next action

Re-run the full 24-candidate batch after pulling the V2 feature-policy fix. The rerun must again use a single frozen market snapshot and must produce both `core` and `core_plus_v2` candidates successfully before any family-level comparison is accepted.

After a clean rerun:

`training -> cheap screening -> strategy-level OOS -> GOLDmicro PF/DD/cost validation -> shadow observation -> lifecycle gates -> Human Gate`

Do not promote from AUC or from this incomplete batch.

## Claude Opus audit note

Claude Opus should treat this batch as a useful failure-mode datapoint. In particular, audit:

- whether the revised explicit V2 feature policy is statistically and semantically correct;
- whether sentinel choices can bias trees or should be accompanied by missingness indicators;
- whether `TradingModelV2.fit()` should retain global `drop_nulls()` behavior or use a more explicit imputation contract;
- whether model split metadata is computed from the same cleaned row set actually seen by XGBoost;
- whether HMM-derived regime features are temporally aligned with the V2 feature construction order;
- whether the rerun preserves true model-family diversity without hidden filtering bias.
