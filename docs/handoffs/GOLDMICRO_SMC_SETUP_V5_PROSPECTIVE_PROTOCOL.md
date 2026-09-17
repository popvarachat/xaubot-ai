# GOLDmicro SMC Setup V5 Prospective Protocol

## Status

Frozen research protocol for fresh chronological evidence after the inspected V3/V4/V5 development history. No live activation, no order placement, no promotion.

## Why this protocol exists

V5 mechanics replay produced a balanced and plausible event stream, but the frozen raw economic baseline rejected at 3/5 positive chronological blocks under both normal and conservative cost profiles. The subsequent instability diagnostic showed that the losing periods were broad time-state failures rather than one obvious mechanical subcomponent: in b02 and b03, BUY/SELL, BOS/CHoCH and FVG/OB were all negative or broadly weak.

Therefore the inspected history must not be mined further for post-hoc segment filters. V5 remains a rejected development candidate until fresh chronological evidence is available.

## Freeze rule

The prospective cutoff is the maximum timestamp present in the source M15 snapshot used by the V5 development study. The cutoff must be materialized once in a manifest before any fresh evaluation. Data at or before the cutoff is development history only.

## Frozen generator

Use the current `GoldmicroSMCSetupV5` mechanics without modification:

- `max_setup_age_bars = 12`
- `max_zone_delay_bars = 6`
- `min_retest_delay_bars = 1`
- `fixed_reward_r = 1.5`
- `atr_stop_floor_multiple = 0.50`
- mutually exclusive break direction
- aligned zone provenance
- zone retest required
- BOS continuation and CHoCH reversal recorded separately
- mirrored BUY/SELL mechanics
- structure-derived stop

No SELL-only, session, confidence, regime, archetype, zone-type or reward filter may be added from the already-inspected development diagnostics.

## Frozen prospective block geometry

The development snapshot contained 24,000 raw M15 rows. Before fresh economic outcomes are opened, that already-frozen geometry is reused to define immutable prospective blocks:

- 5 fixed chronological blocks
- exactly 4,800 fresh M15 entry rows per block
- block 1 = fresh rows 1-4,800
- block 2 = fresh rows 4,801-9,600
- block 3 = fresh rows 9,601-14,400
- block 4 = fresh rows 14,401-19,200
- block 5 = fresh rows 19,201-24,000
- setup events are assigned to a block by ENTRY row only
- block boundaries never move or repartition as fresh data grows
- entry rows after fresh row 24,000 are outside the confirmatory entry window
- bars after fresh row 24,000 may be used only to mature tail events whose entries are inside the frozen entry window
- maturity requires the full 32-bar future horizon; incomplete tail events remain pending

This amendment is based only on the already-inspected 24,000-row development geometry and is frozen before fresh economic outcomes are viewed. It does not change the prospective cutoff or generator.

## Frozen economic evaluation

Reuse the V5 raw baseline contract:

- horizon: 32 bars
- same-bar TP/SL ambiguity: adverse SL-first
- 5 fixed non-overlapping chronological blocks defined above
- minimum 30 completed/matured events per block
- require positive mean after-cost R in at least 4/5 blocks under Normal cost
- require positive mean after-cost R in at least 4/5 blocks under Conservative cost
- if a block has fewer than 30 matured events, classify as prospective evidence insufficient rather than economic rejection
- no ML overlay
- no PF/DD stage unless the fresh raw baseline clears this gate
- no threshold tuning after observing prospective results

Normal and Conservative cost semantics remain those already used by `goldmicro_smc_setup_v5_baseline.py`.

## Freshness / causality rule

A future evaluation dataset may include pre-cutoff bars only as causal warmup context. Any scored setup event must have an entry timestamp strictly greater than the frozen cutoff. Outcomes must be resolved only from bars that occur after that entry. The source cutoff itself must never move forward after results are viewed.

## Evidence hierarchy

1. Already-inspected V3/V4/V5 history: development evidence only.
2. Fresh post-cutoff chronological data: prospective research evidence.
3. Only if the frozen raw gate passes: design a separate PF/DD/capital-risk evaluation while preserving the same setup mechanics.
4. Any later promotion still requires forward shadow evidence, perturbation review, independent audit and explicit Human Gate.

## Safety

- research artifacts only
- no live orders
- no active-model overwrite
- no credentials/auth changes
- no production write path
- no auto-promotion
