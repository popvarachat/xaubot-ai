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

## Frozen economic evaluation

Reuse the V5 raw baseline contract:

- horizon: 32 bars
- same-bar TP/SL ambiguity: adverse SL-first
- 5 non-overlapping chronological blocks over the fresh evidence window
- minimum 30 completed events per block
- require positive mean after-cost R in at least 4/5 blocks under Normal cost
- require positive mean after-cost R in at least 4/5 blocks under Conservative cost
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
