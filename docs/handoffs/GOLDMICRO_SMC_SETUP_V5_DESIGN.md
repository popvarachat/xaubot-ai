# GOLDmicro SMC Setup V5 Design

## Status

Research-only redesign proposal. No live activation, no order placement, no credential/auth changes, no promotion.

## Why V5 exists

Event Target V3 showed some predictive discrimination but no robust PF/DD/cost survivors. Event Edge V4 then predicted realized gross R directly, yet produced 0/24 stable predictive configurations.

The raw causal-SMC setup diagnostic from batch `event_edge_v4_20260916_132710` showed that the underlying setup stream itself drifts across chronological probes:

- s01 mean R +0.0200
- s02 mean R +0.0338
- s03 mean R +0.0114
- s04 mean R -0.0406
- s05 mean R -0.0268

The five probes overlap and are development evidence, not independent confirmation.

Directional descriptive medians differed materially (BUY negative, SELL positive), and several sessions/confidence groups differed, but these inspected segments MUST NOT be converted directly into production filters. They are hypothesis-generation evidence only.

A critical implementation observation is independent of those realized-return segment results: the current generic SMC signal generator is intentionally relaxed. It accepts `(market structure OR recent break) AND (recent FVG OR recent OB)` over a 10-bar lookback, then enters at the current close rather than requiring price to interact with the selected zone. Bullish and bearish eligibility are evaluated with `if ... elif ...`, so simultaneous eligibility resolves to BUY by branch order. These semantics can create stale, temporally unrelated or directionally ambiguous setups.

The diagnostic's apparent split between `1-1.5R` and `1.5-2R` must NOT be treated as evidence for choosing a reward bucket. The current generator sets TP at approximately 1.5R; floating-point values around the 1.5 boundary can be placed on opposite sides of a descriptive bucket.

## V5 research question

Can a stricter, explicitly stateful and directionally symmetric causal-SMC setup definition create a more stationary setup stream before any ML overlay is attempted?

V5 is therefore a setup-generator redesign first, not a new learner.

## Design principles

1. **No inspected-OOS segment filter promotion**
   - Do not hard-code SELL-only, a particular session, confidence bucket, regime, or reward bucket from V4 diagnostics.
   - Those observations may motivate diagnostics but cannot become the confirmatory rule set without fresh evidence.

2. **Mutually exclusive directional setup state**
   - Determine the most recent causally confirmed structural break direction.
   - A setup cannot be simultaneously bullish and bearish.
   - If directional evidence conflicts or ties, emit no setup rather than resolving by branch order.

3. **Break-zone temporal alignment**
   - A candidate zone (FVG or OB) must be directionally aligned with the structural break and occur within a declared causal sequence.
   - Do not combine an arbitrary recent break with an unrelated arbitrary recent zone simply because both are inside the same 10-bar window.

4. **Zone interaction required before entry**
   - Entry eligibility should require a causal revisit/touch/rejection of the selected FVG/OB zone after the break/zone is known.
   - Entry at an arbitrary current close without zone interaction is not a V5 setup.

5. **Explicit setup archetype**
   - Separate at least continuation (BOS-led) and reversal (CHoCH-led) archetypes in the event record.
   - Do not mix them under one opaque `BOS/CHoCH` flag.

6. **Symmetric BUY/SELL mechanics**
   - Same state machine, timing, stop construction and zone rules for both directions, mirrored only by sign.
   - Add property tests that mirrored OHLC input produces mirrored setup semantics.

7. **Structure-derived stop first**
   - Prefer a causally known structural invalidation level tied to the setup archetype/zone.
   - ATR may provide a minimum safety distance but should not silently replace the structure definition with a much farther stop.

8. **Keep reward policy predeclared**
   - Do not choose TP/RR from V4 descriptive buckets.
   - Initial V5 baseline should use one fixed predeclared reward multiple for generator comparison, or evaluate a small predeclared matrix as separate hypotheses with correction/holdout discipline.

9. **Confidence is descriptive until recalibrated**
   - Current confidence weights are heuristic/backtest-derived and V4 diagnostics were not monotonic by confidence bucket.
   - V5 should record component evidence separately (structure, break type, zone type, zone age, retest quality) and avoid treating the scalar confidence as a calibrated probability.

10. **No ML until generator baseline passes stability diagnostics**
   - First test raw setup economics and stationarity.
   - Only after a predeclared generator variant clears the baseline stability gate should a learner be added.

## Proposed V5 state machine

A causal setup should progress through explicit states:

`NO_SETUP -> STRUCTURE_BREAK_CONFIRMED -> ALIGNED_ZONE_AVAILABLE -> ZONE_RETESTED -> ENTRY_EVENT`

A setup expires if:

- opposite structural break occurs,
- zone is invalidated before retest,
- maximum setup age is exceeded,
- direction becomes ambiguous,
- stop/entry geometry is invalid.

Every emitted event should store provenance:

- `setup_direction`
- `setup_archetype` (`BOS_CONTINUATION` or `CHOCH_REVERSAL`)
- `break_index`
- `break_level`
- `zone_type` (`FVG` or `OB`)
- `zone_origin_index`
- `zone_confirm_index`
- `zone_top`, `zone_bottom`
- `retest_index`
- `retest_depth_fraction`
- `entry_index`, `entry_mid`
- `stop_mid`
- `declared_reward_r`
- causal regime/session/context fields for diagnostics only

## First implementation stage

Implement a research-only `GoldmicroSMCSetupV5` generator alongside existing code. Do not alter `SMCAnalyzer.generate_signal()` or any live path.

Initial unit/property tests must cover:

- no setup without a confirmed directional break
- no setup from stale/unrelated zone
- no entry before zone retest
- conflicting bullish/bearish evidence fails closed
- BUY/SELL mirror symmetry
- same-bar and chronology causality
- setup expiry
- stable provenance fields

## Development evaluation protocol

Historical V4 OOS is already inspected, so it is development evidence only.

For implementation debugging, old history may be replayed to verify mechanics and obtain descriptive counts. It must NOT be used to claim confirmatory performance.

Before evaluating economic performance of V5, freeze:

- state-machine rules
- maximum zone/setup age
- retest definition
- stop construction
- reward policy
- cooldown/de-duplication
- session handling
- raw-time split/embargo rules
- baseline stability gates

Then use a fresh chronological holdout not consulted during V5 design, or prospective shadow data.

## Baseline gate before ML

A V5 setup generator should first demonstrate raw setup stability. Predeclare a gate before the first confirmatory run. At minimum require:

- sufficient event count in every chronological sample
- no severe sign reversal in mean realized R across most samples
- positive after-cost expectancy in the required majority of samples
- both BUY and SELL mechanics tested for symmetry; directional asymmetry may be measured but not silently exploited post hoc
- no dependence on one tiny session/regime bucket

Exact numeric thresholds beyond existing risk/PF/DD governance should be frozen before the fresh confirmatory dataset is opened.

## Safety

- research artifacts only
- no active model overwrite
- no production/live order path changes
- no secret/credential changes
- no auto-promotion
- future promotion still requires forward shadow evidence, perturbation review, independent audit, and explicit Human Gate
