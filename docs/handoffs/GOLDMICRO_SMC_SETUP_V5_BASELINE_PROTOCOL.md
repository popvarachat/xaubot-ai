# GOLDmicro SMC Setup V5 Raw Baseline Protocol

## Scope

Research-only. No ML overlay, no PF/DD promotion gate, no live activation, no order placement, no active-model overwrite.

The currently available 24,000-row M15 snapshot has already influenced V3/V4/V5 design. Any replay on it is **development evidence only**. It may be used to reject obviously weak mechanics or diagnose instability, but it cannot establish confirmatory profitability.

## Frozen V5 generator mechanics

Use `SMCSetupV5Config()` unchanged for this baseline:

- max setup age: 12 bars
- max break-to-zone delay: 6 bars
- minimum zone-to-retest delay: 1 bar
- fixed declared reward: 1.5R
- ATR stop floor: 0.50 x ATR
- mutually exclusive causal break direction
- aligned zone confirmed after/on break
- later retest/rejection required
- adverse setup invalidation fails closed
- no session/direction/confidence/regime filter

## Outcome contract

- evaluation starts on the bar after entry
- horizon: 32 raw M15 bars
- declared TP and structure-derived SL are fixed at event emission
- if TP and SL are both touched in the same bar, SL wins (adverse ordering)
- if neither is reached by horizon, exit at horizon close
- realized gross R = signed mid-price move / event risk distance

## Cost profiles

Use the same explicit GOLDmicro research assumptions already used by the project:

- normal: 50 spread points + 1 slippage point at entry and exit
- conservative: 55 spread points + 12 slippage points at entry and exit
- broker/account cash calibration: 33.29 THB per +1.00 price move per 1.0 lot

Costs are converted to event R units through the same broker cost model; no lot-size or PF optimization is performed in this baseline.

## Chronology

Split the full raw M15 snapshot into 5 equal, **non-overlapping raw-time blocks** by row index. Assign an event to the block containing its entry index. Do not event-quantile balance the blocks.

This intentionally lets low setup density fail the sufficiency gate.

## Development baseline gate

Predeclared before viewing V5 realized returns:

- minimum 30 V5 outcome events in **every** chronological block
- positive mean after-cost R in at least 4/5 blocks under the normal cost profile
- positive mean after-cost R in at least 4/5 blocks under the conservative cost profile

Both cost profiles must satisfy the same majority rule. No post-hoc lowering is permitted after seeing the results.

This is a raw-generator stability screen only. Passing it does **not** authorize ML, PF/DD claims, shadow promotion, or live deployment.

## Interpretation rules

- If event count fails, treat the variant as evidence-insufficient; do not loosen the count gate from the same replay.
- If after-cost mean R fails, treat the raw setup generator as economically unstable on development history; do not cherry-pick direction/session/archetype/zone filters from the same history.
- If the development screen passes, freeze V5 mechanics and wait for fresh chronological/prospective evidence before making confirmatory claims.
- V4 descriptive SELL/session/confidence observations remain hypothesis-generation only and are not encoded in V5.

## Safety

Live model remains unchanged. Promotion remains disabled. Any future production change still requires fresh evidence, shadow validation, independent review, and explicit Human Gate.
