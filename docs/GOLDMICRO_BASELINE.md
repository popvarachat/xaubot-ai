# GOLDmicro Broker Baseline

This fork is being adapted from generic XAUUSD assumptions to a broker-specific GOLDmicro workflow.

## Verified broker specification

Captured read-only from MetaTrader 5 on `XMGlobal-MT5 4` for `GOLDmicro` on 2026-09-15:

- Broker/MT5 server: `XMGlobal-MT5 4`
- Symbol: `GOLDmicro`
- Description: `GOLD`
- Path: `Derivatives\\Spot Metals Micro\\GOLDmicro`
- Digits: `2`
- Point: `0.01`
- Trade contract size: `1.0`
- Trade tick size: `0.01`
- Trade tick value: `0.01`
- Trade tick value profit: `0.01`
- Trade tick value loss: `0.01`
- Minimum volume: `0.1`
- Maximum volume: `100.0`
- Volume step: `0.01`
- Trade stops level: `0`
- Trade freeze level: `0`
- Spread: floating
- Symbol base/profit/margin currency: `USD`
- MT5 account currency: `THB`
- Trade calc mode: `4`

Important correction from the initial manual assumption: the broker minimum is `0.1` lot, but the broker-reported volume step is `0.01`, so valid broker-grid sizes can include `0.10`, `0.11`, `0.12`, etc. Runtime code must always prefer `symbol_info()` over manually assumed lot increments.

## Account-currency monetary calibration

MT5 `order_calc_profit()` was used as a read-only account-aware cross-check. The verified output was approximately:

- BUY `0.10` lot, favorable price move `+1.00` -> `+3.33 THB`
- BUY `1.00` lot, favorable price move `+1.00` -> `+33.28 THB`
- SELL `0.10` lot, favorable price move `-1.00` -> `+3.33 THB`
- SELL `1.00` lot, favorable price move `-1.00` -> `+33.28 THB`

`order_calc_profit()` returns P/L in MT5 account currency (`THB`). The observed `33.28-33.30 THB` per `1.00` price move at `1.00` lot is a point-in-time calibration, not a permanent constant.

## Safety rules for sizing

1. Runtime broker specification should be read from MT5 `symbol_info()` rather than assumed.
2. Risk budget = account balance/equity × configured risk percent.
3. Use `order_calc_profit()` for the 1-lot entry-to-stop move to estimate account-currency loss per lot.
4. Raw lot = risk budget / absolute 1-lot stop loss.
5. Floor to broker `volume_step`; never round upward in a way that increases risk.
6. If the risk budget supports less than broker minimum lot (`0.1`), SKIP/HOLD; never force the minimum lot.
7. Recalculate actual monetary risk after lot normalization.

## Observed spread profile

A local MT5 GOLDmicro M1 export covering 2026-01-02 through 2026-09-15 was analyzed without committing the market-data file to GitHub.

Observed `<SPREAD>` distribution across 249,038 M1 rows:

- Mean: `47.51` points
- Median/P50: `50`
- P90: `54`
- P95: `55`
- P99: `59`
- Minimum: `30`
- Maximum spike: `265`
- `88.17%` of rows were within `40-60` points
- `99.06%` of rows were `<=60` points

Operational spread validation therefore uses `40/50/55/60` points. `70/100` are separate stress scenarios.

M1 `<SPREAD>` is a bar-level spread field and is not equivalent to a tick-by-tick Bid/Ask execution stream.

## Capital viability validation

Fine sweep used the existing 739-trade `#24 Final Combined` trade log, current THB calibration, `1%` configured risk, zero commission/swap, and no historical FX reconstruction.

Validation gates:

- Profit Factor `>= 1.30`
- Max Drawdown `<= 10%`
- Skipped trades `<= 20%`

### Operational spread sweep: 40/50/55/60 points

- `11,000 THB`: **FAIL**. Spread 50-60 scenarios exceeded the 20% skip gate; spread 60 skip was `20.97%`.
- `12,000 THB`: **PASS** all operational spread scenarios. Worst operational case at spread 60: PF `1.459`, DD `7.32%`, skip `18.00%`, net `+6,245.89 THB`.
- `15,000 THB`: **PASS** with better reserve. Worst operational case at spread 60: PF `1.463`, DD `7.60%`, skip `12.04%`, net `+8,245.91 THB`.
- `20,000 THB`: **PASS** with stronger execution coverage. Worst operational case at spread 60: PF `1.456`, DD `7.59%`, skip `7.17%`, net `+11,193.68 THB`.

Current classification:

- **Minimum Operational Capital:** `12,000 THB` — minimum tested capital passing all operational 40/50/55/60 spread gates.
- **Recommended Capital:** `15,000 THB` — materially lower skip rate with more operating headroom.
- **Comfortable Capital:** `20,000 THB` — minimum-lot distortion reduced to `<=7.17%` skip across the tested operational range.
- **Full strategy-fidelity reference:** around `75,000 THB+` in the earlier coarse sweep, where tested trades reached `0%` skip under 55/70-point scenarios.

These are validation classifications, not funding advice or profitability guarantees.

## Stress behavior

Separate stress sweep at 70/100 points:

- Spread `70`: `12,000 THB` and above passed the configured PF/DD/skip gates in the tested range.
- Spread `100`: no tested capital from `11,000` through `20,000 THB` passed all gates. The binding failure was Profit Factor, which remained around `1.24-1.27` even as skip rates improved.

The 100-point failure is therefore not primarily a capital-size problem. Adding capital does not restore the strategy edge enough under this provisional cost model.

Candidate runtime-policy interpretation for later design only; **not yet wired into live execution**:

- `<=60` points: normal operational range supported by observed M1 data.
- `61-70` points: degraded/stress range; provisional edge remains positive but should be treated cautiously.
- `>70` points: abnormal-spread region and candidate HOLD/no-new-entry zone, subject to commission/slippage and tick-level validation before any live integration.

## Current non-live tooling

- `src/broker_profile.py`: broker symbol specification abstraction with optional account-currency cash calibration.
- `src/goldmicro_risk.py`: conservative risk-based GOLDmicro sizing.
- `backtests/goldmicro_cost_model.py`: Bid/Ask-aware execution-cost accounting.
- `backtests/goldmicro_replay.py`: sequential-equity replay with PF, expectancy, Sharpe, and DD rebuild.
- `scripts/dump_goldmicro_spec.py`: read-only runtime symbol/account calibration.
- `scripts/run_goldmicro_baseline.py`: legacy XLSX replay with GOLDmicro sizing and cost assumptions.
- `scripts/run_goldmicro_baseline_isolated.py`: isolated bootstrap avoiding the full upstream ML/HMM import chain.
- `scripts/run_goldmicro_capital_sweep_isolated.py`: deterministic #24 Trade Log parser and capital/spread viability sweep.
- `scripts/analyze_goldmicro_spread_profile.py`: local M1 `<SPREAD>` distribution analyzer; market-data file is not committed.

## Known limitations before optimization

- Legacy entry/exit trigger timing originates from a one-price OHLC path, not tick-trigger-accurate Bid/Ask execution.
- Current replay applies current THB cash calibration to historical trades; historical USD/THB conversion is not modeled.
- Commission and swap are currently zero until broker/account-specific values are verified.
- Slippage defaults to zero unless explicitly supplied.
- The legacy workbook is replay input only; its original XAUUSD economics are not accepted as GOLDmicro performance.

## Scope of this branch

`feat/goldmicro-broker-profile-v1` remains non-live:

- No orders are placed, modified, or closed.
- No production execution path has been enabled.
- No credentials or secrets are changed or committed.
- No strategy signal parameters have been optimized yet.

Next validation stage: verify broker/account transaction costs, then proceed to walk-forward/time-series validation before optimization decisions.
