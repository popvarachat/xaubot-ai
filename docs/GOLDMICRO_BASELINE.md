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
- Current observed terminal spread during capture: `56 points` (`0.56` price)
- Profit/margin/base currency: `USD`
- Trade calc mode: `4`

Important correction from the initial manual assumption: the broker minimum is `0.1` lot, but the broker-reported volume step is `0.01`, so valid broker-grid sizes can include `0.10`, `0.11`, `0.12`, etc. Runtime code must always prefer `symbol_info()` over manually assumed lot increments.

## Observed spread baseline

The supplied tick CSV from 2026-09-15 showed approximately `55 points` median spread with point size `0.01`, with roughly `40-70 points` observed in that sample. The live broker-spec capture showed `56 points`, which is consistent with the historical sample.

The supplied tick sample is suitable for broker-cost profiling, but it is only one trading day and must **not** be treated as sufficient evidence for model optimization or profitability.

The server name is documentation/baseline only. Runtime connection settings must continue to come from environment/configuration (for example `MT5_SERVER`) and credentials must never be committed to the repository.

## Safety rules for sizing

1. Runtime broker specification should be read from MT5 `symbol_info()` rather than assumed.
2. Position size must be floored to broker `volume_step`, never rounded upward in a way that increases risk.
3. If the risk budget supports less than the broker minimum lot (`0.1`), the trade must be skipped.
4. Risk calculations should use `trade_tick_size` and `trade_tick_value_loss`/`trade_tick_value`.
5. Backtests must charge bid/ask spread explicitly and should add slippage, commission, and swap where applicable.
6. Broker-reported tick economics should be cross-checked with MT5 `order_calc_profit()` before live enablement.

## Verified monetary interpretation

Using the captured broker specification, a `1.00` USD favorable/adverse price move equals `100` ticks. At `1.0` lot, `100 * 0.01 = 1.00 USD`; at the minimum `0.1` lot this equals approximately `0.10 USD` per `1.00` price move before spread, slippage, commission, and swap. This relationship should still be verified with `order_calc_profit()` as an independent MT5 calculation before any live trading gate is opened.

## Validation plan before optimization

- Replace fixed XAUUSD pip-value assumptions in risk and backtest code paths.
- Use correct BUY Ask entry / Bid exit and SELL Bid entry / Ask exit accounting.
- Recalculate PF, maximum drawdown, expectancy, Sharpe, and equity after realistic costs.
- Verify no look-ahead/data leakage in feature generation and model evaluation.
- Add walk-forward/out-of-sample validation before parameter optimization is considered trustworthy.
- Stress-test spread above the observed baseline, including 70 and 100+ point scenarios.
- Cross-check broker cash P/L with MT5 `order_calc_profit()`.

## Current implementation stage

The branch now contains an isolated GOLDmicro foundation plus a post-trade replay layer:

- `src/broker_profile.py`: broker symbol specification abstraction.
- `src/goldmicro_risk.py`: conservative risk-based GOLDmicro sizing.
- `backtests/goldmicro_cost_model.py`: Bid/Ask-aware execution-cost accounting.
- `backtests/goldmicro_replay.py`: replays legacy trade records with sequential equity, broker-valid lot sizing, spread/slippage/fees, PF, expectancy, Sharpe, and maximum drawdown.
- `.github/workflows/goldmicro-unit-tests.yml`: isolated CI for the new GOLDmicro components.
- `scripts/dump_goldmicro_spec.py`: read-only runtime symbol-spec capture without printing credentials or placing orders.

The replay stage intentionally preserves the legacy signal/entry/exit timing. It is therefore a safer first baseline for measuring the impact of broker sizing and transaction costs, but it is **not yet a full tick-accurate backtest**. In particular, TP/SL/exit trigger timing still originates from the legacy one-price OHLC path.

## Scope of this branch

`feat/goldmicro-broker-profile-v1` does **not** enable live trading or change production execution behavior. The next stage is to cross-check the captured symbol economics with `order_calc_profit()` and replay a sufficiently long historical sample before strategy optimization.
