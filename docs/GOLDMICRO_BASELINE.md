# GOLDmicro Broker Baseline

This fork is being adapted from generic XAUUSD assumptions to a broker-specific GOLDmicro workflow.

## Confirmed user/broker constraints

- Broker/MT5 server: `XMGlobal-MT5 4`
- Symbol: `GOLDmicro`
- Minimum lot: `0.1`
- Lot step: `0.1`
- Observed spread baseline from supplied tick CSV (2026-09-15): approximately `55 points` median when point size is `0.01`
- Observed spread range in that sample: roughly `40-70 points`

The supplied tick sample is suitable for broker-cost profiling, but it is only one trading day and must **not** be treated as sufficient evidence for model optimization or profitability.

The server name is documentation/baseline only. Runtime connection settings must continue to come from environment/configuration (for example `MT5_SERVER`) and credentials must never be committed to the repository.

## Safety rules for sizing

1. Runtime broker specification should be read from MT5 `symbol_info()` rather than assumed.
2. Position size must be floored to broker `volume_step`, never rounded upward in a way that increases risk.
3. If the risk budget supports less than the broker minimum lot (`0.1`), the trade must be skipped.
4. Risk calculations should use `trade_tick_size` and `trade_tick_value_loss`/`trade_tick_value`.
5. Backtests must charge bid/ask spread explicitly and should add slippage, commission, and swap where applicable.

## Validation plan before optimization

- Replace fixed XAUUSD pip-value assumptions in risk and backtest code paths.
- Use correct BUY Ask entry / Bid exit and SELL Bid entry / Ask exit accounting.
- Recalculate PF, maximum drawdown, expectancy, Sharpe, and equity after realistic costs.
- Verify no look-ahead/data leakage in feature generation and model evaluation.
- Add walk-forward/out-of-sample validation before parameter optimization is considered trustworthy.
- Stress-test spread above the observed baseline, including 70 and 100+ point scenarios.

## Current implementation stage

The branch now contains an isolated GOLDmicro foundation plus a post-trade replay layer:

- `src/broker_profile.py`: broker symbol specification abstraction.
- `src/goldmicro_risk.py`: conservative risk-based GOLDmicro sizing.
- `backtests/goldmicro_cost_model.py`: Bid/Ask-aware execution-cost accounting.
- `backtests/goldmicro_replay.py`: replays legacy trade records with sequential equity, broker-valid lot sizing, spread/slippage/fees, PF, expectancy, Sharpe, and maximum drawdown.
- `.github/workflows/goldmicro-unit-tests.yml`: isolated CI for the new GOLDmicro components.

The replay stage intentionally preserves the legacy signal/entry/exit timing. It is therefore a safer first baseline for measuring the impact of broker sizing and transaction costs, but it is **not yet a full tick-accurate backtest**. In particular, TP/SL/exit trigger timing still originates from the legacy one-price OHLC path.

## Scope of this branch

`feat/goldmicro-broker-profile-v1` does **not** enable live trading or change production execution behavior. The next stage is to obtain a trustworthy GOLDmicro broker specification (`tick_size`, `tick_value`, `contract_size`, volume limits) from MT5 and run/replay a sufficiently long historical sample before strategy optimization.
