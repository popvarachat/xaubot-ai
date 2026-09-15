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
- Observed terminal spread during verification: `55-56 points` (`0.55-0.56` price)
- Symbol base/profit/margin currency: `USD`
- MT5 account currency: `THB`
- Trade calc mode: `4`

Important correction from the initial manual assumption: the broker minimum is `0.1` lot, but the broker-reported volume step is `0.01`, so valid broker-grid sizes can include `0.10`, `0.11`, `0.12`, etc. Runtime code must always prefer `symbol_info()` over manually assumed lot increments.

## Account-currency monetary calibration

MT5 `order_calc_profit()` was used as a read-only account-aware cross-check. The verified output was:

- BUY `0.10` lot, favorable price move `+1.00` -> approximately `+3.33 THB`
- BUY `1.00` lot, favorable price move `+1.00` -> approximately `+33.28 THB`
- SELL `0.10` lot, favorable price move `-1.00` -> approximately `+3.33 THB`
- SELL `1.00` lot, favorable price move `-1.00` -> approximately `+33.28 THB`

This confirms that `order_calc_profit()` returns P/L in the MT5 account currency (`THB`). The symbol itself reports USD as its base/profit/margin currency, so the final cash result includes USD->THB conversion inside MT5.

Therefore:

1. `trade_tick_value` remains useful broker metadata.
2. For risk sizing and backtest cash P/L in this THB account, an `order_calc_profit()`-derived account-currency calibration is canonical when available.
3. The observed `33.28 THB` per `1.00` price move at `1.00` lot is a point-in-time calibration, not a permanent constant. It can vary with USD/THB conversion.
4. PF is dimensionless and is unaffected by a constant currency conversion factor. Absolute P/L and drawdown in THB use account-currency values, while DD% remains the preferred cross-period risk measure.

## Observed spread baseline

The supplied tick CSV from 2026-09-15 showed approximately `55 points` median spread with point size `0.01`, with roughly `40-70 points` observed in that sample. Live terminal captures showed `55-56 points`, consistent with the historical sample.

At the verified account-currency calibration (`33.28 THB` per `1.00` price move per `1.00` lot), a flat-price round trip with a `55-point` (`0.55`) spread costs approximately:

- `1.00 lot`: `0.55 * 33.28` ≈ `18.30 THB`
- `0.10 lot`: ≈ `1.83 THB`

before slippage, commission, and swap.

The supplied tick sample is suitable for broker-cost profiling, but it is only one trading day and must **not** be treated as sufficient evidence for model optimization or profitability.

The server name is documentation/baseline only. Runtime connection settings must continue to come from environment/configuration (for example `MT5_SERVER`) and credentials must never be committed to the repository.

## Safety rules for sizing

1. Runtime broker specification should be read from MT5 `symbol_info()` rather than assumed.
2. Position size must be floored to broker `volume_step`, never rounded upward in a way that increases risk.
3. If the risk budget supports less than the broker minimum lot (`0.1`), the trade must be skipped.
4. Account-currency risk/P&L should prefer an `order_calc_profit()`-derived calibration when the account currency differs from the symbol profit currency.
5. Backtests must charge bid/ask spread explicitly and should add slippage, commission, and swap where applicable.
6. Broker-reported tick economics should be cross-checked with MT5 `order_calc_profit()` before live enablement.

## One-click baseline replay

The repository already contains legacy backtest result workbooks, including `backtests/24_final_combined_results/*.xlsx`. `scripts/run_goldmicro_baseline.py` converts those legacy trade records into a GOLDmicro/THB baseline without changing the original signal timing.

From the repository root on the Windows machine where MT5 is already logged in:

```powershell
py -m pip install pandas openpyxl
py scripts\run_goldmicro_baseline.py
```

Default behavior:

- uses the newest `#24 Final Combined` XLSX workbook;
- uses the current MT5 account balance as starting capital (override with `--capital`);
- calibrates account-currency cash P/L with read-only `order_calc_profit()`;
- uses `1.0%` risk per trade by default (override with `--risk`);
- evaluates spread scenarios `55`, `70`, and `100` points;
- reports executed/skipped trades, WR, PF, net P/L, max DD in THB and %, expectancy, and Sharpe;
- saves a local JSON report under `backtests/goldmicro_baseline_results/`.

Example overrides:

```powershell
py scripts\run_goldmicro_baseline.py --capital 100000 --risk 0.5 --spreads 55,70,100 --slippage 5
```

The baseline runner does **not** submit, modify, or close orders. Commission and swap are currently set to zero in this baseline pass until broker/account-specific values are verified, so results must be labeled provisional.

## Validation plan before optimization

- Replace fixed XAUUSD pip-value assumptions in risk and backtest code paths.
- Use correct BUY Ask entry / Bid exit and SELL Bid entry / Ask exit accounting.
- Recalculate PF, maximum drawdown, expectancy, Sharpe, and equity after realistic costs.
- Verify no look-ahead/data leakage in feature generation and model evaluation.
- Add walk-forward/out-of-sample validation before parameter optimization is considered trustworthy.
- Stress-test spread above the observed baseline, including 70 and 100+ point scenarios.
- For long historical tests, either use historical USD/THB conversion or clearly label the use of a fixed account-currency calibration snapshot.

## Current implementation stage

The branch now contains an isolated GOLDmicro foundation plus a post-trade replay layer:

- `src/broker_profile.py`: broker symbol specification abstraction with optional account-currency cash calibration.
- `src/goldmicro_risk.py`: conservative risk-based GOLDmicro sizing.
- `backtests/goldmicro_cost_model.py`: Bid/Ask-aware execution-cost accounting using calibrated account-currency cash P/L when provided.
- `backtests/goldmicro_replay.py`: replays legacy trade records with sequential equity, broker-valid lot sizing, spread/slippage/fees, PF, expectancy, Sharpe, and maximum drawdown.
- `scripts/run_goldmicro_baseline.py`: read-only one-click XLSX replay using live MT5 account-currency calibration and 55/70/100 spread stress scenarios.
- `.github/workflows/goldmicro-unit-tests.yml`: isolated CI for the new GOLDmicro components and syntax-check of the baseline runner.
- `scripts/dump_goldmicro_spec.py`: read-only runtime symbol-spec and account-currency calibration capture without printing credentials or placing orders.

The replay stage intentionally preserves the legacy signal/entry/exit timing. It is therefore a safer first baseline for measuring the impact of broker sizing and transaction costs, but it is **not yet a full tick-accurate backtest**. In particular, TP/SL/exit trigger timing still originates from the legacy one-price OHLC path.

## Scope of this branch

`feat/goldmicro-broker-profile-v1` does **not** enable live trading or change production execution behavior. The next stage is to run the GOLDmicro baseline replay, verify commission/swap assumptions, and only then proceed to walk-forward strategy optimization.
