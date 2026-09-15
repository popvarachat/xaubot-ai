"""Run a read-only GOLDmicro baseline replay from legacy backtest XLSX results.

This script:
- reads the current MT5 GOLDmicro broker specification,
- calibrates account-currency P/L with order_calc_profit(),
- loads legacy trade rows from an XLSX workbook,
- replays them with broker-valid risk sizing and spread scenarios,
- prints PF/DD/expectancy in the MT5 account currency,
- writes a local JSON report.

It NEVER places, modifies, or closes orders.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    raise SystemExit("MetaTrader5 package is required: py -m pip install MetaTrader5")

from backtests.goldmicro_cost_model import BacktestCostConfig, GoldmicroCostModel
from backtests.goldmicro_replay import replay_legacy_trades
from src.broker_profile import BrokerSymbolProfile


SYMBOL = "GOLDmicro"
DEFAULT_PATTERN = "backtests/24_final_combined_results/*.xlsx"


@dataclass(frozen=True)
class LegacyTradeRow:
    direction: str
    entry_price: float
    exit_price: float
    stop_loss: float


def _norm(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


ALIASES = {
    "direction": {"direction", "side", "type", "signal", "trade_type"},
    "entry_price": {"entry_price", "entry", "entryprice", "open_price", "price_open"},
    "exit_price": {"exit_price", "exit", "exitprice", "close_price", "price_close"},
    "stop_loss": {"stop_loss", "stoploss", "sl", "stop_price", "sl_price"},
}


def _resolve_columns(df: pd.DataFrame) -> dict[str, str] | None:
    normalized = {_norm(c): c for c in df.columns}
    resolved: dict[str, str] = {}
    for target, aliases in ALIASES.items():
        for alias in aliases:
            if alias in normalized:
                resolved[target] = normalized[alias]
                break
        if target not in resolved:
            return None
    return resolved


def _extract_trade_sheets(path: Path) -> list[tuple[str, list[LegacyTradeRow]]]:
    xls = pd.ExcelFile(path)
    found: list[tuple[str, list[LegacyTradeRow]]] = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet)
        cols = _resolve_columns(df)
        if not cols:
            continue

        trades: list[LegacyTradeRow] = []
        for _, row in df.iterrows():
            try:
                direction = str(row[cols["direction"]]).upper().strip()
                if direction not in {"BUY", "SELL"}:
                    continue
                entry = float(row[cols["entry_price"]])
                exit_ = float(row[cols["exit_price"]])
                stop = float(row[cols["stop_loss"]])
                if entry <= 0 or exit_ <= 0 or stop <= 0:
                    continue
            except (TypeError, ValueError):
                continue
            trades.append(LegacyTradeRow(direction, entry, exit_, stop))

        if trades:
            found.append((sheet, trades))
    return found


def _latest_default_workbook(repo_root: Path) -> Path:
    matches = sorted(repo_root.glob(DEFAULT_PATTERN), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No workbook found matching {DEFAULT_PATTERN}")
    return matches[0]


def _calibrate_profile(symbol: str) -> tuple[BrokerSymbolProfile, float, str]:
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"Unable to select {symbol}: {mt5.last_error()}")

    info = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    account = mt5.account_info()
    if info is None or tick is None or account is None:
        raise RuntimeError("MT5 symbol/tick/account info unavailable")

    buy = mt5.order_calc_profit(mt5.ORDER_TYPE_BUY, symbol, 1.0, tick.ask, round(tick.ask + 1.0, info.digits))
    sell = mt5.order_calc_profit(mt5.ORDER_TYPE_SELL, symbol, 1.0, tick.bid, round(tick.bid - 1.0, info.digits))
    checks = [abs(float(x)) for x in (buy, sell) if x is not None and float(x) != 0]
    if not checks:
        raise RuntimeError(f"order_calc_profit calibration failed: {mt5.last_error()}")
    cash_per_price_unit_per_lot = sum(checks) / len(checks)

    profile = BrokerSymbolProfile(
        symbol=symbol,
        point=float(info.point),
        tick_size=float(info.trade_tick_size),
        tick_value=float(info.trade_tick_value_loss or info.trade_tick_value),
        contract_size=float(info.trade_contract_size),
        volume_min=float(info.volume_min),
        volume_max=float(info.volume_max),
        volume_step=float(info.volume_step),
        cash_per_price_unit_per_lot=cash_per_price_unit_per_lot,
        cash_currency=str(account.currency),
    )
    profile.validate()
    return profile, float(account.balance), str(account.currency)


def _max_drawdown_cash(initial_capital: float, pnls: Iterable[float]) -> float:
    equity = initial_capital
    peak = initial_capital
    max_dd = 0.0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    return max_dd


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay legacy XAUUSD trades as XM GOLDmicro in account currency")
    parser.add_argument("--input", type=Path, help="Legacy XLSX result file; default: newest #24 final-combined workbook")
    parser.add_argument("--capital", type=float, help="Starting capital in account currency; default: current MT5 balance")
    parser.add_argument("--risk", type=float, default=1.0, help="Risk per trade percent, default 1.0")
    parser.add_argument("--spreads", default="55,70,100", help="Comma-separated spread-point scenarios")
    parser.add_argument("--slippage", type=float, default=0.0, help="Adverse slippage points per side")
    args = parser.parse_args()

    repo_root = REPO_ROOT
    workbook = (args.input if args.input else _latest_default_workbook(repo_root)).resolve()
    if not workbook.exists():
        raise SystemExit(f"Input workbook not found: {workbook}")

    try:
        profile, live_balance, currency = _calibrate_profile(SYMBOL)
        initial_capital = float(args.capital) if args.capital else live_balance
        trade_sheets = _extract_trade_sheets(workbook)
        if not trade_sheets:
            raise RuntimeError("No worksheet contains direction/entry/exit/stop-loss trade columns")

        spreads = [float(x.strip()) for x in args.spreads.split(",") if x.strip()]
        report = {
            "generated_at": datetime.now().isoformat(),
            "input_workbook": str(workbook),
            "symbol": SYMBOL,
            "currency": currency,
            "initial_capital": initial_capital,
            "risk_per_trade_percent": args.risk,
            "slippage_points_per_side": args.slippage,
            "calibration_note": "Uses current MT5 account-currency calibration for historical trades; provisional until historical FX conversion is modeled.",
            "broker_profile": {
                "point": profile.point,
                "tick_size": profile.tick_size,
                "tick_value": profile.tick_value,
                "contract_size": profile.contract_size,
                "volume_min": profile.volume_min,
                "volume_max": profile.volume_max,
                "volume_step": profile.volume_step,
                "cash_per_price_unit_per_lot": profile.cash_per_price_unit_per_lot,
            },
            "results": [],
        }

        print("\n=== GOLDmicro Baseline Replay (READ ONLY) ===")
        print(f"Workbook : {workbook}")
        print(f"Currency : {currency}")
        print(f"Capital  : {initial_capital:,.2f} {currency}")
        print(f"Calibration: {profile.cash_per_price_unit_per_lot:.4f} {currency} per +1.00 price / 1.00 lot")
        print(f"Broker volume: min={profile.volume_min:g}, step={profile.volume_step:g}, max={profile.volume_max:g}")
        print("FX note: historical trades use the current account-currency calibration in this provisional baseline.")

        for sheet, trades in trade_sheets:
            print(f"\nSheet: {sheet} ({len(trades)} legacy trades)")
            for spread in spreads:
                model = GoldmicroCostModel(
                    profile,
                    BacktestCostConfig(
                        spread_points=spread,
                        slippage_points=args.slippage,
                        commission_per_lot_round_turn=0.0,
                        swap_per_lot=0.0,
                    ),
                )
                stats = replay_legacy_trades(
                    trades,
                    profile=profile,
                    cost_model=model,
                    initial_capital=initial_capital,
                    risk_per_trade_percent=args.risk,
                    win_rate=0.55,
                    reward_risk_ratio=2.0,
                )
                win_rate = (stats.wins / stats.executed_trades * 100.0) if stats.executed_trades else 0.0
                max_dd_cash = _max_drawdown_cash(initial_capital, (t.net_pnl for t in stats.trades if not t.skipped))
                row = {
                    "sheet": sheet,
                    "spread_points": spread,
                    "legacy_trades": stats.total_trades,
                    "executed_trades": stats.executed_trades,
                    "skipped_trades": stats.skipped_trades,
                    "wins": stats.wins,
                    "losses": stats.losses,
                    "win_rate_percent": win_rate,
                    "net_profit": stats.net_profit,
                    "final_capital": stats.final_capital,
                    "profit_factor": stats.profit_factor,
                    "max_drawdown_cash": max_dd_cash,
                    "max_drawdown_percent": stats.max_drawdown_percent,
                    "expectancy": stats.expectancy,
                    "sharpe": stats.sharpe_ratio,
                }
                report["results"].append(row)
                print(
                    f"  spread={spread:>5.0f} | exec={stats.executed_trades:>4} skip={stats.skipped_trades:>4} "
                    f"WR={win_rate:>6.2f}% PF={stats.profit_factor:>6.3f} "
                    f"Net={stats.net_profit:>10.2f} {currency} "
                    f"DD={max_dd_cash:>9.2f} {currency} ({stats.max_drawdown_percent:>5.2f}%) "
                    f"Exp={stats.expectancy:>8.2f}"
                )

        out_dir = repo_root / "backtests" / "goldmicro_baseline_results"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"goldmicro_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        out_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved report: {out_file}")
        print("NOTE: commission/swap are 0 and historical FX is not yet modeled; this is a provisional pre-optimization baseline, not a profitability guarantee.")
        return 0
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
