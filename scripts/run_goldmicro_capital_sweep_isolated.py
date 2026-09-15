"""Read-only GOLDmicro capital viability sweep.

Uses the existing legacy #24 trade log plus the isolated GOLDmicro replay layer
to estimate the minimum account capital that satisfies configurable PF/DD/skip
criteria under multiple spread scenarios.

This script does not place, modify, or close MT5 orders.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
BASELINE_RUNNER = REPO_ROOT / "scripts" / "run_goldmicro_baseline.py"
DEFAULT_CAPITALS = "5000,10000,20000,30000,50000,75000,100000,150000,166450,200000"
DEFAULT_SPREADS = "55,70"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Avoid executing upstream src/__init__.py and therefore avoid pulling in the
# complete ML/HMM dependency tree for this read-only analysis.
if "src" not in sys.modules:
    src_package = types.ModuleType("src")
    src_package.__path__ = [str(SRC_DIR)]
    src_package.__package__ = "src"
    sys.modules["src"] = src_package


def _load_baseline_module():
    spec = importlib.util.spec_from_file_location("goldmicro_baseline_runner", BASELINE_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load baseline runner: {BASELINE_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _csv_floats(value: str) -> list[float]:
    parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("Expected at least one numeric value")
    if any(x <= 0 for x in parsed):
        raise argparse.ArgumentTypeError("Values must be positive")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep GOLDmicro capital against PF/DD/skip viability gates")
    parser.add_argument("--input", type=Path, help="Legacy XLSX result file; default: newest #24 final-combined workbook")
    parser.add_argument("--capitals", default=DEFAULT_CAPITALS, help="Comma-separated THB capitals")
    parser.add_argument("--spreads", default=DEFAULT_SPREADS, help="Comma-separated spread-point scenarios")
    parser.add_argument("--risk", type=float, default=1.0, help="Risk per trade percent, default 1.0")
    parser.add_argument("--min-pf", type=float, default=1.30, help="Minimum Profit Factor gate")
    parser.add_argument("--max-dd", type=float, default=10.0, help="Maximum drawdown percent gate")
    parser.add_argument("--max-skip", type=float, default=20.0, help="Maximum skipped-trade percent gate")
    parser.add_argument("--slippage", type=float, default=0.0, help="Adverse slippage points per side")
    args = parser.parse_args()

    if args.risk <= 0:
        raise SystemExit("--risk must be positive")
    if args.min_pf <= 0 or args.max_dd < 0 or args.max_skip < 0:
        raise SystemExit("Invalid viability thresholds")

    capitals = _csv_floats(args.capitals)
    spreads = _csv_floats(args.spreads)
    baseline = _load_baseline_module()

    workbook = (args.input if args.input else baseline._latest_default_workbook(REPO_ROOT)).resolve()
    if not workbook.exists():
        raise SystemExit(f"Input workbook not found: {workbook}")

    try:
        profile, _live_balance, currency = baseline._calibrate_profile(baseline.SYMBOL)
        trade_sheets = baseline._extract_trade_sheets(workbook)
        if not trade_sheets:
            raise RuntimeError("No worksheet contains direction/entry/exit/stop-loss trade columns")

        # The legacy workbook can contain supporting sheets. Sweep every sheet
        # that resolves to actual trade rows and require all to pass.
        report = {
            "generated_at": datetime.now().isoformat(),
            "input_workbook": str(workbook),
            "symbol": baseline.SYMBOL,
            "currency": currency,
            "risk_per_trade_percent": args.risk,
            "spreads": spreads,
            "capitals": capitals,
            "criteria": {
                "min_profit_factor": args.min_pf,
                "max_drawdown_percent": args.max_dd,
                "max_skip_percent": args.max_skip,
            },
            "cash_per_price_unit_per_lot": profile.cash_per_price_unit_per_lot,
            "results": [],
            "minimum_passing_capital": None,
        }

        print("\n=== GOLDmicro Capital Viability Sweep (READ ONLY) ===")
        print(f"Workbook    : {workbook}")
        print(f"Currency    : {currency}")
        print(f"Calibration : {profile.cash_per_price_unit_per_lot:.4f} {currency} per +1.00 price / 1.00 lot")
        print(f"Broker lots : min={profile.volume_min:g}, step={profile.volume_step:g}, max={profile.volume_max:g}")
        print(
            f"Gate        : PF >= {args.min_pf:.2f}, DD <= {args.max_dd:.2f}%, "
            f"Skip <= {args.max_skip:.2f}%"
        )
        print(f"Spreads     : {', '.join(f'{x:g}' for x in spreads)} points")
        print("NOTE: commission/swap=0 and historical FX is not modeled in this provisional sweep.\n")

        first_passing: float | None = None

        for capital in capitals:
            capital_pass = True
            capital_rows = []
            print(f"Capital {capital:,.0f} {currency}")

            for sheet, trades in trade_sheets:
                for spread in spreads:
                    model = baseline.GoldmicroCostModel(
                        profile,
                        baseline.BacktestCostConfig(
                            spread_points=spread,
                            slippage_points=args.slippage,
                            commission_per_lot_round_turn=0.0,
                            swap_per_lot=0.0,
                        ),
                    )
                    stats = baseline.replay_legacy_trades(
                        trades,
                        profile=profile,
                        cost_model=model,
                        initial_capital=capital,
                        risk_per_trade_percent=args.risk,
                        win_rate=0.55,
                        reward_risk_ratio=2.0,
                    )
                    skip_pct = (stats.skipped_trades / stats.total_trades * 100.0) if stats.total_trades else 100.0
                    win_rate = (stats.wins / stats.executed_trades * 100.0) if stats.executed_trades else 0.0
                    max_dd_cash = baseline._max_drawdown_cash(
                        capital, (t.net_pnl for t in stats.trades if not t.skipped)
                    )
                    passed = (
                        stats.executed_trades > 0
                        and stats.profit_factor >= args.min_pf
                        and stats.max_drawdown_percent <= args.max_dd
                        and skip_pct <= args.max_skip
                    )
                    capital_pass = capital_pass and passed

                    row = {
                        "capital": capital,
                        "sheet": sheet,
                        "spread_points": spread,
                        "executed_trades": stats.executed_trades,
                        "skipped_trades": stats.skipped_trades,
                        "skip_percent": skip_pct,
                        "win_rate_percent": win_rate,
                        "profit_factor": stats.profit_factor,
                        "net_profit": stats.net_profit,
                        "max_drawdown_cash": max_dd_cash,
                        "max_drawdown_percent": stats.max_drawdown_percent,
                        "expectancy": stats.expectancy,
                        "passed": passed,
                    }
                    capital_rows.append(row)
                    report["results"].append(row)

                    mark = "PASS" if passed else "FAIL"
                    print(
                        f"  spread={spread:>5.0f} | {mark:4s} | exec={stats.executed_trades:>4} "
                        f"skip={skip_pct:>6.2f}% PF={stats.profit_factor:>6.3f} "
                        f"DD={stats.max_drawdown_percent:>6.2f}% "
                        f"Net={stats.net_profit:>10.2f} {currency}"
                    )

            print(f"  => CAPITAL RESULT: {'PASS' if capital_pass else 'FAIL'}\n")
            if capital_pass and first_passing is None:
                first_passing = capital

        report["minimum_passing_capital"] = first_passing

        print("=== Sweep Decision ===")
        if first_passing is None:
            print("No tested capital passed all configured spread scenarios and risk gates.")
        else:
            print(f"Minimum tested capital passing all gates: {first_passing:,.0f} {currency}")

        out_dir = REPO_ROOT / "backtests" / "goldmicro_capital_sweep_results"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"goldmicro_capital_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        out_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved report: {out_file}")
        return 0
    finally:
        baseline.mt5.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
