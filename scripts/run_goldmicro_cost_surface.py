"""Batch GOLDmicro cost-surface analysis (read only).

Loads the broker calibration and legacy trade workbook once, then evaluates a
grid of spreads, adverse slippage assumptions, risk levels, and starting
capitals.  The goal is to find the execution-cost envelope where the current
legacy strategy still satisfies PF/DD/skip validation gates.

No order is placed, modified, or closed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import types
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
SWEEP_MODULE = REPO_ROOT / "scripts" / "run_goldmicro_capital_sweep_isolated.py"

DEFAULT_CAPITALS = "10000,11000,12000,13000,14000,15000,16000,17000,18000,19000,20000,25000,30000,50000,75000"
DEFAULT_RISKS = "0.5,0.75,1.0"
DEFAULT_SPREADS = "40,50,55,60,70,100"
DEFAULT_SLIPPAGES = "0,1,3,6,9,12,15,18,21,25,35,50,63,103"
REFERENCE_CAPITALS = (12000.0, 15000.0, 20000.0, 30000.0, 50000.0, 75000.0)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Avoid running upstream src/__init__.py and its full ML dependency chain.
if "src" not in sys.modules:
    pkg = types.ModuleType("src")
    pkg.__path__ = [str(SRC_DIR)]
    pkg.__package__ = "src"
    sys.modules["src"] = pkg


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _csv_positive(value: str) -> list[float]:
    vals = [float(x.strip()) for x in value.split(",") if x.strip()]
    if not vals or any(x < 0 for x in vals):
        raise argparse.ArgumentTypeError("Expected comma-separated non-negative numbers")
    return vals


def _fmt_number(x: float) -> str:
    return f"{x:g}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run GOLDmicro spread/slippage/risk/capital cost surface")
    parser.add_argument("--input", type=Path, help="Optional legacy XLSX input")
    parser.add_argument("--capitals", default=DEFAULT_CAPITALS)
    parser.add_argument("--risks", default=DEFAULT_RISKS)
    parser.add_argument("--spreads", default=DEFAULT_SPREADS)
    parser.add_argument("--slippages", default=DEFAULT_SLIPPAGES, help="Adverse slippage points per side")
    parser.add_argument("--min-pf", type=float, default=1.30)
    parser.add_argument("--max-dd", type=float, default=10.0)
    parser.add_argument("--max-skip", type=float, default=20.0)
    args = parser.parse_args()

    capitals = _csv_positive(args.capitals)
    risks = _csv_positive(args.risks)
    spreads = _csv_positive(args.spreads)
    slippages = _csv_positive(args.slippages)
    if any(x <= 0 for x in capitals) or any(x <= 0 for x in risks):
        raise SystemExit("Capitals and risks must be positive")

    sweep = _load(SWEEP_MODULE, "goldmicro_cost_surface_sweep")
    baseline = sweep._load_baseline_module()
    workbook = (args.input if args.input else baseline._latest_default_workbook(REPO_ROOT)).resolve()
    if not workbook.exists():
        raise SystemExit(f"Input workbook not found: {workbook}")

    try:
        profile, _balance, currency = baseline._calibrate_profile(baseline.SYMBOL)
        trade_sheets = sweep._read_trade_sheets_deterministic(workbook, baseline)

        print("\n=== GOLDmicro Cost Surface (READ ONLY) ===")
        print(f"Workbook   : {workbook}")
        print(f"Currency   : {currency}")
        print(f"Calibration: {profile.cash_per_price_unit_per_lot:.4f} {currency} per +1.00 price / 1.00 lot")
        print(f"Spreads    : {','.join(_fmt_number(x) for x in spreads)}")
        print(f"Slippage   : {','.join(_fmt_number(x) for x in slippages)} points/side")
        print(f"Risks      : {','.join(_fmt_number(x) for x in risks)}%")
        print(f"Capitals   : {','.join(_fmt_number(x) for x in capitals)} {currency}")
        print(f"Gate       : PF >= {args.min_pf:.2f}, DD <= {args.max_dd:.2f}%, Skip <= {args.max_skip:.2f}%")
        print("NOTE: constant adverse slippage overlays are stress assumptions, not a forecast of every fill.\n")

        results: list[dict] = []
        state_by_key: dict[tuple[float, float, float, float], dict] = {}

        total_combos = len(spreads) * len(slippages) * len(risks)
        combo_idx = 0
        for spread in spreads:
            for slip in slippages:
                for risk in risks:
                    combo_idx += 1
                    minimum = None
                    minimum_metrics = None
                    for capital in capitals:
                        # A legacy workbook may theoretically contain more than one trade sheet.
                        all_pass = True
                        aggregate = None
                        for _sheet, trades in trade_sheets:
                            model = baseline.GoldmicroCostModel(
                                profile,
                                baseline.BacktestCostConfig(
                                    spread_points=spread,
                                    slippage_points=slip,
                                    commission_per_lot_round_turn=0.0,
                                    swap_per_lot=0.0,
                                ),
                            )
                            stats = baseline.replay_legacy_trades(
                                trades,
                                profile=profile,
                                cost_model=model,
                                initial_capital=capital,
                                risk_per_trade_percent=risk,
                                win_rate=0.55,
                                reward_risk_ratio=2.0,
                            )
                            skip_pct = (stats.skipped_trades / stats.total_trades * 100.0) if stats.total_trades else 100.0
                            passed = (
                                stats.executed_trades > 0
                                and stats.profit_factor >= args.min_pf
                                and stats.max_drawdown_percent <= args.max_dd
                                and skip_pct <= args.max_skip
                            )
                            all_pass = all_pass and passed
                            aggregate = {
                                "capital": capital,
                                "executed_trades": stats.executed_trades,
                                "skip_percent": skip_pct,
                                "profit_factor": stats.profit_factor,
                                "max_drawdown_percent": stats.max_drawdown_percent,
                                "net_profit": stats.net_profit,
                                "passed": passed,
                            }
                            state_by_key[(spread, slip, risk, capital)] = aggregate
                        if all_pass and minimum is None:
                            minimum = capital
                            minimum_metrics = aggregate

                    row = {
                        "spread_points": spread,
                        "slippage_points_per_side": slip,
                        "risk_percent": risk,
                        "minimum_passing_capital": minimum,
                        "minimum_metrics": minimum_metrics,
                    }
                    results.append(row)
                    label = f"{minimum:,.0f} {currency}" if minimum is not None else "NO_PASS"
                    print(
                        f"[{combo_idx:>3}/{total_combos}] spread={spread:>5g} slip={slip:>5g}/side "
                        f"risk={risk:>4g}% -> {label}"
                    )

        # Build a compact frontier: for each spread/risk/reference capital,
        # what is the largest tested slippage per side that still passes?
        frontier: list[dict] = []
        available_caps = set(capitals)
        refs = [c for c in REFERENCE_CAPITALS if c in available_caps]
        for capital in refs:
            for spread in spreads:
                for risk in risks:
                    passing_slips = [
                        slip
                        for slip in slippages
                        if state_by_key.get((spread, slip, risk, capital), {}).get("passed")
                    ]
                    frontier.append(
                        {
                            "capital": capital,
                            "spread_points": spread,
                            "risk_percent": risk,
                            "max_tested_slippage_points_per_side_passing": max(passing_slips) if passing_slips else None,
                        }
                    )

        print("\n=== Cost Frontier: maximum tested slippage/side that still passes ===")
        for capital in refs:
            print(f"\nCapital {capital:,.0f} {currency}")
            print(f"{'SPREAD':>7s} " + " ".join(f"RISK {r:g}%".rjust(12) for r in risks))
            print("-" * (8 + 13 * len(risks)))
            for spread in spreads:
                cells = []
                for risk in risks:
                    row = next(
                        x for x in frontier
                        if x["capital"] == capital and x["spread_points"] == spread and x["risk_percent"] == risk
                    )
                    val = row["max_tested_slippage_points_per_side_passing"]
                    cells.append((f"{val:g} pt" if val is not None else "NO_PASS").rjust(12))
                print(f"{spread:>7g} " + " ".join(cells))

        out_dir = REPO_ROOT / "backtests" / "goldmicro_cost_surface_results"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"goldmicro_cost_surface_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        payload = {
            "generated_at": datetime.now().isoformat(),
            "input_workbook": str(workbook),
            "currency": currency,
            "cash_per_price_unit_per_lot": profile.cash_per_price_unit_per_lot,
            "criteria": {
                "min_profit_factor": args.min_pf,
                "max_drawdown_percent": args.max_dd,
                "max_skip_percent": args.max_skip,
            },
            "capitals": capitals,
            "risks": risks,
            "spreads": spreads,
            "slippages_points_per_side": slippages,
            "results": results,
            "frontier": frontier,
            "limitations": [
                "Slippage overlays are constant adverse scenario values, not per-trade empirical draws.",
                "Entry-side requested-price evidence is sparse in the inspected MT5 history.",
                "Commission/fee observed in the inspected deal history were zero.",
                "Symbol swap metadata exists but no historical swap charge was observed in the inspected deal history.",
                "Legacy trade trigger timing is not tick-accurate.",
                "Historical USD/THB conversion is not modeled.",
            ],
        }
        out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        print(f"\nSaved cost-surface report: {out_file}")
        return 0
    finally:
        baseline.mt5.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
