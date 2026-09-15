"""Run GOLDmicro PF/DD/cost strategy OOS validation for a whole shortlist.

Consumes ``strategy_oos_queue.json`` from the 24 x N multi-sample pre-screen.
Every shortlisted configuration is evaluated across all retained chronological
samples in one command.  No orders are sent and no model is promoted.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.goldmicro_strategy_oos import (
    SampleStrategyResult,
    StrategyOOSThresholds,
    evaluate_strategy_sample,
    summarize_configuration,
)


def _failed_sample(sample: dict, error: Exception) -> SampleStrategyResult:
    return SampleStrategyResult(
        model_id=str(sample.get("model_id", "unknown")),
        sample_index=int(sample.get("sample_index") or 0),
        trades=0,
        wins=0,
        losses=0,
        net_pnl_thb=0.0,
        profit_factor=0.0,
        max_drawdown_percent=100.0,
        expectancy_thb=0.0,
        risk_skips=0,
        risk_skip_percent=100.0,
        model_blocks=0,
        average_lot=0.0,
        status="STRATEGY_SAMPLE_REJECT",
        reasons=(f"evaluation error: {error}",),
    )


def run_queue(queue_path: Path, *, thresholds: StrategyOOSThresholds) -> Path:
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    configs = queue.get("configurations") or []
    if not configs:
        raise ValueError(f"no shortlisted configurations in {queue_path}")

    total_samples = sum(len(c.get("samples") or []) for c in configs)
    print("=== GOLDmicro Strategy OOS / PF-DD-Cost Gate ===")
    print(f"Queue          : {queue_path}")
    print(f"Configurations : {len(configs)}")
    print(f"Sample jobs    : {total_samples}")
    print(f"Capital        : {thresholds.initial_capital_thb:,.0f} THB")
    print(f"Risk cap       : {thresholds.risk_per_trade_percent:.2f}%")
    print(f"PF gate        : >= {thresholds.min_profit_factor:.2f}")
    print(f"DD gate        : <= {thresholds.max_drawdown_percent:.2f}%")
    print("Promotion      : DISABLED")

    summaries = []
    job = 0
    for config in configs:
        base_id = str(config.get("base_model_id"))
        sample_results = []
        for sample in config.get("samples") or []:
            job += 1
            print(f"[{job:02d}/{total_samples:02d}] {sample.get('model_id')}")
            try:
                result = evaluate_strategy_sample(sample, thresholds=thresholds)
            except Exception as exc:
                result = _failed_sample(sample, exc)
            sample_results.append(result)
            print(
                f"  {result.status} | trades={result.trades} | PF={result.profit_factor:.3f} | "
                f"DD={result.max_drawdown_percent:.2f}% | exp={result.expectancy_thb:.2f} THB | "
                f"risk-skip={result.risk_skip_percent:.1f}%"
            )
        summaries.append(
            summarize_configuration(base_id, sample_results, thresholds=thresholds)
        )

    summaries.sort(
        key=lambda x: (
            x["status"] == "STRATEGY_OOS_PASS",
            x["sample_pass_rate"],
            x["median_pf"],
            x["median_expectancy_thb"],
            -x["worst_dd_percent"],
        ),
        reverse=True,
    )

    report_dir = queue_path.parent
    report_path = report_dir / "strategy_oos_report.json"
    shadow_path = report_dir / "shadow_queue.json"
    report = {
        "generated_at": datetime.now().isoformat(),
        "batch_id": queue.get("batch_id"),
        "state": "STRATEGY_OOS_COMPLETE",
        "thresholds": thresholds.__dict__,
        "configurations": summaries,
        "pass_count": sum(1 for x in summaries if x["status"] == "STRATEGY_OOS_PASS"),
        "promotion_performed": False,
        "warning": (
            "This is a research strategy proxy using causal candidate artifacts and broker-correct "
            "costs. Walk-forward/perturbation/shadow evidence and Human Gate remain mandatory."
        ),
    }
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    shadow = {
        "batch_id": queue.get("batch_id"),
        "state": "AWAITING_SHADOW_AND_INDEPENDENT_AUDIT",
        "configurations": [x for x in summaries if x["status"] == "STRATEGY_OOS_PASS"],
        "promotion_performed": False,
    }
    shadow_path.write_text(json.dumps(shadow, indent=2, default=str), encoding="utf-8")

    print("\n=== Strategy OOS Summary ===")
    for item in summaries:
        print(
            f"{item['status']:20s} pass={item['sample_pass_count']}/{item['sample_count']} "
            f"medianPF={item['median_pf']:.3f} worstDD={item['worst_dd_percent']:.2f}% "
            f"{item['base_model_id']}"
        )
    print(f"\nReport         : {report_path}")
    print(f"Shadow queue   : {shadow_path}")
    print("Live model     : UNCHANGED")
    print("Promotion      : DISABLED")
    return report_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True, type=Path)
    ap.add_argument("--capital", type=float, default=20000.0)
    ap.add_argument("--risk", type=float, default=1.0)
    ap.add_argument("--min-pf", type=float, default=1.30)
    ap.add_argument("--max-dd", type=float, default=10.0)
    ap.add_argument("--max-skip", type=float, default=20.0)
    ap.add_argument("--min-trades", type=int, default=30)
    args = ap.parse_args()
    thresholds = StrategyOOSThresholds(
        initial_capital_thb=args.capital,
        risk_per_trade_percent=args.risk,
        min_profit_factor=args.min_pf,
        max_drawdown_percent=args.max_dd,
        max_risk_skip_percent=args.max_skip,
        min_trades=args.min_trades,
    )
    run_queue(args.queue, thresholds=thresholds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
