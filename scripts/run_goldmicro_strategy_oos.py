"""Run GOLDmicro PF/DD/cost strategy OOS validation for a whole shortlist.

Consumes ``strategy_oos_queue.json`` from the 24 x N multi-sample pre-screen.
Every shortlisted predictive configuration is evaluated across all retained
chronological samples under BOTH normal and conservative execution-cost profiles
in one command. A configuration reaches the shadow queue only when both cost
profiles pass their multi-sample gates.

An empty shortlist is a valid research outcome, not an execution failure. In
that case this runner writes explicit NO_ELIGIBLE_CONFIGURATIONS evidence and
returns success without fabricating PF/DD results or weakening the AUC gate.

No orders are sent and no model is promoted.
"""
from __future__ import annotations

import argparse
import json
import re
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

COST_PROFILES = ("normal", "conservative")


def _sample_for_cost(sample: dict, cost_profile: str) -> dict:
    """Reuse one trained model artifact under a different execution-cost scenario."""
    if cost_profile not in COST_PROFILES:
        raise ValueError(f"unknown cost profile: {cost_profile}")
    row = dict(sample)
    model_id = str(row.get("model_id", "unknown"))
    pattern = r"-(normal|conservative)-s(\d+)$"
    if re.search(pattern, model_id) is None:
        raise ValueError(f"cannot rewrite cost profile in model id: {model_id}")
    row["model_id"] = re.sub(pattern, rf"-{cost_profile}-s\2", model_id)
    row["evaluation_cost_profile"] = cost_profile
    return row


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


def _robust_summary(base_id: str, by_cost: dict[str, dict]) -> dict:
    reasons = []
    for cost in COST_PROFILES:
        summary = by_cost[cost]
        if summary["status"] != "STRATEGY_OOS_PASS":
            reasons.append(f"{cost} cost profile failed multi-sample strategy gate")
    return {
        "base_model_id": base_id,
        "status": "STRATEGY_OOS_ROBUST_PASS" if not reasons else "STRATEGY_OOS_ROBUST_REJECT",
        "cost_profiles": by_cost,
        "robust_min_sample_pass_rate": min(
            by_cost[c]["sample_pass_rate"] for c in COST_PROFILES
        ),
        "robust_min_median_pf": min(by_cost[c]["median_pf"] for c in COST_PROFILES),
        "robust_worst_dd_percent": max(
            by_cost[c]["worst_dd_percent"] for c in COST_PROFILES
        ),
        "robust_min_median_expectancy_thb": min(
            by_cost[c]["median_expectancy_thb"] for c in COST_PROFILES
        ),
        "reasons": reasons if reasons else [
            "normal and conservative GOLDmicro cost profiles both passed"
        ],
    }


def _write_no_eligible_outputs(
    queue_path: Path,
    queue: dict,
    thresholds: StrategyOOSThresholds,
) -> Path:
    """Persist an explicit terminal result when the AUC pre-screen yields zero models."""
    report_dir = queue_path.parent
    report_path = report_dir / "strategy_oos_report.json"
    shadow_path = report_dir / "shadow_queue.json"
    reason = (
        "No configuration passed the upstream multi-sample AUC/generalization pre-screen. "
        "Strategy PF/DD/cost evaluation was intentionally not run. Do not lower the predictive "
        "gate merely to manufacture a shortlist; redesign or re-align the predictive target/features first."
    )
    report = {
        "generated_at": datetime.now().isoformat(),
        "batch_id": queue.get("batch_id"),
        "state": "NO_ELIGIBLE_CONFIGURATIONS",
        "thresholds": thresholds.__dict__,
        "cost_profiles": list(COST_PROFILES),
        "configurations": [],
        "robust_pass_count": 0,
        "strategy_oos_executed": False,
        "promotion_performed": False,
        "reason": reason,
    }
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    shadow = {
        "batch_id": queue.get("batch_id"),
        "state": "NO_ELIGIBLE_CONFIGURATIONS",
        "configurations": [],
        "promotion_performed": False,
        "reason": reason,
    }
    shadow_path.write_text(json.dumps(shadow, indent=2, default=str), encoding="utf-8")

    print("=== GOLDmicro Strategy OOS / PF-DD-Cost Robustness Gate ===")
    print(f"Queue          : {queue_path}")
    print("Configurations : 0")
    print("OOS jobs       : 0")
    print("State          : NO_ELIGIBLE_CONFIGURATIONS")
    print("Action         : PF/DD/cost gate skipped; predictive research must be redesigned first")
    print(f"Report         : {report_path}")
    print(f"Shadow queue   : {shadow_path}")
    print("Live model     : UNCHANGED")
    print("Promotion      : DISABLED")
    return report_path


def run_queue(queue_path: Path, *, thresholds: StrategyOOSThresholds) -> Path:
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    configs = queue.get("configurations") or []
    if not configs:
        return _write_no_eligible_outputs(queue_path, queue, thresholds)

    chronological_samples = sum(len(c.get("samples") or []) for c in configs)
    total_jobs = chronological_samples * len(COST_PROFILES)
    print("=== GOLDmicro Strategy OOS / PF-DD-Cost Robustness Gate ===")
    print(f"Queue          : {queue_path}")
    print(f"Configurations : {len(configs)}")
    print(f"Chron samples  : {chronological_samples}")
    print(f"Cost profiles  : {', '.join(COST_PROFILES)}")
    print(f"OOS jobs       : {chronological_samples} x {len(COST_PROFILES)} = {total_jobs}")
    print(f"Capital        : {thresholds.initial_capital_thb:,.0f} THB")
    print(f"Risk cap       : {thresholds.risk_per_trade_percent:.2f}%")
    print(f"PF gate        : >= {thresholds.min_profit_factor:.2f}")
    print(f"DD gate        : <= {thresholds.max_drawdown_percent:.2f}%")
    print("Promotion      : DISABLED")

    summaries = []
    job = 0
    for config in configs:
        base_id = str(config.get("base_model_id"))
        by_cost: dict[str, dict] = {}
        for cost_profile in COST_PROFILES:
            sample_results = []
            for raw_sample in config.get("samples") or []:
                job += 1
                try:
                    sample = _sample_for_cost(raw_sample, cost_profile)
                except Exception as exc:
                    sample = dict(raw_sample)
                    result = _failed_sample(sample, exc)
                    sample_results.append(result)
                    print(f"[{job:03d}/{total_jobs:03d}] {raw_sample.get('model_id')} | cost={cost_profile}")
                    print(f"  {result.status} | {result.reasons[0]}")
                    continue

                print(
                    f"[{job:03d}/{total_jobs:03d}] {raw_sample.get('model_id')} | "
                    f"cost={cost_profile}"
                )
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

            by_cost[cost_profile] = summarize_configuration(
                f"{base_id}::{cost_profile}",
                sample_results,
                thresholds=thresholds,
            )
        summaries.append(_robust_summary(base_id, by_cost))

    summaries.sort(
        key=lambda x: (
            x["status"] == "STRATEGY_OOS_ROBUST_PASS",
            x["robust_min_sample_pass_rate"],
            x["robust_min_median_pf"],
            x["robust_min_median_expectancy_thb"],
            -x["robust_worst_dd_percent"],
        ),
        reverse=True,
    )

    report_dir = queue_path.parent
    report_path = report_dir / "strategy_oos_report.json"
    shadow_path = report_dir / "shadow_queue.json"
    robust_pass = [x for x in summaries if x["status"] == "STRATEGY_OOS_ROBUST_PASS"]
    report = {
        "generated_at": datetime.now().isoformat(),
        "batch_id": queue.get("batch_id"),
        "state": "STRATEGY_OOS_COMPLETE",
        "thresholds": thresholds.__dict__,
        "cost_profiles": list(COST_PROFILES),
        "configurations": summaries,
        "robust_pass_count": len(robust_pass),
        "strategy_oos_executed": True,
        "promotion_performed": False,
        "warning": (
            "This is a research strategy proxy using causal candidate artifacts and broker-correct "
            "normal/conservative costs. Chronological probes may overlap; independent forward shadow "
            "evidence, perturbation review and Human Gate remain mandatory."
        ),
    }
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    shadow = {
        "batch_id": queue.get("batch_id"),
        "state": "AWAITING_SHADOW_AND_INDEPENDENT_AUDIT",
        "configurations": robust_pass,
        "promotion_performed": False,
    }
    shadow_path.write_text(json.dumps(shadow, indent=2, default=str), encoding="utf-8")

    print("\n=== Strategy OOS Robustness Summary ===")
    for item in summaries:
        print(
            f"{item['status']:28s} minPass={item['robust_min_sample_pass_rate']:.0%} "
            f"minMedianPF={item['robust_min_median_pf']:.3f} "
            f"worstDD={item['robust_worst_dd_percent']:.2f}% "
            f"{item['base_model_id']}"
        )
    print(f"\nRobust pass    : {len(robust_pass)}/{len(summaries)} configurations")
    print(f"Report         : {report_path}")
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
