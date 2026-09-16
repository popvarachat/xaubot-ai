"""Run PF/DD/cost robustness for GOLDmicro event-success candidates.

This runner is intentionally separate from run_goldmicro_strategy_oos.py because
event-target XGBoost predicts setup success probability, not BUY/SELL direction.
SMC remains the source of direction/SL/TP; the model is a fixed p>=0.50 entry gate.
No threshold tuning, live activation, order placement or promotion is performed.
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

from src.goldmicro_event_strategy_oos import (
    DEFAULT_MIN_SUCCESS_PROBABILITY,
    EVENT_MODEL_SEMANTICS,
    evaluate_event_strategy_sample,
)
from src.goldmicro_strategy_oos import (
    SampleStrategyResult,
    StrategyOOSThresholds,
    summarize_configuration,
)

COST_PROFILES = ("normal", "conservative")


def _failed_sample(sample: dict, cost_profile: str, error: Exception) -> SampleStrategyResult:
    return SampleStrategyResult(
        model_id=f"{sample.get('model_id', 'unknown')}::cost={cost_profile}",
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
    reasons: list[str] = []
    for cost in COST_PROFILES:
        if by_cost[cost]["status"] != "STRATEGY_OOS_PASS":
            reasons.append(f"{cost} cost profile failed multi-sample strategy gate")
    return {
        "base_model_id": base_id,
        "status": "EVENT_STRATEGY_OOS_ROBUST_PASS" if not reasons else "EVENT_STRATEGY_OOS_ROBUST_REJECT",
        "cost_profiles": by_cost,
        "robust_min_sample_pass_rate": min(by_cost[c]["sample_pass_rate"] for c in COST_PROFILES),
        "robust_min_median_pf": min(by_cost[c]["median_pf"] for c in COST_PROFILES),
        "robust_worst_dd_percent": max(by_cost[c]["worst_dd_percent"] for c in COST_PROFILES),
        "robust_min_median_expectancy_thb": min(
            by_cost[c]["median_expectancy_thb"] for c in COST_PROFILES
        ),
        "reasons": reasons if reasons else [
            "normal and conservative GOLDmicro event-strategy cost profiles both passed"
        ],
    }


def run_queue(
    queue_path: Path,
    *,
    thresholds: StrategyOOSThresholds,
    min_success_probability: float = DEFAULT_MIN_SUCCESS_PROBABILITY,
) -> Path:
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    configs = queue.get("configurations") or []
    report_dir = queue_path.parent
    snapshot = report_dir / "market_snapshot_m15_master.parquet"
    if not snapshot.exists():
        raise FileNotFoundError(f"master M15 snapshot missing: {snapshot}")
    if not configs:
        raise ValueError("event strategy queue is empty; nothing to evaluate")

    chronological_samples = sum(len(c.get("samples") or []) for c in configs)
    total_jobs = chronological_samples * len(COST_PROFILES)
    print("=== GOLDmicro Event Strategy OOS / PF-DD-Cost Robustness Gate ===")
    print(f"Queue          : {queue_path}")
    print(f"Model semantics: {EVENT_MODEL_SEMANTICS}")
    print("Direction      : causal SMC only")
    print(f"Entry gate      : event success probability >= {min_success_probability:.2f}")
    print("Threshold tune : DISABLED")
    print(f"Configurations : {len(configs)}")
    print(f"Chron samples  : {chronological_samples}")
    print(f"Cost profiles  : {', '.join(COST_PROFILES)}")
    print(f"OOS jobs       : {chronological_samples} x {len(COST_PROFILES)} = {total_jobs}")
    print(f"Capital        : {thresholds.initial_capital_thb:,.0f} THB")
    print(f"Risk cap       : {thresholds.risk_per_trade_percent:.2f}%")
    print(f"PF gate        : >= {thresholds.min_profit_factor:.2f}")
    print(f"DD gate        : <= {thresholds.max_drawdown_percent:.2f}%")
    print("Live model     : UNCHANGED")
    print("Promotion      : DISABLED")

    summaries: list[dict] = []
    job = 0
    for config in configs:
        base_id = str(config.get("base_model_id"))
        by_cost: dict[str, dict] = {}
        for cost_profile in COST_PROFILES:
            sample_results: list[SampleStrategyResult] = []
            for sample in config.get("samples") or []:
                job += 1
                print(f"[{job:03d}/{total_jobs:03d}] {sample.get('model_id')} | cost={cost_profile}")
                try:
                    result = evaluate_event_strategy_sample(
                        sample,
                        market_snapshot_path=snapshot,
                        cost_profile=cost_profile,
                        thresholds=thresholds,
                        min_success_probability=min_success_probability,
                    )
                except Exception as exc:
                    result = _failed_sample(sample, cost_profile, exc)
                sample_results.append(result)
                print(
                    f"  {result.status} | trades={result.trades} | PF={result.profit_factor:.3f} | "
                    f"DD={result.max_drawdown_percent:.2f}% | exp={result.expectancy_thb:.2f} THB | "
                    f"risk-skip={result.risk_skip_percent:.1f}% | model-blocks={result.model_blocks}"
                )
                if result.reasons and result.status != "STRATEGY_SAMPLE_PASS":
                    print(f"  reason: {'; '.join(result.reasons)}")

            by_cost[cost_profile] = summarize_configuration(
                f"{base_id}::{cost_profile}",
                sample_results,
                thresholds=thresholds,
            )
        summaries.append(_robust_summary(base_id, by_cost))

    summaries.sort(
        key=lambda item: (
            item["status"] == "EVENT_STRATEGY_OOS_ROBUST_PASS",
            item["robust_min_sample_pass_rate"],
            item["robust_min_median_pf"],
            item["robust_min_median_expectancy_thb"],
            -item["robust_worst_dd_percent"],
        ),
        reverse=True,
    )

    robust_pass = [x for x in summaries if x["status"] == "EVENT_STRATEGY_OOS_ROBUST_PASS"]
    report_path = report_dir / "event_strategy_oos_report.json"
    shadow_path = report_dir / "event_shadow_queue.json"
    report = {
        "generated_at": datetime.now().isoformat(),
        "batch_id": queue.get("batch_id"),
        "state": "EVENT_STRATEGY_OOS_COMPLETE",
        "model_semantics": EVENT_MODEL_SEMANTICS,
        "direction_source": "causal_smc",
        "event_probability_gate": min_success_probability,
        "threshold_tuning_performed": False,
        "thresholds": thresholds.__dict__,
        "cost_profiles": list(COST_PROFILES),
        "configurations": summaries,
        "robust_pass_count": len(robust_pass),
        "strategy_oos_executed": True,
        "promotion_performed": False,
        "warning": (
            "Research-only event meta-label strategy proxy. Event XGBoost gates SMC setups and is never "
            "interpreted as BUY/SELL. Chronological probes may overlap; independent forward shadow evidence, "
            "perturbation review and Human Gate remain mandatory."
        ),
    }
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    shadow_path.write_text(json.dumps({
        "batch_id": queue.get("batch_id"),
        "state": "AWAITING_SHADOW_AND_INDEPENDENT_AUDIT" if robust_pass else "NO_EVENT_STRATEGY_ROBUST_SURVIVORS",
        "model_semantics": EVENT_MODEL_SEMANTICS,
        "event_probability_gate": min_success_probability,
        "configurations": robust_pass,
        "promotion_performed": False,
    }, indent=2, default=str), encoding="utf-8")

    print("\n=== Event Strategy OOS Robustness Summary ===")
    for item in summaries:
        print(
            f"{item['status']:34s} minPass={item['robust_min_sample_pass_rate']:.0%} "
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
    ap.add_argument("--event-prob", type=float, default=DEFAULT_MIN_SUCCESS_PROBABILITY)
    args = ap.parse_args()
    thresholds = StrategyOOSThresholds(
        initial_capital_thb=args.capital,
        risk_per_trade_percent=args.risk,
        min_profit_factor=args.min_pf,
        max_drawdown_percent=args.max_dd,
        max_risk_skip_percent=args.max_skip,
        min_trades=args.min_trades,
    )
    run_queue(
        args.queue,
        thresholds=thresholds,
        min_success_probability=args.event_prob,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
