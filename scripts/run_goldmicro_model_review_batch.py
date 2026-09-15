"""Batch-review Champion + many Challenger metrics for GOLDmicro lifecycle research.

Input JSON example:
{
  "champion": { ... ModelWindowMetrics fields ... },
  "challengers": [{...}, {...}]
}

The script is read-only with respect to models and trading. It never promotes,
replaces, or reloads a live model. It only emits an auditable decision report.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from src.model_lifecycle import (
    LifecycleThresholds,
    ModelWindowMetrics,
    evaluate_champion,
    rank_challengers,
)


def _metrics(payload: dict) -> ModelWindowMetrics:
    return ModelWindowMetrics(
        model_id=str(payload["model_id"]),
        trades=int(payload["trades"]),
        profit_factor=float(payload["profit_factor"]),
        max_drawdown_pct=float(payload["max_drawdown_pct"]),
        expectancy=float(payload.get("expectancy", 0.0)),
        skip_pct=float(payload.get("skip_pct", 0.0)),
        execution_cost_points=float(payload.get("execution_cost_points", 0.0)),
        stable_windows=int(payload.get("stable_windows", 1)),
        shadow_trades=int(payload.get("shadow_trades", 0)),
        shadow_days=int(payload.get("shadow_days", 0)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate many GOLDmicro model candidates in one batch")
    parser.add_argument("--input", required=True, type=Path, help="Champion/challenger metrics JSON")
    parser.add_argument("--output", type=Path, help="Optional output JSON path")
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    champion = _metrics(payload["champion"])
    challengers = [_metrics(x) for x in payload.get("challengers", [])]
    thresholds = LifecycleThresholds()

    champion_decision = evaluate_champion(champion, thresholds)
    ranked = rank_challengers(challengers, champion, thresholds)

    print("\n=== GOLDmicro MODEL LIFECYCLE BATCH REVIEW (READ ONLY) ===")
    print(f"Champion : {champion.model_id}")
    print(f"State    : {champion_decision.state}")
    print(f"Entries  : {'ALLOW' if champion_decision.allow_new_entries else 'HOLD'}")
    print(f"Retrain  : {'YES' if champion_decision.retrain_challenger else 'NO'}")
    if champion_decision.reasons:
        print("Reasons  : " + "; ".join(champion_decision.reasons))

    print("\nChallenger ranking")
    print(f"{'MODEL':28s} {'PF':>7s} {'DD%':>7s} {'TRADES':>8s} {'SHADOW':>8s} {'STATE':>28s}")
    print("-" * 94)
    for metrics, decision in ranked:
        print(
            f"{metrics.model_id[:28]:28s} {metrics.profit_factor:7.3f} "
            f"{metrics.max_drawdown_pct:7.2f} {metrics.trades:8d} "
            f"{metrics.shadow_trades:8d} {decision.state:>28s}"
        )
        if decision.reasons:
            print("  -> " + "; ".join(decision.reasons))

    report = {
        "generated_at": datetime.now().isoformat(),
        "champion": champion.__dict__,
        "champion_decision": champion_decision.to_dict(),
        "challengers": [
            {"metrics": metrics.__dict__, "decision": decision.to_dict()}
            for metrics, decision in ranked
        ],
        "note": "Eligibility is not promotion. Promotion requires Human Gate and a flat-position activation boundary.",
    }

    output = args.output
    if output is None:
        out_dir = Path("backtests/goldmicro_model_lifecycle_results")
        out_dir.mkdir(parents=True, exist_ok=True)
        output = out_dir / f"model_lifecycle_review_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved review report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
