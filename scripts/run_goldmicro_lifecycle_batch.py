"""Generate a non-live GOLDmicro lifecycle batch plan and evaluate review inputs.

One command creates many challenger specs, optionally evaluates champion/challenger
metrics JSON, and optionally summarizes shadow-trade JSON. It never trains into
active paths and never promotes a model.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.challenger_batch import build_challenger_specs, write_batch_plan, assert_isolated_output
from src.model_lifecycle import ModelWindowMetrics, rank_challengers, evaluate_champion
from src.shadow_evaluator import ShadowTrade, summarize_many_shadow_models
from src.model_registry import ensure_lifecycle_dirs


def _metric(row: dict) -> ModelWindowMetrics:
    return ModelWindowMetrics(
        model_id=row["model_id"],
        trades=int(row["trades"]),
        profit_factor=float(row["profit_factor"]),
        max_drawdown_pct=float(row["max_drawdown_pct"]),
        expectancy=float(row["expectancy"]),
        skip_pct=float(row.get("skip_pct", 0.0)),
        execution_cost_points=float(row.get("execution_cost_points", 0.0)),
        stable_windows=int(row.get("stable_windows", 1)),
        shadow_trades=int(row.get("shadow_trades", 0)),
        shadow_days=int(row.get("shadow_days", 0)),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=96, help="max challenger specs in this batch")
    ap.add_argument("--metrics-json", help="optional JSON with champion and challengers arrays")
    ap.add_argument("--shadow-json", help="optional JSON array of shadow trade rows")
    ap.add_argument("--starting-equity", type=float, default=20000.0)
    args = ap.parse_args()

    paths = ensure_lifecycle_dirs(ROOT / "models")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = paths.reports / f"lifecycle_{stamp}"
    report_dir.mkdir(parents=True, exist_ok=True)

    specs = build_challenger_specs(root=ROOT / "models", limit=args.limit)
    for spec in specs:
        assert_isolated_output(spec, ROOT / "models")
    plan_path = write_batch_plan(specs, report_dir / "challenger_batch_plan.json")

    output = {
        "generated_at": datetime.now().isoformat(),
        "mode": "READ_ONLY_NON_LIVE",
        "challenger_count": len(specs),
        "plan_path": str(plan_path),
        "champion_review": None,
        "challenger_ranking": [],
        "shadow_summaries": [],
        "promotion_performed": False,
    }

    if args.metrics_json:
        data = json.loads(Path(args.metrics_json).read_text(encoding="utf-8"))
        champion = _metric(data["champion"])
        output["champion_review"] = evaluate_champion(champion).to_dict()
        challengers = [_metric(x) for x in data.get("challengers", [])]
        for metrics, decision in rank_challengers(challengers, champion):
            output["challenger_ranking"].append({
                "metrics": {
                    "model_id": metrics.model_id,
                    "trades": metrics.trades,
                    "profit_factor": metrics.profit_factor,
                    "max_drawdown_pct": metrics.max_drawdown_pct,
                    "expectancy": metrics.expectancy,
                    "skip_pct": metrics.skip_pct,
                    "execution_cost_points": metrics.execution_cost_points,
                    "stable_windows": metrics.stable_windows,
                    "shadow_trades": metrics.shadow_trades,
                    "shadow_days": metrics.shadow_days,
                },
                "decision": decision.to_dict(),
            })

    if args.shadow_json:
        rows = json.loads(Path(args.shadow_json).read_text(encoding="utf-8"))
        trades = [ShadowTrade(**row) for row in rows]
        output["shadow_summaries"] = [
            s.to_dict() for s in summarize_many_shadow_models(trades, starting_equity=args.starting_equity)
        ]

    out_path = report_dir / "lifecycle_batch_report.json"
    out_path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")

    print("=== GOLDmicro Lifecycle Batch (NON-LIVE) ===")
    print(f"Challenger specs : {len(specs)}")
    print(f"Plan             : {plan_path}")
    print(f"Report           : {out_path}")
    print("Promotion        : NEVER automatic; Human Gate required")
    print("Active model     : UNCHANGED")


if __name__ == "__main__":
    main()
