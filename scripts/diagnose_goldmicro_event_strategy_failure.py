"""Summarize why calibrated event-strategy OOS candidates failed.

Consumes event_strategy_oos_report.json only.  This diagnostic does not change
thresholds, rerun PF/DD, optimize a gate, train a model, or promote anything.
Its purpose is to separate economic-edge failure from sample-size, drawdown,
risk-sizing, and execution-cost failure before deciding what research to do next.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median


def _reason_bucket(reason: str) -> str:
    text = reason.lower()
    if text.startswith("trades "):
        return "MIN_TRADES"
    if text.startswith("pf "):
        return "PROFIT_FACTOR"
    if text.startswith("dd ") or "hard dd breach" in text:
        return "DRAWDOWN"
    if text.startswith("risk skip "):
        return "RISK_SKIP"
    if text.startswith("expectancy "):
        return "EXPECTANCY"
    if "no event candidates cleared" in text:
        return "NO_ACCEPTED_EVENTS"
    if "evaluation error" in text:
        return "EVALUATION_ERROR"
    return "OTHER"


def _sample_rows(report: dict) -> list[dict]:
    rows: list[dict] = []
    for cfg in report.get("configurations") or []:
        base_id = str(cfg.get("base_model_id") or "unknown")
        for cost_name, cost_summary in (cfg.get("cost_profiles") or {}).items():
            for sample in cost_summary.get("samples") or []:
                row = dict(sample)
                row["base_model_id"] = base_id
                row["cost_profile"] = cost_name
                rows.append(row)
    return rows


def run(report_path: Path) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = _sample_rows(report)
    if not rows:
        raise ValueError("strategy report contains no sample rows")

    reasons = Counter()
    for row in rows:
        for reason in row.get("reasons") or []:
            reasons[_reason_bucket(str(reason))] += 1

    print("=== GOLDmicro Event Strategy Failure Diagnostic ===")
    print(f"Report          : {report_path}")
    print(f"Configurations  : {len(report.get('configurations') or [])}")
    print(f"Sample results  : {len(rows)}")
    print(f"Strategy passes : {sum(r.get('status') == 'STRATEGY_SAMPLE_PASS' for r in rows)}/{len(rows)}")
    print("Threshold tune  : DISABLED")
    print("Promotion       : DISABLED")
    print("\nFailure buckets:")
    for name in ("MIN_TRADES", "PROFIT_FACTOR", "EXPECTANCY", "DRAWDOWN", "RISK_SKIP", "NO_ACCEPTED_EVENTS", "EVALUATION_ERROR", "OTHER"):
        print(f"  {name:18s} {reasons.get(name, 0):3d}")

    print("\nPer configuration / cost profile:")
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["base_model_id"], row["cost_profile"]), []).append(row)
    for (base_id, cost_name), group in grouped.items():
        trades = [int(r.get("trades") or 0) for r in group]
        pfs = [float(r.get("profit_factor") or 0.0) for r in group]
        exps = [float(r.get("expectancy_thb") or 0.0) for r in group]
        dds = [float(r.get("max_drawdown_percent") or 0.0) for r in group]
        sample_passes = sum(r.get("status") == "STRATEGY_SAMPLE_PASS" for r in group)
        positive_exp = sum(float(r.get("expectancy_thb") or 0.0) > 0 for r in group)
        pf_ge_1 = sum(float(r.get("profit_factor") or 0.0) >= 1.0 for r in group)
        pf_ge_gate = sum(float(r.get("profit_factor") or 0.0) >= 1.30 for r in group)
        print(
            f"{cost_name:12s} pass={sample_passes}/{len(group)} "
            f"trades med/min/max={median(trades):.0f}/{min(trades)}/{max(trades)} "
            f"PF med/max={median(pfs):.3f}/{max(pfs):.3f} "
            f"PF>=1={pf_ge_1}/{len(group)} PF>=1.30={pf_ge_gate}/{len(group)} "
            f"exp>0={positive_exp}/{len(group)} DDmax={max(dds):.2f}% "
            f"{base_id}"
        )

    print("\nInterpretation guard:")
    print("  This report diagnoses failure modes only. Do not lower PF/min-trade gates or tune")
    print("  the probability/economic gate from these realized strategy returns.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, type=Path)
    args = ap.parse_args()
    run(args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
