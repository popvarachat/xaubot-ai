"""Diagnose where V5 raw setup economics drift across chronological blocks.

Development-only.  This script does not alter the frozen V5 generator, tune any
threshold, run ML, place orders, or promote artifacts.  It replays the frozen V5
setup mechanics and decomposes already-inspected historical outcomes by simple
pre-existing provenance fields so the next research question can be chosen
without pretending the same history is confirmatory evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from statistics import median

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.goldmicro_causal_smc import GoldmicroCausalSMCAnalyzer
from src.goldmicro_smc_setup_v5 import GoldmicroSMCSetupV5, SMCSetupV5Config
from src.goldmicro_smc_setup_v5_baseline import SMCSetupV5BaselineThresholds, simulate_event


def _stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "median": None, "positive_rate": None}
    return {
        "n": len(values),
        "mean": float(sum(values) / len(values)),
        "median": float(median(values)),
        "positive_rate": float(sum(v > 0.0 for v in values) / len(values)),
    }


def _segment(block_rows: list[dict], key: str) -> dict[str, dict]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in block_rows:
        groups[str(row[key])].append(float(row["conservative_net_r"]))
    return {name: _stats(values) for name, values in sorted(groups.items())}


def run(report_path: Path) -> Path:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    snapshot_path = Path(str(((report.get("snapshots") or {}).get("m15") or {}).get("path") or ""))
    if not snapshot_path.exists():
        raise FileNotFoundError(f"M15 snapshot not found: {snapshot_path}")

    raw = pl.read_parquet(snapshot_path)
    causal = GoldmicroCausalSMCAnalyzer(swing_length=5).calculate_all(raw)
    config = SMCSetupV5Config()
    events = GoldmicroSMCSetupV5(config).generate(causal)
    thresholds = SMCSetupV5BaselineThresholds()
    outcomes = [
        simulate_event(raw, event, horizon_bars=thresholds.horizon_bars)
        for event in events
        if event.entry_index + 1 < len(raw)
    ]

    event_by_entry = {event.entry_index: event for event in events}
    rows = []
    for out in outcomes:
        event = event_by_entry[out.entry_index]
        rows.append({
            "entry_index": out.entry_index,
            "direction": out.direction,
            "archetype": out.archetype,
            "zone_type": out.zone_type,
            "exit_reason": out.exit_reason,
            "normal_net_r": out.normal_net_r,
            "conservative_net_r": out.conservative_net_r,
            "gross_r": out.gross_r,
            "setup_age_bars": event.setup_age_bars,
            "break_to_zone_bars": event.break_to_zone_bars,
            "zone_to_retest_bars": event.zone_to_retest_bars,
            "retest_depth_fraction": event.retest_depth_fraction,
        })

    n_rows = len(raw)
    block_reports = []
    for block_idx in range(thresholds.chronological_blocks):
        start = (n_rows * block_idx) // thresholds.chronological_blocks
        end = (n_rows * (block_idx + 1)) // thresholds.chronological_blocks
        block = [row for row in rows if start <= row["entry_index"] < end]
        block_reports.append({
            "block": block_idx + 1,
            "events": len(block),
            "overall_conservative": _stats([r["conservative_net_r"] for r in block]),
            "direction": _segment(block, "direction"),
            "archetype": _segment(block, "archetype"),
            "zone_type": _segment(block, "zone_type"),
            "exit_reason_counts": dict(Counter(r["exit_reason"] for r in block)),
            "median_setup_age_bars": float(median([r["setup_age_bars"] for r in block])) if block else None,
            "median_break_to_zone_bars": float(median([r["break_to_zone_bars"] for r in block])) if block else None,
            "median_zone_to_retest_bars": float(median([r["zone_to_retest_bars"] for r in block])) if block else None,
            "median_retest_depth_fraction": float(median([r["retest_depth_fraction"] for r in block])) if block else None,
        })

    def across(segment_name: str) -> dict[str, dict]:
        names = sorted({name for b in block_reports for name in b[segment_name]})
        result: dict[str, dict] = {}
        for name in names:
            segment_rows = [b[segment_name][name] for b in block_reports if name in b[segment_name]]
            means = [r["mean"] for r in segment_rows if r["mean"] is not None]
            ns = [r["n"] for r in segment_rows]
            result[name] = {
                "blocks": len(segment_rows),
                "positive_blocks": sum((r["mean"] or 0.0) > 0.0 for r in segment_rows),
                "median_n": int(median(ns)) if ns else 0,
                "median_mean_conservative_r": float(median(means)) if means else None,
                "min_mean_conservative_r": float(min(means)) if means else None,
                "max_mean_conservative_r": float(max(means)) if means else None,
            }
        return result

    payload = {
        "source_report": str(report_path),
        "source_snapshot": str(snapshot_path),
        "evidence_status": "DEVELOPMENT_HISTORY_ALREADY_INSPECTED",
        "threshold_tuning": "DISABLED",
        "generator_changes": "NONE",
        "pf_dd": "NOT_EVALUATED",
        "ml": "DISABLED",
        "blocks": block_reports,
        "across_blocks": {
            "direction": across("direction"),
            "archetype": across("archetype"),
            "zone_type": across("zone_type"),
        },
        "guard": (
            "Descriptive decomposition only. Do not convert the best inspected segment into a filter and then "
            "call this same history confirmatory. Any redesigned rule requires a predeclared protocol and fresh evidence."
        ),
    }

    out = report_path.parent / "smc_setup_v5_instability_diagnostic.json"
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print("=== GOLDmicro SMC Setup V5 Instability Diagnostic ===")
    print(f"Source report       : {report_path}")
    print(f"Events              : {len(rows)}")
    print("Evidence            : DEVELOPMENT / HISTORY ALREADY INSPECTED")
    print("Generator changes   : NONE")
    print("Threshold tuning    : DISABLED")
    print("PF/DD               : NOT EVALUATED")
    print("ML                  : DISABLED")
    print("\nPer block decomposition (conservative net R):")
    for b in block_reports:
        overall = b["overall_conservative"]
        print(f"\n  b{b['block']:02d} events={b['events']:3d} overall={overall['mean']:+.4f}R")
        for title, key in (("Direction", "direction"), ("Archetype", "archetype"), ("Zone", "zone_type")):
            parts = []
            for name, stat in b[key].items():
                parts.append(f"{name} n={stat['n']} mean={stat['mean']:+.4f}R")
            print(f"    {title:<10}: " + " | ".join(parts))
        print(f"    Exits      : {b['exit_reason_counts']}")

    print("\nAcross-block segment stability (conservative net R; descriptive only):")
    for title, key in (("Direction", "direction"), ("Archetype", "archetype"), ("Zone", "zone_type")):
        print(f"\n{title}:")
        for name, stat in payload["across_blocks"][key].items():
            print(
                f"  {name:20s} +blocks={stat['positive_blocks']}/{stat['blocks']} "
                f"medN={stat['median_n']:3d} medMean={stat['median_mean_conservative_r']:+.4f}R "
                f"range={stat['min_mean_conservative_r']:+.4f}..{stat['max_mean_conservative_r']:+.4f}R"
            )

    print(f"\nReport              : {out}")
    print("Interpretation guard:")
    print("  Diagnose instability only. Do not promote or post-hoc filter from these inspected segments.")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()
    run(args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
