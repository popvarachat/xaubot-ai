"""Diagnose raw causal-SMC setup economics after V4 structural failure.

This is a development-only diagnostic. It does not train a model, tune a gate,
run PF/DD, place orders, or alter live artifacts. It reads the V4 report and the
already-produced event-edge parquet files, reconstructs each job's untouched-OOS
event slice, de-duplicates repeated setup events within each chronological probe,
and summarizes realized gross R by simple setup structure.

Five chronological probes overlap in market time; they are NOT independent trials.
Results are for research direction only and must not be used as confirmatory OOS.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

import numpy as np
import polars as pl

from src.goldmicro_event_edge_target import EDGE_TARGET_COLUMN
from src.goldmicro_strategy_oos import _session


def _reward_r(row: dict) -> float:
    entry = float(row["event_entry"])
    stop = float(row["event_stop_loss"])
    target = float(row["event_take_profit"])
    risk = abs(entry - stop)
    return abs(target - entry) / risk if risk > 0 else float("nan")


def _confidence_bucket(value: float) -> str:
    if value < 0.65:
        return "conf<0.65"
    if value < 0.75:
        return "0.65-0.75"
    if value < 0.85:
        return "0.75-0.85"
    return "conf>=0.85"


def _reward_bucket(value: float) -> str:
    if value < 1.0:
        return "rewardR<1"
    if value < 1.5:
        return "1-1.5R"
    if value < 2.0:
        return "1.5-2R"
    return "rewardR>=2"


def _stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean_r": None, "median_r": None, "positive_rate": None,
                "p10_r": None, "p90_r": None}
    a = np.asarray(values, dtype=float)
    return {
        "n": int(a.size),
        "mean_r": float(np.mean(a)),
        "median_r": float(np.median(a)),
        "positive_rate": float(np.mean(a > 0.0)),
        "p10_r": float(np.quantile(a, 0.10)),
        "p90_r": float(np.quantile(a, 0.90)),
    }


def _segment_probe(rows: list[dict], key_fn) -> dict[str, dict]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        key = str(key_fn(row))
        groups[key].append(float(row[EDGE_TARGET_COLUMN]))
    return {k: _stats(v) for k, v in sorted(groups.items())}


def _median_segment_across_probes(probe_segments: list[dict[str, dict]]) -> dict[str, dict]:
    keys = sorted({key for seg in probe_segments for key in seg})
    out: dict[str, dict] = {}
    for key in keys:
        rows = [seg[key] for seg in probe_segments if key in seg and seg[key]["n"] > 0]
        if not rows:
            continue
        out[key] = {
            "probes": len(rows),
            "median_n": int(median([r["n"] for r in rows])),
            "median_mean_r": float(median([r["mean_r"] for r in rows])),
            "median_positive_rate": float(median([r["positive_rate"] for r in rows])),
        }
    return out


def run(report_path: Path) -> Path:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    jobs = report.get("jobs") or []

    # Within a probe many configurations reuse the same historical setup events.
    # De-duplicate by setup identity so model-matrix breadth does not multiply the
    # same SMC event. Across probes we keep separate summaries because windows overlap.
    by_probe: dict[int, dict[tuple, dict]] = defaultdict(dict)
    source_jobs = 0
    for job in jobs:
        if not job.get("success"):
            continue
        path = Path(str(job.get("training_data_path") or ""))
        split = job.get("split") or {}
        first_idx = int(split.get("test_first_event_index") or -1)
        if first_idx < 0 or not path.exists():
            continue
        frame = pl.read_parquet(path)
        if "event_index" not in frame.columns or EDGE_TARGET_COLUMN not in frame.columns:
            continue
        oos = frame.filter(pl.col("event_index") >= first_idx).sort("time")
        probe = int(job.get("sample_index") or 0)
        for row in oos.iter_rows(named=True):
            key = (
                str(row.get("time")), str(row.get("event_direction")),
                round(float(row.get("event_entry")), 6),
                round(float(row.get("event_stop_loss")), 6),
                round(float(row.get("event_take_profit")), 6),
            )
            by_probe[probe].setdefault(key, row)
        source_jobs += 1

    if not by_probe:
        raise ValueError("no readable untouched-OOS event artifacts found")

    probe_reports = []
    for probe in sorted(by_probe):
        rows = list(by_probe[probe].values())
        overall = _stats([float(r[EDGE_TARGET_COLUMN]) for r in rows])
        direction = _segment_probe(rows, lambda r: r.get("event_direction") or "unknown")
        outcome = _segment_probe(rows, lambda r: r.get("event_outcome_reason") or "unknown")
        session = _segment_probe(rows, lambda r: _session(r["time"])[0])
        confidence = _segment_probe(rows, lambda r: _confidence_bucket(float(r.get("event_smc_confidence") or 0.0)))
        reward = _segment_probe(rows, lambda r: _reward_bucket(_reward_r(r)))
        regime = _segment_probe(rows, lambda r: r.get("regime_name") or "unknown")
        probe_reports.append({
            "probe": probe,
            "events": len(rows),
            "overall": overall,
            "direction": direction,
            "outcome": outcome,
            "session": session,
            "confidence": confidence,
            "declared_reward_r": reward,
            "regime": regime,
        })

    def seg(name: str) -> dict[str, dict]:
        return _median_segment_across_probes([p[name] for p in probe_reports])

    result = {
        "source_report": str(report_path),
        "evidence_status": "DEVELOPMENT_OOS_ALREADY_INSPECTED",
        "overlap_note": "chronological probes overlap and are not independent trials",
        "source_successful_jobs": source_jobs,
        "probes": probe_reports,
        "across_probe_medians": {
            "overall_mean_r": float(median([p["overall"]["mean_r"] for p in probe_reports])),
            "overall_positive_rate": float(median([p["overall"]["positive_rate"] for p in probe_reports])),
            "direction": seg("direction"),
            "session": seg("session"),
            "confidence": seg("confidence"),
            "declared_reward_r": seg("declared_reward_r"),
            "regime": seg("regime"),
        },
        "guard": (
            "Descriptive SMC structure only. Do not choose a live filter from these inspected OOS segments. "
            "Any new filter/learner hypothesis requires a predeclared protocol and fresh future evidence."
        ),
    }
    out_path = report_path.parent / "smc_setup_structure_diagnostic.json"
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    print("=== GOLDmicro Causal-SMC Setup Structure Diagnostic ===")
    print(f"Report              : {report_path}")
    print(f"Source trained jobs : {source_jobs}")
    print(f"Probes              : {len(probe_reports)} (overlapping; development evidence)")
    print("Gate changes        : DISABLED")
    print("Retraining          : NONE")
    print("PF/DD               : NOT EVALUATED")
    print("Promotion           : DISABLED")
    print("\nPer probe baseline:")
    for p in probe_reports:
        s = p["overall"]
        print(f"  s{p['probe']:02d} events={p['events']:4d} meanR={s['mean_r']:+.4f} medR={s['median_r']:+.4f} positive={s['positive_rate']:.1%}")

    print("\nAcross-probe segment medians (descriptive only):")
    for title, key in [
        ("Direction", "direction"), ("Session", "session"),
        ("SMC confidence", "confidence"), ("Declared reward R", "declared_reward_r"),
        ("Regime", "regime"),
    ]:
        print(f"\n{title}:")
        for name, row in result["across_probe_medians"][key].items():
            print(
                f"  {name:32s} probes={row['probes']} medN={row['median_n']:4d} "
                f"meanR={row['median_mean_r']:+.4f} positive={row['median_positive_rate']:.1%}"
            )

    print(f"\nReport              : {out_path}")
    print("Interpretation guard:")
    print("  Do not turn the best inspected segment into a trading filter. Use this only to decide")
    print("  whether the next research phase should redesign SMC setup generation or ML representation.")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, type=Path)
    args = ap.parse_args()
    run(args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
