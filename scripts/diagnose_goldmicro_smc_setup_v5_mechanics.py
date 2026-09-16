"""Replay GOLDmicro SMC Setup V5 mechanics on an existing research snapshot.

Development-only.  This script does not evaluate returns, PF/DD, or promotion.
It exists to verify that the new state machine emits plausible, attributable
setup events before any economic study is designed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.goldmicro_causal_smc import GoldmicroCausalSMCAnalyzer
from src.goldmicro_smc_setup_v5 import GoldmicroSMCSetupV5, SMCSetupV5Config


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def run(report_path: Path) -> Path:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    snapshot_path = Path(str(((report.get("snapshots") or {}).get("m15") or {}).get("path") or ""))
    if not snapshot_path.exists():
        raise FileNotFoundError(f"M15 snapshot not found: {snapshot_path}")

    raw = pl.read_parquet(snapshot_path)
    required = {"open", "high", "low", "close"}
    if not required.issubset(set(raw.columns)):
        raise ValueError(f"snapshot missing OHLC columns: {sorted(required - set(raw.columns))}")

    analyzer = GoldmicroCausalSMCAnalyzer(swing_length=5)
    causal = analyzer.calculate_all(raw)
    config = SMCSetupV5Config()
    events = GoldmicroSMCSetupV5(config).generate(causal)

    directions = Counter(e.setup_direction for e in events)
    archetypes = Counter(e.setup_archetype for e in events)
    zones = Counter(e.zone_type for e in events)
    combinations = Counter((e.setup_direction, e.setup_archetype, e.zone_type) for e in events)
    setup_ages = [float(e.setup_age_bars) for e in events]
    zone_delays = [float(e.break_to_zone_bars) for e in events]
    retest_delays = [float(e.zone_to_retest_bars) for e in events]
    retest_depths = [float(e.retest_depth_fraction) for e in events]

    payload = {
        "source_report": str(report_path),
        "source_snapshot": str(snapshot_path),
        "evidence_status": "MECHANICS_ONLY_DEVELOPMENT",
        "config": {
            "max_setup_age_bars": config.max_setup_age_bars,
            "max_zone_delay_bars": config.max_zone_delay_bars,
            "min_retest_delay_bars": config.min_retest_delay_bars,
            "fixed_reward_r": config.fixed_reward_r,
            "atr_stop_floor_multiple": config.atr_stop_floor_multiple,
        },
        "raw_rows": len(raw),
        "setup_events": len(events),
        "direction_counts": dict(directions),
        "archetype_counts": dict(archetypes),
        "zone_counts": dict(zones),
        "combination_counts": {"|".join(k): v for k, v in sorted(combinations.items())},
        "median_setup_age_bars": _median(setup_ages),
        "median_break_to_zone_bars": _median(zone_delays),
        "median_zone_to_retest_bars": _median(retest_delays),
        "median_retest_depth_fraction": _median(retest_depths),
        "first_events": [e.to_dict() for e in events[:20]],
        "guard": "Mechanics/provenance only. Do not infer edge or choose filters from this replay.",
    }
    out = report_path.parent / "smc_setup_v5_mechanics.json"
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print("=== GOLDmicro SMC Setup V5 Mechanics Replay ===")
    print(f"Source report       : {report_path}")
    print(f"Snapshot rows       : {len(raw):,}")
    print(f"Setup events        : {len(events):,}")
    print(f"Direction           : BUY={directions.get('BUY', 0)} SELL={directions.get('SELL', 0)}")
    print(f"Archetype           : BOS={archetypes.get('BOS_CONTINUATION', 0)} CHOCH={archetypes.get('CHOCH_REVERSAL', 0)}")
    print(f"Zone                : FVG={zones.get('FVG', 0)} OB={zones.get('OB', 0)}")
    print(f"Median setup age    : {payload['median_setup_age_bars']}")
    print(f"Median break->zone  : {payload['median_break_to_zone_bars']}")
    print(f"Median zone->retest : {payload['median_zone_to_retest_bars']}")
    print(f"Median retest depth : {payload['median_retest_depth_fraction']}")
    print("Economic evaluation : NOT PERFORMED")
    print("Promotion           : DISABLED")
    print(f"Report              : {out}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()
    run(args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
