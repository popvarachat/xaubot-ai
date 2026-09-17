"""Freeze the GOLDmicro SMC Setup V5 prospective evidence cutoff.

Research-only. This script reads the already-inspected development source report,
finds the maximum timestamp in its M15 snapshot, and writes a manifest that locks
that timestamp as the prospective cutoff. It does not evaluate returns or alter
any live path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.goldmicro_smc_setup_v5 import SMCSetupV5Config
from src.goldmicro_smc_setup_v5_baseline import SMCSetupV5BaselineThresholds


def _resolve_time_column(df: pl.DataFrame) -> str:
    for name in ("time", "datetime", "timestamp"):
        if name in df.columns:
            return name
    raise ValueError("M15 snapshot has no supported time column: time/datetime/timestamp")


def run(report_path: Path) -> Path:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    snapshot_path = Path(str(((report.get("snapshots") or {}).get("m15") or {}).get("path") or ""))
    if not snapshot_path.exists():
        raise FileNotFoundError(f"M15 snapshot not found: {snapshot_path}")

    df = pl.read_parquet(snapshot_path)
    if df.is_empty():
        raise ValueError("M15 snapshot is empty")
    time_col = _resolve_time_column(df)
    max_time = df.select(pl.col(time_col).max()).item()
    min_time = df.select(pl.col(time_col).min()).item()
    if max_time is None:
        raise ValueError("Could not determine prospective cutoff timestamp")

    setup = SMCSetupV5Config()
    gate = SMCSetupV5BaselineThresholds()
    payload = {
        "protocol": "GOLDMICRO_SMC_SETUP_V5_PROSPECTIVE_V1",
        "source_report": str(report_path),
        "source_snapshot": str(snapshot_path),
        "source_rows": len(df),
        "source_time_column": time_col,
        "source_min_time": str(min_time),
        "prospective_cutoff_exclusive": str(max_time),
        "fresh_event_rule": "entry_timestamp > prospective_cutoff_exclusive",
        "generator": {
            "max_setup_age_bars": setup.max_setup_age_bars,
            "max_zone_delay_bars": setup.max_zone_delay_bars,
            "min_retest_delay_bars": setup.min_retest_delay_bars,
            "fixed_reward_r": setup.fixed_reward_r,
            "atr_stop_floor_multiple": setup.atr_stop_floor_multiple,
        },
        "raw_baseline_gate": {
            "chronological_blocks": gate.chronological_blocks,
            "horizon_bars": gate.horizon_bars,
            "min_events_per_block": gate.min_events_per_block,
            "min_positive_blocks": gate.min_positive_blocks,
            "required_cost_profiles": ["normal", "conservative"],
            "same_bar_ordering": "SL_FIRST",
        },
        "ml": "DISABLED_UNTIL_FRESH_RAW_BASELINE_PASS",
        "pf_dd": "DISABLED_UNTIL_FRESH_RAW_BASELINE_PASS",
        "promotion": "DISABLED",
        "guard": (
            "Freeze once before prospective evaluation. Do not move the cutoff after viewing fresh outcomes. "
            "Pre-cutoff bars may be used only as causal warmup; scored entries must be strictly post-cutoff."
        ),
    }

    out = report_path.parent / "smc_setup_v5_prospective_manifest.json"
    if out.exists():
        existing = json.loads(out.read_text(encoding="utf-8"))
        prior = str(existing.get("prospective_cutoff_exclusive"))
        if prior and prior != str(max_time):
            raise RuntimeError(
                f"Prospective manifest already exists with cutoff {prior}; refusing to move it to {max_time}"
            )
        print(f"Prospective manifest already frozen: {out}")
        print(f"Cutoff (exclusive)   : {prior}")
        return out

    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print("=== GOLDmicro SMC Setup V5 Prospective Freeze ===")
    print(f"Source report        : {report_path}")
    print(f"Source snapshot      : {snapshot_path}")
    print(f"Source rows          : {len(df):,}")
    print(f"Time column          : {time_col}")
    print(f"Cutoff (exclusive)   : {max_time}")
    print("Fresh event rule     : entry_timestamp > cutoff")
    print(f"Raw gate             : {gate.min_positive_blocks}/{gate.chronological_blocks} positive blocks, min {gate.min_events_per_block} events/block, BOTH costs")
    print("ML                   : DISABLED")
    print("PF/DD                : DISABLED")
    print("Promotion            : DISABLED")
    print(f"Manifest             : {out}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()
    run(args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
