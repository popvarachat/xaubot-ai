"""Blind readiness check for GOLDmicro SMC Setup V5 prospective evidence.

This command never resolves outcomes and never prints P/L, R, PF, DD, or win rate.
It only counts strictly post-cutoff setup events and requires them to have a full
future horizon available before declaring the evidence set ready for the one-shot
predeclared economic evaluation.
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

from src.goldmicro_causal_smc import GoldmicroCausalSMCAnalyzer
from src.goldmicro_smc_setup_v5 import GoldmicroSMCSetupV5, SMCSetupV5Config
from src.goldmicro_smc_setup_v5_readiness import (
    ProspectiveReadinessThresholds,
    evaluate_readiness,
    parse_cutoff,
)


def _resolve_time_column(df: pl.DataFrame) -> str:
    for name in ("time", "datetime", "timestamp"):
        if name in df.columns:
            return name
    raise ValueError("snapshot has no supported time column: time/datetime/timestamp")


def run(manifest_path: Path, snapshot_override: Path | None = None) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != "GOLDMICRO_SMC_SETUP_V5_PROSPECTIVE_V1":
        raise ValueError("unsupported or missing V5 prospective protocol")

    source_snapshot = Path(str(manifest.get("source_snapshot") or ""))
    snapshot_path = snapshot_override or source_snapshot
    if not snapshot_path.exists():
        raise FileNotFoundError(f"prospective snapshot not found: {snapshot_path}")

    df = pl.read_parquet(snapshot_path)
    if df.is_empty():
        raise ValueError("prospective snapshot is empty")
    time_col = _resolve_time_column(df)
    timestamps = df[time_col].to_list()
    cutoff = parse_cutoff(str(manifest.get("prospective_cutoff_exclusive") or ""))

    generator_cfg = manifest.get("generator") or {}
    setup_cfg = SMCSetupV5Config(
        max_setup_age_bars=int(generator_cfg.get("max_setup_age_bars", 12)),
        max_zone_delay_bars=int(generator_cfg.get("max_zone_delay_bars", 6)),
        min_retest_delay_bars=int(generator_cfg.get("min_retest_delay_bars", 1)),
        fixed_reward_r=float(generator_cfg.get("fixed_reward_r", 1.5)),
        atr_stop_floor_multiple=float(generator_cfg.get("atr_stop_floor_multiple", 0.50)),
    )

    gate = manifest.get("raw_baseline_gate") or {}
    thresholds = ProspectiveReadinessThresholds(
        chronological_blocks=int(gate.get("chronological_blocks", 5)),
        horizon_bars=int(gate.get("horizon_bars", 32)),
        min_events_per_block=int(gate.get("min_events_per_block", 30)),
    )

    causal = GoldmicroCausalSMCAnalyzer(swing_length=5).calculate_all(df)
    events = GoldmicroSMCSetupV5(setup_cfg).generate(causal)
    result = evaluate_readiness(
        timestamps=timestamps,
        event_entry_indices=[event.entry_index for event in events],
        cutoff=cutoff,
        thresholds=thresholds,
    )

    payload = {
        "protocol": manifest["protocol"],
        "manifest": str(manifest_path),
        "snapshot": str(snapshot_path),
        "cutoff_exclusive": str(cutoff),
        "blind": True,
        **result,
        "economic_outcomes": "NOT_EVALUATED",
        "ml": "DISABLED",
        "pf_dd": "DISABLED",
        "promotion": "DISABLED",
    }
    out = manifest_path.parent / "smc_setup_v5_prospective_readiness.json"
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print("=== GOLDmicro SMC Setup V5 Prospective Readiness ===")
    print(f"Manifest            : {manifest_path}")
    print(f"Snapshot            : {snapshot_path}")
    print(f"Cutoff (exclusive)  : {cutoff}")
    print(f"Fresh rows          : {result['fresh_rows']:,}")
    print(f"Fresh setup events  : {result['fresh_setup_events']:,}")
    print(f"Matured setup events: {result['matured_setup_events']:,}")
    print(f"Per-block matured   : {result['block_event_counts']}")
    print(f"Min / required      : {result.get('min_block_events', 0)} / {thresholds.min_events_per_block}")
    print(f"State               : {result['status']}")
    print("Economic outcomes   : HIDDEN / NOT EVALUATED")
    print("ML                  : DISABLED")
    print("PF/DD               : DISABLED")
    print("Promotion           : DISABLED")
    print(f"Report              : {out}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument(
        "--snapshot",
        type=Path,
        default=None,
        help="Updated M15 parquet containing pre-cutoff warmup plus fresh bars. Defaults to frozen source snapshot.",
    )
    args = ap.parse_args()
    run(args.manifest, args.snapshot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
