"""Run the development-only raw economic baseline for GOLDmicro SMC Setup V5."""
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
from src.goldmicro_smc_setup_v5_baseline import (
    SMCSetupV5BaselineThresholds,
    evaluate_baseline,
)


def run(report_path: Path) -> Path:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    snapshot_path = Path(str(((report.get("snapshots") or {}).get("m15") or {}).get("path") or ""))
    if not snapshot_path.exists():
        raise FileNotFoundError(f"M15 snapshot not found: {snapshot_path}")

    raw = pl.read_parquet(snapshot_path)
    analyzer = GoldmicroCausalSMCAnalyzer(swing_length=5)
    causal = analyzer.calculate_all(raw)
    setup_config = SMCSetupV5Config()
    events = GoldmicroSMCSetupV5(setup_config).generate(causal)
    thresholds = SMCSetupV5BaselineThresholds()
    result = evaluate_baseline(raw, events, thresholds=thresholds)

    payload = {
        "source_report": str(report_path),
        "source_snapshot": str(snapshot_path),
        "evidence_status": "DEVELOPMENT_OOS_ALREADY_INSPECTED",
        "setup_config": {
            "max_setup_age_bars": setup_config.max_setup_age_bars,
            "max_zone_delay_bars": setup_config.max_zone_delay_bars,
            "min_retest_delay_bars": setup_config.min_retest_delay_bars,
            "fixed_reward_r": setup_config.fixed_reward_r,
            "atr_stop_floor_multiple": setup_config.atr_stop_floor_multiple,
        },
        **result,
        "pf_dd": "NOT_EVALUATED",
        "ml": "DISABLED",
        "promotion": "DISABLED",
    }
    out = report_path.parent / "smc_setup_v5_raw_baseline.json"
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print("=== GOLDmicro SMC Setup V5 Raw Baseline ===")
    print(f"Source report        : {report_path}")
    print(f"Snapshot rows        : {len(raw):,}")
    print(f"Setup events         : {len(events):,}")
    print(f"Outcome events       : {result['events']:,}")
    print(f"Blocks               : {thresholds.chronological_blocks} non-overlapping raw-time blocks")
    print(f"Min events / block   : {thresholds.min_events_per_block}")
    print(f"Required +mean blocks: {thresholds.min_positive_blocks}/{thresholds.chronological_blocks} under BOTH cost profiles")
    print("Evidence status      : DEVELOPMENT / HISTORY ALREADY INSPECTED")
    print("PF/DD                : NOT EVALUATED")
    print("ML                   : DISABLED")
    print("\nPer chronological block:")
    for block in result["blocks"]:
        gross = block["gross_r"]
        normal = block["normal_net_r"]
        cons = block["conservative_net_r"]
        print(
            f"  b{block['block']:02d} events={block['events']:3d} "
            f"gross={gross['mean']:+.4f}R normal={normal['mean']:+.4f}R "
            f"conservative={cons['mean']:+.4f}R "
            f"BUY/SELL={block['direction_counts']['BUY']}/{block['direction_counts']['SELL']} "
            f"BOS/CHOCH={block['archetype_counts']['BOS_CONTINUATION']}/{block['archetype_counts']['CHOCH_REVERSAL']}"
        )
    print("\nGate summary:")
    print(f"  min observed events     : {result.get('min_events_observed', 0)}")
    print(f"  positive normal blocks  : {result['positive_blocks_normal']}/{thresholds.chronological_blocks}")
    print(f"  positive conserv blocks : {result['positive_blocks_conservative']}/{thresholds.chronological_blocks}")
    print(f"  state                   : {result['status']}")
    print("  threshold tuning        : DISABLED")
    print("  promotion               : DISABLED")
    print(f"\nReport                 : {out}")
    print("Interpretation guard:")
    print("  This is development evidence on already-inspected history. Do not alter the frozen")
    print("  setup rules from these realized returns and then call the same history confirmatory.")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()
    run(args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
