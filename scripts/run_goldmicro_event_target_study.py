"""One-run GOLDmicro SMC event-target research matrix.

Default: 24 predictive configurations x 5 chronological probes = 120 jobs.
The model predicts P(SMC setup TP-before-SL within 32 M15 bars). The predictive
hard gate remains AUC >= 0.55 on >=4/5 probes; no threshold is relaxed after
seeing results. Artifacts are research-only and never activated.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.candidate_screening import ScreeningThresholds
from src.challenger_batch import build_challenger_specs
from src.goldmicro_event_target import EventTargetConfig
from src.goldmicro_event_trainer import train_event_candidate
from src.model_registry import candidate_dir, sha256_file
from src.mt5_research_connector import MT5ResearchConnector
from src.multisample_screening import select_stable_shortlist, summarize_multisample


def _resolve_git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _clip_h1(df: pl.DataFrame, cutoff) -> pl.DataFrame:
    return df.filter(pl.col("time") <= cutoff) if "time" in df.columns else df


def _snapshot(df: pl.DataFrame, path: Path) -> dict:
    df.write_parquet(path)
    times = df["time"].to_list() if "time" in df.columns and len(df) else []
    return {
        "rows": len(df),
        "start": str(times[0]) if times else None,
        "end": str(times[-1]) if times else None,
        "path": str(path),
        "sha256": sha256_file(path),
    }


def _sample_spec(base_spec, sample_index: int):
    base_id = f"{base_spec.model_id}-event32"
    model_id = f"{base_id}-s{sample_index:02d}"
    return replace(
        base_spec,
        model_id=model_id,
        output_dir=str(candidate_dir(model_id, ROOT / "models")),
    ), base_id


def run_study(
    *,
    limit: int = 24,
    samples: int = 5,
    sample_stride_bars: int = 1000,
    top_k: int = 6,
    symbol: str = "GOLDmicro",
    timeframe: str = "M15",
    min_test_auc: float = 0.55,
    max_gap: float = 0.12,
    min_test_events: int = 100,
    min_pass_rate: float = 0.80,
) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    git_sha = _resolve_git_sha()
    report_dir = ROOT / "models" / "reports" / f"event_target_{stamp}"
    report_dir.mkdir(parents=True, exist_ok=True)

    specs = build_challenger_specs(
        root=ROOT / "models",
        limit=limit,
        batch_id=f"event-{stamp}",
        confidence_thresholds=(0.60,),
        cost_profiles=("normal",),
    )
    total_jobs = len(specs) * samples
    event_cfg = EventTargetConfig(max_holding_bars=32, event_cooldown_bars=10)

    print("=== GOLDmicro Event Target V2 Study ===")
    print(f"Git SHA        : {git_sha}")
    print(f"Configurations : {len(specs)}")
    print(f"Samples/config : {samples}")
    print(f"Training jobs  : {len(specs)} x {samples} = {total_jobs}")
    print("Target         : P(SMC TP before SMC SL within 32 M15 bars)")
    print("Same-bar rule  : adverse / SL first")
    print(f"AUC hard gate  : >= {min_test_auc:.2f} on >= {min_pass_rate:.0%} samples")
    print("Gate policy    : NO AUTO-RELAXATION")
    print("Live model     : UNCHANGED")
    print("Promotion      : DISABLED")

    connector = MT5ResearchConnector()
    results: list[dict] = []
    sample_windows: list[dict] = []
    connector.connect()
    try:
        max_bars = max(s.train_bars for s in specs)
        master_bars = max_bars + sample_stride_bars * (samples - 1)
        print(f"Freezing master {timeframe} snapshot: {master_bars:,} bars")
        raw_m15_master = connector.get_market_data(symbol, timeframe, master_bars)
        if len(raw_m15_master) < master_bars:
            raise RuntimeError(f"need {master_bars:,} {timeframe} bars, received {len(raw_m15_master):,}")
        h1_bars = min(max(master_bars // 4 + 100, 1000), 10000)
        print(f"Freezing master H1 snapshot: {h1_bars:,} bars")
        raw_h1_master = connector.get_market_data(symbol, "H1", h1_bars)
        if len(raw_h1_master) < 100:
            raise RuntimeError("insufficient H1 history")

        m15_meta = _snapshot(raw_m15_master, report_dir / "market_snapshot_m15_master.parquet")
        h1_meta = _snapshot(raw_h1_master, report_dir / "market_snapshot_h1_master.parquet")
        master_fp = f"{m15_meta['sha256']}:{h1_meta['sha256']}"

        job_no = 0
        for sample_index in range(1, samples + 1):
            bars_to_drop = sample_stride_bars * (samples - sample_index)
            raw_m15 = raw_m15_master.head(len(raw_m15_master) - bars_to_drop)
            cutoff = raw_m15["time"][-1]
            raw_h1 = _clip_h1(raw_h1_master, cutoff)
            sample_windows.append({
                "sample_index": sample_index,
                "cutoff": str(cutoff),
                "m15_rows": len(raw_m15),
                "h1_rows": len(raw_h1),
            })
            print(f"\n--- Probe {sample_index}/{samples} | cutoff={cutoff} ---")

            for base_spec in specs:
                job_no += 1
                spec, base_id = _sample_spec(base_spec, sample_index)
                print(f"[{job_no:03d}/{total_jobs:03d}] {spec.model_id}")
                try:
                    result = train_event_candidate(
                        spec,
                        connector=connector,
                        symbol=symbol,
                        timeframe=timeframe,
                        git_sha=git_sha,
                        raw_m15=raw_m15,
                        raw_h1=raw_h1,
                        data_fingerprint=f"{master_fp}:probe={sample_index}:cutoff={cutoff}:event32",
                        event_config=event_cfg,
                    )
                    result.update({
                        "base_model_id": base_id,
                        "sample_index": sample_index,
                        "sample_cutoff": str(cutoff),
                        "cost_profile": "normal",
                        "xgb_path": result["event_model_path"],
                        "training_data_path": result["event_data_path"],
                    })
                    results.append(result)
                    tm = result["train_metrics"]
                    print(
                        f"  PASS train | AUC={tm['xgb_test_score']:.4f} | "
                        f"PR-AUC={tm['pr_auc']:.4f} | Brier={tm['brier']:.4f} | "
                        f"events={result['train_event_count']}/{result['test_event_count']}"
                    )
                except Exception as exc:
                    print(f"  FAIL train | {exc}")
                    results.append({
                        "model_id": spec.model_id,
                        "base_model_id": base_id,
                        "sample_index": sample_index,
                        "sample_cutoff": str(cutoff),
                        "success": False,
                        "xgb_profile": spec.xgb_profile,
                        "feature_profile": spec.feature_profile,
                        "cost_profile": "normal",
                        "train_metrics": {},
                        "error": str(exc),
                    })
    finally:
        connector.disconnect()

    thresholds = ScreeningThresholds(
        min_test_auc=min_test_auc,
        max_generalization_gap=max_gap,
        min_test_samples=min_test_events,
    )
    summaries = summarize_multisample(
        results,
        thresholds=thresholds,
        expected_samples=samples,
        min_pass_rate=min_pass_rate,
    )
    shortlist = select_stable_shortlist(summaries, top_k=top_k)
    state = "EVENT_TARGET_GATE_PASS" if shortlist else "NO_EVENT_CONFIGURATION_CLEARS_GATE"

    report = {
        "generated_at": datetime.now().isoformat(),
        "batch_id": stamp,
        "state": state,
        "git_sha": git_sha,
        "target": {
            "unit": "causal_smc_setup_event",
            "positive": "TP before SL within 32 raw M15 bars",
            "negative": "SL first, same-bar ambiguity, or timeout/no-TP-first",
            "same_bar_policy": "adverse_sl_first",
            "max_holding_bars": 32,
            "event_cooldown_bars": 10,
            "split_policy": "raw-bar split with 32-bar embargo on each side",
        },
        "thresholds": {
            "min_test_auc": min_test_auc,
            "max_generalization_gap": max_gap,
            "min_test_events": min_test_events,
            "min_pass_rate": min_pass_rate,
        },
        "snapshots": {"m15": m15_meta, "h1": h1_meta},
        "sample_windows": sample_windows,
        "jobs": results,
        "configurations": [x.to_dict() for x in summaries],
        "shortlist": [x.to_dict() for x in shortlist],
        "strategy_pf_dd_status": "NOT_YET_EVALUATED" if shortlist else "SKIPPED_NO_PREDICTIVE_SURVIVORS",
        "promotion_performed": False,
        "interpretation_guard": (
            "Event-target AUC is only a predictive pre-screen. Do not promote from this report. "
            "PF/DD/cost, forward shadow evidence, independent review and Human Gate remain mandatory."
        ),
    }
    path = report_dir / "event_target_report.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    queue = report_dir / "event_strategy_oos_queue.json"
    queue.write_text(json.dumps({
        "batch_id": stamp,
        "state": "AWAITING_EVENT_STRATEGY_OOS" if shortlist else "NO_ELIGIBLE_EVENT_CONFIGURATION",
        "configurations": [x.to_dict() for x in shortlist],
        "promotion_performed": False,
    }, indent=2, default=str), encoding="utf-8")

    print("\n=== Event Target Summary ===")
    print(f"Stable pass     : {sum(x.status == 'STABLE_SCREEN_PASS' for x in summaries)}/{len(summaries)}")
    print(f"Shortlist       : {len(shortlist)}")
    print(f"State           : {state}")
    print(f"Report          : {path}")
    print(f"Strategy queue  : {queue}")
    print("Live model      : UNCHANGED")
    print("Promotion       : DISABLED")
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--sample-stride-bars", type=int, default=1000)
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--min-test-auc", type=float, default=0.55)
    ap.add_argument("--max-gap", type=float, default=0.12)
    ap.add_argument("--min-test-events", type=int, default=100)
    ap.add_argument("--min-pass-rate", type=float, default=0.80)
    args = ap.parse_args()
    run_study(
        limit=args.limit,
        samples=args.samples,
        sample_stride_bars=args.sample_stride_bars,
        top_k=args.top_k,
        min_test_auc=args.min_test_auc,
        max_gap=args.max_gap,
        min_test_events=args.min_test_events,
        min_pass_rate=args.min_pass_rate,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
