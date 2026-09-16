"""One-run GOLDmicro target-alignment study.

The study holds the causal feature pipeline fixed and asks whether predicting a
later close (1/4/8/16 M15 bars by default) produces a repeatable signal.  It uses
24 predictive configurations x 5 chronological samples x N target horizons in a
single frozen market snapshot.  Existing AUC/generalization gates are NOT
relaxed.  If no horizon clears the hard gate, strategy PF/DD is skipped cleanly.

No live model is changed, no order is sent, and no promotion occurs.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
import sys

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.candidate_screening import ScreeningThresholds
from src.challenger_batch import build_challenger_specs
from src.goldmicro_candidate_trainer import OOS_GAP_BARS
from src.goldmicro_target_trainer import train_target_aligned_candidate
from src.model_registry import candidate_dir, sha256_file
from src.mt5_research_connector import MT5ResearchConnector
from src.target_alignment import (
    select_target_shortlist,
    summarize_target_alignment,
)


def _resolve_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _parse_horizons(text: str) -> tuple[int, ...]:
    vals = tuple(dict.fromkeys(int(x.strip()) for x in text.split(",") if x.strip()))
    if not vals:
        raise ValueError("at least one target horizon is required")
    if any(v < 1 for v in vals):
        raise ValueError("target horizons must be >= 1")
    if any(v >= OOS_GAP_BARS for v in vals):
        raise ValueError(
            f"all target horizons must be < OOS gap {OOS_GAP_BARS}; got {vals}"
        )
    return vals


def _snapshot_metadata(df: pl.DataFrame, *, path: Path, label: str) -> dict:
    df.write_parquet(path)
    times = df["time"].to_list() if "time" in df.columns and len(df) else []
    return {
        "label": label,
        "rows": len(df),
        "start": str(times[0]) if times else None,
        "end": str(times[-1]) if times else None,
        "path": str(path),
        "sha256": sha256_file(path),
    }


def _clip_h1_to_cutoff(df_h1: pl.DataFrame, cutoff) -> pl.DataFrame:
    if df_h1 is None or len(df_h1) == 0 or "time" not in df_h1.columns:
        return df_h1
    return df_h1.filter(pl.col("time") <= cutoff)


def _target_base_id(base_model_id: str, horizon: int) -> str:
    """Insert tXX before the cost profile so cross-cost OOS rewriting still works."""
    pattern = r"-(normal|conservative)$"
    if re.search(pattern, base_model_id) is None:
        raise ValueError(f"unexpected base model id: {base_model_id}")
    return re.sub(pattern, rf"-t{horizon:02d}-\1", base_model_id)


def _target_sample_spec(base_spec, *, horizon: int, sample_index: int):
    target_base = _target_base_id(base_spec.model_id, horizon)
    model_id = f"{target_base}-s{sample_index:02d}"
    return replace(
        base_spec,
        model_id=model_id,
        output_dir=str(candidate_dir(model_id, ROOT / "models")),
    ), target_base


def run_study(
    *,
    limit: int,
    samples: int,
    sample_stride_bars: int,
    horizons: tuple[int, ...],
    top_k: int,
    symbol: str,
    timeframe: str,
    min_test_auc: float,
    max_gap: float,
    min_test_samples: int,
    min_pass_rate: float,
    capital: float,
    risk: float,
    min_pf: float,
    max_dd: float,
    max_skip: float,
    min_trades: int,
) -> Path:
    if limit < 1 or samples < 1 or top_k < 1:
        raise ValueError("limit, samples and top_k must be >= 1")
    if sample_stride_bars < 1:
        raise ValueError("sample_stride_bars must be >= 1")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    git_sha = _resolve_git_sha()
    report_dir = ROOT / "models" / "reports" / f"target_alignment_{stamp}"
    report_dir.mkdir(parents=True, exist_ok=True)

    # Cost belongs to the strategy gate, not XGBoost training. Confidence threshold
    # also does not change AUC training quality. Hold both fixed here so all 24
    # configurations explore genuine predictive/model-feature variation.
    specs = build_challenger_specs(
        root=ROOT / "models",
        limit=limit,
        batch_id=f"target-{stamp}",
        confidence_thresholds=(0.60,),
        cost_profiles=("normal",),
    )
    total_jobs = len(specs) * samples * len(horizons)

    print("=== GOLDmicro Target Alignment Study ===")
    print(f"Git SHA        : {git_sha}")
    print(f"Configurations : {len(specs)}")
    print(f"Samples/config : {samples}")
    print(f"Target horizons: {', '.join(map(str, horizons))} {timeframe} bars")
    print(
        f"Training jobs  : {len(specs)} x {samples} x {len(horizons)} = {total_jobs}"
    )
    print(f"AUC hard gate  : >= {min_test_auc:.2f} on >= {min_pass_rate:.0%} samples")
    print("Gate policy     : NO AUTO-RELAXATION")
    print("Promotion       : DISABLED")

    connector = MT5ResearchConnector()
    results: list[dict] = []
    sample_windows: list[dict] = []
    connector.connect()
    try:
        max_bars = max(spec.train_bars for spec in specs)
        master_bars = max_bars + sample_stride_bars * (samples - 1)
        print(f"Freezing master {timeframe} snapshot: {master_bars:,} bars")
        raw_m15_master = connector.get_market_data(symbol, timeframe, master_bars)
        if len(raw_m15_master) < master_bars:
            raise RuntimeError(
                f"insufficient {timeframe} history: need {master_bars:,}, "
                f"received {len(raw_m15_master):,}"
            )

        h1_bars = min(max(master_bars // 4 + 100, 1000), 10000)
        print(f"Freezing master H1 snapshot: {h1_bars:,} bars")
        raw_h1_master = connector.get_market_data(symbol, "H1", h1_bars)
        if len(raw_h1_master) < 100:
            raise RuntimeError(f"insufficient H1 snapshot: {len(raw_h1_master)} bars")

        m15_meta = _snapshot_metadata(
            raw_m15_master,
            path=report_dir / "market_snapshot_m15_master.parquet",
            label=timeframe,
        )
        h1_meta = _snapshot_metadata(
            raw_h1_master,
            path=report_dir / "market_snapshot_h1_master.parquet",
            label="H1",
        )
        master_fingerprint = f"{m15_meta['sha256']}:{h1_meta['sha256']}"

        job_no = 0
        for sample_index in range(1, samples + 1):
            bars_to_drop = sample_stride_bars * (samples - sample_index)
            sample_rows = len(raw_m15_master) - bars_to_drop
            raw_m15 = raw_m15_master.head(sample_rows)
            cutoff = raw_m15["time"][-1]
            raw_h1 = _clip_h1_to_cutoff(raw_h1_master, cutoff)
            sample_windows.append(
                {
                    "sample_index": sample_index,
                    "m15_rows": len(raw_m15),
                    "h1_rows": len(raw_h1),
                    "cutoff": str(cutoff),
                }
            )
            print(
                f"\n--- Sample {sample_index}/{samples} | cutoff={cutoff} | "
                f"M15 rows={len(raw_m15):,} ---"
            )

            for horizon in horizons:
                print(f"  Target horizon: +{horizon} {timeframe} bars")
                for base_spec in specs:
                    job_no += 1
                    spec, target_base = _target_sample_spec(
                        base_spec,
                        horizon=horizon,
                        sample_index=sample_index,
                    )
                    fingerprint = (
                        f"{master_fingerprint}:sample={sample_index}:cutoff={cutoff}:"
                        f"target={horizon}"
                    )
                    print(f"[{job_no:03d}/{total_jobs:03d}] {spec.model_id}")
                    try:
                        row = train_target_aligned_candidate(
                            spec,
                            target_lookahead_bars=horizon,
                            connector=connector,
                            symbol=symbol,
                            timeframe=timeframe,
                            git_sha=git_sha,
                            raw_m15=raw_m15,
                            raw_h1=raw_h1,
                            data_fingerprint=fingerprint,
                        )
                        row.update(
                            {
                                "base_model_id": target_base,
                                "predictive_config_id": base_spec.model_id,
                                "sample_index": sample_index,
                                "sample_count": samples,
                                "sample_cutoff": str(cutoff),
                            }
                        )
                        results.append(row)
                        metrics = row.get("train_metrics") or {}
                        print(
                            "  PASS train | "
                            f"test AUC={metrics.get('xgb_test_score')} | "
                            f"target+={row.get('target_positive_rate')}"
                        )
                    except Exception as exc:
                        results.append(
                            {
                                "model_id": spec.model_id,
                                "base_model_id": target_base,
                                "predictive_config_id": base_spec.model_id,
                                "success": False,
                                "error": str(exc),
                                "target_lookahead_bars": horizon,
                                "xgb_profile": spec.xgb_profile,
                                "feature_profile": spec.feature_profile,
                                "cost_profile": spec.cost_profile,
                                "hmm_lookback": spec.hmm_lookback,
                                "confidence_threshold": spec.confidence_threshold,
                                "seed": spec.seed,
                                "train_bars": spec.train_bars,
                                "sample_index": sample_index,
                                "sample_count": samples,
                                "sample_cutoff": str(cutoff),
                            }
                        )
                        print(f"  FAIL train | {exc}")
    finally:
        connector.disconnect()

    training_path = report_dir / "target_alignment_training_results.json"
    training_payload = {
        "batch_id": stamp,
        "generated_at": datetime.now().isoformat(),
        "git_sha": git_sha,
        "symbol": symbol,
        "timeframe": timeframe,
        "configurations": len(specs),
        "samples_per_configuration": samples,
        "sample_stride_bars": sample_stride_bars,
        "target_horizons": list(horizons),
        "total_training_jobs": total_jobs,
        "snapshot": {
            "master_m15": m15_meta,
            "master_h1": h1_meta,
            "sample_windows": sample_windows,
        },
        "results": results,
        "promotion_performed": False,
    }
    training_path.write_text(
        json.dumps(training_payload, indent=2, default=str), encoding="utf-8"
    )

    thresholds = ScreeningThresholds(
        min_test_auc=min_test_auc,
        max_generalization_gap=max_gap,
        min_test_samples=min_test_samples,
    )
    config_summaries, horizon_summaries = summarize_target_alignment(
        results,
        expected_samples=samples,
        thresholds=thresholds,
        min_pass_rate=min_pass_rate,
    )
    horizon_by_base = {
        str(row.get("base_model_id")): int(row.get("target_lookahead_bars") or 1)
        for row in results
    }
    shortlist = select_target_shortlist(
        config_summaries,
        horizon_by_base=horizon_by_base,
        top_k=top_k,
    )

    report_path = report_dir / "target_alignment_report.json"
    state = "TARGET_HORIZON_GATE_PASS" if shortlist else "NO_TARGET_HORIZON_CLEARS_GATE"
    report_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(),
                "batch_id": stamp,
                "state": state,
                "thresholds": {
                    **thresholds.__dict__,
                    "min_pass_rate": min_pass_rate,
                    "oos_gap_bars": OOS_GAP_BARS,
                },
                "horizons": [x.to_dict() for x in horizon_summaries],
                "configurations": [x.to_dict() for x in config_summaries],
                "shortlist": [x.to_dict() for x in shortlist],
                "promotion_performed": False,
                "interpretation_guard": (
                    "Target-horizon selection is diagnostic and subject to multiple-testing bias. "
                    "No threshold is relaxed because of observed results. Strategy PF/DD/cost, "
                    "forward shadow evidence, independent review and Human Gate remain mandatory."
                ),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    strategy_queue = report_dir / "strategy_oos_queue.json"
    strategy_queue.write_text(
        json.dumps(
            {
                "batch_id": stamp,
                "state": (
                    "AWAITING_STRATEGY_OOS_PF_DD_COST"
                    if shortlist
                    else "NO_ELIGIBLE_TARGET_HORIZON"
                ),
                "samples_per_configuration": samples,
                "configurations": [
                    {
                        "base_model_id": item.base_model_id,
                        "target_lookahead_bars": horizon_by_base[item.base_model_id],
                        "pass_rate": item.pass_rate,
                        "median_test_auc": item.median_test_auc,
                        "min_test_auc": item.min_test_auc,
                        "max_gap": item.max_gap,
                        "samples": [
                            {
                                **sample,
                                "target_lookahead_bars": horizon_by_base[item.base_model_id],
                            }
                            for sample in item.samples
                        ],
                        "pf_dd_status": "NOT_YET_EVALUATED",
                        "shadow_status": "NOT_YET_EVALUATED",
                    }
                    for item in shortlist
                ],
                "promotion_performed": False,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print("\n=== Target Alignment Summary ===")
    for item in horizon_summaries:
        print(
            f"t+{item.target_lookahead_bars:02d}: {item.status:20s} "
            f"stable={item.stable_pass_count}/{item.configuration_count} "
            f"median={item.median_of_config_median_auc:.4f} "
            f"bestMedian={item.max_config_median_auc:.4f} "
            f"maxSingle={item.max_single_sample_auc:.4f}"
        )
    print(f"Shortlist      : {len(shortlist)}")
    print(f"Report         : {report_path}")

    strategy_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_goldmicro_strategy_oos.py"),
        "--queue", str(strategy_queue),
        "--capital", str(capital),
        "--risk", str(risk),
        "--min-pf", str(min_pf),
        "--max-dd", str(max_dd),
        "--max-skip", str(max_skip),
        "--min-trades", str(min_trades),
    ]
    strategy = subprocess.run(strategy_cmd, cwd=ROOT)
    if strategy.returncode != 0:
        raise RuntimeError(f"strategy OOS stage failed with code {strategy.returncode}")

    print("\n=== TARGET ALIGNMENT STUDY COMPLETE ===")
    print(f"Training       : {training_path}")
    print(f"Alignment      : {report_path}")
    print(f"Strategy queue : {strategy_queue}")
    print(f"Strategy OOS   : {report_dir / 'strategy_oos_report.json'}")
    print(f"Shadow queue   : {report_dir / 'shadow_queue.json'}")
    print("Live model     : UNCHANGED")
    print("Promotion      : DISABLED")
    return report_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--sample-stride-bars", type=int, default=1000)
    ap.add_argument("--horizons", default="1,4,8,16")
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--symbol", default="GOLDmicro")
    ap.add_argument("--timeframe", default="M15")
    ap.add_argument("--min-test-auc", type=float, default=0.55)
    ap.add_argument("--max-gap", type=float, default=0.12)
    ap.add_argument("--min-test-samples", type=int, default=500)
    ap.add_argument("--min-pass-rate", type=float, default=0.80)
    ap.add_argument("--capital", type=float, default=20000.0)
    ap.add_argument("--risk", type=float, default=1.0)
    ap.add_argument("--min-pf", type=float, default=1.30)
    ap.add_argument("--max-dd", type=float, default=10.0)
    ap.add_argument("--max-skip", type=float, default=20.0)
    ap.add_argument("--min-trades", type=int, default=30)
    args = ap.parse_args()

    run_study(
        limit=args.limit,
        samples=args.samples,
        sample_stride_bars=args.sample_stride_bars,
        horizons=_parse_horizons(args.horizons),
        top_k=args.top_k,
        symbol=args.symbol,
        timeframe=args.timeframe,
        min_test_auc=args.min_test_auc,
        max_gap=args.max_gap,
        min_test_samples=args.min_test_samples,
        min_pass_rate=args.min_pass_rate,
        capital=args.capital,
        risk=args.risk,
        min_pf=args.min_pf,
        max_dd=args.max_dd,
        max_skip=args.max_skip,
        min_trades=args.min_trades,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
