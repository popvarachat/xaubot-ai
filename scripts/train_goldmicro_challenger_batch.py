"""Train many GOLDmicro challengers across one or more chronological samples.

Default mode is PLAN ONLY. Use --execute to perform local read-only market-data
training. Candidate models are written only under models/candidates and never to
active Champion paths. No orders are sent and no promotion occurs.

For multi-sample research the runner freezes one larger M15/H1 master snapshot,
then evaluates the same challenger configuration set at several historical
cutoffs. This gives 24 x N chronological training/evaluation jobs in one command
without pretending that repeated runs on the same latest snapshot are independent.

Research MT5 access deliberately attaches to the operator's already logged-in
terminal session. It does not require MT5_LOGIN / MT5_PASSWORD / MT5_SERVER in
.env and does not expose an order-send path.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import sys
from datetime import datetime
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.challenger_batch import build_challenger_specs, write_batch_plan
from src.goldmicro_candidate_trainer import train_candidate
from src.mt5_research_connector import MT5ResearchConnector
from src.model_registry import candidate_dir, sha256_file


def _snapshot_metadata(df, *, path: Path, label: str) -> dict:
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


def _sample_spec(base_spec, *, sample_index: int):
    """Clone one base configuration into an isolated per-sample artifact path."""
    sample_tag = f"s{sample_index:02d}"
    model_id = f"{base_spec.model_id}-{sample_tag}"
    return replace(
        base_spec,
        model_id=model_id,
        output_dir=str(candidate_dir(model_id, ROOT / "models")),
    )


def _clip_h1_to_cutoff(df_h1: pl.DataFrame, cutoff) -> pl.DataFrame:
    if df_h1 is None or len(df_h1) == 0 or "time" not in df_h1.columns:
        return df_h1
    return df_h1.filter(pl.col("time") <= cutoff)


def run_training_batch(
    *,
    limit: int = 24,
    execute: bool = False,
    symbol: str = "GOLDmicro",
    timeframe: str = "M15",
    git_sha: str = "unknown",
    batch_id: str | None = None,
    mt5_path: str | None = None,
    samples: int = 1,
    sample_stride_bars: int = 1000,
) -> Path:
    if samples < 1:
        raise ValueError("samples must be >= 1")
    if sample_stride_bars < 1:
        raise ValueError("sample_stride_bars must be >= 1")

    stamp = batch_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = ROOT / "models" / "reports" / f"challenger_train_{stamp}"
    report_dir.mkdir(parents=True, exist_ok=True)
    specs = build_challenger_specs(
        root=ROOT / "models",
        limit=limit,
        batch_id=stamp,
    )
    plan_path = write_batch_plan(specs, report_dir / "plan.json")
    total_jobs = len(specs) * samples

    print("=== GOLDmicro Multi-Challenger Training Batch ===")
    print(f"Batch ID       : {stamp}")
    print(f"Configurations : {len(specs)}")
    print(f"Samples/config : {samples}")
    print(f"Training jobs  : {len(specs)} x {samples} = {total_jobs}")
    if samples > 1:
        print(f"Sample stride  : {sample_stride_bars:,} {timeframe} bars")
    print(f"Plan           : {plan_path}")
    print("Active         : NEVER overwritten")
    print("Promotion      : NEVER automatic")

    if not execute:
        payload = {
            "batch_id": stamp,
            "mode": "PLAN_ONLY",
            "plan": str(plan_path),
            "configurations": len(specs),
            "samples_per_configuration": samples,
            "sample_stride_bars": sample_stride_bars,
            "total_training_jobs": total_jobs,
            "results": [],
        }
        out = report_dir / "batch_training_results.json"
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("Mode           : PLAN ONLY")
        print("Add --execute when ready to train the complete matrix.")
        return out

    print("Mode           : EXECUTE TRAINING (NON-LIVE)")
    print("MT5 access     : READ-ONLY ATTACH TO LOGGED-IN TERMINAL SESSION")
    connector = MT5ResearchConnector(path=mt5_path)
    results = []
    snapshot = {}
    sample_windows: list[dict] = []
    connector.connect()
    try:
        max_bars = max(spec.train_bars for spec in specs)
        extra_bars = sample_stride_bars * (samples - 1)
        master_bars = max_bars + extra_bars
        print(f"Freezing master {timeframe} snapshot: {master_bars:,} bars")
        raw_m15_master = connector.get_market_data(symbol, timeframe, master_bars)
        if len(raw_m15_master) < master_bars:
            raise RuntimeError(
                f"insufficient {timeframe} history for {samples} samples: "
                f"need {master_bars:,}, received {len(raw_m15_master):,}. "
                "Reduce --samples or --sample-stride-bars."
            )

        # H1 timestamps are later shifted by +1h by the causal V2 wrapper before
        # joining into M15, so clipping by bar-open timestamp is safe here.
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
        snapshot = {
            "symbol": symbol,
            "timeframe": timeframe,
            "master_m15": m15_meta,
            "master_h1": h1_meta,
            "master_fingerprint": master_fingerprint,
            "samples_per_configuration": samples,
            "sample_stride_bars": sample_stride_bars,
            "mt5_access_mode": "logged_in_terminal_session_read_only",
        }

        job_no = 0
        for sample_index in range(1, samples + 1):
            bars_to_drop = sample_stride_bars * (samples - sample_index)
            sample_rows = len(raw_m15_master) - bars_to_drop
            raw_m15 = raw_m15_master.head(sample_rows)
            cutoff = raw_m15["time"][-1]
            raw_h1 = _clip_h1_to_cutoff(raw_h1_master, cutoff)
            fingerprint = (
                f"{master_fingerprint}:sample={sample_index}:"
                f"m15_rows={len(raw_m15)}:h1_rows={len(raw_h1)}:cutoff={cutoff}"
            )
            window = {
                "sample_index": sample_index,
                "m15_rows": len(raw_m15),
                "h1_rows": len(raw_h1),
                "cutoff": str(cutoff),
                "fingerprint": fingerprint,
            }
            sample_windows.append(window)
            print(
                f"\n--- Sample {sample_index}/{samples} | cutoff={cutoff} | "
                f"M15 rows={len(raw_m15):,} ---"
            )

            for base_spec in specs:
                job_no += 1
                spec = _sample_spec(base_spec, sample_index=sample_index)
                print(f"[{job_no:03d}/{total_jobs:03d}] {spec.model_id}")
                try:
                    result = train_candidate(
                        spec,
                        connector=connector,
                        symbol=symbol,
                        timeframe=timeframe,
                        git_sha=git_sha,
                        raw_m15=raw_m15,
                        raw_h1=raw_h1,
                        data_fingerprint=fingerprint,
                    )
                    result.update(
                        {
                            "base_model_id": base_spec.model_id,
                            "sample_index": sample_index,
                            "sample_count": samples,
                            "sample_cutoff": str(cutoff),
                        }
                    )
                    results.append(result)
                    auc = result.get("train_metrics", {}).get("xgb_test_score")
                    gap = None
                    train_auc = result.get("train_metrics", {}).get("xgb_train_score")
                    if isinstance(auc, (int, float)) and isinstance(train_auc, (int, float)):
                        gap = train_auc - auc
                    print(f"  PASS train | test AUC={auc} | gap={gap}")
                except Exception as exc:
                    results.append(
                        {
                            "model_id": spec.model_id,
                            "base_model_id": base_spec.model_id,
                            "batch_id": stamp,
                            "success": False,
                            "error": str(exc),
                            "xgb_profile": spec.xgb_profile,
                            "feature_profile": spec.feature_profile,
                            "cost_profile": spec.cost_profile,
                            "hmm_lookback": spec.hmm_lookback,
                            "confidence_threshold": spec.confidence_threshold,
                            "seed": spec.seed,
                            "train_bars": spec.train_bars,
                            "data_fingerprint": fingerprint,
                            "sample_index": sample_index,
                            "sample_count": samples,
                            "sample_cutoff": str(cutoff),
                        }
                    )
                    print(f"  FAIL train | {exc}")
    finally:
        connector.disconnect()

    snapshot["sample_windows"] = sample_windows
    (report_dir / "market_snapshot.json").write_text(
        json.dumps(snapshot, indent=2, default=str), encoding="utf-8"
    )

    out = report_dir / "batch_training_results.json"
    out.write_text(
        json.dumps(
            {
                "batch_id": stamp,
                "mode": "EXECUTE_NON_LIVE",
                "generated_at": datetime.now().isoformat(),
                "snapshot": snapshot,
                "plan": str(plan_path),
                "configurations": len(specs),
                "samples_per_configuration": samples,
                "sample_stride_bars": sample_stride_bars,
                "total_training_jobs": total_jobs,
                "results": results,
                "promotion_performed": False,
                "note": (
                    "AUC is only a multi-sample pre-screen. PF/DD/cost/walk-forward/"
                    "shadow gates remain mandatory."
                ),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    passed = sum(1 for r in results if r.get("success"))
    print(f"Completed      : {passed}/{len(results)} training jobs successfully")
    print(f"Results        : {out}")
    print("Next           : multi-sample screen -> strategy PF/DD/cost -> shadow")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--execute", action="store_true", help="actually train candidate models locally")
    ap.add_argument("--symbol", default="GOLDmicro")
    ap.add_argument("--timeframe", default="M15")
    ap.add_argument("--git-sha", default="unknown")
    ap.add_argument("--batch-id", help="optional stable id for this research batch")
    ap.add_argument("--samples", type=int, default=1, help="chronological samples per configuration")
    ap.add_argument(
        "--sample-stride-bars",
        type=int,
        default=1000,
        help="distance between chronological sample cutoffs in execution-timeframe bars",
    )
    ap.add_argument(
        "--mt5-path",
        help="optional MT5 terminal executable path; credentials are never accepted by this research runner",
    )
    args = ap.parse_args()

    run_training_batch(
        limit=args.limit,
        execute=args.execute,
        symbol=args.symbol,
        timeframe=args.timeframe,
        git_sha=args.git_sha,
        batch_id=args.batch_id,
        mt5_path=args.mt5_path,
        samples=args.samples,
        sample_stride_bars=args.sample_stride_bars,
    )


if __name__ == "__main__":
    main()
