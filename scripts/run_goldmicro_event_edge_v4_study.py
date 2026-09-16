"""One-run GOLDmicro Event Economic-Edge V4 regression study.

Default matrix: 24 configurations x 5 chronological probes = 120 jobs.
Pre-screen is frozen before the first V4 run and uses only untouched-OOS
regression diagnostics: positive rank correlation, positive calibration slope,
positive observed mean in the predicted-positive gross-R tail, and >=100 OOS
events on at least 4/5 probes.  PF/DD/cost is not evaluated here.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
from statistics import median
import subprocess
import sys

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.challenger_batch import build_challenger_specs
from src.goldmicro_event_edge_trainer import train_event_edge_candidate
from src.goldmicro_event_target import EventTargetConfig
from src.model_registry import candidate_dir, sha256_file
from src.mt5_research_connector import MT5ResearchConnector

EXPECTED_SAMPLES = 5
MIN_PASS_RATE = 0.80
MIN_OOS_EVENTS = 100


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
        "rows": len(df), "start": str(times[0]) if times else None,
        "end": str(times[-1]) if times else None, "path": str(path),
        "sha256": sha256_file(path),
    }


def _sample_spec(base_spec, sample_index: int):
    base_id = f"{base_spec.model_id}-edgev4"
    model_id = f"{base_id}-s{sample_index:02d}"
    return replace(
        base_spec,
        model_id=model_id,
        output_dir=str(candidate_dir(model_id, ROOT / "models")),
    ), base_id


def _probe_pass(row: dict) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not row.get("success"):
        return False, [f"training failed: {row.get('error', 'unknown')}"]
    metrics = row.get("metrics") or {}
    if int(row.get("test_event_count") or 0) < MIN_OOS_EVENTS:
        reasons.append(f"OOS events {row.get('test_event_count')} < {MIN_OOS_EVENTS}")
    if float(metrics.get("oos_rank_correlation") or 0.0) <= 0.0:
        reasons.append("OOS rank correlation <= 0")
    if float(metrics.get("oos_calibration_slope") or 0.0) <= 0.0:
        reasons.append("OOS calibration slope <= 0")
    if not bool(metrics.get("positive_pred_tail_sign_consistent")):
        reasons.append("predicted-positive gross-R tail has non-positive realized mean")
    return not reasons, reasons


def _summarize(results: list[dict], expected_samples: int, min_pass_rate: float) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for row in results:
        groups.setdefault(str(row.get("base_model_id") or "unknown"), []).append(row)
    required = int(expected_samples * min_pass_rate + 0.999999)
    summaries: list[dict] = []
    for base_id, group in groups.items():
        group = sorted(group, key=lambda x: int(x.get("sample_index") or 0))
        samples = []
        passes = 0
        ranks: list[float] = []
        slopes: list[float] = []
        maes: list[float] = []
        rmses: list[float] = []
        for row in group:
            passed, reasons = _probe_pass(row)
            passes += int(passed)
            metrics = row.get("metrics") or {}
            if row.get("success"):
                ranks.append(float(metrics.get("oos_rank_correlation") or 0.0))
                slopes.append(float(metrics.get("oos_calibration_slope") or 0.0))
                maes.append(float(metrics.get("oos_mae") or 0.0))
                rmses.append(float(metrics.get("oos_rmse") or 0.0))
            samples.append({
                "model_id": row.get("model_id"),
                "sample_index": row.get("sample_index"),
                "sample_cutoff": row.get("sample_cutoff"),
                "success": bool(row.get("success")),
                "screen_status": "EDGE_SCREEN_PASS" if passed else "EDGE_SCREEN_REJECT",
                "screen_reasons": reasons,
                "metrics": metrics,
                "xgb_path": row.get("xgb_path"),
                "calibration_path": row.get("calibration_path"),
                "hmm_path": row.get("hmm_path"),
                "training_data_path": row.get("training_data_path"),
                "split": row.get("split"),
                "data_fingerprint": row.get("data_fingerprint"),
            })
        first = group[0] if group else {}
        reasons: list[str] = []
        if len(group) != expected_samples:
            reasons.append(f"sample records {len(group)} != expected {expected_samples}")
        trained = sum(1 for x in group if x.get("success"))
        if trained != expected_samples:
            reasons.append(f"trained {trained}/{expected_samples}")
        if passes < required:
            reasons.append(f"screen pass {passes}/{expected_samples} < required {required}/{expected_samples}")
        summaries.append({
            "base_model_id": base_id,
            "status": "EDGE_STABLE_SCREEN_PASS" if not reasons else "EDGE_STABLE_SCREEN_REJECT",
            "sample_count": len(group),
            "trained_count": trained,
            "pass_count": passes,
            "pass_rate": passes / expected_samples,
            "median_oos_rank_correlation": median(ranks) if ranks else 0.0,
            "min_oos_rank_correlation": min(ranks) if ranks else 0.0,
            "median_oos_calibration_slope": median(slopes) if slopes else 0.0,
            "median_oos_mae": median(maes) if maes else 0.0,
            "median_oos_rmse": median(rmses) if rmses else 0.0,
            "xgb_profile": first.get("xgb_profile"),
            "feature_profile": first.get("feature_profile"),
            "cost_profile": first.get("cost_profile"),
            "reasons": reasons if reasons else [f"passed >= {required}/{expected_samples} frozen V4 regression screens"],
            "samples": samples,
        })
    return sorted(
        summaries,
        key=lambda x: (
            x["status"] == "EDGE_STABLE_SCREEN_PASS",
            x["pass_rate"],
            x["median_oos_rank_correlation"],
            x["min_oos_rank_correlation"],
            x["median_oos_calibration_slope"],
            -x["median_oos_rmse"],
        ), reverse=True,
    )


def _shortlist(summaries: list[dict], top_k: int) -> list[dict]:
    passed = [x for x in summaries if x["status"] == "EDGE_STABLE_SCREEN_PASS"]
    chosen: list[dict] = []
    used: set[tuple[str, str]] = set()
    used_ids: set[str] = set()
    for item in passed:
        sig = (str(item.get("xgb_profile")), str(item.get("feature_profile")))
        if sig in used:
            continue
        chosen.append(item); used.add(sig); used_ids.add(item["base_model_id"])
        if len(chosen) >= top_k:
            return chosen
    for item in passed:
        if item["base_model_id"] in used_ids:
            continue
        chosen.append(item)
        if len(chosen) >= top_k:
            break
    return chosen


def run_study(*, limit: int = 24, samples: int = EXPECTED_SAMPLES,
              sample_stride_bars: int = 1000, top_k: int = 6,
              symbol: str = "GOLDmicro", timeframe: str = "M15") -> Path:
    if samples != EXPECTED_SAMPLES:
        raise ValueError(f"V4 frozen protocol requires exactly {EXPECTED_SAMPLES} probes")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    git_sha = _resolve_git_sha()
    report_dir = ROOT / "models" / "reports" / f"event_edge_v4_{stamp}"
    report_dir.mkdir(parents=True, exist_ok=True)
    specs = build_challenger_specs(
        root=ROOT / "models", limit=limit, batch_id=f"edgev4-{stamp}",
        confidence_thresholds=(0.60,), cost_profiles=("normal",),
    )
    event_cfg = EventTargetConfig(max_holding_bars=32, event_cooldown_bars=10)
    total_jobs = len(specs) * samples

    print("=== GOLDmicro Event Economic-Edge V4 Study ===")
    print(f"Git SHA        : {git_sha}")
    print(f"Configurations : {len(specs)}")
    print(f"Samples/config : {samples}")
    print(f"Training jobs  : {len(specs)} x {samples} = {total_jobs}")
    print("Target         : realized gross R of causal SMC setup")
    print("Split          : FIT -> embargo -> CALIBRATION -> embargo -> UNTOUCHED OOS")
    print("Screen         : rank>0, OOS slope>0, positive-tail realized mean>0 on >=4/5 probes")
    print("PF/DD          : NOT EVALUATED")
    print("Live model     : UNCHANGED")
    print("Promotion      : DISABLED")

    connector = MT5ResearchConnector(); results: list[dict] = []; sample_windows: list[dict] = []
    connector.connect()
    try:
        max_bars = max(s.train_bars for s in specs)
        master_bars = max_bars + sample_stride_bars * (samples - 1)
        raw_m15_master = connector.get_market_data(symbol, timeframe, master_bars)
        if len(raw_m15_master) < master_bars:
            raise RuntimeError(f"need {master_bars:,} {timeframe} bars, received {len(raw_m15_master):,}")
        h1_bars = min(max(master_bars // 4 + 100, 1000), 10000)
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
            sample_windows.append({"sample_index": sample_index, "cutoff": str(cutoff),
                                   "m15_rows": len(raw_m15), "h1_rows": len(raw_h1)})
            print(f"\n--- Probe {sample_index}/{samples} | cutoff={cutoff} ---")
            for base_spec in specs:
                job_no += 1
                spec, base_id = _sample_spec(base_spec, sample_index)
                print(f"[{job_no:03d}/{total_jobs:03d}] {spec.model_id}")
                try:
                    result = train_event_edge_candidate(
                        spec, connector=connector, symbol=symbol, timeframe=timeframe,
                        git_sha=git_sha, raw_m15=raw_m15, raw_h1=raw_h1,
                        data_fingerprint=f"{master_fp}:probe={sample_index}:cutoff={cutoff}:edgev4",
                        event_config=event_cfg,
                    )
                    result.update({"base_model_id": base_id, "sample_index": sample_index,
                                   "sample_cutoff": str(cutoff), "cost_profile": "normal"})
                    results.append(result)
                    m = result["metrics"]
                    print(f"  PASS train | rank={m['oos_rank_correlation']:.4f} | slope={m['oos_calibration_slope']:.4f} | "
                          f"MAE={m['oos_mae']:.4f}R | RMSE={m['oos_rmse']:.4f}R | "
                          f"tail={m['positive_pred_tail_count']} obs={m['positive_pred_tail_mean_realized_r']}")
                except Exception as exc:
                    print(f"  FAIL train | {exc}")
                    results.append({"model_id": spec.model_id, "base_model_id": base_id,
                                    "sample_index": sample_index, "sample_cutoff": str(cutoff),
                                    "success": False, "xgb_profile": spec.xgb_profile,
                                    "feature_profile": spec.feature_profile, "cost_profile": "normal",
                                    "metrics": {}, "error": str(exc)})
    finally:
        connector.disconnect()

    summaries = _summarize(results, samples, MIN_PASS_RATE)
    shortlist = _shortlist(summaries, top_k)
    state = "EVENT_EDGE_V4_GATE_PASS" if shortlist else "NO_EVENT_EDGE_CONFIGURATION_CLEARS_GATE"
    report = {
        "generated_at": datetime.now().isoformat(), "batch_id": stamp, "state": state,
        "git_sha": git_sha,
        "target": {"unit": "causal_smc_setup_event", "value": "realized_gross_R",
                   "timeout": "horizon_close", "same_bar_policy": "adverse_sl_first",
                   "max_holding_bars": 32, "event_cooldown_bars": 10},
        "frozen_screen": {"expected_samples": samples, "required_passes": 4,
                          "min_oos_events": MIN_OOS_EVENTS, "rank_correlation": ">0",
                          "oos_calibration_slope": ">0",
                          "predicted_positive_tail_realized_mean": ">0"},
        "snapshots": {"m15": m15_meta, "h1": h1_meta}, "sample_windows": sample_windows,
        "jobs": results, "configurations": summaries, "shortlist": shortlist,
        "strategy_pf_dd_status": "NOT_YET_EVALUATED" if shortlist else "SKIPPED_NO_PREDICTIVE_SURVIVORS",
        "promotion_performed": False,
        "interpretation_guard": "Regression screen is predictive evidence only; no PF/DD or promotion from this report.",
    }
    report_path = report_dir / "event_edge_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    queue_path = report_dir / "event_edge_strategy_oos_queue.json"
    queue_path.write_text(json.dumps({"batch_id": stamp,
        "state": "AWAITING_EVENT_EDGE_STRATEGY_OOS" if shortlist else "NO_ELIGIBLE_EVENT_EDGE_CONFIGURATION",
        "economic_gate": "calibrated_predicted_net_R_gt_0",
        "configurations": shortlist, "promotion_performed": False}, indent=2, default=str), encoding="utf-8")

    print("\n=== Event Economic-Edge V4 Summary ===")
    print(f"Stable pass     : {sum(x['status']=='EDGE_STABLE_SCREEN_PASS' for x in summaries)}/{len(summaries)}")
    print(f"Shortlist       : {len(shortlist)}")
    print(f"State           : {state}")
    print(f"Report          : {report_path}")
    print(f"Strategy queue  : {queue_path}")
    print("Live model      : UNCHANGED")
    print("Promotion       : DISABLED")
    return report_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--sample-stride-bars", type=int, default=1000)
    ap.add_argument("--top-k", type=int, default=6)
    args = ap.parse_args()
    run_study(limit=args.limit, samples=args.samples,
              sample_stride_bars=args.sample_stride_bars, top_k=args.top_k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
