"""Diagnose why Event Economic-Edge V4 predictive screening produced no stable survivors.

Reads an existing event_edge_report.json only. It does not retrain, change gates,
run PF/DD, or promote any model.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import median


def _f(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, type=Path)
    args = ap.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    configs = report.get("configurations") or []

    buckets = Counter()
    total_samples = 0
    trained_samples = 0
    passed_samples = 0

    print("=== GOLDmicro Event Edge V4 Predictive Failure Diagnostic ===")
    print(f"Report          : {args.report}")
    print(f"Configurations  : {len(configs)}")
    print("Gate changes    : DISABLED")
    print("PF/DD           : NOT EVALUATED")
    print("Promotion       : DISABLED")
    print()

    rows = []
    for cfg in configs:
        samples = cfg.get("samples") or []
        total_samples += len(samples)
        trained_samples += sum(bool(s.get("success")) for s in samples)
        passed_samples += sum(s.get("screen_status") == "EDGE_SCREEN_PASS" for s in samples)
        ranks, slopes, tail_obs, tail_pred, maes, rmses = [], [], [], [], [], []
        sample_passes = 0
        for sample in samples:
            if sample.get("screen_status") == "EDGE_SCREEN_PASS":
                sample_passes += 1
            for reason in sample.get("screen_reasons") or []:
                text = str(reason)
                if "training failed" in text:
                    buckets["TRAINING_FAILED"] += 1
                elif "OOS events" in text:
                    buckets["MIN_OOS_EVENTS"] += 1
                elif "rank correlation" in text:
                    buckets["RANK_NONPOSITIVE"] += 1
                elif "calibration slope" in text:
                    buckets["SLOPE_NONPOSITIVE"] += 1
                elif "tail" in text:
                    buckets["POSITIVE_TAIL_NONPOSITIVE_REALIZED"] += 1
                else:
                    buckets["OTHER"] += 1
            m = sample.get("metrics") or {}
            if sample.get("success"):
                ranks.append(_f(m.get("oos_rank_correlation")))
                slopes.append(_f(m.get("oos_calibration_slope")))
                maes.append(_f(m.get("oos_mae")))
                rmses.append(_f(m.get("oos_rmse")))
                if m.get("positive_pred_tail_mean_realized_r") is not None:
                    tail_obs.append(_f(m.get("positive_pred_tail_mean_realized_r")))
                if m.get("positive_pred_tail_mean_predicted_r") is not None:
                    tail_pred.append(_f(m.get("positive_pred_tail_mean_predicted_r")))
        rows.append((
            sample_passes,
            median(ranks) if ranks else 0.0,
            min(ranks) if ranks else 0.0,
            median(slopes) if slopes else 0.0,
            median(tail_obs) if tail_obs else None,
            median(tail_pred) if tail_pred else None,
            median(maes) if maes else 0.0,
            median(rmses) if rmses else 0.0,
            cfg.get("base_model_id"),
        ))

    print(f"Samples         : {total_samples}")
    print(f"Trained         : {trained_samples}/{total_samples}")
    print(f"Probe passes    : {passed_samples}/{total_samples}")
    print("\nFailure buckets:")
    for key in ["TRAINING_FAILED", "MIN_OOS_EVENTS", "RANK_NONPOSITIVE", "SLOPE_NONPOSITIVE", "POSITIVE_TAIL_NONPOSITIVE_REALIZED", "OTHER"]:
        print(f"  {key:38s} {buckets[key]:3d}")

    print("\nPer configuration:")
    rows.sort(key=lambda x: (x[0], x[1], x[3]), reverse=True)
    for p, med_rank, min_rank, med_slope, med_tail_obs, med_tail_pred, med_mae, med_rmse, base_id in rows:
        obs = "n/a" if med_tail_obs is None else f"{med_tail_obs:.3f}R"
        pred = "n/a" if med_tail_pred is None else f"{med_tail_pred:.3f}R"
        print(
            f"pass={p}/5 rank med/min={med_rank:+.3f}/{min_rank:+.3f} "
            f"slope med={med_slope:+.3f} tail obs/pred={obs}/{pred} "
            f"MAE={med_mae:.3f}R RMSE={med_rmse:.3f}R {base_id}"
        )

    print("\nInterpretation guard:")
    print("  Diagnostic only. Do not relax the frozen V4 screen from these results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
