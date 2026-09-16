"""Diagnose structural failure modes in a completed GOLDmicro Event Edge V4 report.

Read-only diagnostic. It does not retrain models, tune gates, evaluate PF/DD, or
change promotion state. The purpose is to separate three questions:
1) why training failed before OOS,
2) whether the gross-R target is stable across chronological partitions, and
3) whether apparent rank/calibration signal survives into untouched OOS.

Once an OOS batch has been inspected it is development evidence, not pristine
confirmatory evidence. This script makes that status explicit.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


def _median(values):
    values = [float(v) for v in values if v is not None]
    return median(values) if values else 0.0


def _sign(value: float, eps: float = 1e-12) -> int:
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def _failure_kind(error: str) -> str:
    text = (error or "").lower()
    if "non-positive slope" in text:
        return "PRE_OOS_CALIBRATION_NONPOSITIVE_SLOPE"
    if "degenerate prediction variance" in text:
        return "PRE_OOS_CALIBRATION_DEGENERATE_PREDICTION"
    if "fit events" in text or "calibration events" in text or "test events" in text or "usable fit/calibration/oos" in text:
        return "INSUFFICIENT_EVENT_COUNT"
    if "hmm" in text:
        return "HMM_FAILURE"
    if "feature" in text:
        return "FEATURE_FAILURE"
    return "OTHER_TRAINING_FAILURE"


def run(report_path: Path) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    jobs = report.get("jobs") or []
    configs = report.get("configurations") or []
    trained = [j for j in jobs if j.get("success")]
    failed = [j for j in jobs if not j.get("success")]

    print("=== GOLDmicro Event Edge V4 Structural Diagnostic ===")
    print(f"Report              : {report_path}")
    print(f"Configurations      : {len(configs)}")
    print(f"Jobs                : {len(jobs)}")
    print(f"Trained             : {len(trained)}/{len(jobs)}")
    print("Gate changes        : DISABLED")
    print("PF/DD               : NOT EVALUATED")
    print("Promotion           : DISABLED")
    print("Evidence status     : DEVELOPMENT / OOS ALREADY INSPECTED")

    kinds = Counter(_failure_kind(str(j.get("error") or "")) for j in failed)
    exact = Counter(str(j.get("error") or "unknown") for j in failed)
    print("\nTraining failure classes:")
    for key, count in kinds.most_common():
        print(f"  {key:44s} {count:3d}")
    print("\nTop exact training errors:")
    for msg, count in exact.most_common(10):
        print(f"  {count:3d}  {msg}")

    if trained:
        fit_mean = [j.get("fit_target_mean_r") for j in trained]
        cal_mean = [j.get("calibration_target_mean_r") for j in trained]
        test_mean = [j.get("test_target_mean_r") for j in trained]
        pre_slope = [(j.get("calibration") or {}).get("slope") for j in trained]
        oos_slope = [(j.get("metrics") or {}).get("oos_calibration_slope") for j in trained]
        rank = [(j.get("metrics") or {}).get("oos_rank_correlation") for j in trained]
        tail_real = [(j.get("metrics") or {}).get("positive_pred_tail_mean_realized_r") for j in trained]
        tail_pred = [(j.get("metrics") or {}).get("positive_pred_tail_mean_predicted_r") for j in trained]

        sign_flips_fit_test = sum(
            1 for a, b in zip(fit_mean, test_mean)
            if a is not None and b is not None and _sign(float(a)) != _sign(float(b))
        )
        sign_flips_cal_test = sum(
            1 for a, b in zip(cal_mean, test_mean)
            if a is not None and b is not None and _sign(float(a)) != _sign(float(b))
        )
        negative_oos_slope = sum(1 for x in oos_slope if x is not None and float(x) <= 0)
        nonpositive_rank = sum(1 for x in rank if x is not None and float(x) <= 0)
        nonpositive_tail = sum(1 for x in tail_real if x is None or float(x) <= 0)

        print("\nTarget / chronology diagnostics (trained jobs):")
        print(f"  fit target mean R median           : {_median(fit_mean):+.4f}")
        print(f"  calibration target mean R median   : {_median(cal_mean):+.4f}")
        print(f"  OOS target mean R median           : {_median(test_mean):+.4f}")
        print(f"  fit->OOS mean sign flips           : {sign_flips_fit_test}/{len(trained)}")
        print(f"  calibration->OOS mean sign flips   : {sign_flips_cal_test}/{len(trained)}")
        print(f"  pre-OOS affine slope median        : {_median(pre_slope):+.4f}")
        print(f"  OOS calibration slope median       : {_median(oos_slope):+.4f}")
        print(f"  OOS slope <= 0                     : {negative_oos_slope}/{len(trained)}")
        print(f"  OOS rank <= 0                      : {nonpositive_rank}/{len(trained)}")
        print(f"  positive-pred tail realized <= 0   : {nonpositive_tail}/{len(trained)}")
        print(f"  tail predicted mean R median       : {_median(tail_pred):+.4f}")
        print(f"  tail realized mean R median        : {_median(tail_real):+.4f}")

    grouped = defaultdict(list)
    for j in jobs:
        key = (str(j.get("xgb_profile") or ""), str(j.get("feature_profile") or ""))
        grouped[key].append(j)

    print("\nBy model/feature profile:")
    for (xgb_profile, feature_profile), group in sorted(grouped.items()):
        ok = [j for j in group if j.get("success")]
        probe_pass = 0
        for j in ok:
            m = j.get("metrics") or {}
            if (
                float(m.get("oos_rank_correlation") or 0.0) > 0.0
                and float(m.get("oos_calibration_slope") or 0.0) > 0.0
                and bool(m.get("positive_pred_tail_sign_consistent"))
                and int(j.get("test_event_count") or 0) >= 100
            ):
                probe_pass += 1
        print(
            f"  {xgb_profile:12s} {feature_profile:14s} "
            f"trained={len(ok):2d}/{len(group):2d} probePass={probe_pass:2d}/{len(group):2d} "
            f"rankMed={_median([(j.get('metrics') or {}).get('oos_rank_correlation') for j in ok]):+.4f} "
            f"tailObsMed={_median([(j.get('metrics') or {}).get('positive_pred_tail_mean_realized_r') for j in ok]):+.4f}R"
        )

    print("\nInterpretation guard:")
    print("  Do not relax V4 gates from this diagnostic. The inspected historical OOS is now")
    print("  development evidence; any redesigned learner/target requires fresh future evidence")
    print("  before confirmatory PF/DD or promotion claims.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, type=Path)
    args = ap.parse_args()
    run(args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
