"""Score 24 x N GOLDmicro chronological challenger samples in one pass.

The output shortlist contains *configurations*. Each shortlisted configuration
retains all of its chronological sample artifacts for the later strategy
PF/DD/cost walk-forward gate. Nothing here promotes or activates a model.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.candidate_screening import ScreeningThresholds
from src.multisample_screening import summarize_multisample, select_stable_shortlist


def score_multisample_batch(
    input_path: Path,
    *,
    top_k: int = 6,
    min_test_auc: float = 0.55,
    max_gap: float = 0.12,
    min_test_samples: int = 500,
    min_pass_rate: float = 0.80,
) -> Path:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    rows = payload.get("results", [])
    expected_samples = int(payload.get("samples_per_configuration") or 1)
    thresholds = ScreeningThresholds(
        min_test_auc=min_test_auc,
        max_generalization_gap=max_gap,
        min_test_samples=min_test_samples,
    )
    summaries = summarize_multisample(
        rows,
        thresholds=thresholds,
        expected_samples=expected_samples,
        min_pass_rate=min_pass_rate,
    )
    shortlist = select_stable_shortlist(summaries, top_k=top_k)

    report_dir = input_path.parent
    output = report_dir / "multisample_screening.json"
    strategy_queue = report_dir / "strategy_oos_queue.json"

    report = {
        "generated_at": datetime.now().isoformat(),
        "batch_id": payload.get("batch_id"),
        "configurations": len(summaries),
        "samples_per_configuration": expected_samples,
        "training_jobs": len(rows),
        "thresholds": {
            **thresholds.__dict__,
            "min_pass_rate": min_pass_rate,
        },
        "summaries": [x.to_dict() for x in summaries],
        "shortlist": [x.to_dict() for x in shortlist],
        "promotion_performed": False,
        "warning": (
            "Multi-sample AUC consistency is still not strategy validation. "
            "PF/DD, GOLDmicro execution cost, walk-forward stability, shadow "
            "observation and Human Gate remain mandatory."
        ),
    }
    output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    queue = {
        "batch_id": payload.get("batch_id"),
        "state": "AWAITING_STRATEGY_OOS_PF_DD_COST",
        "samples_per_configuration": expected_samples,
        "configurations": [
            {
                "base_model_id": item.base_model_id,
                "pass_rate": item.pass_rate,
                "median_test_auc": item.median_test_auc,
                "min_test_auc": item.min_test_auc,
                "max_gap": item.max_gap,
                "samples": list(item.samples),
                "pf_dd_status": "NOT_YET_EVALUATED",
                "shadow_status": "NOT_YET_EVALUATED",
            }
            for item in shortlist
        ],
        "promotion_performed": False,
    }
    strategy_queue.write_text(
        json.dumps(queue, indent=2, default=str), encoding="utf-8"
    )

    print("\n=== GOLDmicro Multi-Sample Challenger Screen ===")
    print(f"Input          : {input_path}")
    print(f"Configurations : {len(summaries)}")
    print(f"Samples/config : {expected_samples}")
    print(f"Training jobs  : {len(rows)}")
    print(
        f"Stable pass    : "
        f"{sum(1 for x in summaries if x.status == 'STABLE_SCREEN_PASS')}"
    )
    print(f"Shortlist      : {len(shortlist)} configurations")
    print(f"Report         : {output}")
    print(f"Strategy queue : {strategy_queue}")
    print("NOTE           : this is a pre-screen, not promotion eligibility")
    return output


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--min-test-auc", type=float, default=0.55)
    ap.add_argument("--max-gap", type=float, default=0.12)
    ap.add_argument("--min-test-samples", type=int, default=500)
    ap.add_argument("--min-pass-rate", type=float, default=0.80)
    args = ap.parse_args()
    score_multisample_batch(
        args.input,
        top_k=args.top_k,
        min_test_auc=args.min_test_auc,
        max_gap=args.max_gap,
        min_test_samples=args.min_test_samples,
        min_pass_rate=args.min_pass_rate,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
