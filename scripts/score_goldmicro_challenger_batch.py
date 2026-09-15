"""Screen many trained GOLDmicro challengers in one pass.

This is a cheap research funnel only. It uses training/test AUC, train-test gap,
and sample count to remove obviously weak/overfit candidates before expensive
strategy PF/DD/cost/shadow validation. It never promotes a model.
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

from src.candidate_screening import (
    ScreeningThresholds,
    screen_many,
    select_diverse_shortlist,
)


def score_batch(
    input_path: Path,
    *,
    top_k: int = 6,
    min_test_auc: float = 0.55,
    max_gap: float = 0.12,
    min_test_samples: int = 500,
) -> Path:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    rows = payload.get("results", [])
    thresholds = ScreeningThresholds(
        min_test_auc=min_test_auc,
        max_generalization_gap=max_gap,
        min_test_samples=min_test_samples,
    )
    screened = screen_many(rows, thresholds)
    shortlist = select_diverse_shortlist(screened, top_k=top_k)

    report_dir = input_path.parent
    output = report_dir / "candidate_screening.json"
    shadow_queue = report_dir / "shadow_queue.json"

    report = {
        "generated_at": datetime.now().isoformat(),
        "batch_id": payload.get("batch_id"),
        "thresholds": thresholds.__dict__,
        "screened": [x.to_dict() for x in screened],
        "shortlist": [x.to_dict() for x in shortlist],
        "promotion_performed": False,
        "warning": (
            "AUC screening is not strategy validation. PF/DD, GOLDmicro execution-cost, "
            "walk-forward stability and shadow gates are still required."
        ),
    }
    output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    queue = {
        "batch_id": payload.get("batch_id"),
        "state": "AWAITING_STRATEGY_OOS_AND_SHADOW",
        "candidates": [
            {
                "model_id": x.model_id,
                "candidate_dir": str(x.source.get("xgb_path", "")).replace("\\xgboost_model.pkl", "").replace("/xgboost_model.pkl", ""),
                "xgb_path": x.source.get("xgb_path"),
                "hmm_path": x.source.get("hmm_path"),
                "training_data_path": x.source.get("training_data_path"),
                "split": x.source.get("split"),
                "data_fingerprint": x.source.get("data_fingerprint"),
                "screen_adjusted_score": x.adjusted_score,
                "screen_test_auc": x.test_auc,
                "screen_gap": x.generalization_gap,
                "pf_dd_status": "NOT_YET_EVALUATED",
                "shadow_status": "NOT_YET_EVALUATED",
            }
            for x in shortlist
        ],
        "promotion_performed": False,
    }
    shadow_queue.write_text(json.dumps(queue, indent=2, default=str), encoding="utf-8")

    print("\n=== GOLDmicro Challenger Batch Screen ===")
    print(f"Input      : {input_path}")
    print(f"Candidates : {len(screened)}")
    print(f"Pass       : {sum(1 for x in screened if x.status == 'SCREEN_PASS')}")
    print(f"Shortlist  : {len(shortlist)}")
    print(f"Report     : {output}")
    print(f"Shadow Q   : {shadow_queue}")
    print("NOTE       : shortlist is NOT promotion eligibility; PF/DD is still pending")
    return output


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--min-test-auc", type=float, default=0.55)
    ap.add_argument("--max-gap", type=float, default=0.12)
    ap.add_argument("--min-test-samples", type=int, default=500)
    args = ap.parse_args()
    score_batch(
        args.input,
        top_k=args.top_k,
        min_test_auc=args.min_test_auc,
        max_gap=args.max_gap,
        min_test_samples=args.min_test_samples,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
