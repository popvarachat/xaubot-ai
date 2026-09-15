"""Train many GOLDmicro challengers in one isolated batch.

Default mode is PLAN ONLY. Use --execute to perform local read-only market-data
training. Candidate models are written only under models/candidates and never to
active Champion paths. No orders are sent and no promotion occurs.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.challenger_batch import build_challenger_specs, write_batch_plan
from src.goldmicro_candidate_trainer import train_candidate
from src.config import get_config
from src.mt5_connector import MT5Connector


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--execute", action="store_true", help="actually train candidate models locally")
    ap.add_argument("--symbol", default="GOLDmicro")
    ap.add_argument("--timeframe", default="M15")
    ap.add_argument("--git-sha", default="unknown")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = ROOT / "models" / "reports" / f"challenger_train_{stamp}"
    report_dir.mkdir(parents=True, exist_ok=True)
    specs = build_challenger_specs(root=ROOT / "models", limit=args.limit)
    plan_path = write_batch_plan(specs, report_dir / "plan.json")

    print("=== GOLDmicro Multi-Challenger Training Batch ===")
    print(f"Candidates : {len(specs)}")
    print(f"Plan       : {plan_path}")
    print("Active     : NEVER overwritten")
    print("Promotion  : NEVER automatic")

    if not args.execute:
        print("Mode       : PLAN ONLY")
        print("Add --execute when ready to train all candidates in this batch.")
        return

    print("Mode       : EXECUTE TRAINING (NON-LIVE)")
    config = get_config()
    connector = MT5Connector(
        login=config.mt5_login,
        password=config.mt5_password,
        server=config.mt5_server,
        path=config.mt5_path,
    )
    results = []
    connector.connect()
    try:
        for i, spec in enumerate(specs, start=1):
            print(f"[{i:02d}/{len(specs):02d}] {spec.model_id}")
            try:
                result = train_candidate(
                    spec,
                    connector=connector,
                    symbol=args.symbol,
                    timeframe=args.timeframe,
                    git_sha=args.git_sha,
                )
                results.append(result)
                auc = result.get("train_metrics", {}).get("xgb_test_score")
                print(f"  PASS train | test score={auc}")
            except Exception as exc:
                results.append({"model_id": spec.model_id, "success": False, "error": str(exc)})
                print(f"  FAIL train | {exc}")
    finally:
        connector.disconnect()

    out = report_dir / "batch_training_results.json"
    out.write_text(json.dumps({"results": results}, indent=2, default=str), encoding="utf-8")
    passed = sum(1 for r in results if r.get("success"))
    print(f"Completed  : {passed}/{len(results)} candidates trained successfully")
    print(f"Results    : {out}")
    print("Next       : validate/OOS/cost/shadow; do NOT promote from training score alone")


if __name__ == "__main__":
    main()
