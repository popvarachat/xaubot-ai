"""Train many GOLDmicro challengers in one isolated batch.

Default mode is PLAN ONLY. Use --execute to perform local read-only market-data
training. Candidate models are written only under models/candidates and never to
active Champion paths. No orders are sent and no promotion occurs.

The execute path freezes one M15/H1 market snapshot for the whole batch so every
candidate is trained and screened against the same data cut. This avoids unfair
candidate-to-candidate drift caused by repeatedly fetching moving MT5 history.
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
from src.model_registry import sha256_file


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


def run_training_batch(
    *,
    limit: int = 24,
    execute: bool = False,
    symbol: str = "GOLDmicro",
    timeframe: str = "M15",
    git_sha: str = "unknown",
    batch_id: str | None = None,
) -> Path:
    stamp = batch_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = ROOT / "models" / "reports" / f"challenger_train_{stamp}"
    report_dir.mkdir(parents=True, exist_ok=True)
    specs = build_challenger_specs(
        root=ROOT / "models",
        limit=limit,
        batch_id=stamp,
    )
    plan_path = write_batch_plan(specs, report_dir / "plan.json")

    print("=== GOLDmicro Multi-Challenger Training Batch ===")
    print(f"Batch ID   : {stamp}")
    print(f"Candidates : {len(specs)}")
    print(f"Plan       : {plan_path}")
    print("Active     : NEVER overwritten")
    print("Promotion  : NEVER automatic")

    if not execute:
        payload = {
            "batch_id": stamp,
            "mode": "PLAN_ONLY",
            "plan": str(plan_path),
            "results": [],
        }
        out = report_dir / "batch_training_results.json"
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("Mode       : PLAN ONLY")
        print("Add --execute when ready to train all candidates in this batch.")
        return out

    print("Mode       : EXECUTE TRAINING (NON-LIVE)")
    config = get_config()
    connector = MT5Connector(
        login=config.mt5_login,
        password=config.mt5_password,
        server=config.mt5_server,
        path=config.mt5_path,
    )
    results = []
    snapshot = {}
    connector.connect()
    try:
        max_bars = max(spec.train_bars for spec in specs)
        print(f"Freezing one {timeframe} snapshot: {max_bars:,} bars")
        raw_m15 = connector.get_market_data(symbol, timeframe, max_bars)
        if len(raw_m15) < 1000:
            raise RuntimeError(f"insufficient {timeframe} snapshot: {len(raw_m15)} bars")

        h1_bars = min(max_bars // 4, 5000)
        print(f"Freezing one H1 snapshot: {h1_bars:,} bars")
        raw_h1 = connector.get_market_data(symbol, "H1", h1_bars)

        m15_meta = _snapshot_metadata(
            raw_m15,
            path=report_dir / "market_snapshot_m15.parquet",
            label=timeframe,
        )
        h1_meta = _snapshot_metadata(
            raw_h1,
            path=report_dir / "market_snapshot_h1.parquet",
            label="H1",
        )
        fingerprint = f"{m15_meta['sha256']}:{h1_meta['sha256']}"
        snapshot = {
            "symbol": symbol,
            "timeframe": timeframe,
            "m15": m15_meta,
            "h1": h1_meta,
            "fingerprint": fingerprint,
        }
        (report_dir / "market_snapshot.json").write_text(
            json.dumps(snapshot, indent=2), encoding="utf-8"
        )

        for i, spec in enumerate(specs, start=1):
            print(f"[{i:02d}/{len(specs):02d}] {spec.model_id}")
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
                results.append(result)
                auc = result.get("train_metrics", {}).get("xgb_test_score")
                gap = None
                train_auc = result.get("train_metrics", {}).get("xgb_train_score")
                if isinstance(auc, (int, float)) and isinstance(train_auc, (int, float)):
                    gap = train_auc - auc
                print(f"  PASS train | test AUC={auc} | gap={gap}")
            except Exception as exc:
                results.append({
                    "model_id": spec.model_id,
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
                    "data_fingerprint": snapshot.get("fingerprint", ""),
                })
                print(f"  FAIL train | {exc}")
    finally:
        connector.disconnect()

    out = report_dir / "batch_training_results.json"
    out.write_text(
        json.dumps(
            {
                "batch_id": stamp,
                "mode": "EXECUTE_NON_LIVE",
                "generated_at": datetime.now().isoformat(),
                "snapshot": snapshot,
                "plan": str(plan_path),
                "results": results,
                "promotion_performed": False,
                "note": (
                    "Training score is only a pre-screen. PF/DD/cost/shadow gates remain mandatory."
                ),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    passed = sum(1 for r in results if r.get("success"))
    print(f"Completed  : {passed}/{len(results)} candidates trained successfully")
    print(f"Results    : {out}")
    print("Next       : batch screen -> OOS strategy/cost -> shadow; do NOT promote from AUC")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--execute", action="store_true", help="actually train candidate models locally")
    ap.add_argument("--symbol", default="GOLDmicro")
    ap.add_argument("--timeframe", default="M15")
    ap.add_argument("--git-sha", default="unknown")
    ap.add_argument("--batch-id", help="optional stable id for this research batch")
    args = ap.parse_args()

    run_training_batch(
        limit=args.limit,
        execute=args.execute,
        symbol=args.symbol,
        timeframe=args.timeframe,
        git_sha=args.git_sha,
        batch_id=args.batch_id,
    )


if __name__ == "__main__":
    main()
