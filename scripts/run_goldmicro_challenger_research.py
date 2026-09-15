"""One-command GOLDmicro multi-challenger research pipeline.

Stages in one run:
1) train the same challenger configuration set across N chronological samples,
2) cheap AUC/generalization screening,
3) aggregate stability across samples,
4) emit a configuration shortlist for strategy PF/DD/cost validation.

No candidate is promoted and no live trading behavior is changed.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def _resolve_git_sha(explicit: str | None) -> str:
    if explicit and explicit != "auto":
        return explicit
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--sample-stride-bars", type=int, default=1000)
    ap.add_argument("--execute", action="store_true", help="train candidates locally; otherwise plan only")
    ap.add_argument("--symbol", default="GOLDmicro")
    ap.add_argument("--timeframe", default="M15")
    ap.add_argument("--git-sha", default="auto")
    ap.add_argument("--batch-id")
    args = ap.parse_args()

    git_sha = _resolve_git_sha(args.git_sha)
    batch_id = args.batch_id
    train_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train_goldmicro_challenger_batch.py"),
        "--limit", str(args.limit),
        "--samples", str(args.samples),
        "--sample-stride-bars", str(args.sample_stride_bars),
        "--symbol", args.symbol,
        "--timeframe", args.timeframe,
        "--git-sha", git_sha,
    ]
    if batch_id:
        train_cmd += ["--batch-id", batch_id]
    if args.execute:
        train_cmd.append("--execute")

    total_jobs = args.limit * args.samples
    print("=== GOLDmicro Challenger Research Pipeline ===")
    print(f"Git SHA       : {git_sha}")
    print(f"Configurations: {args.limit}")
    print(f"Samples/config: {args.samples}")
    print(f"Training jobs : {args.limit} x {args.samples} = {total_jobs}")
    print(f"Shortlist     : {args.top_k}")
    print(f"Mode          : {'EXECUTE NON-LIVE' if args.execute else 'PLAN ONLY'}")
    print("Promotion     : DISABLED")
    result = subprocess.run(train_cmd, cwd=ROOT)
    if result.returncode != 0:
        return result.returncode

    if not args.execute:
        print("\nPLAN COMPLETE. Re-run once with --execute for the complete matrix.")
        return 0

    reports_root = ROOT / "models" / "reports"
    if batch_id:
        report_dir = reports_root / f"challenger_train_{batch_id}"
    else:
        candidates = sorted(
            reports_root.glob("challenger_train_*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise RuntimeError("training report directory not found")
        report_dir = candidates[0]

    training_results = report_dir / "batch_training_results.json"
    if not training_results.exists():
        raise RuntimeError(f"training results not found: {training_results}")

    if args.samples > 1:
        score_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "score_goldmicro_multisample_batch.py"),
            "--input", str(training_results),
            "--top-k", str(args.top_k),
        ]
    else:
        score_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "score_goldmicro_challenger_batch.py"),
            "--input", str(training_results),
            "--top-k", str(args.top_k),
        ]

    result = subprocess.run(score_cmd, cwd=ROOT)
    if result.returncode != 0:
        return result.returncode

    print("\n=== RESEARCH MATRIX COMPLETE ===")
    print(f"Training      : {training_results}")
    if args.samples > 1:
        print(f"Screening     : {report_dir / 'multisample_screening.json'}")
        print(f"Strategy queue: {report_dir / 'strategy_oos_queue.json'}")
    else:
        print(f"Screening     : {report_dir / 'candidate_screening.json'}")
        print(f"Shadow Q      : {report_dir / 'shadow_queue.json'}")
    print("Next gate     : strategy OOS + GOLDmicro PF/DD/cost + shadow observation")
    print("Live model    : UNCHANGED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
