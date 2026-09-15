"""One-command GOLDmicro multi-challenger research pipeline.

Stages in one run:
1) train the same challenger configuration set across N chronological samples,
   using prefix-causal research SMC/HMM/V2 features,
2) cheap AUC/generalization screening,
3) aggregate stability across samples,
4) shortlist configurations,
5) evaluate the shortlist across every retained OOS sample with GOLDmicro
   broker-correct sizing, execution cost, PF/DD/expectancy gates,
6) emit a shadow queue for later observation/audit.

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


def _latest_report_dir(reports_root: Path, batch_id: str | None) -> Path:
    if batch_id:
        return reports_root / f"challenger_train_{batch_id}"
    candidates = sorted(
        reports_root.glob("challenger_train_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError("training report directory not found")
    return candidates[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--sample-stride-bars", type=int, default=1000)
    ap.add_argument("--execute", action="store_true", help="train/evaluate candidates locally; otherwise plan only")
    ap.add_argument("--symbol", default="GOLDmicro")
    ap.add_argument("--timeframe", default="M15")
    ap.add_argument("--git-sha", default="auto")
    ap.add_argument("--batch-id")
    ap.add_argument("--capital", type=float, default=20000.0)
    ap.add_argument("--risk", type=float, default=1.0)
    ap.add_argument("--min-pf", type=float, default=1.30)
    ap.add_argument("--max-dd", type=float, default=10.0)
    ap.add_argument("--max-skip", type=float, default=20.0)
    ap.add_argument("--min-trades", type=int, default=30)
    args = ap.parse_args()

    if args.limit < 1 or args.samples < 1 or args.top_k < 1:
        raise ValueError("limit, samples and top-k must be >= 1")

    git_sha = _resolve_git_sha(args.git_sha)
    batch_id = args.batch_id
    train_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train_goldmicro_causal_challenger_batch.py"),
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
    print("=== GOLDmicro Full Research Validation Pipeline ===")
    print(f"Git SHA       : {git_sha}")
    print(f"Configurations: {args.limit}")
    print(f"Samples/config: {args.samples}")
    print(f"Training jobs : {args.limit} x {args.samples} = {total_jobs}")
    print(f"Shortlist     : {args.top_k}")
    print("SMC history   : PREFIX-CAUSAL RESEARCH WRAPPER")
    print(f"Mode          : {'EXECUTE NON-LIVE' if args.execute else 'PLAN ONLY'}")
    print("Promotion     : DISABLED")

    result = subprocess.run(train_cmd, cwd=ROOT)
    if result.returncode != 0:
        return result.returncode

    if not args.execute:
        print("\nPLAN COMPLETE. Re-run once with --execute for the complete matrix and OOS gates.")
        return 0

    reports_root = ROOT / "models" / "reports"
    report_dir = _latest_report_dir(reports_root, batch_id)
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

    strategy_report = None
    shadow_queue = report_dir / "shadow_queue.json"
    if args.samples > 1:
        strategy_queue = report_dir / "strategy_oos_queue.json"
        if not strategy_queue.exists():
            raise RuntimeError(f"strategy queue not found: {strategy_queue}")
        strategy_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "run_goldmicro_strategy_oos.py"),
            "--queue", str(strategy_queue),
            "--capital", str(args.capital),
            "--risk", str(args.risk),
            "--min-pf", str(args.min_pf),
            "--max-dd", str(args.max_dd),
            "--max-skip", str(args.max_skip),
            "--min-trades", str(args.min_trades),
        ]
        result = subprocess.run(strategy_cmd, cwd=ROOT)
        if result.returncode != 0:
            return result.returncode
        strategy_report = report_dir / "strategy_oos_report.json"

    print("\n=== FULL RESEARCH VALIDATION COMPLETE ===")
    print(f"Training       : {training_results}")
    if args.samples > 1:
        print(f"AUC screening  : {report_dir / 'multisample_screening.json'}")
        print(f"Strategy OOS   : {strategy_report}")
        print(f"Shadow queue   : {shadow_queue}")
        print("Next gate      : independent review + non-executing shadow observation")
    else:
        print(f"Screening      : {report_dir / 'candidate_screening.json'}")
        print(f"Shadow queue   : {shadow_queue}")
        print("Next gate      : multi-sample strategy OOS validation")
    print("Live model     : UNCHANGED")
    print("Promotion      : DISABLED")
    print("Human Gate     : REQUIRED before any future activation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
