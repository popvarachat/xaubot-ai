"""Run the full non-live GOLDmicro research batch in one command.

This orchestrator intentionally runs multiple validation models together:
1) multi-model execution-cost matrix,
2) full spread/slippage/risk/capital cost surface.

It is read-only with respect to MT5 trading: the child scripts only inspect broker
metadata/history and replay legacy backtest trades. No orders are placed,
modified, or closed.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_MATRIX = REPO_ROOT / "scripts" / "run_goldmicro_model_matrix.py"
COST_SURFACE = REPO_ROOT / "scripts" / "run_goldmicro_cost_surface.py"


def _run(label: str, cmd: list[str]) -> int:
    print("\n" + "=" * 88)
    print(label)
    print("=" * 88)
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    if proc.returncode != 0:
        print(f"\n{label}: FAILED with exit code {proc.returncode}", file=sys.stderr)
    else:
        print(f"\n{label}: COMPLETE")
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the complete GOLDmicro 0.10-lot multi-model research batch"
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Optional legacy XLSX input passed to all child models",
    )
    parser.add_argument(
        "--capitals",
        default=None,
        help="Optional comma-separated THB capital grid passed to all child models",
    )
    parser.add_argument(
        "--risks",
        default=None,
        help="Optional comma-separated risk percentages, e.g. 0.5,0.75,1.0",
    )
    parser.add_argument("--min-pf", type=float, default=1.30)
    parser.add_argument("--max-dd", type=float, default=10.0)
    parser.add_argument("--max-skip", type=float, default=20.0)
    parser.add_argument(
        "--skip-cost-surface",
        action="store_true",
        help="Run only the named model matrix",
    )
    args = parser.parse_args()

    common: list[str] = []
    if args.input:
        common.extend(["--input", str(args.input.resolve())])
    if args.capitals:
        common.extend(["--capitals", args.capitals])
    common.extend(
        [
            "--min-pf",
            str(args.min_pf),
            "--max-dd",
            str(args.max_dd),
            "--max-skip",
            str(args.max_skip),
        ]
    )

    print("\n=== GOLDmicro FULL MULTI-MODEL BATCH (READ ONLY) ===")
    print(f"Started : {datetime.now().isoformat(timespec='seconds')}")
    print("Lot grid: strategy minimum 0.10 lot, increment 0.10 lot")
    print("Scope   : multiple models in one run; no live orders")
    print("Includes:")
    print("  A) Named execution-cost model matrix (M0-M5 across multiple risks/capitals)")
    print("  B) Full cost surface (spread x slippage x risk x capital)")

    matrix_cmd = [sys.executable, str(MODEL_MATRIX)] + common
    if args.risks:
        matrix_cmd.extend(["--risks", args.risks])

    rc_matrix = _run("A) MULTI-MODEL MATRIX", matrix_cmd)
    if rc_matrix != 0:
        return rc_matrix

    if args.skip_cost_surface:
        print("\nCost surface skipped by request.")
        return 0

    surface_cmd = [sys.executable, str(COST_SURFACE)] + common
    if args.risks:
        surface_cmd.extend(["--risks", args.risks])

    rc_surface = _run("B) FULL COST SURFACE", surface_cmd)
    if rc_surface != 0:
        return rc_surface

    print("\n" + "=" * 88)
    print("FULL BATCH COMPLETE")
    print("Use the two generated JSON reports together for the 0.10-lot GOLDmicro baseline.")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
