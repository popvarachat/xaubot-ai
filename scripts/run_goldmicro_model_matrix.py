"""Run multiple read-only GOLDmicro execution-cost scenario models in one command.

This is a research orchestrator. It calls the existing isolated capital sweep
runner repeatedly with evidence-based spread/slippage assumptions and aggregates
the minimum passing capital for each model/risk combination.

It does not place, modify, or close MT5 orders.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SWEEP = REPO_ROOT / "scripts" / "run_goldmicro_capital_sweep_isolated.py"
DEFAULT_CAPITALS = "10000,11000,12000,13000,14000,15000,16000,17000,18000,19000,20000,25000,30000,50000,75000"
DEFAULT_RISKS = "0.5,0.75,1.0"


@dataclass(frozen=True)
class ScenarioModel:
    name: str
    spreads: str
    slippage_points_per_side: float
    basis: str


MODELS = (
    ScenarioModel(
        "M0_OPERATIONAL_BAND",
        "40,50,55,60",
        0.0,
        "Observed operational spread band; no extra slippage overlay",
    ),
    ScenarioModel(
        "M1_TYPICAL_SIGNED_MEDIAN",
        "50",
        1.0,
        "Observed median spread plus signed execution-slippage median ~= 1 point",
    ),
    ScenarioModel(
        "M2_CONSERVATIVE_ADVERSE_MEDIAN",
        "55",
        12.0,
        "Observed P95 spread plus adverse-slippage median ~= 12 points",
    ),
    ScenarioModel(
        "M3_UPPER_OPERATIONAL_P75",
        "60",
        25.0,
        "Upper operational spread plus adverse-slippage P75 ~= 25 points",
    ),
    ScenarioModel(
        "M4_STRESS_P90",
        "70",
        63.0,
        "Stress spread plus adverse-slippage P90 ~= 63 points",
    ),
    ScenarioModel(
        "M5_TAIL_STRESS_P95",
        "100",
        103.0,
        "Tail spread plus adverse-slippage P95 ~= 103 points",
    ),
)


def _csv_positive_floats(value: str) -> list[float]:
    vals = [float(x.strip()) for x in value.split(",") if x.strip()]
    if not vals or any(x <= 0 for x in vals):
        raise argparse.ArgumentTypeError("Expected comma-separated positive numbers")
    return vals


def _extract_report_path(stdout: str) -> Path | None:
    match = re.search(r"^Saved report:\s*(.+)$", stdout, flags=re.MULTILINE)
    if not match:
        return None
    return Path(match.group(1).strip())


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a GOLDmicro multi-model research matrix")
    parser.add_argument("--capitals", default=DEFAULT_CAPITALS, help="Comma-separated THB capitals")
    parser.add_argument("--risks", default=DEFAULT_RISKS, help="Comma-separated risk percentages")
    parser.add_argument("--min-pf", type=float, default=1.30)
    parser.add_argument("--max-dd", type=float, default=10.0)
    parser.add_argument("--max-skip", type=float, default=20.0)
    parser.add_argument("--input", type=Path, help="Optional legacy XLSX input")
    args = parser.parse_args()

    risks = _csv_positive_floats(args.risks)
    rows = []

    print("\n=== GOLDmicro Multi-Model Matrix (READ ONLY) ===")
    print(f"Models   : {len(MODELS)}")
    print(f"Risks    : {', '.join(f'{r:g}%' for r in risks)}")
    print(f"Capitals : {args.capitals}")
    print("NOTE: M1-M5 use constant adverse slippage overlays per side for scenario testing.")
    print("      They are validation scenarios, not predictive ML models and not live-trading settings.\n")

    for model in MODELS:
        for risk in risks:
            cmd = [
                sys.executable,
                str(SWEEP),
                "--capitals",
                args.capitals,
                "--spreads",
                model.spreads,
                "--risk",
                str(risk),
                "--min-pf",
                str(args.min_pf),
                "--max-dd",
                str(args.max_dd),
                "--max-skip",
                str(args.max_skip),
                "--slippage",
                str(model.slippage_points_per_side),
            ]
            if args.input:
                cmd.extend(["--input", str(args.input.resolve())])

            print(
                f"Running {model.name} | risk={risk:g}% | spread={model.spreads} | "
                f"slip={model.slippage_points_per_side:g}/side"
            )
            proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True)
            if proc.returncode != 0:
                print(proc.stdout)
                print(proc.stderr, file=sys.stderr)
                rows.append(
                    {
                        "model": model.name,
                        "risk_percent": risk,
                        "spreads": model.spreads,
                        "slippage_points_per_side": model.slippage_points_per_side,
                        "basis": model.basis,
                        "status": "ERROR",
                        "minimum_passing_capital": None,
                    }
                )
                continue

            report_path = _extract_report_path(proc.stdout)
            minimum = None
            if report_path and report_path.exists():
                report = json.loads(report_path.read_text(encoding="utf-8"))
                minimum = report.get("minimum_passing_capital")

            status = "PASS_FOUND" if minimum is not None else "NO_PASS"
            min_text = f"{minimum:,.0f} THB" if minimum is not None else "NONE"
            print(f"  => {status}: {min_text}")
            rows.append(
                {
                    "model": model.name,
                    "risk_percent": risk,
                    "spreads": model.spreads,
                    "slippage_points_per_side": model.slippage_points_per_side,
                    "basis": model.basis,
                    "status": status,
                    "minimum_passing_capital": minimum,
                    "report_path": str(report_path) if report_path else None,
                }
            )

    print("\n=== Matrix Summary ===")
    print(f"{'MODEL':32s} {'RISK':>6s} {'SPREAD':>11s} {'SLIP/SIDE':>10s} {'MIN PASS CAPITAL':>18s}")
    print("-" * 84)
    for row in rows:
        minimum = row.get("minimum_passing_capital")
        min_text = f"{minimum:,.0f} THB" if minimum is not None else row["status"]
        print(
            f"{row['model'][:32]:32s} {row['risk_percent']:>5.2f}% "
            f"{row['spreads']:>11s} {row['slippage_points_per_side']:>10.1f} {min_text:>18s}"
        )

    out_dir = REPO_ROOT / "backtests" / "goldmicro_model_matrix_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"goldmicro_model_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    payload = {
        "generated_at": datetime.now().isoformat(),
        "criteria": {
            "min_profit_factor": args.min_pf,
            "max_drawdown_percent": args.max_dd,
            "max_skip_percent": args.max_skip,
        },
        "capitals": args.capitals,
        "risks": risks,
        "models": [model.__dict__ for model in MODELS],
        "results": rows,
        "limitations": [
            "Entry-side slippage evidence is sparse relative to exit-side evidence.",
            "Scenario slippage is applied as a constant adverse amount per side.",
            "Commission/fee historical deal fields were zero in the inspected account history.",
            "Swap exists in symbol specification but no historical swap charge was observed in the inspected deal history.",
            "Legacy trade trigger timing is not tick-accurate.",
            "Historical USD/THB conversion is not modeled.",
        ],
    }
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved matrix report: {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
