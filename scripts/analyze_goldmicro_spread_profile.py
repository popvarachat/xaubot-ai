"""Analyze an MT5 GOLDmicro M1 CSV spread profile locally.

Expected MT5 export columns include ``<DATE>``, ``<TIME>`` and ``<SPREAD>``.
The source CSV is intentionally not committed; this script only reads a local
file supplied with ``--input`` and prints distribution statistics that can be
used to choose realistic backtest spread scenarios.

No MT5 connection and no order actions are performed.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def _percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("No spread values")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze GOLDmicro M1 <SPREAD> distribution")
    parser.add_argument("--input", type=Path, required=True, help="MT5 M1 CSV export")
    args = parser.parse_args()

    path = args.input.resolve()
    if not path.exists():
        raise SystemExit(f"Input file not found: {path}")

    spreads: list[float] = []
    first_stamp: str | None = None
    last_stamp: str | None = None

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or "<SPREAD>" not in reader.fieldnames:
            raise SystemExit(f"Expected <SPREAD> column. Found: {reader.fieldnames}")

        for row in reader:
            try:
                spread = float(row["<SPREAD>"])
            except (TypeError, ValueError):
                continue
            if spread < 0:
                continue
            spreads.append(spread)
            stamp = f"{row.get('<DATE>', '')} {row.get('<TIME>', '')}".strip()
            if first_stamp is None:
                first_stamp = stamp
            last_stamp = stamp

    if not spreads:
        raise SystemExit("No usable spread rows found")

    n = len(spreads)
    mean = sum(spreads) / n
    in_40_60 = sum(1 for x in spreads if 40 <= x <= 60) / n * 100.0
    le_60 = sum(1 for x in spreads if x <= 60) / n * 100.0
    gt_60 = 100.0 - le_60

    print("\n=== GOLDmicro M1 Spread Profile ===")
    print(f"File       : {path}")
    print(f"Rows       : {n:,}")
    if first_stamp or last_stamp:
        print(f"Range      : {first_stamp} -> {last_stamp}")
    print(f"Min        : {min(spreads):.0f} points")
    print(f"Mean       : {mean:.2f} points")
    print(f"Median/P50 : {_percentile(spreads, 0.50):.0f} points")
    print(f"P90        : {_percentile(spreads, 0.90):.0f} points")
    print(f"P95        : {_percentile(spreads, 0.95):.0f} points")
    print(f"P99        : {_percentile(spreads, 0.99):.0f} points")
    print(f"Max        : {max(spreads):.0f} points")
    print(f"40-60      : {in_40_60:.2f}% of rows")
    print(f"<=60       : {le_60:.2f}% of rows")
    print(f">60        : {gt_60:.2f}% of rows")
    print("Suggested operational sweep: 40,50,55,60 points")
    print("Suggested stress sweep     : 70,100 points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
