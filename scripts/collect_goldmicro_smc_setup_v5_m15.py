"""Collect a read-only GOLDmicro M15 prospective snapshot from MT5.

Safety properties:
- initializes an already logged-in MT5 terminal without credentials
- fetches rates only; never calls order_send or any trading mutation
- preserves the frozen development snapshot unchanged
- appends only fully closed bars strictly newer than the frozen cutoff
- writes a separate prospective parquet plus metadata JSON
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.goldmicro_smc_setup_v5_collector import (
    drop_forming_m15_bar,
    merge_prospective_snapshot,
    normalize_mt5_rates,
)
from src.goldmicro_smc_setup_v5_readiness import parse_cutoff


def _load_manifest(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"prospective manifest not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("protocol") != "GOLDMICRO_SMC_SETUP_V5_PROSPECTIVE_V1":
        raise ValueError("unsupported or missing prospective protocol")
    return data


def _epoch_to_utc_naive(epoch_seconds: int | float) -> datetime:
    """Convert an MT5 epoch timestamp to the same UTC-naive basis used by Polars."""
    return datetime.fromtimestamp(float(epoch_seconds), tz=timezone.utc).replace(tzinfo=None)


def run(manifest_path: Path, symbol: str, count: int, output: Path | None) -> Path:
    manifest = _load_manifest(manifest_path)
    cutoff = parse_cutoff(str(manifest["prospective_cutoff_exclusive"]))
    source = Path(str(manifest["source_snapshot"]))
    if not source.exists():
        raise FileNotFoundError(f"frozen source snapshot not found: {source}")
    frozen = pl.read_parquet(source)

    try:
        import MetaTrader5 as mt5
    except ImportError as exc:
        raise RuntimeError("MetaTrader5 package is not installed in this Python environment") from exc

    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        terminal = mt5.terminal_info()
        if terminal is None or not bool(getattr(terminal, "connected", False)):
            raise RuntimeError("MT5 terminal is not connected to broker server")
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"could not select symbol {symbol}: {mt5.last_error()}")
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, count)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"no M15 rates returned for {symbol}: {mt5.last_error()}")

        # IMPORTANT: anchor bar-closure logic to the broker/MT5 epoch clock, not
        # the workstation wall clock. On some installations the terminal/server
        # time basis differs from the OS timezone. Comparing rate epochs against
        # datetime.now() can therefore misclassify many valid bars as "future".
        tick = mt5.symbol_info_tick(symbol)
        if tick is None or not getattr(tick, "time", None):
            raise RuntimeError(f"no reference tick returned for {symbol}: {mt5.last_error()}")
        market_reference_time = _epoch_to_utc_naive(tick.time)

        fetched = drop_forming_m15_bar(
            normalize_mt5_rates(rates),
            now_utc=market_reference_time,
        )
    finally:
        mt5.shutdown()

    merged, fetched_fresh = merge_prospective_snapshot(frozen, fetched, cutoff=cutoff)
    out = output or (manifest_path.parent / "market_snapshot_m15_v5_prospective.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.write_parquet(out)

    fresh_rows = merged.filter(pl.col("time") > pl.lit(cutoff)).height
    latest = merged.select(pl.col("time").max()).item() if len(merged) else None
    meta = {
        "protocol": manifest.get("protocol"),
        "manifest": str(manifest_path),
        "symbol": symbol,
        "cutoff_exclusive": str(cutoff),
        "frozen_source": str(source),
        "output_snapshot": str(out),
        "frozen_rows": len(frozen),
        "mt5_closed_rows_fetched": len(fetched),
        "mt5_post_cutoff_rows_in_fetch": fetched_fresh,
        "prospective_rows_total": len(merged),
        "fresh_rows_total": fresh_rows,
        "latest_closed_bar_time": str(latest),
        "market_reference_time": str(market_reference_time),
        "bar_close_clock": "MT5_SYMBOL_TICK_EPOCH",
        "economic_outcomes": "NOT_EVALUATED",
        "orders": "NEVER_SENT",
    }
    meta_path = out.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("=== GOLDmicro SMC Setup V5 Read-Only M15 Collector ===")
    print(f"Manifest             : {manifest_path}")
    print(f"Symbol               : {symbol}")
    print(f"Cutoff (exclusive)   : {cutoff}")
    print(f"Frozen rows          : {len(frozen):,}")
    print(f"Closed MT5 rows read : {len(fetched):,}")
    print(f"Fresh rows in fetch  : {fetched_fresh:,}")
    print(f"Fresh rows total     : {fresh_rows:,}")
    print(f"Market reference     : {market_reference_time}")
    print(f"Latest closed bar    : {latest}")
    print("Economic outcomes    : NOT EVALUATED")
    print("Orders               : NEVER SENT")
    print(f"Snapshot             : {out}")
    print(f"Metadata             : {meta_path}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--symbol", default="GOLDmicro")
    ap.add_argument("--count", type=int, default=50000, help="Recent M15 bars to read from MT5; read-only")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    if args.count < 100:
        raise SystemExit("--count must be >= 100")
    run(args.manifest, args.symbol, args.count, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
