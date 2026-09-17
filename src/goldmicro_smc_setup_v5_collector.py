"""Read-only helpers for GOLDmicro SMC Setup V5 prospective M15 snapshots.

This module contains only pure dataframe handling. It never talks to MT5 and never
places orders. The companion CLI performs a read-only rates fetch from an already
logged-in MT5 terminal, drops the still-forming M15 bar, and appends only bars that
are strictly newer than the frozen prospective cutoff.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import polars as pl

REQUIRED_M15_COLUMNS = ("time", "open", "high", "low", "close")


def normalize_mt5_rates(rates: Any) -> pl.DataFrame:
    """Convert an MT5 structured rates array into the project M15 schema."""
    if rates is None or len(rates) == 0:
        return pl.DataFrame(schema={
            "time": pl.Datetime,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Int64,
            "spread": pl.Int64,
            "real_volume": pl.Int64,
        })
    df = pl.DataFrame({
        "time": rates["time"],
        "open": rates["open"],
        "high": rates["high"],
        "low": rates["low"],
        "close": rates["close"],
        "volume": rates["tick_volume"],
        "spread": rates["spread"],
        "real_volume": rates["real_volume"],
    })
    return df.with_columns([
        pl.from_epoch(pl.col("time"), time_unit="s").alias("time"),
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("volume").cast(pl.Int64),
        pl.col("spread").cast(pl.Int64),
        pl.col("real_volume").cast(pl.Int64),
    ])


def drop_forming_m15_bar(df: pl.DataFrame, now_utc: datetime | None = None) -> pl.DataFrame:
    """Keep only bars whose 15-minute interval has fully closed."""
    if df.is_empty():
        return df
    now = now_utc or datetime.now(timezone.utc)
    if now.tzinfo is not None:
        now = now.astimezone(timezone.utc).replace(tzinfo=None)
    latest_closed_open = now.replace(second=0, microsecond=0)
    minute = (latest_closed_open.minute // 15) * 15
    current_bar_open = latest_closed_open.replace(minute=minute)
    return df.filter(pl.col("time") < pl.lit(current_bar_open))


def merge_prospective_snapshot(
    frozen_source: pl.DataFrame,
    fetched: pl.DataFrame,
    *,
    cutoff: datetime,
) -> tuple[pl.DataFrame, int]:
    """Append strictly post-cutoff fetched bars without mutating the frozen source."""
    for name in REQUIRED_M15_COLUMNS:
        if name not in frozen_source.columns:
            raise ValueError(f"frozen source missing required column: {name}")
        if name not in fetched.columns:
            raise ValueError(f"fetched rates missing required column: {name}")
    if cutoff.tzinfo is not None:
        cutoff = cutoff.astimezone(timezone.utc).replace(tzinfo=None)
    fresh = fetched.filter(pl.col("time") > pl.lit(cutoff))
    fresh_count = len(fresh)
    if fresh_count == 0:
        return frozen_source.sort("time"), 0

    # Align to the union of columns so historical snapshots with extra metadata remain usable.
    columns = list(dict.fromkeys([*frozen_source.columns, *fresh.columns]))
    def align(df: pl.DataFrame) -> pl.DataFrame:
        exprs = []
        for c in columns:
            if c in df.columns:
                exprs.append(pl.col(c))
            else:
                exprs.append(pl.lit(None).alias(c))
        return df.select(exprs)

    merged = pl.concat([align(frozen_source), align(fresh)], how="vertical_relaxed")
    merged = merged.unique(subset=["time"], keep="last").sort("time")
    return merged, fresh_count
