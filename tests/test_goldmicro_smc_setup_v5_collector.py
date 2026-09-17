from __future__ import annotations

from datetime import datetime, timezone

import polars as pl

from src.goldmicro_smc_setup_v5_collector import drop_forming_m15_bar, merge_prospective_snapshot


def _frame(times):
    n = len(times)
    return pl.DataFrame({
        "time": times,
        "open": [100.0] * n,
        "high": [101.0] * n,
        "low": [99.0] * n,
        "close": [100.5] * n,
        "volume": [10] * n,
        "spread": [50] * n,
        "real_volume": [0] * n,
    })


def test_drop_forming_m15_bar_keeps_only_closed_intervals():
    df = _frame([
        datetime(2026, 9, 16, 9, 30),
        datetime(2026, 9, 16, 9, 45),
        datetime(2026, 9, 16, 10, 0),
    ])
    out = drop_forming_m15_bar(df, datetime(2026, 9, 16, 10, 7, tzinfo=timezone.utc))
    assert out["time"].to_list() == [
        datetime(2026, 9, 16, 9, 30),
        datetime(2026, 9, 16, 9, 45),
    ]


def test_merge_appends_only_strictly_post_cutoff_and_deduplicates():
    cutoff = datetime(2026, 9, 16, 9, 0)
    frozen = _frame([
        datetime(2026, 9, 16, 8, 45),
        datetime(2026, 9, 16, 9, 0),
    ])
    fetched = _frame([
        datetime(2026, 9, 16, 9, 0),
        datetime(2026, 9, 16, 9, 15),
        datetime(2026, 9, 16, 9, 30),
    ])
    merged, fresh_count = merge_prospective_snapshot(frozen, fetched, cutoff=cutoff)
    assert fresh_count == 2
    assert merged["time"].to_list() == [
        datetime(2026, 9, 16, 8, 45),
        datetime(2026, 9, 16, 9, 0),
        datetime(2026, 9, 16, 9, 15),
        datetime(2026, 9, 16, 9, 30),
    ]


def test_merge_does_not_modify_frozen_source_when_no_fresh_rows():
    cutoff = datetime(2026, 9, 16, 9, 0)
    frozen = _frame([datetime(2026, 9, 16, 9, 0)])
    fetched = _frame([datetime(2026, 9, 16, 8, 45)])
    merged, fresh_count = merge_prospective_snapshot(frozen, fetched, cutoff=cutoff)
    assert fresh_count == 0
    assert merged.frame_equal(frozen) if hasattr(merged, "frame_equal") else merged.equals(frozen)
