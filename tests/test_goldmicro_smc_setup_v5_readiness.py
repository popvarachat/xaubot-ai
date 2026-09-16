from __future__ import annotations

from datetime import datetime, timedelta

from src.goldmicro_smc_setup_v5_readiness import (
    ProspectiveReadinessThresholds,
    evaluate_readiness,
    parse_cutoff,
)


def _times(n: int, start: datetime) -> list[datetime]:
    return [start + timedelta(minutes=15 * i) for i in range(n)]


def test_no_fresh_rows_waits_without_outcomes():
    cutoff = datetime(2026, 9, 16, 9, 0)
    timestamps = _times(20, datetime(2026, 9, 16, 4, 0))
    result = evaluate_readiness(
        timestamps=timestamps,
        event_entry_indices=[5, 10],
        cutoff=cutoff,
    )
    assert result["status"] == "WAITING_FOR_FRESH_ROWS"
    assert result["ready"] is False
    assert result["fresh_rows"] == 0


def test_unmatured_tail_events_do_not_count():
    cutoff = datetime(2026, 9, 16, 9, 0)
    timestamps = _times(80, datetime(2026, 9, 16, 0, 0))
    # 15:45 and later entries have less than 32 bars of future path here.
    fresh_events = [37, 40, 45, 50, 60, 70]
    result = evaluate_readiness(
        timestamps=timestamps,
        event_entry_indices=fresh_events,
        cutoff=cutoff,
        thresholds=ProspectiveReadinessThresholds(
            chronological_blocks=2,
            horizon_bars=32,
            min_events_per_block=1,
        ),
    )
    assert result["fresh_setup_events"] == len(fresh_events)
    assert result["matured_setup_events"] < result["fresh_setup_events"]
    assert result["ready"] is False


def test_ready_only_when_every_block_has_minimum_matured_events():
    cutoff = datetime(2026, 9, 16, 9, 0)
    timestamps = _times(220, datetime(2026, 9, 16, 0, 0))
    # Fresh starts at index 37. With 32-bar maturity, entries through 187 can count.
    events = [45, 55, 95, 105, 145, 155]
    result = evaluate_readiness(
        timestamps=timestamps,
        event_entry_indices=events,
        cutoff=cutoff,
        thresholds=ProspectiveReadinessThresholds(
            chronological_blocks=3,
            horizon_bars=32,
            min_events_per_block=2,
        ),
    )
    assert result["block_event_counts"] == [2, 2, 2]
    assert result["ready"] is True
    assert result["status"] == "READY_FOR_ONE_SHOT_ECONOMIC_EVALUATION"


def test_parse_cutoff_accepts_frozen_iso_style():
    assert parse_cutoff("2026-09-16 09:00:00") == datetime(2026, 9, 16, 9, 0)
