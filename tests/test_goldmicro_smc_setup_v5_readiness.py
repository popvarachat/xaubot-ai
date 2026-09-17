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
    assert result["window_sealed"] is False


def test_unmatured_tail_events_do_not_count():
    cutoff = datetime(2026, 9, 16, 9, 0)
    timestamps = _times(80, datetime(2026, 9, 16, 0, 0))
    fresh_events = [37, 40, 45, 50, 60, 70]
    result = evaluate_readiness(
        timestamps=timestamps,
        event_entry_indices=fresh_events,
        cutoff=cutoff,
        thresholds=ProspectiveReadinessThresholds(
            chronological_blocks=2,
            rows_per_block=100,
            confirmatory_fresh_rows=200,
            horizon_bars=32,
            min_events_per_block=1,
        ),
    )
    assert result["fresh_setup_events"] == len(fresh_events)
    assert result["matured_setup_events"] < result["fresh_setup_events"]
    assert result["ready"] is False


def test_fixed_blocks_do_not_repartition_as_fresh_data_grows():
    cutoff = datetime(2026, 9, 16, 9, 0)
    # Fresh starts at index 37. Fixed blocks are [37,47), [47,57), [57,67).
    thresholds = ProspectiveReadinessThresholds(
        chronological_blocks=3,
        rows_per_block=10,
        confirmatory_fresh_rows=30,
        horizon_bars=2,
        min_events_per_block=1,
    )
    events = [40, 50, 60]

    early = evaluate_readiness(
        timestamps=_times(65, datetime(2026, 9, 16, 0, 0)),
        event_entry_indices=events,
        cutoff=cutoff,
        thresholds=thresholds,
    )
    later = evaluate_readiness(
        timestamps=_times(80, datetime(2026, 9, 16, 0, 0)),
        event_entry_indices=events,
        cutoff=cutoff,
        thresholds=thresholds,
    )

    assert early["block_event_counts"] == [1, 1, 1]
    assert later["block_event_counts"] == [1, 1, 1]


def test_ready_requires_sealed_window_and_minimum_in_every_fixed_block():
    cutoff = datetime(2026, 9, 16, 9, 0)
    thresholds = ProspectiveReadinessThresholds(
        chronological_blocks=3,
        rows_per_block=10,
        confirmatory_fresh_rows=30,
        horizon_bars=2,
        min_events_per_block=2,
    )
    # Fresh starts at 37; fixed blocks [37,47), [47,57), [57,67).
    events = [40, 41, 50, 51, 60, 61]
    result = evaluate_readiness(
        timestamps=_times(75, datetime(2026, 9, 16, 0, 0)),
        event_entry_indices=events,
        cutoff=cutoff,
        thresholds=thresholds,
    )
    assert result["block_event_counts"] == [2, 2, 2]
    assert result["window_sealed"] is True
    assert result["ready"] is True
    assert result["status"] == "READY_FOR_ONE_SHOT_ECONOMIC_EVALUATION"


def test_events_after_confirmatory_entry_window_are_not_scored():
    cutoff = datetime(2026, 9, 16, 9, 0)
    thresholds = ProspectiveReadinessThresholds(
        chronological_blocks=2,
        rows_per_block=10,
        confirmatory_fresh_rows=20,
        horizon_bars=2,
        min_events_per_block=1,
    )
    # Fresh starts at 37; entry window ends before index 57.
    events = [40, 50, 60]
    result = evaluate_readiness(
        timestamps=_times(70, datetime(2026, 9, 16, 0, 0)),
        event_entry_indices=events,
        cutoff=cutoff,
        thresholds=thresholds,
    )
    assert result["fresh_setup_events"] == 2
    assert result["block_event_counts"] == [1, 1]


def test_parse_cutoff_accepts_frozen_iso_style():
    assert parse_cutoff("2026-09-16 09:00:00") == datetime(2026, 9, 16, 9, 0)
