"""Blind prospective-readiness logic for GOLDmicro SMC Setup V5.

This module deliberately does NOT resolve trade outcomes or calculate P/L. It
only determines whether enough post-freeze, fully matured setup events exist to
open the predeclared economic evaluation once.

Prospective block geometry is frozen from the already-inspected 24,000-row
M15 development snapshot before any fresh economic outcomes are viewed:
- 5 fixed chronological blocks
- 4,800 fresh M15 rows per block
- 24,000 fresh M15 entry rows total
- event assignment by ENTRY row only
- later bars may mature tail events but never change block membership
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class ProspectiveReadinessThresholds:
    chronological_blocks: int = 5
    rows_per_block: int = 4800
    confirmatory_fresh_rows: int = 24000
    horizon_bars: int = 32
    min_events_per_block: int = 30

    def validate(self) -> None:
        if self.chronological_blocks < 2:
            raise ValueError("chronological_blocks must be >= 2")
        if self.rows_per_block < 1:
            raise ValueError("rows_per_block must be >= 1")
        if self.confirmatory_fresh_rows < 1:
            raise ValueError("confirmatory_fresh_rows must be >= 1")
        if self.chronological_blocks * self.rows_per_block != self.confirmatory_fresh_rows:
            raise ValueError(
                "chronological_blocks * rows_per_block must equal confirmatory_fresh_rows"
            )
        if self.horizon_bars < 1:
            raise ValueError("horizon_bars must be >= 1")
        if self.min_events_per_block < 1:
            raise ValueError("min_events_per_block must be >= 1")


def parse_cutoff(value: str) -> datetime:
    text = str(value).strip()
    if not text:
        raise ValueError("prospective cutoff is empty")
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid prospective cutoff: {value}") from exc


def _comparable(value: Any, cutoff: datetime) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    # The frozen research snapshot currently uses naive MT5 timestamps. Refuse
    # mixed aware/naive comparison instead of silently shifting time zones.
    if (dt.tzinfo is None) != (cutoff.tzinfo is None):
        raise ValueError("timestamp timezone awareness differs from frozen cutoff")
    return dt


def evaluate_readiness(
    *,
    timestamps: list[Any],
    event_entry_indices: list[int],
    cutoff: datetime,
    thresholds: ProspectiveReadinessThresholds = ProspectiveReadinessThresholds(),
) -> dict[str, Any]:
    """Return counts/readiness only; no return, PF/DD, or outcome information."""
    thresholds.validate()
    if not timestamps:
        raise ValueError("timestamps cannot be empty")

    times = [_comparable(value, cutoff) for value in timestamps]
    fresh_indices = [idx for idx, ts in enumerate(times) if ts > cutoff]
    if not fresh_indices:
        return {
            "status": "WAITING_FOR_FRESH_ROWS",
            "thresholds": asdict(thresholds),
            "fresh_rows": 0,
            "confirmatory_entry_rows_observed": 0,
            "window_sealed": False,
            "fresh_setup_events": 0,
            "matured_setup_events": 0,
            "block_event_counts": [0] * thresholds.chronological_blocks,
            "ready": False,
        }

    fresh_start = fresh_indices[0]
    n_rows = len(times)
    fresh_rows = n_rows - fresh_start
    entry_window_end = fresh_start + thresholds.confirmatory_fresh_rows
    confirmatory_rows_observed = min(fresh_rows, thresholds.confirmatory_fresh_rows)
    window_sealed = fresh_rows >= thresholds.confirmatory_fresh_rows
    last_mature_entry = n_rows - 1 - thresholds.horizon_bars

    # Scored entry events are limited to the immutable 24,000-row confirmatory
    # entry window. Bars after that window may exist only to mature tail events.
    fresh_events = sorted(
        idx
        for idx in event_entry_indices
        if 0 <= idx < n_rows
        and times[idx] > cutoff
        and idx < entry_window_end
    )
    matured_events = [idx for idx in fresh_events if idx <= last_mature_entry]

    counts: list[int] = []
    for block_idx in range(thresholds.chronological_blocks):
        start = fresh_start + block_idx * thresholds.rows_per_block
        end = start + thresholds.rows_per_block
        counts.append(sum(start <= idx < end for idx in matured_events))

    enough_events = all(count >= thresholds.min_events_per_block for count in counts)
    ready = window_sealed and enough_events
    return {
        "status": "READY_FOR_ONE_SHOT_ECONOMIC_EVALUATION" if ready else "ACCUMULATING_FRESH_EVIDENCE",
        "thresholds": asdict(thresholds),
        "fresh_rows": fresh_rows,
        "confirmatory_entry_rows_observed": confirmatory_rows_observed,
        "window_sealed": window_sealed,
        "fresh_setup_events": len(fresh_events),
        "matured_setup_events": len(matured_events),
        "block_event_counts": counts,
        "min_block_events": min(counts) if counts else 0,
        "latest_timestamp": str(times[-1]),
        "ready": ready,
        "guard": (
            "Blind readiness only. Do not calculate or inspect realized returns until ready. "
            "Blocks are fixed 4,800-row fresh M15 entry windows and never repartition as data grows. "
            "Only events with a full post-entry outcome horizon are counted as matured."
        ),
    }
