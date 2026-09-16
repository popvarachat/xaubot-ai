"""Research-only realized-R target for GOLDmicro causal SMC setup events.

V4 keeps SMC as the sole source of direction/entry/SL/TP, but replaces the V3
binary TP-before-SL label with realized gross payoff in setup-risk units.  The
future-path contract is unchanged: next-bar scan, 32 raw M15 bars, adverse
same-bar ordering, and horizon-close timeout.
"""
from __future__ import annotations

import polars as pl

from src.goldmicro_event_target import EventTargetConfig, build_event_target_frame


EDGE_TARGET_COLUMN = "event_gross_r"
EDGE_EXIT_COLUMN = "event_edge_exit_mid"


def _signed_price_move(direction: str, entry: float, exit_mid: float) -> float:
    if direction == "BUY":
        return exit_mid - entry
    if direction == "SELL":
        return entry - exit_mid
    raise ValueError(f"unknown event direction: {direction}")


def build_event_edge_frame(
    df: pl.DataFrame,
    *,
    config: EventTargetConfig = EventTargetConfig(),
) -> pl.DataFrame:
    """Attach realized gross R to the existing causal SMC event contract.

    TP/SL events exit at their declared boundary. Timeout events exit at the
    actual close at ``event_outcome_index``. Broker costs are intentionally not
    included in the training target; they remain explicit at strategy evaluation.
    """
    events = build_event_target_frame(df, config=config)
    if events.is_empty():
        return events

    closes = df["close"].to_list()
    gross_r: list[float] = []
    exits: list[float] = []
    risk_distances: list[float] = []

    for row in events.iter_rows(named=True):
        direction = str(row["event_direction"])
        entry = float(row["event_entry"])
        stop = float(row["event_stop_loss"])
        target = float(row["event_take_profit"])
        reason = str(row["event_outcome_reason"])
        outcome_idx = int(row["event_outcome_index"])
        risk_distance = abs(entry - stop)
        if risk_distance <= 0.0:
            raise ValueError("event risk distance must be positive")

        if reason == "TAKE_PROFIT_FIRST":
            exit_mid = target
        elif reason in {"STOP_LOSS_FIRST", "AMBIGUOUS_BAR_SL_FIRST"}:
            exit_mid = stop
        elif reason == "TIMEOUT_NO_TP_FIRST":
            if not 0 <= outcome_idx < len(closes):
                raise ValueError(f"event outcome index outside raw frame: {outcome_idx}")
            exit_mid = float(closes[outcome_idx])
        else:
            raise ValueError(f"unknown event outcome reason: {reason}")

        exits.append(exit_mid)
        risk_distances.append(risk_distance)
        gross_r.append(_signed_price_move(direction, entry, exit_mid) / risk_distance)

    return events.with_columns(
        pl.Series(EDGE_EXIT_COLUMN, exits, dtype=pl.Float64),
        pl.Series("event_risk_distance", risk_distances, dtype=pl.Float64),
        pl.Series(EDGE_TARGET_COLUMN, gross_r, dtype=pl.Float64),
    )
