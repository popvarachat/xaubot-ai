"""Research-only event-conditioned target construction for GOLDmicro.

The unit of prediction is a causal SMC setup event, not every M15 bar.
A positive label means the setup's own take-profit is touched before its own
stop-loss within a predeclared raw-bar horizon. Same-bar TP/SL ambiguity is
resolved adversely. No live execution code imports this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from src.goldmicro_causal_smc import GoldmicroCausalSMCAnalyzer


@dataclass(frozen=True)
class EventTargetConfig:
    max_holding_bars: int = 32
    event_cooldown_bars: int = 10
    warmup_bars: int = 100


def _resolve_outcome(
    df: pl.DataFrame,
    *,
    event_index: int,
    direction: str,
    stop_loss: float,
    take_profit: float,
    max_holding_bars: int,
) -> tuple[int, str, int]:
    """Return (target, outcome_reason, outcome_index) using adverse OHLC ordering."""
    highs = df["high"].to_list()
    lows = df["low"].to_list()
    end = min(event_index + max_holding_bars, len(df) - 1)
    for idx in range(event_index + 1, end + 1):
        high = float(highs[idx])
        low = float(lows[idx])
        if direction == "BUY":
            tp_hit = high >= take_profit
            sl_hit = low <= stop_loss
        else:
            tp_hit = low <= take_profit
            sl_hit = high >= stop_loss
        if sl_hit:
            return 0, ("AMBIGUOUS_BAR_SL_FIRST" if tp_hit else "STOP_LOSS_FIRST"), idx
        if tp_hit:
            return 1, "TAKE_PROFIT_FIRST", idx
    return 0, "TIMEOUT_NO_TP_FIRST", end


def build_event_target_frame(
    df: pl.DataFrame,
    *,
    config: EventTargetConfig = EventTargetConfig(),
) -> pl.DataFrame:
    """Extract de-duplicated causal SMC setup rows and attach event outcomes.

    ``df`` must already contain prefix-causal SMC columns. ``generate_signal`` is
    called on a prefix ending exactly at each decision bar, so no future row is
    available to the signal API. Consecutive persistent setups are de-duplicated
    with a fixed raw-bar cooldown to avoid counting the same setup every candle.
    """
    if config.max_holding_bars < 1:
        raise ValueError("max_holding_bars must be >= 1")
    if config.event_cooldown_bars < 0:
        raise ValueError("event_cooldown_bars must be >= 0")
    if len(df) <= config.warmup_bars + config.max_holding_bars:
        raise ValueError("insufficient raw bars for event target construction")

    smc = GoldmicroCausalSMCAnalyzer(swing_length=5)
    rows: list[dict[str, Any]] = []
    last_event_index = -10**9
    stop_at = len(df) - config.max_holding_bars

    for idx in range(config.warmup_bars, stop_at):
        if idx - last_event_index < config.event_cooldown_bars:
            continue
        prefix = df.slice(0, idx + 1)
        signal = smc.generate_signal(prefix)
        if signal is None:
            continue

        target, reason, outcome_idx = _resolve_outcome(
            df,
            event_index=idx,
            direction=signal.signal_type,
            stop_loss=float(signal.stop_loss),
            take_profit=float(signal.take_profit),
            max_holding_bars=config.max_holding_bars,
        )
        row = df.row(idx, named=True)
        row.update(
            {
                "event_index": idx,
                "event_target": int(target),
                "event_direction": signal.signal_type,
                "event_entry": float(signal.entry_price),
                "event_stop_loss": float(signal.stop_loss),
                "event_take_profit": float(signal.take_profit),
                "event_smc_confidence": float(signal.confidence),
                "event_outcome_reason": reason,
                "event_outcome_index": int(outcome_idx),
            }
        )
        rows.append(row)
        last_event_index = idx

    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows)


def split_events_by_raw_time(
    events: pl.DataFrame,
    *,
    raw_split_index: int,
    label_horizon_bars: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split event rows with a raw-bar embargo around the train/OOS boundary.

    Training decisions must finish their label horizon before the raw split.
    OOS decisions begin only after an equal forward embargo. The gap is therefore
    measured in market bars, not in filtered event-row units.
    """
    if events.is_empty():
        return events, events
    if label_horizon_bars < 1:
        raise ValueError("label_horizon_bars must be >= 1")
    train_last_decision = raw_split_index - label_horizon_bars - 1
    test_first_decision = raw_split_index + label_horizon_bars
    train = events.filter(pl.col("event_index") <= train_last_decision)
    test = events.filter(pl.col("event_index") >= test_first_decision)
    return train, test
