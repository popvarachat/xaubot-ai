from datetime import datetime, timedelta

import polars as pl

from src.goldmicro_event_target import (
    _event_rows_to_frame,
    _resolve_outcome,
    split_events_by_raw_time,
)


def _ohlc(highs, lows):
    n = len(highs)
    return pl.DataFrame({
        "time": [datetime(2026, 1, 1) + timedelta(minutes=15 * i) for i in range(n)],
        "open": [100.0] * n,
        "high": highs,
        "low": lows,
        "close": [100.0] * n,
    })


def test_same_bar_tp_sl_is_adverse_for_buy():
    df = _ohlc([100.0, 103.0, 100.0], [100.0, 97.0, 100.0])
    target, reason, idx = _resolve_outcome(
        df,
        event_index=0,
        direction="BUY",
        stop_loss=98.0,
        take_profit=102.0,
        max_holding_bars=2,
    )
    assert target == 0
    assert reason == "AMBIGUOUS_BAR_SL_FIRST"
    assert idx == 1


def test_tp_before_sl_is_positive():
    df = _ohlc([100.0, 103.0, 100.0], [100.0, 99.0, 97.0])
    target, reason, idx = _resolve_outcome(
        df,
        event_index=0,
        direction="BUY",
        stop_loss=98.0,
        take_profit=102.0,
        max_holding_bars=2,
    )
    assert target == 1
    assert reason == "TAKE_PROFIT_FIRST"
    assert idx == 1


def test_raw_time_embargo_is_measured_in_bar_indices_not_event_rows():
    events = pl.DataFrame({
        "event_index": [60, 67, 68, 99, 100, 131, 132, 150],
        "event_target": [1, 0, 1, 0, 1, 0, 1, 0],
    })
    train, test = split_events_by_raw_time(
        events,
        raw_split_index=100,
        label_horizon_bars=32,
    )
    assert train["event_index"].to_list() == [60, 67]
    assert test["event_index"].to_list() == [132, 150]


def test_timeout_is_negative():
    df = _ohlc([100.0, 101.0, 101.0], [100.0, 99.0, 99.0])
    target, reason, idx = _resolve_outcome(
        df,
        event_index=0,
        direction="BUY",
        stop_loss=95.0,
        take_profit=105.0,
        max_holding_bars=2,
    )
    assert target == 0
    assert reason == "TIMEOUT_NO_TP_FIRST"
    assert idx == 2


def test_event_frame_scans_all_rows_before_inferring_numeric_schema():
    rows = [{"mixed_numeric": i, "event_index": i} for i in range(120)]
    rows[-1]["mixed_numeric"] = 5138.891338

    frame = _event_rows_to_frame(rows)

    assert frame.height == 120
    assert frame["mixed_numeric"].dtype == pl.Float64
    assert frame["mixed_numeric"][-1] == 5138.891338
