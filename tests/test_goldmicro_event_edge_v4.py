from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from src.goldmicro_event_edge_calibration import (
    apply_affine_calibrator,
    fit_affine_calibrator,
)
from src.goldmicro_event_edge_target import EDGE_TARGET_COLUMN, build_event_edge_frame
from src.goldmicro_event_target import EventTargetConfig


class _Signal:
    def __init__(self, signal_type, entry, stop, target, confidence=0.75):
        self.signal_type = signal_type
        self.entry_price = entry
        self.stop_loss = stop
        self.take_profit = target
        self.confidence = confidence


def test_affine_calibration_recovers_positive_mapping():
    raw = np.linspace(-1.0, 1.0, 100)
    realized = 2.0 * raw + 0.25
    calibration = fit_affine_calibrator(raw, realized)
    assert calibration["slope"] == pytest.approx(2.0, rel=1e-6)
    assert calibration["intercept"] == pytest.approx(0.25, rel=1e-6)
    out = apply_affine_calibrator(np.array([0.0, 1.0]), calibration)
    assert out.tolist() == pytest.approx([0.25, 2.25])


def test_affine_calibration_fails_closed_on_degenerate_predictions():
    with pytest.raises(ValueError, match="degenerate prediction variance"):
        fit_affine_calibrator(np.ones(30), np.linspace(-1.0, 1.0, 30))


def test_affine_calibration_fails_closed_on_nonpositive_slope():
    raw = np.linspace(-1.0, 1.0, 30)
    realized = -raw
    with pytest.raises(ValueError, match="non-positive slope"):
        fit_affine_calibrator(raw, realized)


def _frame_with_minimal_features(n=150):
    start = datetime(2026, 1, 5, 10, 0)
    return pl.DataFrame({
        "time": [start + timedelta(minutes=15 * i) for i in range(n)],
        "open": [100.0] * n,
        "high": [100.0] * n,
        "low": [100.0] * n,
        "close": [100.0] * n,
    })


def test_timeout_realized_r_uses_horizon_close(monkeypatch):
    df = _frame_with_minimal_features()
    # Decision at warmup=100. Risk distance is 2.0. Horizon close at 102 is 101,
    # so BUY timeout payoff must be +0.5R rather than a binary loss.
    closes = df["close"].to_list()
    closes[102] = 101.0
    df = df.with_columns(pl.Series("close", closes))

    calls = {"n": 0}
    def fake_signal(self, prefix):
        idx = len(prefix) - 1
        if idx == 100 and calls["n"] == 0:
            calls["n"] += 1
            return _Signal("BUY", 100.0, 98.0, 104.0)
        return None

    monkeypatch.setattr(
        "src.goldmicro_event_target.GoldmicroCausalSMCAnalyzer.generate_signal",
        fake_signal,
    )
    out = build_event_edge_frame(
        df,
        config=EventTargetConfig(max_holding_bars=2, event_cooldown_bars=10, warmup_bars=100),
    )
    assert out.height == 1
    assert out["event_outcome_reason"][0] == "TIMEOUT_NO_TP_FIRST"
    assert out[EDGE_TARGET_COLUMN][0] == pytest.approx(0.5)


def test_stop_loss_is_minus_one_r(monkeypatch):
    df = _frame_with_minimal_features()
    lows = df["low"].to_list()
    lows[101] = 97.0
    df = df.with_columns(pl.Series("low", lows))

    def fake_signal(self, prefix):
        return _Signal("BUY", 100.0, 98.0, 104.0) if len(prefix) - 1 == 100 else None

    monkeypatch.setattr(
        "src.goldmicro_event_target.GoldmicroCausalSMCAnalyzer.generate_signal",
        fake_signal,
    )
    out = build_event_edge_frame(
        df,
        config=EventTargetConfig(max_holding_bars=2, event_cooldown_bars=10, warmup_bars=100),
    )
    assert out.height == 1
    assert out[EDGE_TARGET_COLUMN][0] == pytest.approx(-1.0)


def test_take_profit_uses_declared_reward_r(monkeypatch):
    df = _frame_with_minimal_features()
    highs = df["high"].to_list()
    highs[101] = 104.5
    df = df.with_columns(pl.Series("high", highs))

    def fake_signal(self, prefix):
        return _Signal("BUY", 100.0, 98.0, 104.0) if len(prefix) - 1 == 100 else None

    monkeypatch.setattr(
        "src.goldmicro_event_target.GoldmicroCausalSMCAnalyzer.generate_signal",
        fake_signal,
    )
    out = build_event_edge_frame(
        df,
        config=EventTargetConfig(max_holding_bars=2, event_cooldown_bars=10, warmup_bars=100),
    )
    assert out.height == 1
    assert out[EDGE_TARGET_COLUMN][0] == pytest.approx(2.0)
