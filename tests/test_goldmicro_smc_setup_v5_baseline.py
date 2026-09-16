from __future__ import annotations

import polars as pl

from src.goldmicro_smc_setup_v5 import SMCSetupV5Event
from src.goldmicro_smc_setup_v5_baseline import (
    SMCSetupV5BaselineThresholds,
    evaluate_baseline,
    simulate_event,
)


def _event(*, direction: str = "BUY", entry_index: int = 0) -> SMCSetupV5Event:
    entry = 100.0
    stop = 99.0 if direction == "BUY" else 101.0
    tp = 101.5 if direction == "BUY" else 98.5
    return SMCSetupV5Event(
        setup_direction=direction,  # type: ignore[arg-type]
        setup_archetype="BOS_CONTINUATION",
        break_index=max(0, entry_index - 2),
        break_level=100.0,
        zone_type="FVG",
        zone_origin_index=max(0, entry_index - 2),
        zone_confirm_index=max(0, entry_index - 1),
        zone_top=100.0,
        zone_bottom=99.5,
        retest_index=entry_index,
        retest_depth_fraction=0.5,
        entry_index=entry_index,
        entry_mid=entry,
        stop_mid=stop,
        take_profit_mid=tp,
        declared_reward_r=1.5,
        break_to_zone_bars=1,
        zone_to_retest_bars=1,
        setup_age_bars=2,
        component_evidence=("BOS_CONTINUATION", "ZONE_FVG", "ZONE_RETEST_REJECTION", "STRUCTURE_DERIVED_STOP"),
    )


def test_same_bar_tp_and_sl_is_adverse_stop_first():
    df = pl.DataFrame({
        "open": [100.0, 100.0],
        "high": [100.2, 102.0],
        "low": [99.8, 98.0],
        "close": [100.0, 100.5],
    })
    out = simulate_event(df, _event(), horizon_bars=1)
    assert out.exit_reason == "SAME_BAR_SL_FIRST"
    assert out.gross_r == -1.0
    assert out.normal_net_r < out.gross_r
    assert out.conservative_net_r < out.normal_net_r


def test_take_profit_resolves_to_declared_reward_r_before_cost():
    df = pl.DataFrame({
        "open": [100.0, 100.2],
        "high": [100.2, 101.6],
        "low": [99.8, 99.5],
        "close": [100.0, 101.5],
    })
    out = simulate_event(df, _event(), horizon_bars=1)
    assert out.exit_reason == "TAKE_PROFIT"
    assert out.gross_r == 1.5
    assert out.normal_net_r < 1.5
    assert out.conservative_net_r < out.normal_net_r


def test_baseline_uses_non_overlapping_raw_time_blocks_and_min_event_gate():
    rows = 50
    df = pl.DataFrame({
        "open": [100.0] * rows,
        "high": [102.0] * rows,
        "low": [99.5] * rows,
        "close": [101.5] * rows,
    })
    events = [_event(entry_index=i) for i in (1, 11, 21, 31, 41)]
    thresholds = SMCSetupV5BaselineThresholds(
        chronological_blocks=5,
        horizon_bars=1,
        min_events_per_block=2,
        min_positive_blocks=4,
    )
    result = evaluate_baseline(df, events, thresholds=thresholds)
    assert [b["events"] for b in result["blocks"]] == [1, 1, 1, 1, 1]
    assert result["gate_pass"] is False
    assert result["status"] == "V5_RAW_BASELINE_REJECT"
