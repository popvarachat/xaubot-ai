from __future__ import annotations

from copy import deepcopy

import polars as pl

from src.goldmicro_smc_setup_v5 import GoldmicroSMCSetupV5, SMCSetupV5Config


def _row(**overrides):
    row = {
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.0,
        "bos": 0,
        "choch": 0,
        "market_structure": 0,
        "swing_high_level": None,
        "swing_low_level": None,
        "last_swing_high": 104.0,
        "last_swing_low": 96.0,
        "is_fvg_bull": False,
        "is_fvg_bear": False,
        "fvg_top": None,
        "fvg_bottom": None,
        "ob": 0,
        "ob_top": None,
        "ob_bottom": None,
        "ob_origin_index": -1,
        "atr": 2.0,
    }
    row.update(overrides)
    return row


def _frame(rows):
    return pl.from_dicts(rows, infer_schema_length=None)


def test_no_setup_without_confirmed_directional_break():
    rows = [
        _row(is_fvg_bull=True, fvg_top=100.0, fvg_bottom=99.0),
        _row(low=99.5, high=101.0, close=100.5),
    ]
    assert GoldmicroSMCSetupV5().generate(_frame(rows)) == []


def test_no_entry_before_zone_retest():
    rows = [
        _row(bos=1, last_swing_high=99.0),
        _row(is_fvg_bull=True, fvg_top=100.0, fvg_bottom=99.0),
    ]
    assert GoldmicroSMCSetupV5().generate(_frame(rows)) == []


def test_stale_zone_expires_before_entry():
    cfg = SMCSetupV5Config(max_setup_age_bars=5, max_zone_delay_bars=2)
    rows = [_row(bos=1)] + [_row() for _ in range(3)]
    rows.append(_row(is_fvg_bull=True, fvg_top=100.0, fvg_bottom=99.0))
    rows.append(_row(low=99.5, high=101.0, close=100.5))
    assert GoldmicroSMCSetupV5(cfg).generate(_frame(rows)) == []


def test_conflicting_break_evidence_fails_closed():
    rows = [
        _row(bos=1, choch=-1, is_fvg_bull=True, fvg_top=100.0, fvg_bottom=99.0),
        _row(low=99.5, high=101.0, close=100.5),
    ]
    assert GoldmicroSMCSetupV5().generate(_frame(rows)) == []


def test_bullish_break_zone_retest_emits_provenance():
    rows = [
        _row(bos=1, last_swing_high=99.0),
        _row(is_fvg_bull=True, fvg_top=100.0, fvg_bottom=99.0),
        _row(low=99.4, high=101.0, close=100.6, last_swing_low=98.0),
    ]
    events = GoldmicroSMCSetupV5().generate(_frame(rows))
    assert len(events) == 1
    event = events[0]
    assert event.setup_direction == "BUY"
    assert event.setup_archetype == "BOS_CONTINUATION"
    assert event.break_index == 0
    assert event.zone_confirm_index == 1
    assert event.retest_index == 2
    assert event.entry_index == 2
    assert event.zone_type == "FVG"
    assert event.stop_mid < event.entry_mid < event.take_profit_mid
    rr = (event.take_profit_mid - event.entry_mid) / (event.entry_mid - event.stop_mid)
    assert rr == 1.5
    assert "ZONE_RETEST_REJECTION" in event.component_evidence


def test_choch_reversal_is_recorded_separately():
    rows = [
        _row(choch=-1, last_swing_low=101.0),
        _row(ob=-1, ob_top=101.0, ob_bottom=100.0, ob_origin_index=0),
        _row(low=99.0, high=100.7, close=99.6, last_swing_high=102.0),
    ]
    events = GoldmicroSMCSetupV5().generate(_frame(rows))
    assert len(events) == 1
    assert events[0].setup_direction == "SELL"
    assert events[0].setup_archetype == "CHOCH_REVERSAL"
    assert events[0].zone_type == "OB"


def _mirror_row(row):
    mirrored = deepcopy(row)
    for key in ("open", "high", "low", "close", "fvg_top", "fvg_bottom", "ob_top", "ob_bottom",
                "last_swing_high", "last_swing_low", "swing_high_level", "swing_low_level"):
        if mirrored.get(key) is not None:
            mirrored[key] = -float(mirrored[key])
    mirrored["high"], mirrored["low"] = -float(row["low"]), -float(row["high"])
    if row.get("fvg_top") is not None and row.get("fvg_bottom") is not None:
        mirrored["fvg_top"], mirrored["fvg_bottom"] = -float(row["fvg_bottom"]), -float(row["fvg_top"])
    if row.get("ob_top") is not None and row.get("ob_bottom") is not None:
        mirrored["ob_top"], mirrored["ob_bottom"] = -float(row["ob_bottom"]), -float(row["ob_top"])
    mirrored["bos"] = -int(row.get("bos") or 0)
    mirrored["choch"] = -int(row.get("choch") or 0)
    mirrored["market_structure"] = -int(row.get("market_structure") or 0)
    mirrored["is_fvg_bull"] = bool(row.get("is_fvg_bear"))
    mirrored["is_fvg_bear"] = bool(row.get("is_fvg_bull"))
    mirrored["ob"] = -int(row.get("ob") or 0)
    mirrored["last_swing_high"] = -float(row["last_swing_low"]) if row.get("last_swing_low") is not None else None
    mirrored["last_swing_low"] = -float(row["last_swing_high"]) if row.get("last_swing_high") is not None else None
    return mirrored


def test_buy_sell_mirror_symmetry():
    buy_rows = [
        _row(bos=1, last_swing_high=99.0),
        _row(is_fvg_bull=True, fvg_top=100.0, fvg_bottom=99.0),
        _row(low=99.4, high=101.0, close=100.6, last_swing_low=98.0),
    ]
    sell_rows = [_mirror_row(row) for row in buy_rows]
    buy = GoldmicroSMCSetupV5().generate(_frame(buy_rows))[0]
    sell = GoldmicroSMCSetupV5().generate(_frame(sell_rows))[0]
    assert sell.setup_direction == "SELL"
    assert sell.setup_archetype == buy.setup_archetype
    assert sell.entry_index == buy.entry_index
    assert sell.zone_confirm_index == buy.zone_confirm_index
    assert sell.entry_mid == -buy.entry_mid
    assert sell.stop_mid == -buy.stop_mid
    assert sell.take_profit_mid == -buy.take_profit_mid
    assert sell.retest_depth_fraction == buy.retest_depth_fraction


def test_zone_invalidation_cancels_pending_setup():
    rows = [
        _row(bos=1),
        _row(is_fvg_bull=True, fvg_top=100.0, fvg_bottom=99.0),
        _row(low=98.0, high=99.4, close=98.5),
        _row(low=99.3, high=101.0, close=100.5),
    ]
    assert GoldmicroSMCSetupV5().generate(_frame(rows)) == []
