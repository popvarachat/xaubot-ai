"""Research-only causal SMC setup generator for GOLDmicro V5.

V5 is intentionally separate from ``SMCAnalyzer.generate_signal`` and all live
execution paths.  It turns already-causal SMC columns into explicit setup events
through a small state machine:

NO_SETUP -> STRUCTURE_BREAK_CONFIRMED -> ALIGNED_ZONE_AVAILABLE
         -> ZONE_RETESTED -> ENTRY_EVENT

The generator fails closed on conflicting break evidence, requires a zone that is
confirmed after (or on) the structural break, requires a later zone retest, and
uses mirrored BUY/SELL mechanics.  No ML score is used here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import math
import polars as pl

Direction = Literal["BUY", "SELL"]
Archetype = Literal["BOS_CONTINUATION", "CHOCH_REVERSAL"]
ZoneType = Literal["FVG", "OB"]


@dataclass(frozen=True)
class SMCSetupV5Config:
    max_setup_age_bars: int = 12
    max_zone_delay_bars: int = 6
    min_retest_delay_bars: int = 1
    fixed_reward_r: float = 1.5
    atr_stop_floor_multiple: float = 0.50

    def validate(self) -> None:
        if self.max_setup_age_bars < 2:
            raise ValueError("max_setup_age_bars must be >= 2")
        if self.max_zone_delay_bars < 0:
            raise ValueError("max_zone_delay_bars must be >= 0")
        if self.min_retest_delay_bars < 1:
            raise ValueError("min_retest_delay_bars must be >= 1")
        if self.fixed_reward_r <= 0:
            raise ValueError("fixed_reward_r must be positive")
        if self.atr_stop_floor_multiple < 0:
            raise ValueError("atr_stop_floor_multiple cannot be negative")


@dataclass(frozen=True)
class SMCSetupV5Event:
    setup_direction: Direction
    setup_archetype: Archetype
    break_index: int
    break_level: float | None
    zone_type: ZoneType
    zone_origin_index: int
    zone_confirm_index: int
    zone_top: float
    zone_bottom: float
    retest_index: int
    retest_depth_fraction: float
    entry_index: int
    entry_mid: float
    stop_mid: float
    take_profit_mid: float
    declared_reward_r: float
    break_to_zone_bars: int
    zone_to_retest_bars: int
    setup_age_bars: int
    component_evidence: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["component_evidence"] = list(self.component_evidence)
        return data


@dataclass
class _PendingSetup:
    direction: Direction
    archetype: Archetype
    break_index: int
    break_level: float | None
    zone_type: ZoneType | None = None
    zone_origin_index: int | None = None
    zone_confirm_index: int | None = None
    zone_top: float | None = None
    zone_bottom: float | None = None


def _finite(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _required_columns() -> tuple[str, ...]:
    return ("open", "high", "low", "close", "bos", "choch", "is_fvg_bull", "is_fvg_bear",
            "fvg_top", "fvg_bottom", "ob", "ob_top", "ob_bottom")


def _break_evidence(row: dict[str, Any]) -> tuple[Direction | None, Archetype | None, bool]:
    """Return (direction, archetype, conflict) for one causal confirmation row."""
    bos = int(row.get("bos") or 0)
    choch = int(row.get("choch") or 0)
    bullish = (bos == 1) or (choch == 1)
    bearish = (bos == -1) or (choch == -1)
    if bullish and bearish:
        return None, None, True
    if bullish:
        # If both point the same direction, reversal semantics are more specific.
        return "BUY", ("CHOCH_REVERSAL" if choch == 1 else "BOS_CONTINUATION"), False
    if bearish:
        return "SELL", ("CHOCH_REVERSAL" if choch == -1 else "BOS_CONTINUATION"), False
    return None, None, False


def _break_level(row: dict[str, Any], direction: Direction) -> float | None:
    if direction == "BUY":
        candidates = (row.get("last_swing_high"), row.get("swing_high_level"))
    else:
        candidates = (row.get("last_swing_low"), row.get("swing_low_level"))
    for value in candidates:
        if _finite(value):
            return float(value)
    return None


def _aligned_zone(row: dict[str, Any], direction: Direction, index: int) -> tuple[ZoneType, int, float, float] | None:
    """Return the zone confirmed on this row only, never a stale forward-filled zone."""
    if direction == "BUY":
        if bool(row.get("is_fvg_bull")) and _finite(row.get("fvg_top")) and _finite(row.get("fvg_bottom")):
            top, bottom = float(row["fvg_top"]), float(row["fvg_bottom"])
            return "FVG", index - 2, max(top, bottom), min(top, bottom)
        if int(row.get("ob") or 0) == 1 and _finite(row.get("ob_top")) and _finite(row.get("ob_bottom")):
            origin = int(row.get("ob_origin_index") if row.get("ob_origin_index") is not None else index)
            top, bottom = float(row["ob_top"]), float(row["ob_bottom"])
            return "OB", origin, max(top, bottom), min(top, bottom)
    else:
        if bool(row.get("is_fvg_bear")) and _finite(row.get("fvg_top")) and _finite(row.get("fvg_bottom")):
            top, bottom = float(row["fvg_top"]), float(row["fvg_bottom"])
            return "FVG", index - 2, max(top, bottom), min(top, bottom)
        if int(row.get("ob") or 0) == -1 and _finite(row.get("ob_top")) and _finite(row.get("ob_bottom")):
            origin = int(row.get("ob_origin_index") if row.get("ob_origin_index") is not None else index)
            top, bottom = float(row["ob_top"]), float(row["ob_bottom"])
            return "OB", origin, max(top, bottom), min(top, bottom)
    return None


def _zone_invalidated(row: dict[str, Any], pending: _PendingSetup) -> bool:
    assert pending.zone_top is not None and pending.zone_bottom is not None
    close = float(row["close"])
    return close < pending.zone_bottom if pending.direction == "BUY" else close > pending.zone_top


def _retest_depth(row: dict[str, Any], pending: _PendingSetup) -> float | None:
    """Return zone penetration fraction [0,1] when touched and directionally rejected."""
    assert pending.zone_top is not None and pending.zone_bottom is not None
    top, bottom = pending.zone_top, pending.zone_bottom
    width = top - bottom
    if width <= 0:
        return None
    high, low, close = float(row["high"]), float(row["low"]), float(row["close"])
    touched = high >= bottom and low <= top
    if not touched:
        return None
    midpoint = (top + bottom) / 2.0
    if pending.direction == "BUY":
        if close < midpoint:
            return None
        penetration = max(0.0, top - low)
    else:
        if close > midpoint:
            return None
        penetration = max(0.0, high - bottom)
    return min(1.0, penetration / width)


def _structure_stop(row: dict[str, Any], pending: _PendingSetup, config: SMCSetupV5Config) -> float | None:
    assert pending.zone_top is not None and pending.zone_bottom is not None
    entry = float(row["close"])
    atr = float(row.get("atr")) if _finite(row.get("atr")) and float(row.get("atr")) > 0 else 0.0
    floor_distance = config.atr_stop_floor_multiple * atr
    if pending.direction == "BUY":
        candidates = [pending.zone_bottom]
        if _finite(row.get("last_swing_low")) and float(row["last_swing_low"]) < entry:
            candidates.append(float(row["last_swing_low"]))
        stop = min(candidates)
        if floor_distance > 0 and entry - stop < floor_distance:
            stop = entry - floor_distance
        return stop if stop < entry else None
    candidates = [pending.zone_top]
    if _finite(row.get("last_swing_high")) and float(row["last_swing_high"]) > entry:
        candidates.append(float(row["last_swing_high"]))
    stop = max(candidates)
    if floor_distance > 0 and stop - entry < floor_distance:
        stop = entry + floor_distance
    return stop if stop > entry else None


class GoldmicroSMCSetupV5:
    """Deterministic research state machine over causal SMC feature rows."""

    def __init__(self, config: SMCSetupV5Config = SMCSetupV5Config()):
        config.validate()
        self.config = config

    def generate(self, df: pl.DataFrame) -> list[SMCSetupV5Event]:
        missing = [name for name in _required_columns() if name not in df.columns]
        if missing:
            raise ValueError(f"SMC V5 frame missing required columns: {missing}")
        events: list[SMCSetupV5Event] = []
        pending: _PendingSetup | None = None

        for idx, row in enumerate(df.iter_rows(named=True)):
            direction, archetype, conflict = _break_evidence(row)
            if conflict:
                pending = None
                continue

            # Any newly confirmed break supersedes prior pending state. Opposite or
            # same-direction new breaks both start a new causally attributable setup.
            if direction is not None and archetype is not None:
                pending = _PendingSetup(
                    direction=direction,
                    archetype=archetype,
                    break_index=idx,
                    break_level=_break_level(row, direction),
                )

            if pending is None:
                continue
            age = idx - pending.break_index
            if age > self.config.max_setup_age_bars:
                pending = None
                continue

            if pending.zone_confirm_index is None:
                if age > self.config.max_zone_delay_bars:
                    pending = None
                    continue
                zone = _aligned_zone(row, pending.direction, idx)
                if zone is not None:
                    ztype, origin, top, bottom = zone
                    if top <= bottom:
                        pending = None
                        continue
                    pending.zone_type = ztype
                    pending.zone_origin_index = origin
                    pending.zone_confirm_index = idx
                    pending.zone_top = top
                    pending.zone_bottom = bottom
                continue

            assert pending.zone_confirm_index is not None
            if idx - pending.zone_confirm_index < self.config.min_retest_delay_bars:
                continue
            if _zone_invalidated(row, pending):
                pending = None
                continue
            depth = _retest_depth(row, pending)
            if depth is None:
                continue
            stop = _structure_stop(row, pending, self.config)
            if stop is None:
                pending = None
                continue
            entry = float(row["close"])
            risk = abs(entry - stop)
            if risk <= 0 or not math.isfinite(risk):
                pending = None
                continue
            tp = entry + self.config.fixed_reward_r * risk if pending.direction == "BUY" else entry - self.config.fixed_reward_r * risk
            evidence = (
                pending.archetype,
                f"ZONE_{pending.zone_type}",
                "ZONE_RETEST_REJECTION",
                "STRUCTURE_DERIVED_STOP",
            )
            events.append(SMCSetupV5Event(
                setup_direction=pending.direction,
                setup_archetype=pending.archetype,
                break_index=pending.break_index,
                break_level=pending.break_level,
                zone_type=pending.zone_type,  # type: ignore[arg-type]
                zone_origin_index=int(pending.zone_origin_index if pending.zone_origin_index is not None else pending.zone_confirm_index),
                zone_confirm_index=pending.zone_confirm_index,
                zone_top=float(pending.zone_top),
                zone_bottom=float(pending.zone_bottom),
                retest_index=idx,
                retest_depth_fraction=float(depth),
                entry_index=idx,
                entry_mid=entry,
                stop_mid=float(stop),
                take_profit_mid=float(tp),
                declared_reward_r=self.config.fixed_reward_r,
                break_to_zone_bars=pending.zone_confirm_index - pending.break_index,
                zone_to_retest_bars=idx - pending.zone_confirm_index,
                setup_age_bars=idx - pending.break_index,
                component_evidence=evidence,
            ))
            pending = None

        return events
