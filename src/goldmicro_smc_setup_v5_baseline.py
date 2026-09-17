"""Development-only raw economic baseline for GOLDmicro SMC Setup V5.

This module evaluates the frozen V5 setup mechanics without any ML overlay.
It is intentionally separate from live execution. Historical data already
inspected during V3/V4/V5 design is development evidence only; outputs from
this module cannot be used as confirmatory promotion evidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import median
from typing import Any, Literal

import polars as pl

from backtests.goldmicro_cost_model import BacktestCostConfig, GoldmicroCostModel
from src.broker_profile import BrokerSymbolProfile
from src.goldmicro_smc_setup_v5 import SMCSetupV5Event

ExitReason = Literal["TAKE_PROFIT", "STOP_LOSS", "SAME_BAR_SL_FIRST", "TIMEOUT"]


@dataclass(frozen=True)
class SMCSetupV5BaselineThresholds:
    chronological_blocks: int = 5
    horizon_bars: int = 32
    min_events_per_block: int = 30
    min_positive_blocks: int = 4

    def validate(self) -> None:
        if self.chronological_blocks < 2:
            raise ValueError("chronological_blocks must be >= 2")
        if self.horizon_bars < 1:
            raise ValueError("horizon_bars must be >= 1")
        if self.min_events_per_block < 1:
            raise ValueError("min_events_per_block must be >= 1")
        if not 1 <= self.min_positive_blocks <= self.chronological_blocks:
            raise ValueError("min_positive_blocks must be within chronological_blocks")


@dataclass(frozen=True)
class SMCSetupV5Outcome:
    entry_index: int
    direction: str
    archetype: str
    zone_type: str
    exit_index: int
    exit_mid: float
    exit_reason: ExitReason
    gross_r: float
    normal_net_r: float
    conservative_net_r: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_goldmicro_profile() -> BrokerSymbolProfile:
    return BrokerSymbolProfile(
        symbol="GOLDmicro",
        point=0.01,
        tick_size=0.01,
        tick_value=0.01,
        contract_size=1.0,
        volume_min=0.10,
        volume_max=100.0,
        volume_step=0.01,
        cash_per_price_unit_per_lot=33.29,
        cash_currency="THB",
    )


def cost_config(name: str) -> BacktestCostConfig:
    if name == "normal":
        return BacktestCostConfig(spread_points=50.0, slippage_points=1.0)
    if name == "conservative":
        return BacktestCostConfig(spread_points=55.0, slippage_points=12.0)
    raise ValueError(f"unknown cost profile: {name}")


def _signed_move(direction: str, entry: float, exit_mid: float) -> float:
    if direction == "BUY":
        return exit_mid - entry
    if direction == "SELL":
        return entry - exit_mid
    raise ValueError(f"unknown direction: {direction}")


def _net_r(
    *,
    event: SMCSetupV5Event,
    exit_mid: float,
    model: GoldmicroCostModel,
    profile: BrokerSymbolProfile,
) -> float:
    risk_distance = abs(event.entry_mid - event.stop_mid)
    if risk_distance <= 0:
        raise ValueError("event risk distance must be positive")
    risk_cash = abs(profile.cash_pnl_for_price_delta(risk_distance, 1.0))
    if risk_cash <= 0:
        raise ValueError("event risk cash must be positive")
    pnl = model.pnl_from_mid(
        side=event.setup_direction,
        entry_mid=event.entry_mid,
        exit_mid=exit_mid,
        lot_size=1.0,
    )
    return pnl.net_pnl / risk_cash


def simulate_event(
    df: pl.DataFrame,
    event: SMCSetupV5Event,
    *,
    horizon_bars: int = 32,
    profile: BrokerSymbolProfile | None = None,
) -> SMCSetupV5Outcome:
    """Resolve one V5 event with adverse same-bar ordering and 32-bar timeout."""
    profile = profile or default_goldmicro_profile()
    normal_model = GoldmicroCostModel(profile, cost_config("normal"))
    conservative_model = GoldmicroCostModel(profile, cost_config("conservative"))

    highs = df["high"].to_list()
    lows = df["low"].to_list()
    closes = df["close"].to_list()
    n = len(df)
    if not 0 <= event.entry_index < n:
        raise ValueError("entry_index outside frame")
    end = min(event.entry_index + horizon_bars, n - 1)
    exit_index = end
    exit_mid = float(closes[end])
    reason: ExitReason = "TIMEOUT"

    for idx in range(event.entry_index + 1, end + 1):
        high = float(highs[idx])
        low = float(lows[idx])
        if event.setup_direction == "BUY":
            hit_sl = low <= event.stop_mid
            hit_tp = high >= event.take_profit_mid
        else:
            hit_sl = high >= event.stop_mid
            hit_tp = low <= event.take_profit_mid
        if hit_sl and hit_tp:
            exit_index = idx
            exit_mid = event.stop_mid
            reason = "SAME_BAR_SL_FIRST"
            break
        if hit_sl:
            exit_index = idx
            exit_mid = event.stop_mid
            reason = "STOP_LOSS"
            break
        if hit_tp:
            exit_index = idx
            exit_mid = event.take_profit_mid
            reason = "TAKE_PROFIT"
            break

    risk_distance = abs(event.entry_mid - event.stop_mid)
    if risk_distance <= 0:
        raise ValueError("event risk distance must be positive")
    gross_r = _signed_move(event.setup_direction, event.entry_mid, exit_mid) / risk_distance
    return SMCSetupV5Outcome(
        entry_index=event.entry_index,
        direction=event.setup_direction,
        archetype=event.setup_archetype,
        zone_type=event.zone_type,
        exit_index=exit_index,
        exit_mid=float(exit_mid),
        exit_reason=reason,
        gross_r=float(gross_r),
        normal_net_r=float(_net_r(event=event, exit_mid=exit_mid, model=normal_model, profile=profile)),
        conservative_net_r=float(_net_r(event=event, exit_mid=exit_mid, model=conservative_model, profile=profile)),
    )


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "positive_rate": None}
    return {
        "n": len(values),
        "mean": float(sum(values) / len(values)),
        "median": float(median(values)),
        "positive_rate": float(sum(v > 0 for v in values) / len(values)),
    }


def evaluate_baseline(
    df: pl.DataFrame,
    events: list[SMCSetupV5Event],
    *,
    thresholds: SMCSetupV5BaselineThresholds = SMCSetupV5BaselineThresholds(),
) -> dict[str, Any]:
    """Evaluate non-overlapping raw-time blocks; no ML, PF/DD, or threshold tuning."""
    thresholds.validate()
    if not events:
        return {
            "status": "NO_EVENTS",
            "thresholds": asdict(thresholds),
            "events": 0,
            "blocks": [],
            "positive_blocks_normal": 0,
            "positive_blocks_conservative": 0,
            "gate_pass": False,
        }

    outcomes = [
        simulate_event(df, event, horizon_bars=thresholds.horizon_bars)
        for event in events
        if event.entry_index + 1 < len(df)
    ]
    blocks: list[dict[str, Any]] = []
    n_rows = len(df)
    for block_idx in range(thresholds.chronological_blocks):
        start = (n_rows * block_idx) // thresholds.chronological_blocks
        end = (n_rows * (block_idx + 1)) // thresholds.chronological_blocks
        block = [o for o in outcomes if start <= o.entry_index < end]
        gross = [o.gross_r for o in block]
        normal = [o.normal_net_r for o in block]
        conservative = [o.conservative_net_r for o in block]
        blocks.append({
            "block": block_idx + 1,
            "raw_start_index": start,
            "raw_end_index_exclusive": end,
            "events": len(block),
            "gross_r": _stats(gross),
            "normal_net_r": _stats(normal),
            "conservative_net_r": _stats(conservative),
            "direction_counts": {
                "BUY": sum(o.direction == "BUY" for o in block),
                "SELL": sum(o.direction == "SELL" for o in block),
            },
            "archetype_counts": {
                "BOS_CONTINUATION": sum(o.archetype == "BOS_CONTINUATION" for o in block),
                "CHOCH_REVERSAL": sum(o.archetype == "CHOCH_REVERSAL" for o in block),
            },
            "zone_counts": {
                "FVG": sum(o.zone_type == "FVG" for o in block),
                "OB": sum(o.zone_type == "OB" for o in block),
            },
        })

    enough_events = all(b["events"] >= thresholds.min_events_per_block for b in blocks)
    positive_normal = sum((b["normal_net_r"]["mean"] or 0.0) > 0.0 for b in blocks)
    positive_conservative = sum((b["conservative_net_r"]["mean"] or 0.0) > 0.0 for b in blocks)
    gate_pass = (
        enough_events
        and positive_normal >= thresholds.min_positive_blocks
        and positive_conservative >= thresholds.min_positive_blocks
    )
    return {
        "status": "V5_RAW_BASELINE_PASS" if gate_pass else "V5_RAW_BASELINE_REJECT",
        "thresholds": asdict(thresholds),
        "events": len(outcomes),
        "blocks": blocks,
        "positive_blocks_normal": positive_normal,
        "positive_blocks_conservative": positive_conservative,
        "min_events_observed": min((b["events"] for b in blocks), default=0),
        "gate_pass": gate_pass,
        "guard": (
            "Development evidence only on already-inspected history. Do not tune V5 rules from these realized returns. "
            "Any confirmatory claim requires a frozen protocol and fresh chronological/prospective evidence."
        ),
    }
