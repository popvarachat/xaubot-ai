"""Execution-cost model for GOLDmicro backtests.

The legacy backtest uses one OHLC price stream for both BUY and SELL. This
module makes spread/slippage/fees explicit and can work either with observed
Bid/Ask ticks or with a mid-price plus an assumed spread profile.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.broker_profile import BrokerSymbolProfile

Side = Literal["BUY", "SELL"]


@dataclass(frozen=True)
class BacktestCostConfig:
    spread_points: float = 55.0
    slippage_points: float = 0.0
    commission_per_lot_round_turn: float = 0.0
    swap_per_lot: float = 0.0


@dataclass(frozen=True)
class TradeCostBreakdown:
    gross_pnl: float
    commission: float
    swap: float
    net_pnl: float
    entry_exec_price: float
    exit_exec_price: float


class GoldmicroCostModel:
    def __init__(self, profile: BrokerSymbolProfile, config: BacktestCostConfig | None = None):
        self.profile = profile
        self.profile.validate()
        self.config = config or BacktestCostConfig()
        if self.config.spread_points < 0 or self.config.slippage_points < 0:
            raise ValueError("spread/slippage points cannot be negative")

    def _half_spread_price(self) -> float:
        return self.config.spread_points * self.profile.point / 2.0

    def _slippage_price(self) -> float:
        return self.config.slippage_points * self.profile.point

    def execution_prices_from_mid(self, side: Side, entry_mid: float, exit_mid: float) -> tuple[float, float]:
        """Convert a mid-price backtest path into adverse executable prices.

        BUY enters at Ask and exits at Bid; SELL enters at Bid and exits at Ask.
        Slippage is applied adversely at both entry and exit.
        """
        hs = self._half_spread_price()
        slip = self._slippage_price()
        if side == "BUY":
            return entry_mid + hs + slip, exit_mid - hs - slip
        if side == "SELL":
            return entry_mid - hs - slip, exit_mid + hs + slip
        raise ValueError(f"Unsupported side: {side}")

    def pnl_from_observed_bid_ask(
        self,
        *,
        side: Side,
        entry_bid: float,
        entry_ask: float,
        exit_bid: float,
        exit_ask: float,
        lot_size: float,
        swap_days: float = 0.0,
    ) -> TradeCostBreakdown:
        """Calculate P/L from actual Bid/Ask observations.

        No synthetic spread is added because spread is already embedded in the
        supplied Bid/Ask prices. Optional configured slippage is still charged.
        """
        slip = self._slippage_price()
        if side == "BUY":
            entry_exec = entry_ask + slip
            exit_exec = exit_bid - slip
            price_delta = exit_exec - entry_exec
        elif side == "SELL":
            entry_exec = entry_bid - slip
            exit_exec = exit_ask + slip
            price_delta = entry_exec - exit_exec
        else:
            raise ValueError(f"Unsupported side: {side}")
        return self._cash_breakdown(price_delta, lot_size, entry_exec, exit_exec, swap_days)

    def pnl_from_mid(
        self,
        *,
        side: Side,
        entry_mid: float,
        exit_mid: float,
        lot_size: float,
        swap_days: float = 0.0,
    ) -> TradeCostBreakdown:
        entry_exec, exit_exec = self.execution_prices_from_mid(side, entry_mid, exit_mid)
        price_delta = (exit_exec - entry_exec) if side == "BUY" else (entry_exec - exit_exec)
        return self._cash_breakdown(price_delta, lot_size, entry_exec, exit_exec, swap_days)

    def _cash_breakdown(
        self,
        favorable_price_delta: float,
        lot_size: float,
        entry_exec: float,
        exit_exec: float,
        swap_days: float,
    ) -> TradeCostBreakdown:
        if lot_size <= 0:
            raise ValueError("lot_size must be positive")
        ticks = favorable_price_delta / self.profile.tick_size
        gross = ticks * self.profile.tick_value * lot_size
        commission = self.config.commission_per_lot_round_turn * lot_size
        swap = self.config.swap_per_lot * lot_size * swap_days
        net = gross - commission - swap
        return TradeCostBreakdown(
            gross_pnl=gross,
            commission=commission,
            swap=swap,
            net_pnl=net,
            entry_exec_price=entry_exec,
            exit_exec_price=exit_exec,
        )
