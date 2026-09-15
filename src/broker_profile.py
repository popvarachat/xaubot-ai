"""Broker symbol profile helpers.

This module removes hard-coded XAUUSD sizing assumptions from new code paths.
It can be populated from MT5 symbol_info() at runtime, while tests/backtests can
use explicit historical broker specifications.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True)
class BrokerSymbolProfile:
    symbol: str
    point: float
    tick_size: float
    tick_value: float
    contract_size: float
    volume_min: float
    volume_max: float
    volume_step: float
    cash_per_price_unit_per_lot: float | None = None
    cash_currency: str | None = None

    @classmethod
    def from_mt5_symbol_info(cls, symbol: str, info: Any) -> "BrokerSymbolProfile":
        """Build a profile from MetaTrader5.symbol_info(symbol).

        Uses trade_tick_value_loss when available because sizing is risk-side.
        Falls back to trade_tick_value for brokers that expose only one value.

        Note: some CFD symbols/brokers expose tick metadata that does not match
        ``order_calc_profit()`` in the account currency. In that case set
        ``cash_per_price_unit_per_lot`` from a read-only MT5 calibration and the
        profile will prefer that calibrated cash model.
        """
        if info is None:
            raise ValueError(f"No MT5 symbol info available for {symbol}")

        tick_value_loss = float(getattr(info, "trade_tick_value_loss", 0.0) or 0.0)
        tick_value = tick_value_loss or float(getattr(info, "trade_tick_value", 0.0) or 0.0)

        profile = cls(
            symbol=symbol,
            point=float(getattr(info, "point", 0.0) or 0.0),
            tick_size=float(getattr(info, "trade_tick_size", 0.0) or 0.0),
            tick_value=tick_value,
            contract_size=float(getattr(info, "trade_contract_size", 0.0) or 0.0),
            volume_min=float(getattr(info, "volume_min", 0.0) or 0.0),
            volume_max=float(getattr(info, "volume_max", 0.0) or 0.0),
            volume_step=float(getattr(info, "volume_step", 0.0) or 0.0),
        )
        profile.validate()
        return profile

    def validate(self) -> None:
        values = {
            "point": self.point,
            "tick_size": self.tick_size,
            "tick_value": self.tick_value,
            "contract_size": self.contract_size,
            "volume_min": self.volume_min,
            "volume_max": self.volume_max,
            "volume_step": self.volume_step,
        }
        invalid = [name for name, value in values.items() if value <= 0]
        if invalid:
            raise ValueError(f"Invalid broker profile fields for {self.symbol}: {', '.join(invalid)}")
        if self.volume_max < self.volume_min:
            raise ValueError("volume_max must be >= volume_min")
        if self.cash_per_price_unit_per_lot is not None and self.cash_per_price_unit_per_lot <= 0:
            raise ValueError("cash_per_price_unit_per_lot must be positive when provided")

    def spread_points(self, bid: float, ask: float) -> float:
        if ask < bid:
            raise ValueError("ask must be >= bid")
        return (ask - bid) / self.point

    def cash_pnl_for_price_delta(self, price_delta: float, lot_size: float) -> float:
        """Convert a favorable price delta into account-currency P/L.

        If a read-only MT5 ``order_calc_profit()`` calibration is available,
        prefer it over raw tick metadata because the broker/account conversion
        path can differ from the naive ``ticks * tick_value`` calculation.
        """
        if lot_size < 0:
            raise ValueError("lot_size cannot be negative")
        if self.cash_per_price_unit_per_lot is not None:
            return price_delta * self.cash_per_price_unit_per_lot * lot_size
        ticks = price_delta / self.tick_size
        return ticks * self.tick_value * lot_size

    def cash_risk_per_lot(self, entry_price: float, stop_price: float) -> float:
        """Return monetary loss for 1.0 lot if stop is hit, excluding slippage/fees."""
        distance = abs(entry_price - stop_price)
        if distance <= 0:
            return 0.0
        return abs(self.cash_pnl_for_price_delta(distance, 1.0))

    def raw_lot_for_risk(self, risk_amount: float, entry_price: float, stop_price: float) -> float:
        if risk_amount <= 0:
            return 0.0
        per_lot = self.cash_risk_per_lot(entry_price, stop_price)
        if per_lot <= 0:
            return 0.0
        return risk_amount / per_lot

    def normalize_volume_down(self, raw_lot: float) -> float:
        """Floor volume to broker step so rounding never increases risk.

        Returns 0.0 when the risk budget cannot support the broker minimum lot.
        """
        if raw_lot < self.volume_min:
            return 0.0
        capped = min(raw_lot, self.volume_max)
        steps = math.floor((capped + 1e-12) / self.volume_step)
        normalized = steps * self.volume_step
        if normalized < self.volume_min:
            return 0.0
        return round(normalized, 10)

    def lot_for_risk(self, risk_amount: float, entry_price: float, stop_price: float) -> float:
        return self.normalize_volume_down(
            self.raw_lot_for_risk(risk_amount, entry_price, stop_price)
        )
