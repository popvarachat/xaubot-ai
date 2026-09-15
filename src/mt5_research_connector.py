"""Read-only MT5 connector for GOLDmicro research batches.

This connector intentionally attaches to an already logged-in MetaTrader 5 terminal
session and exposes market-data reads only. It does not accept account credentials,
does not implement order_send, and is separate from the production MT5Connector.

The goal is to let offline/non-live model research reuse the operator's existing MT5
terminal session without requiring MT5_LOGIN / MT5_PASSWORD / MT5_SERVER secrets in
the research environment.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os

import polars as pl

try:
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover - exercised on machines without MT5 package
    mt5 = None


@dataclass(frozen=True)
class ResearchTerminalStatus:
    connected: bool
    terminal_connected: bool
    account_available: bool
    server: str = ""


class MT5ResearchConnector:
    """Minimal market-data-only connector bound to an existing MT5 login session."""

    TIMEFRAMES = {
        "M1": "TIMEFRAME_M1",
        "M5": "TIMEFRAME_M5",
        "M15": "TIMEFRAME_M15",
        "M30": "TIMEFRAME_M30",
        "H1": "TIMEFRAME_H1",
        "H4": "TIMEFRAME_H4",
        "D1": "TIMEFRAME_D1",
        "W1": "TIMEFRAME_W1",
    }

    def __init__(self, *, path: Optional[str] = None, timeout: int = 60000) -> None:
        self.path = path or os.getenv("MT5_PATH") or None
        self.timeout = int(timeout)
        self._connected = False

    def connect(self) -> bool:
        """Attach to the locally configured/logged-in MT5 terminal without credentials."""
        if mt5 is None:
            raise RuntimeError("MetaTrader5 Python package is not installed")

        kwargs = {"timeout": self.timeout}
        if self.path:
            kwargs["path"] = str(Path(self.path))

        # No login/password/server parameters by design. MT5 uses the terminal's
        # currently saved/logged-in session. This keeps research secrets out of .env.
        if not mt5.initialize(**kwargs):
            raise ConnectionError(
                "Could not attach to MT5 terminal session. Open MetaTrader 5, log in "
                "to the intended account, then rerun the research batch. "
                f"MT5 error: {mt5.last_error()}"
            )

        terminal = mt5.terminal_info()
        account = mt5.account_info()
        if terminal is None or not bool(getattr(terminal, "connected", False)):
            mt5.shutdown()
            raise ConnectionError("MT5 terminal is open but not connected to the broker server")
        if account is None:
            mt5.shutdown()
            raise ConnectionError("MT5 terminal has no logged-in account session")

        self._connected = True
        return True

    def status(self) -> ResearchTerminalStatus:
        if mt5 is None:
            return ResearchTerminalStatus(False, False, False, "")
        terminal = mt5.terminal_info() if self._connected else None
        account = mt5.account_info() if self._connected else None
        return ResearchTerminalStatus(
            connected=self._connected,
            terminal_connected=bool(terminal and getattr(terminal, "connected", False)),
            account_available=account is not None,
            server=str(getattr(account, "server", "")) if account is not None else "",
        )

    def disconnect(self) -> None:
        if self._connected and mt5 is not None:
            mt5.shutdown()
        self._connected = False

    def _timeframe_constant(self, timeframe: str):
        if mt5 is None:
            raise RuntimeError("MetaTrader5 Python package is not installed")
        attr = self.TIMEFRAMES.get(timeframe.upper())
        if attr is None:
            raise ValueError(f"Invalid timeframe: {timeframe}")
        return getattr(mt5, attr)

    def get_market_data(self, symbol: str, timeframe: str = "M15", count: int = 1000) -> pl.DataFrame:
        """Read historical bars from the attached terminal. No trading methods exist."""
        if not self._connected:
            raise ConnectionError("Research MT5 connector is not connected")
        if count <= 0:
            raise ValueError("count must be positive")

        tf = self._timeframe_constant(timeframe)
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"MT5 symbol not available/selectable: {symbol}; error={mt5.last_error()}")

        rates = mt5.copy_rates_from_pos(symbol, tf, 0, int(count))
        if rates is None or len(rates) == 0:
            raise RuntimeError(
                f"No MT5 market data returned for {symbol} {timeframe}; error={mt5.last_error()}"
            )

        df = pl.DataFrame(
            {
                "time": rates["time"],
                "open": rates["open"],
                "high": rates["high"],
                "low": rates["low"],
                "close": rates["close"],
                "tick_volume": rates["tick_volume"],
                "spread": rates["spread"],
                "real_volume": rates["real_volume"],
            }
        )
        return df.with_columns(
            [
                pl.from_epoch(pl.col("time"), time_unit="s").alias("time"),
                pl.col("open").cast(pl.Float64),
                pl.col("high").cast(pl.Float64),
                pl.col("low").cast(pl.Float64),
                pl.col("close").cast(pl.Float64),
                pl.col("tick_volume").cast(pl.Int64).alias("volume"),
            ]
        ).drop("tick_volume")
