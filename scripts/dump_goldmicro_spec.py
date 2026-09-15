"""Print non-sensitive MT5 broker specifications for GOLDmicro.

Read-only diagnostic. It does not place, modify, or close orders and does not
print account login/password. Run it on the Windows machine where MetaTrader 5
is installed and already logged in.
"""

from __future__ import annotations

import json
import sys

try:
    import MetaTrader5 as mt5
except ImportError:
    print("MetaTrader5 Python package is not installed.", file=sys.stderr)
    raise SystemExit(2)


SYMBOL = "GOLDmicro"
SAFE_FIELDS = (
    "name",
    "description",
    "path",
    "digits",
    "point",
    "trade_contract_size",
    "trade_tick_size",
    "trade_tick_value",
    "trade_tick_value_profit",
    "trade_tick_value_loss",
    "volume_min",
    "volume_max",
    "volume_step",
    "trade_stops_level",
    "trade_freeze_level",
    "spread",
    "spread_float",
    "currency_base",
    "currency_profit",
    "currency_margin",
    "trade_calc_mode",
)


def _profit_check(action: int, volume: float, open_price: float, close_price: float):
    value = mt5.order_calc_profit(action, SYMBOL, volume, open_price, close_price)
    return None if value is None else float(value)


def main() -> int:
    if not mt5.initialize():
        print(f"MT5 initialize failed: {mt5.last_error()}", file=sys.stderr)
        return 1

    try:
        terminal = mt5.terminal_info()
        if terminal is None:
            print("MT5 terminal_info() unavailable", file=sys.stderr)
            return 1

        account = mt5.account_info()

        if not mt5.symbol_select(SYMBOL, True):
            print(f"Unable to select {SYMBOL}: {mt5.last_error()}", file=sys.stderr)
            return 1

        info = mt5.symbol_info(SYMBOL)
        tick = mt5.symbol_info_tick(SYMBOL)
        if info is None:
            print(f"symbol_info({SYMBOL}) returned None", file=sys.stderr)
            return 1

        payload = {
            "terminal_connected": bool(getattr(terminal, "connected", False)),
            "account_currency": getattr(account, "currency", None) if account is not None else None,
            "symbol": SYMBOL,
            "spec": {field: getattr(info, field, None) for field in SAFE_FIELDS},
            "current_tick": None,
            "order_calc_profit_check": None,
        }
        if tick is not None:
            payload["current_tick"] = {
                "bid": tick.bid,
                "ask": tick.ask,
                "last": tick.last,
                "time_msc": tick.time_msc,
                "spread_price": tick.ask - tick.bid,
                "spread_points": (tick.ask - tick.bid) / info.point if info.point else None,
            }

            # Independent MT5 P/L cross-checks. Values are returned by MT5 in
            # the trading account currency reported above. These are calculations
            # only and do not submit any trade request.
            plus_one = round(tick.ask + 1.0, info.digits)
            minus_one = round(tick.bid - 1.0, info.digits)
            payload["order_calc_profit_check"] = {
                "currency": payload["account_currency"],
                "buy_0_10_lot_plus_1_price": _profit_check(
                    mt5.ORDER_TYPE_BUY, 0.10, tick.ask, plus_one
                ),
                "buy_1_00_lot_plus_1_price": _profit_check(
                    mt5.ORDER_TYPE_BUY, 1.00, tick.ask, plus_one
                ),
                "sell_0_10_lot_minus_1_price": _profit_check(
                    mt5.ORDER_TYPE_SELL, 0.10, tick.bid, minus_one
                ),
                "sell_1_00_lot_minus_1_price": _profit_check(
                    mt5.ORDER_TYPE_SELL, 1.00, tick.bid, minus_one
                ),
            }

        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
        return 0
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
