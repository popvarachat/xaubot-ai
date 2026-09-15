"""Read-only GOLDmicro transaction-cost diagnostic for MetaTrader 5.

Collects broker swap metadata plus recent historical deal charges for GOLDmicro.
It does not place, modify, or close orders and does not print credentials or
account-login identifiers.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timedelta

try:
    import MetaTrader5 as mt5
except ImportError:
    print("MetaTrader5 Python package is not installed.", file=sys.stderr)
    raise SystemExit(2)


SYMBOL = "GOLDmicro"
SWAP_FIELDS = (
    "swap_mode",
    "swap_long",
    "swap_short",
    "swap_rollover3days",
)


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _summary(values: list[float]):
    if not values:
        return None
    ordered = sorted(values)
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
        "p95": ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Dump read-only GOLDmicro transaction-cost evidence")
    parser.add_argument("--days", type=int, default=365, help="History window in days, default 365")
    args = parser.parse_args()
    if args.days <= 0:
        raise SystemExit("--days must be positive")

    if not mt5.initialize():
        print(f"MT5 initialize failed: {mt5.last_error()}", file=sys.stderr)
        return 1

    try:
        account = mt5.account_info()
        if not mt5.symbol_select(SYMBOL, True):
            print(f"Unable to select {SYMBOL}: {mt5.last_error()}", file=sys.stderr)
            return 1

        info = mt5.symbol_info(SYMBOL)
        if info is None:
            print(f"symbol_info({SYMBOL}) returned None", file=sys.stderr)
            return 1

        end = datetime.now()
        start = end - timedelta(days=args.days)
        deals = mt5.history_deals_get(start, end, group=f"*{SYMBOL}*")
        if deals is None:
            deals = ()

        rows = []
        commission_per_lot_per_deal = []
        swap_per_lot_per_deal = []
        fee_per_lot_per_deal = []
        execution_slippage_points = []

        point = _safe_float(getattr(info, "point", None)) or 0.0

        for deal in deals:
            volume = _safe_float(getattr(deal, "volume", None)) or 0.0
            commission = _safe_float(getattr(deal, "commission", None)) or 0.0
            swap = _safe_float(getattr(deal, "swap", None)) or 0.0
            fee = _safe_float(getattr(deal, "fee", None)) or 0.0
            price = _safe_float(getattr(deal, "price", None))
            order_ticket = getattr(deal, "order", None)

            if volume > 0:
                if commission != 0:
                    commission_per_lot_per_deal.append(abs(commission) / volume)
                if swap != 0:
                    swap_per_lot_per_deal.append(abs(swap) / volume)
                if fee != 0:
                    fee_per_lot_per_deal.append(abs(fee) / volume)

            requested_price = None
            slippage_points = None
            if order_ticket and price is not None and point > 0:
                hist_orders = mt5.history_orders_get(ticket=order_ticket)
                if hist_orders:
                    order = hist_orders[0]
                    requested_price = _safe_float(getattr(order, "price_open", None))
                    if requested_price and requested_price > 0:
                        slippage_points = abs(price - requested_price) / point
                        execution_slippage_points.append(slippage_points)

            rows.append(
                {
                    "time": datetime.fromtimestamp(getattr(deal, "time", 0)).isoformat()
                    if getattr(deal, "time", 0)
                    else None,
                    "type": getattr(deal, "type", None),
                    "entry": getattr(deal, "entry", None),
                    "volume": volume,
                    "price": price,
                    "commission": commission,
                    "swap": swap,
                    "fee": fee,
                    "requested_order_price": requested_price,
                    "absolute_slippage_points": slippage_points,
                }
            )

        payload = {
            "symbol": SYMBOL,
            "account_currency": getattr(account, "currency", None) if account is not None else None,
            "history_window": {
                "from": start.isoformat(),
                "to": end.isoformat(),
                "days": args.days,
            },
            "symbol_swap_spec": {field: getattr(info, field, None) for field in SWAP_FIELDS},
            "deal_count": len(rows),
            "charge_summary": {
                "commission_abs_account_currency_per_lot_per_deal": _summary(commission_per_lot_per_deal),
                "swap_abs_account_currency_per_lot_per_deal": _summary(swap_per_lot_per_deal),
                "fee_abs_account_currency_per_lot_per_deal": _summary(fee_per_lot_per_deal),
                "absolute_execution_slippage_points": _summary(execution_slippage_points),
            },
            "totals": {
                "commission": sum(row["commission"] for row in rows),
                "swap": sum(row["swap"] for row in rows),
                "fee": sum(row["fee"] for row in rows),
            },
            "notes": [
                "Commission/swap/fee values are MT5 historical deal fields in account currency.",
                "Per-lot commission is reported per deal, not assumed round-turn; charging conventions vary by broker/account.",
                "Slippage compares historical deal execution price with the linked historical order price_open only when available.",
                "A zero/empty statistic means the selected history contains no usable evidence for that cost component.",
            ],
            "recent_deals": rows[-20:],
        }

        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
        return 0
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
