"""GOLDmicro post-trade replay for legacy backtest output.

This module intentionally does not alter signal generation. It takes the
legacy SimulatedTrade records, recalculates broker-valid GOLDmicro position
sizes, reprices entry/exit with explicit spread/slippage/fees, and rebuilds
portfolio statistics. This gives a safer baseline before modifying the legacy
signal engine itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Iterable, Protocol

from backtests.goldmicro_cost_model import GoldmicroCostModel
from src.broker_profile import BrokerSymbolProfile
from src.goldmicro_risk import size_goldmicro_position


class TradeLike(Protocol):
    direction: str
    entry_price: float
    exit_price: float
    stop_loss: float


@dataclass(frozen=True)
class ReplayedTrade:
    direction: str
    entry_price_mid: float
    exit_price_mid: float
    entry_exec_price: float
    exit_exec_price: float
    lot_size: float
    requested_risk_amount: float
    actual_risk_amount: float
    net_pnl: float
    skipped: bool = False
    skip_reason: str = ""


@dataclass
class ReplayStats:
    initial_capital: float
    final_capital: float
    total_trades: int = 0
    executed_trades: int = 0
    skipped_trades: int = 0
    wins: int = 0
    losses: int = 0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_profit: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_percent: float = 0.0
    expectancy: float = 0.0
    sharpe_ratio: float = 0.0
    trades: list[ReplayedTrade] = field(default_factory=list)


def _compute_summary(stats: ReplayStats, returns: list[float]) -> None:
    stats.net_profit = stats.final_capital - stats.initial_capital
    stats.profit_factor = (
        stats.gross_profit / stats.gross_loss if stats.gross_loss > 0 else float("inf")
    )
    stats.expectancy = (
        stats.net_profit / stats.executed_trades if stats.executed_trades else 0.0
    )

    if len(returns) >= 2:
        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
        std = sqrt(variance)
        stats.sharpe_ratio = (mean / std) * sqrt(len(returns)) if std > 0 else 0.0


def replay_legacy_trades(
    trades: Iterable[TradeLike],
    *,
    profile: BrokerSymbolProfile,
    cost_model: GoldmicroCostModel,
    initial_capital: float,
    risk_per_trade_percent: float = 1.0,
    win_rate: float = 0.55,
    reward_risk_ratio: float = 2.0,
    regime_multiplier: float = 1.0,
) -> ReplayStats:
    """Reprice and resize legacy trades using GOLDmicro broker constraints.

    Notes:
    - Entry and exit prices supplied by the legacy backtest are treated as MID
      prices. The cost model converts them to executable Bid/Ask prices.
    - Trade order and original entry/exit timing are preserved.
    - A trade is skipped when the broker minimum lot would exceed the allowed
      risk budget.
    - Capital is updated sequentially, so later position sizes and drawdown use
      the replayed equity path rather than the original legacy P/L path.
    """
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive")

    stats = ReplayStats(initial_capital=initial_capital, final_capital=initial_capital)
    capital = initial_capital
    peak = initial_capital
    returns: list[float] = []

    for trade in trades:
        stats.total_trades += 1

        sizing = size_goldmicro_position(
            profile=profile,
            account_balance=capital,
            entry_price=float(trade.entry_price),
            stop_price=float(trade.stop_loss),
            risk_per_trade_percent=risk_per_trade_percent,
            win_rate=win_rate,
            reward_risk_ratio=reward_risk_ratio,
            regime_multiplier=regime_multiplier,
        )

        if not sizing.approved:
            stats.skipped_trades += 1
            stats.trades.append(
                ReplayedTrade(
                    direction=str(trade.direction),
                    entry_price_mid=float(trade.entry_price),
                    exit_price_mid=float(trade.exit_price),
                    entry_exec_price=0.0,
                    exit_exec_price=0.0,
                    lot_size=0.0,
                    requested_risk_amount=sizing.requested_risk_amount,
                    actual_risk_amount=0.0,
                    net_pnl=0.0,
                    skipped=True,
                    skip_reason=sizing.reason,
                )
            )
            continue

        before = capital
        costs = cost_model.pnl_from_mid(
            side=str(trade.direction).upper(),
            entry_mid=float(trade.entry_price),
            exit_mid=float(trade.exit_price),
            lot_size=sizing.lot_size,
        )
        pnl = costs.net_pnl
        capital += pnl

        stats.executed_trades += 1
        if pnl > 0:
            stats.wins += 1
            stats.gross_profit += pnl
        elif pnl < 0:
            stats.losses += 1
            stats.gross_loss += abs(pnl)

        trade_return = pnl / before if before > 0 else 0.0
        returns.append(trade_return)

        if capital > peak:
            peak = capital
        if peak > 0:
            dd = (peak - capital) / peak * 100.0
            stats.max_drawdown_percent = max(stats.max_drawdown_percent, dd)

        stats.trades.append(
            ReplayedTrade(
                direction=str(trade.direction),
                entry_price_mid=float(trade.entry_price),
                exit_price_mid=float(trade.exit_price),
                entry_exec_price=costs.entry_exec_price,
                exit_exec_price=costs.exit_exec_price,
                lot_size=sizing.lot_size,
                requested_risk_amount=sizing.requested_risk_amount,
                actual_risk_amount=sizing.actual_risk_amount,
                net_pnl=pnl,
            )
        )

    stats.final_capital = capital
    _compute_summary(stats, returns)
    return stats
