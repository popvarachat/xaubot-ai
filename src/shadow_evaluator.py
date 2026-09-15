"""Shadow-model aggregation for GOLDmicro.

Shadow observations are hypothetical only. This module never sends orders and never
changes the active model. It summarizes candidate outcomes for lifecycle gates.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from math import sqrt
from typing import Iterable


@dataclass(frozen=True)
class ShadowTrade:
    model_id: str
    day: str
    pnl: float
    skipped: bool = False
    execution_cost_points: float = 0.0


@dataclass(frozen=True)
class ShadowSummary:
    model_id: str
    trades: int
    days: int
    wins: int
    losses: int
    profit_factor: float
    expectancy: float
    max_drawdown_pct: float
    skip_pct: float
    mean_execution_cost_points: float
    sharpe_like: float

    def to_dict(self) -> dict:
        return asdict(self)


def summarize_shadow(
    trades: Iterable[ShadowTrade],
    *,
    starting_equity: float = 20000.0,
) -> ShadowSummary:
    rows = list(trades)
    if not rows:
        raise ValueError("shadow trade set is empty")
    model_ids = {r.model_id for r in rows}
    if len(model_ids) != 1:
        raise ValueError("summarize_shadow expects one model_id")
    model_id = next(iter(model_ids))

    executed = [r for r in rows if not r.skipped]
    skipped = len(rows) - len(executed)
    pnl = [float(r.pnl) for r in executed]
    gross_profit = sum(x for x in pnl if x > 0)
    gross_loss = abs(sum(x for x in pnl if x < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    expectancy = sum(pnl) / len(pnl) if pnl else 0.0

    equity = starting_equity
    peak = starting_equity
    max_dd = 0.0
    returns: list[float] = []
    for x in pnl:
        before = equity
        equity += x
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak * 100.0)
        if before > 0:
            returns.append(x / before)

    sharpe_like = 0.0
    if len(returns) >= 2:
        mean = sum(returns) / len(returns)
        var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
        std = sqrt(var)
        if std > 0:
            sharpe_like = mean / std * sqrt(len(returns))

    cost_rows = [r.execution_cost_points for r in executed]
    mean_cost = sum(cost_rows) / len(cost_rows) if cost_rows else 0.0
    return ShadowSummary(
        model_id=model_id,
        trades=len(executed),
        days=len({r.day for r in rows}),
        wins=sum(1 for x in pnl if x > 0),
        losses=sum(1 for x in pnl if x < 0),
        profit_factor=pf,
        expectancy=expectancy,
        max_drawdown_pct=max_dd,
        skip_pct=(skipped / len(rows) * 100.0),
        mean_execution_cost_points=mean_cost,
        sharpe_like=sharpe_like,
    )


def summarize_many_shadow_models(
    trades: Iterable[ShadowTrade],
    *,
    starting_equity: float = 20000.0,
) -> list[ShadowSummary]:
    grouped: dict[str, list[ShadowTrade]] = {}
    for row in trades:
        grouped.setdefault(row.model_id, []).append(row)
    summaries = [summarize_shadow(rows, starting_equity=starting_equity) for rows in grouped.values()]
    return sorted(summaries, key=lambda s: (s.profit_factor, -s.max_drawdown_pct, s.expectancy), reverse=True)
