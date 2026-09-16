from dataclasses import dataclass

from backtests.goldmicro_cost_model import BacktestCostConfig, GoldmicroCostModel
from backtests.goldmicro_replay import replay_legacy_trades
from src.broker_profile import BrokerSymbolProfile


PROFILE = BrokerSymbolProfile(
    symbol="GOLDmicro",
    point=0.01,
    tick_size=0.01,
    tick_value=0.01,
    contract_size=1.0,
    volume_min=0.1,
    volume_max=100.0,
    volume_step=0.01,
    cash_per_price_unit_per_lot=33.28,
    cash_currency="THB",
)


@dataclass
class DummyTrade:
    direction: str
    entry_price: float
    exit_price: float
    stop_loss: float


def test_replay_charges_55_point_spread_and_uses_broker_lot_step():
    model = GoldmicroCostModel(PROFILE, BacktestCostConfig(spread_points=55.0))
    trades = [DummyTrade("BUY", 4300.0, 4302.0, 4295.0)]

    stats = replay_legacy_trades(
        trades,
        profile=PROFILE,
        cost_model=model,
        initial_capital=5000.0,
        risk_per_trade_percent=1.0,
        win_rate=0.60,
        reward_risk_ratio=2.0,
    )

    assert stats.executed_trades == 1
    assert stats.skipped_trades == 0
    assert stats.trades[0].lot_size >= 0.1
    assert round(stats.trades[0].lot_size * 100) == stats.trades[0].lot_size * 100
    assert stats.trades[0].entry_exec_price > trades[0].entry_price
    assert stats.trades[0].exit_exec_price < trades[0].exit_price


def test_replay_skips_trade_when_minimum_lot_exceeds_risk_budget():
    model = GoldmicroCostModel(PROFILE, BacktestCostConfig(spread_points=55.0))
    trades = [DummyTrade("BUY", 4300.0, 4301.0, 4200.0)]

    stats = replay_legacy_trades(
        trades,
        profile=PROFILE,
        cost_model=model,
        initial_capital=100.0,
        risk_per_trade_percent=0.5,
        win_rate=0.60,
        reward_risk_ratio=2.0,
    )

    assert stats.total_trades == 1
    assert stats.executed_trades == 0
    assert stats.skipped_trades == 1
    assert stats.final_capital == 100.0
    assert "minimum" in stats.trades[0].skip_reason.lower()


def test_replay_rebuilds_pf_and_drawdown_from_net_costed_pnl():
    model = GoldmicroCostModel(PROFILE, BacktestCostConfig(spread_points=55.0))
    trades = [
        DummyTrade("BUY", 4300.0, 4303.0, 4295.0),
        DummyTrade("SELL", 4300.0, 4302.0, 4305.0),
        DummyTrade("SELL", 4300.0, 4296.0, 4305.0),
    ]

    stats = replay_legacy_trades(
        trades,
        profile=PROFILE,
        cost_model=model,
        initial_capital=5000.0,
        risk_per_trade_percent=1.0,
        win_rate=0.60,
        reward_risk_ratio=2.0,
    )

    assert stats.executed_trades == 3
    assert stats.wins == 2
    assert stats.losses == 1
    assert stats.profit_factor > 0
    assert stats.max_drawdown_percent > 0
    assert abs(stats.net_profit - (stats.final_capital - stats.initial_capital)) < 1e-9
