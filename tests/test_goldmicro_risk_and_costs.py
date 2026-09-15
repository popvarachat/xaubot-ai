from src.broker_profile import BrokerSymbolProfile
from src.goldmicro_risk import half_kelly_fraction, size_goldmicro_position
from backtests.goldmicro_cost_model import BacktestCostConfig, GoldmicroCostModel


def _profile() -> BrokerSymbolProfile:
    return BrokerSymbolProfile(
        symbol="GOLDmicro",
        point=0.01,
        tick_size=0.01,
        tick_value=0.10,
        contract_size=10.0,
        volume_min=0.1,
        volume_max=100.0,
        volume_step=0.1,
    )


def test_half_kelly_cannot_be_negative():
    assert half_kelly_fraction(0.30, 1.0) == 0.0


def test_sizing_floors_to_point_one_step_without_increasing_risk():
    profile = _profile()
    result = size_goldmicro_position(
        profile=profile,
        account_balance=10_000,
        entry_price=4300.0,
        stop_price=4295.0,
        risk_per_trade_percent=1.0,
        win_rate=0.55,
        reward_risk_ratio=2.0,
    )
    assert result.approved is True
    assert result.lot_size >= 0.1
    assert round(result.lot_size * 10) == result.lot_size * 10
    assert result.actual_risk_amount <= result.requested_risk_amount + 1e-9


def test_sizing_rejects_when_minimum_lot_exceeds_risk_budget():
    profile = _profile()
    result = size_goldmicro_position(
        profile=profile,
        account_balance=100,
        entry_price=4300.0,
        stop_price=4200.0,
        risk_per_trade_percent=0.5,
        win_rate=0.60,
        reward_risk_ratio=2.0,
    )
    assert result.approved is False
    assert result.lot_size == 0.0
    assert "Broker minimum 0.1 lot" in result.reason


def test_mid_price_buy_is_charged_55_point_spread_round_trip():
    profile = _profile()
    model = GoldmicroCostModel(profile, BacktestCostConfig(spread_points=55))
    result = model.pnl_from_mid(
        side="BUY",
        entry_mid=4300.00,
        exit_mid=4300.00,
        lot_size=0.1,
    )
    # Flat mid-price must lose exactly the synthetic spread.
    expected = -(55 * profile.point / profile.tick_size) * profile.tick_value * 0.1
    assert abs(result.net_pnl - expected) < 1e-9


def test_observed_bid_ask_uses_actual_spread_not_synthetic_spread():
    profile = _profile()
    model = GoldmicroCostModel(profile, BacktestCostConfig(spread_points=999))
    result = model.pnl_from_observed_bid_ask(
        side="BUY",
        entry_bid=4300.00,
        entry_ask=4300.55,
        exit_bid=4300.00,
        exit_ask=4300.55,
        lot_size=0.1,
    )
    expected = -(0.55 / profile.tick_size) * profile.tick_value * 0.1
    assert abs(result.net_pnl - expected) < 1e-9


def test_sell_uses_bid_entry_and_ask_exit():
    profile = _profile()
    model = GoldmicroCostModel(profile, BacktestCostConfig(spread_points=55))
    entry, exit_ = model.execution_prices_from_mid("SELL", 4300.0, 4299.0)
    assert entry < 4300.0
    assert exit_ > 4299.0
