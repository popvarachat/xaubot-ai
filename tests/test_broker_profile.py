from types import SimpleNamespace

import pytest

from src.broker_profile import BrokerSymbolProfile


def make_goldmicro_profile() -> BrokerSymbolProfile:
    # Tick value/contract size are illustrative test fixtures only.
    # Runtime values must come from MT5 symbol_info() for the connected broker.
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


def test_goldmicro_volume_flooring_is_risk_conservative():
    p = make_goldmicro_profile()
    assert p.normalize_volume_down(0.09) == 0.0
    assert p.normalize_volume_down(0.10) == 0.1
    assert p.normalize_volume_down(0.19) == 0.1
    assert p.normalize_volume_down(0.20) == 0.2
    assert p.normalize_volume_down(0.29) == 0.2


def test_spread_points_uses_broker_point_size():
    p = make_goldmicro_profile()
    assert p.spread_points(4300.86, 4301.41) == pytest.approx(55.0)


def test_risk_sizing_rejects_when_minimum_lot_exceeds_budget():
    p = make_goldmicro_profile()
    # $10 price stop => 1000 ticks; fixture tick value $0.10 => $100 risk/lot.
    # $5 risk budget supports 0.05 lot, below GOLDmicro minimum 0.1 => skip.
    assert p.raw_lot_for_risk(5.0, 4300.0, 4290.0) == pytest.approx(0.05)
    assert p.lot_for_risk(5.0, 4300.0, 4290.0) == 0.0


def test_profile_can_be_created_from_mt5_symbol_info_shape():
    info = SimpleNamespace(
        point=0.01,
        trade_tick_size=0.01,
        trade_tick_value=0.09,
        trade_tick_value_loss=0.10,
        trade_contract_size=10.0,
        volume_min=0.1,
        volume_max=50.0,
        volume_step=0.1,
    )
    p = BrokerSymbolProfile.from_mt5_symbol_info("GOLDmicro", info)
    assert p.tick_value == 0.10
    assert p.volume_min == 0.1
    assert p.volume_step == 0.1
