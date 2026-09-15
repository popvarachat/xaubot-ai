from types import SimpleNamespace

import numpy as np

import src.mt5_research_connector as research_mod
from src.mt5_research_connector import MT5ResearchConnector


class FakeMT5:
    TIMEFRAME_M1 = 1
    TIMEFRAME_M5 = 5
    TIMEFRAME_M15 = 15
    TIMEFRAME_M30 = 30
    TIMEFRAME_H1 = 16385
    TIMEFRAME_H4 = 16388
    TIMEFRAME_D1 = 16408
    TIMEFRAME_W1 = 32769

    def __init__(self):
        self.initialize_kwargs = None
        self.shutdown_called = False

    def initialize(self, **kwargs):
        self.initialize_kwargs = dict(kwargs)
        return True

    def terminal_info(self):
        return SimpleNamespace(connected=True)

    def account_info(self):
        return SimpleNamespace(server="TEST-SERVER")

    def shutdown(self):
        self.shutdown_called = True
        return True

    def symbol_select(self, symbol, selected):
        return symbol == "GOLDmicro" and selected is True

    def copy_rates_from_pos(self, symbol, timeframe, start, count):
        dtype = [
            ("time", "i8"),
            ("open", "f8"),
            ("high", "f8"),
            ("low", "f8"),
            ("close", "f8"),
            ("tick_volume", "i8"),
            ("spread", "i8"),
            ("real_volume", "i8"),
        ]
        rows = np.zeros(min(count, 2), dtype=dtype)
        rows["time"] = [1_700_000_000, 1_700_000_900][: len(rows)]
        rows["open"] = 4300.0
        rows["high"] = 4301.0
        rows["low"] = 4299.0
        rows["close"] = 4300.5
        rows["tick_volume"] = 10
        rows["spread"] = 55
        rows["real_volume"] = 0
        return rows

    def last_error(self):
        return (0, "ok")


def test_research_connector_attaches_without_credentials(monkeypatch):
    fake = FakeMT5()
    monkeypatch.setattr(research_mod, "mt5", fake)

    connector = MT5ResearchConnector()
    assert connector.connect() is True
    assert fake.initialize_kwargs == {"timeout": 60000}
    assert "login" not in fake.initialize_kwargs
    assert "password" not in fake.initialize_kwargs
    assert "server" not in fake.initialize_kwargs

    df = connector.get_market_data("GOLDmicro", "M15", 2)
    assert len(df) == 2
    assert "volume" in df.columns
    assert "tick_volume" not in df.columns

    connector.disconnect()
    assert fake.shutdown_called is True


def test_research_connector_has_no_order_send_surface():
    connector = MT5ResearchConnector()
    assert not hasattr(connector, "order_send")
    assert not hasattr(connector, "place_order")
