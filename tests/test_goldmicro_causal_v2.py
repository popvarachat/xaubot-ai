from datetime import datetime, timedelta

import polars as pl

from src.goldmicro_causal_v2 import GoldmicroCausalV2FeatureEngineer


def test_consecutive_direction_is_prefix_causal():
    df = pl.DataFrame(
        {
            "open": [1.0, 1.0, 1.0, 1.0, 1.0],
            "high": [2.2, 3.2, 4.2, 1.2, 1.2],
            "low": [0.8, 0.8, 0.8, -0.2, -0.2],
            "close": [2.0, 3.0, 4.0, 0.0, 0.0],
            "atr": [1.0] * 5,
        }
    )

    fe = GoldmicroCausalV2FeatureEngineer()
    full = fe.add_price_action_features(df)

    assert full["consecutive_direction"].to_list() == [1, 2, 3, 1, 2]

    # Prefix invariance: appending future bars must not change a feature value that
    # was already knowable at the prefix boundary.
    for end in range(1, len(df) + 1):
        prefix = fe.add_price_action_features(df.head(end))
        assert prefix["consecutive_direction"][-1] == full["consecutive_direction"][end - 1]


def test_regime_duration_is_prefix_causal():
    df = pl.DataFrame(
        {
            "regime": [0, 0, 1, 1, 1],
            "atr": [1.0, 1.1, 1.2, 1.3, 1.4],
        }
    )

    fe = GoldmicroCausalV2FeatureEngineer()
    full = fe.add_regime_features(df)

    assert full["regime_duration_bars"].to_list() == [1, 2, 1, 2, 3]
    assert full["regime_transition_prob"].to_list() == [1.0, 0.5, 1.0, 0.5, 1.0 / 3.0]

    for end in range(1, len(df) + 1):
        prefix = fe.add_regime_features(df.head(end))
        assert prefix["regime_duration_bars"][-1] == full["regime_duration_bars"][end - 1]


def test_h1_features_are_not_visible_before_h1_close():
    base = datetime(2026, 9, 15, 9, 0)
    m15 = pl.DataFrame(
        {
            "time": [
                base + timedelta(hours=1, minutes=15),
                base + timedelta(hours=1, minutes=45),
                base + timedelta(hours=2),
            ],
            "close": [100.0, 101.0, 102.0],
            "atr": [1.0, 1.0, 1.0],
            "rsi": [40.0, 41.0, 42.0],
            "market_structure": [0, 0, 0],
            "bos": [0, 0, 0],
        }
    )
    h1 = pl.DataFrame(
        {
            # Historical MT5 rate timestamps are treated as bar-open boundaries.
            "time": [base, base + timedelta(hours=1)],
            "close": [90.0, 190.0],
            "atr": [2.0, 3.0],
            "rsi": [10.0, 90.0],
            "market_structure": [-1, 1],
            "bos": [-1, 1],
        }
    )

    out = GoldmicroCausalV2FeatureEngineer().add_h1_features(m15, h1)

    # The 10:00 H1 row is not fully known at 10:15 or 10:45.  It only becomes
    # eligible at 11:00, so those earlier M15 rows must still see the 09:00 H1 row.
    assert out["h1_rsi"].to_list() == [10.0, 10.0, 90.0]
