from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from src.goldmicro_causal_smc import (
    GoldmicroCausalSMCAnalyzer,
    causal_smc_feature_columns,
)


def _market_frame(n: int = 220) -> pl.DataFrame:
    rng = np.random.default_rng(20260916)
    returns = rng.normal(0.0, 0.8, n)
    close = 3600.0 + np.cumsum(returns)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + rng.uniform(0.1, 1.2, n)
    low = np.minimum(open_, close) - rng.uniform(0.1, 1.2, n)
    return pl.DataFrame(
        {
            "time": [datetime(2026, 1, 1) + timedelta(minutes=15 * i) for i in range(n)],
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.integers(100, 1000, n),
        }
    )


def _assert_values_equal(left: pl.Series, right: pl.Series) -> None:
    assert left.len() == right.len()
    a = left.to_list()
    b = right.to_list()
    for x, y in zip(a, b):
        if x is None or y is None:
            assert x is None and y is None
        elif isinstance(x, float) and np.isnan(x):
            assert isinstance(y, float) and np.isnan(y)
        else:
            assert x == y


def test_causal_smc_is_prefix_invariant() -> None:
    raw = _market_frame()
    analyzer = GoldmicroCausalSMCAnalyzer(swing_length=5, ob_lookback=10)
    full = analyzer.calculate_all(raw)

    for cutoff in (60, 100, 150, 200):
        prefix = analyzer.calculate_all(raw.head(cutoff))
        full_prefix = full.head(cutoff)
        for column in causal_smc_feature_columns():
            _assert_values_equal(prefix[column], full_prefix[column])


def test_order_block_origin_never_points_to_future() -> None:
    raw = _market_frame()
    analyzer = GoldmicroCausalSMCAnalyzer(swing_length=5, ob_lookback=10)
    out = analyzer.calculate_all(raw)

    origins = out["ob_origin_index"].to_list()
    events = out["ob"].to_list()
    for idx, (event, origin) in enumerate(zip(events, origins)):
        if event == 0:
            assert origin == -1
        else:
            assert 0 <= origin < idx
