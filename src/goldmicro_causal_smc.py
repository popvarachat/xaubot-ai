"""Prefix-causal SMC wrapper for GOLDmicro research.

The generic SMC analyzer detects an order-block origin candle only after a later
confirmation bar is observed, but historically stores the OB marker back on the
origin candle.  That is convenient for chart annotation, yet it is not causal
when the resulting frame is used as machine-learning training data: an earlier
row can contain information that was only knowable several bars later.

This research-only wrapper keeps all existing SMC logic except order-block event
placement.  A confirmed order block is emitted on the confirmation bar, while
its zone prices still refer to the historical origin candle.  Appending future
bars therefore never rewrites prior OB feature rows.

No live trading code imports this module by default.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from src.smc_polars import SMCAnalyzer


class GoldmicroCausalSMCAnalyzer(SMCAnalyzer):
    """SMC analyzer whose order-block columns are prefix invariant."""

    def calculate_order_blocks(self, df: pl.DataFrame) -> pl.DataFrame:
        """Emit an OB event when it becomes knowable, never on an earlier row.

        The source/origin candle is searched exactly as in the legacy analyzer,
        but the event marker and zone values are written at confirmation index
        ``i`` rather than retroactively at origin index ``j``.
        """
        if "swing_high" not in df.columns:
            df = self.calculate_swing_points(df)

        opens = df["open"].to_numpy()
        highs = df["high"].to_numpy()
        lows = df["low"].to_numpy()
        closes = df["close"].to_numpy()
        swing_highs = df["swing_high"].to_numpy()
        swing_lows = df["swing_low"].to_numpy()

        n = len(df)
        ob = np.zeros(n, dtype=np.int8)
        ob_top = np.full(n, np.nan)
        ob_bottom = np.full(n, np.nan)
        ob_origin_index = np.full(n, -1, dtype=np.int64)

        for i in range(self.ob_lookback, n):
            # Bullish OB: a confirmed swing low plus a prior bearish origin
            # candle whose high has been broken by the CURRENT close.
            if swing_lows[i] == -1:
                for j in range(i - 1, max(0, i - self.ob_lookback), -1):
                    if closes[j] < opens[j] and closes[i] > highs[j]:
                        ob[i] = 1
                        ob_top[i] = highs[j]
                        ob_bottom[i] = lows[j]
                        ob_origin_index[i] = j
                        break

            # Bearish OB: symmetrical confirmation on the CURRENT bar.
            if swing_highs[i] == 1:
                for j in range(i - 1, max(0, i - self.ob_lookback), -1):
                    if closes[j] > opens[j] and closes[i] < lows[j]:
                        ob[i] = -1
                        ob_top[i] = highs[j]
                        ob_bottom[i] = lows[j]
                        ob_origin_index[i] = j
                        break

        # ``ob_mitigated`` is intentionally conservative here.  The event row
        # represents first causal availability of the zone; later persistent-zone
        # lifecycle is not an ML feature in the GOLDmicro candidate policy.
        ob_mitigated = np.zeros(n, dtype=bool)

        return df.with_columns(
            pl.Series("ob", ob),
            pl.Series("ob_top", ob_top),
            pl.Series("ob_bottom", ob_bottom),
            pl.Series("ob_mitigated", ob_mitigated),
            pl.Series("ob_origin_index", ob_origin_index),
        )


def causal_smc_feature_columns() -> tuple[str, ...]:
    """Columns expected to remain prefix invariant in GOLDmicro research."""
    return (
        "swing_high",
        "swing_low",
        "swing_high_level",
        "swing_low_level",
        "last_swing_high",
        "last_swing_low",
        "is_fvg_bull",
        "is_fvg_bear",
        "fvg_signal",
        "ob",
        "ob_top",
        "ob_bottom",
        "bos",
        "choch",
        "market_structure",
    )
