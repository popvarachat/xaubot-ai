"""Causality-safe V2 feature wrapper for GOLDmicro challenger research.

The upstream/general V2 feature engineer is kept intact for compatibility.  GOLDmicro
research uses this subclass to enforce two time-series invariants before any model is
allowed into the expensive PF/DD/cost validation stage:

1. H1 features become visible to an M15 row only after the H1 bar has closed.  MT5
   rate timestamps represent the bar timestamp/open boundary, so the H1 availability
   timestamp is shifted forward by one hour before the backward as-of join.
2. Run-length features are prefix-causal.  Their value at row t must be identical
   whether features are calculated on data ending at t or on a longer future suffix.

This module is research-only.  It does not place orders, promote models, or touch
active/live model paths.
"""
from __future__ import annotations

import polars as pl

from backtests.ml_v2.ml_v2_feature_eng import MLV2FeatureEngineer


class GoldmicroCausalV2FeatureEngineer(MLV2FeatureEngineer):
    """ML V2 features with explicit closed-bar and prefix-causality guarantees."""

    H1_AVAILABILITY_HOURS = 1

    def add_h1_features(self, df_m15: pl.DataFrame, df_h1: pl.DataFrame) -> pl.DataFrame:
        """Expose an H1 row only from its close boundary onward.

        MT5 historical rates label bars by their opening timestamp.  A historical H1
        row stamped 10:00 contains the final high/low/close accumulated through the
        10:00-10:59 hour.  Joining that row directly to a 10:15 M15 bar would reveal
        information that was not yet available at 10:15.  Shifting the H1 timestamp
        to 11:00 converts it to an information-availability timestamp.
        """
        if df_h1 is None or len(df_h1) == 0 or "time" not in df_h1.columns:
            return super().add_h1_features(df_m15, df_h1)

        closed_h1 = df_h1.with_columns(
            (pl.col("time") + pl.duration(hours=self.H1_AVAILABILITY_HOURS)).alias("time")
        )
        return super().add_h1_features(df_m15, closed_h1)

    def add_regime_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Keep regime run length causal instead of using final group size."""
        out = super().add_regime_features(df)
        if "regime" not in df.columns:
            return out

        out = out.with_row_count("_causal_row")
        out = out.with_columns(
            (pl.col("regime") != pl.col("regime").shift(1))
            .fill_null(True)
            .alias("_causal_regime_change")
        )
        out = out.with_columns(
            pl.col("_causal_regime_change").cum_sum().alias("_causal_regime_group")
        )
        out = out.with_columns(
            pl.col("_causal_row")
            .cum_count()
            .over("_causal_regime_group")
            .alias("regime_duration_bars")
        )
        out = out.with_columns(
            (1.0 / pl.col("regime_duration_bars")).alias("regime_transition_prob")
        )
        return out.drop(["_causal_row", "_causal_regime_change", "_causal_regime_group"])

    def add_price_action_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Keep candle-direction run length causal instead of using final group size."""
        out = super().add_price_action_features(df)
        if "close" not in out.columns or "open" not in out.columns:
            return out

        out = out.with_columns(
            pl.when(pl.col("close") > pl.col("open"))
            .then(1)
            .when(pl.col("close") < pl.col("open"))
            .then(-1)
            .otherwise(0)
            .alias("_causal_direction")
        )
        out = out.with_columns(
            (pl.col("_causal_direction") != pl.col("_causal_direction").shift(1))
            .fill_null(True)
            .alias("_causal_dir_change")
        )
        out = out.with_columns(
            pl.col("_causal_dir_change").cum_sum().alias("_causal_dir_group")
        )
        out = out.with_row_count("_causal_row")
        out = out.with_columns(
            pl.col("_causal_row")
            .cum_count()
            .over("_causal_dir_group")
            .alias("consecutive_direction")
        )
        return out.drop(
            ["_causal_direction", "_causal_dir_change", "_causal_dir_group", "_causal_row"]
        )
