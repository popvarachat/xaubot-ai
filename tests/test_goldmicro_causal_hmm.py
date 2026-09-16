import numpy as np
import polars as pl

from src.goldmicro_causal_hmm import causal_confirm_states, predict_causal_regimes
from src.goldmicro_causal_v2 import GoldmicroCausalV2FeatureEngineer
from src.regime_detector import MarketRegime


class _FakeModel:
    startprob_ = np.array([0.99, 0.01], dtype=float)
    transmat_ = np.array([[0.92, 0.08], [0.08, 0.92]], dtype=float)

    def _compute_log_likelihood(self, features):
        # Negative observations favor state 0; positive observations favor state 1.
        x = np.asarray(features, dtype=float)[:, 0]
        return np.column_stack(
            [
                np.where(x <= 0, 0.0, -7.0),
                np.where(x > 0, 0.0, -7.0),
            ]
        )


class _FakeDetector:
    fitted = True
    model = _FakeModel()
    scaler = None
    n_regimes = 2
    smoothing_enabled = True
    smoothing_min_duration = 2
    regime_mapping = {
        0: MarketRegime.LOW_VOLATILITY,
        1: MarketRegime.HIGH_VOLATILITY,
    }

    def prepare_features(self, df):
        return df.select("x").to_numpy()


def test_causal_confirmation_never_rewrites_earlier_states():
    raw = np.array([0, 0, 1, 1, 1, 0, 0], dtype=int)
    confirmed = causal_confirm_states(raw, min_duration=2)
    assert confirmed.tolist() == [0, 0, 0, 1, 1, 1, 0]


def test_hmm_forward_filter_is_prefix_invariant():
    df = pl.DataFrame({"x": [-1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0]})
    detector = _FakeDetector()
    full = predict_causal_regimes(detector, df)

    # Appending future observations must not change any historical regime output.
    for end in range(1, len(df) + 1):
        prefix = predict_causal_regimes(detector, df.head(end))
        assert prefix["regime"].to_list() == full["regime"].head(end).to_list()
        assert prefix["regime_name"].to_list() == full["regime_name"].head(end).to_list()
        np.testing.assert_allclose(
            prefix["regime_confidence"].to_numpy(),
            full["regime_confidence"].head(end).to_numpy(),
            rtol=1e-12,
            atol=1e-12,
        )


def test_v2_regime_run_length_uses_causal_hmm_states_not_constant_defaults():
    base = pl.DataFrame({"x": [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0]})
    with_regime = predict_causal_regimes(_FakeDetector(), base)
    with_regime = with_regime.with_columns(pl.lit(1.0).alias("atr"))

    out = GoldmicroCausalV2FeatureEngineer().add_regime_features(with_regime)
    durations = out["regime_duration_bars"].to_list()

    assert max(durations) > 1
    for end in range(1, len(with_regime) + 1):
        prefix = GoldmicroCausalV2FeatureEngineer().add_regime_features(with_regime.head(end))
        assert prefix["regime_duration_bars"][-1] == durations[end - 1]
