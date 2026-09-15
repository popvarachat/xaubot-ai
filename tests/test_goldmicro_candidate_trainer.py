import polars as pl

from src.goldmicro_candidate_trainer import (
    feature_columns_for_profile,
    impute_v2_feature_nulls,
    xgb_params_for_profile,
)


def test_xgb_profiles_have_distinct_complexity_and_seed():
    conservative = xgb_params_for_profile("conservative", 11)
    balanced = xgb_params_for_profile("balanced", 29)
    responsive = xgb_params_for_profile("responsive", 47)

    assert conservative["max_depth"] < balanced["max_depth"] < responsive["max_depth"]
    assert conservative["reg_lambda"] > balanced["reg_lambda"] > responsive["reg_lambda"]
    assert conservative["seed"] == 11
    assert balanced["seed"] == 29
    assert responsive["seed"] == 47


def test_unknown_xgb_profile_is_rejected():
    try:
        xgb_params_for_profile("magic", 1)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_core_plus_v2_uses_explicit_engineered_features_not_sparse_raw_smc_columns():
    df = pl.DataFrame({
        "rsi": [50.0, 51.0],
        "atr": [1.0, 1.1],
        "fvg_top": [None, 4300.0],
        "ob_top": [None, None],
        "h1_swing_proximity": [None, 2.0],
        "h1_ob_proximity": [None, 3.0],
        "fvg_age_bars": [None, 4],
        "wick_ratio": [0.2, 0.3],
        "target": [0, 1],
    })

    features = feature_columns_for_profile(df, "core_plus_v2")

    assert "rsi" in features
    assert "atr" in features
    assert "h1_swing_proximity" in features
    assert "h1_ob_proximity" in features
    assert "fvg_age_bars" in features
    assert "wick_ratio" in features
    assert "fvg_top" not in features
    assert "ob_top" not in features


def test_v2_imputation_preserves_semantic_sentinels_without_touching_raw_sparse_columns():
    df = pl.DataFrame({
        "h1_swing_proximity": [None, 2.0],
        "h1_ob_proximity": [None, 3.0],
        "fvg_age_bars": [None, 4],
        "h1_rsi": [None, 55.0],
        "fvg_top": [None, 4300.0],
    })
    features = [
        "h1_swing_proximity",
        "h1_ob_proximity",
        "fvg_age_bars",
        "h1_rsi",
    ]

    out = impute_v2_feature_nulls(df, features)

    assert out["h1_swing_proximity"][0] == 999.0
    assert out["h1_ob_proximity"][0] == 999.0
    assert out["fvg_age_bars"][0] == 999
    assert out["h1_rsi"][0] == 50.0
    assert out["fvg_top"][0] is None
