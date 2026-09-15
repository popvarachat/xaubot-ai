from src.goldmicro_candidate_trainer import xgb_params_for_profile


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
