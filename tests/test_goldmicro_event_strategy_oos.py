from datetime import datetime, timedelta
import json

import numpy as np
import polars as pl
import xgboost as xgb

from backtests.goldmicro_cost_model import GoldmicroCostModel
from src.goldmicro_event_calibration import CALIBRATION_METHOD
from src.goldmicro_event_strategy_oos import (
    ECONOMIC_GATE_POLICY,
    EVENT_MODEL_SEMANTICS,
    _break_even_probability,
    _simulate_smc_exit,
    evaluate_event_strategy_sample,
)
from src.goldmicro_strategy_oos import (
    StrategyOOSThresholds,
    cost_config_for_profile,
    default_goldmicro_profile,
)


def test_event_contract_is_success_probability_not_direction() -> None:
    assert EVENT_MODEL_SEMANTICS == "p_smc_setup_tp_before_sl_within_32_bars"
    assert ECONOMIC_GATE_POLICY == "calibrated_probability_gte_cost_break_even"


def test_event_exit_uses_adverse_same_bar_ordering() -> None:
    market = pl.DataFrame({
        "time": [datetime(2026, 1, 5, 10, 0) + timedelta(minutes=15 * i) for i in range(3)],
        "open": [100.0, 100.0, 100.0],
        "high": [100.0, 103.0, 100.0],
        "low": [100.0, 97.0, 100.0],
        "close": [100.0, 100.0, 100.0],
    })
    cost_model = GoldmicroCostModel(
        default_goldmicro_profile(),
        cost_config_for_profile("normal"),
    )
    _, exit_mid, _, reason = _simulate_smc_exit(
        market=market,
        event_market_index=0,
        direction="BUY",
        entry_mid=100.0,
        stop_mid=98.0,
        target_mid=102.0,
        lot=0.10,
        cost_model=cost_model,
        max_holding_bars=2,
    )
    assert exit_mid == 98.0
    assert reason == "AMBIGUOUS_BAR_STOP_FIRST"


def test_break_even_probability_uses_broker_cost_and_payoff() -> None:
    profile = default_goldmicro_profile()
    normal = GoldmicroCostModel(profile, cost_config_for_profile("normal"))
    conservative = GoldmicroCostModel(profile, cost_config_for_profile("conservative"))
    p_normal = _break_even_probability(
        direction="BUY",
        entry_mid=100.0,
        stop_mid=98.0,
        target_mid=103.0,
        cost_model=normal,
    )
    p_conservative = _break_even_probability(
        direction="BUY",
        entry_mid=100.0,
        stop_mid=98.0,
        target_mid=103.0,
        cost_model=conservative,
    )
    assert 0.0 < p_normal < 1.0
    assert 0.0 < p_conservative < 1.0
    assert p_conservative >= p_normal


def _write_fixture(tmp_path, *, calibration_intercept: float):
    start = datetime(2026, 1, 5, 10, 0)
    times = [start + timedelta(minutes=15 * i) for i in range(40)]
    market = pl.DataFrame({
        "time": times,
        "open": [100.0] * 40,
        "high": [100.0] * 6 + [102.0] + [100.0] * 33,
        "low": [100.0] * 40,
        "close": [100.0] * 40,
    })
    snapshot = tmp_path / "market_snapshot_m15_master.parquet"
    market.write_parquet(snapshot)

    events = pl.DataFrame({
        "time": [times[5]],
        "event_index": [5],
        "event_target": [1],
        "event_direction": ["BUY"],
        "event_entry": [100.0],
        "event_stop_loss": [98.0],
        "event_take_profit": [101.5],
        "event_smc_confidence": [0.75],
        "regime_name": ["medium_volatility"],
        "f1": [1.0],
    })
    event_path = tmp_path / "event_data.parquet"
    events.write_parquet(event_path)

    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    y = np.array([0, 1, 1, 1], dtype=float)
    booster = xgb.train(
        {"objective": "binary:logistic", "max_depth": 1, "eta": 1.0, "seed": 7},
        xgb.DMatrix(X, label=y, feature_names=["f1"]),
        num_boost_round=3,
    )
    model_path = tmp_path / "event_xgb.json"
    booster.save_model(model_path)

    calibration_path = tmp_path / "event_calibration.json"
    calibration_path.write_text(
        json.dumps({
            "method": CALIBRATION_METHOD,
            "coef": 0.0,
            "intercept": calibration_intercept,
            "sample_count": 100,
            "positive_rate": 0.25,
        }),
        encoding="utf-8",
    )

    sample = {
        "model_id": "gold-event-test-core-normal-event32-s01",
        "sample_index": 1,
        "xgb_path": str(model_path),
        "calibration_path": str(calibration_path),
        "training_data_path": str(event_path),
        "split": {"test_first_event_index": 5},
    }
    return snapshot, sample


def _permissive_thresholds() -> StrategyOOSThresholds:
    return StrategyOOSThresholds(
        initial_capital_thb=20000.0,
        risk_per_trade_percent=1.0,
        min_profit_factor=0.0,
        max_drawdown_percent=100.0,
        max_risk_skip_percent=100.0,
        min_trades=1,
        cooldown_bars=0,
    )


def test_event_strategy_accepts_probability_above_cost_break_even(tmp_path) -> None:
    snapshot, sample = _write_fixture(tmp_path, calibration_intercept=10.0)
    result = evaluate_event_strategy_sample(
        sample,
        market_snapshot_path=snapshot,
        cost_profile="normal",
        thresholds=_permissive_thresholds(),
    )
    assert result.trades == 1
    assert result.model_id.endswith("::cost=normal::gate=break-even")
    assert result.status == "STRATEGY_SAMPLE_PASS"


def test_event_strategy_blocks_probability_below_cost_break_even(tmp_path) -> None:
    snapshot, sample = _write_fixture(tmp_path, calibration_intercept=-10.0)
    result = evaluate_event_strategy_sample(
        sample,
        market_snapshot_path=snapshot,
        cost_profile="normal",
        thresholds=StrategyOOSThresholds(
            initial_capital_thb=20000.0,
            risk_per_trade_percent=1.0,
            min_profit_factor=1.30,
            max_drawdown_percent=10.0,
            max_risk_skip_percent=20.0,
            min_trades=1,
            cooldown_bars=0,
        ),
    )
    assert result.trades == 0
    assert result.model_blocks == 1
    assert result.risk_skips == 0
    assert result.risk_skip_percent == 0.0
    assert any("cost break-even probability gate" in reason for reason in result.reasons)


def test_missing_calibration_artifact_is_rejected(tmp_path) -> None:
    snapshot, sample = _write_fixture(tmp_path, calibration_intercept=10.0)
    sample["calibration_path"] = str(tmp_path / "missing.json")
    try:
        evaluate_event_strategy_sample(
            sample,
            market_snapshot_path=snapshot,
            cost_profile="normal",
        )
    except FileNotFoundError as exc:
        assert "requires calibration artifact" in str(exc)
    else:
        raise AssertionError("missing calibration artifact must fail closed")
