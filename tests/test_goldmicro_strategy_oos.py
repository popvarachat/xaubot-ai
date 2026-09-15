from __future__ import annotations

from backtests.goldmicro_cost_model import BacktestCostConfig
from src.goldmicro_strategy_oos import (
    SampleStrategyResult,
    StrategyOOSThresholds,
    cost_config_for_profile,
    default_goldmicro_profile,
    infer_cost_profile,
    size_with_execution_cost,
    summarize_configuration,
)


def _sample(idx: int, *, pf: float, dd: float, status: str = "STRATEGY_SAMPLE_PASS") -> SampleStrategyResult:
    return SampleStrategyResult(
        model_id=f"model-s{idx:02d}",
        sample_index=idx,
        trades=60,
        wins=36,
        losses=24,
        net_pnl_thb=500.0,
        profit_factor=pf,
        max_drawdown_percent=dd,
        expectancy_thb=8.0,
        risk_skips=2,
        risk_skip_percent=5.0,
        model_blocks=1,
        average_lot=0.1,
        status=status,
        reasons=("ok",),
    )


def test_cost_profiles_match_research_scenarios() -> None:
    normal = cost_config_for_profile("normal")
    conservative = cost_config_for_profile("conservative")
    assert normal.spread_points == 50.0
    assert normal.slippage_points == 1.0
    assert conservative.spread_points == 55.0
    assert conservative.slippage_points == 12.0


def test_cost_adjusted_lot_never_exceeds_risk_budget() -> None:
    profile = default_goldmicro_profile()
    cost = BacktestCostConfig(spread_points=55.0, slippage_points=12.0)
    lot, budget, modeled = size_with_execution_cost(
        profile=profile,
        cost=cost,
        balance_thb=20000.0,
        entry_mid=3600.0,
        stop_mid=3585.0,
        risk_percent=1.0,
    )
    assert lot >= 0.10
    assert round(lot * 10) == lot * 10
    assert modeled <= budget + 1e-9


def test_minimum_lot_skips_when_cost_adjusted_risk_is_too_large() -> None:
    profile = default_goldmicro_profile()
    cost = BacktestCostConfig(spread_points=55.0, slippage_points=12.0)
    lot, budget, min_loss = size_with_execution_cost(
        profile=profile,
        cost=cost,
        balance_thb=1000.0,
        entry_mid=3600.0,
        stop_mid=3500.0,
        risk_percent=1.0,
    )
    assert lot == 0.0
    assert min_loss > budget


def test_configuration_requires_four_of_five_and_no_hard_dd_breach() -> None:
    thresholds = StrategyOOSThresholds()
    results = [_sample(i, pf=1.5, dd=5.0) for i in range(1, 6)]
    summary = summarize_configuration("base", results, thresholds=thresholds)
    assert summary["status"] == "STRATEGY_OOS_PASS"

    one_bad_dd = list(results)
    one_bad_dd[-1] = _sample(5, pf=1.5, dd=11.0, status="STRATEGY_SAMPLE_REJECT")
    summary = summarize_configuration("base", one_bad_dd, thresholds=thresholds)
    assert summary["status"] == "STRATEGY_OOS_REJECT"
    assert any("hard DD breach" in r for r in summary["reasons"])


def test_cost_profile_inference_from_sample_model_id() -> None:
    assert infer_cost_profile("gold-x-core-normal-s01") == "normal"
    assert infer_cost_profile("gold-x-core-conservative-s05") == "conservative"
