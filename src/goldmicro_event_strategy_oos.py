"""Research-only PF/DD/cost evaluator for GOLDmicro event-success models.

The event model predicts calibrated P(SMC setup reaches its own TP before its own
SL within 32 raw M15 bars). It MUST NOT be interpreted as BUY/SELL direction.
SMC owns direction, stop and target. The calibrated event probability is compared
with that setup's broker-cost break-even probability; no return-driven probability
threshold is tuned. No orders are sent and no live model is modified.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import xgboost as xgb

from backtests.goldmicro_cost_model import GoldmicroCostModel
from src.goldmicro_event_calibration import apply_platt_calibrator, read_calibration
from src.goldmicro_strategy_oos import (
    SampleStrategyResult,
    StrategyOOSThresholds,
    _session,
    cost_config_for_profile,
    default_goldmicro_profile,
    size_with_execution_cost,
)


EVENT_MODEL_SEMANTICS = "p_smc_setup_tp_before_sl_within_32_bars"
ECONOMIC_GATE_POLICY = "calibrated_probability_gte_cost_break_even"


def _simulate_smc_exit(
    *,
    market: pl.DataFrame,
    event_market_index: int,
    direction: str,
    entry_mid: float,
    stop_mid: float,
    target_mid: float,
    lot: float,
    cost_model: GoldmicroCostModel,
    max_holding_bars: int,
) -> tuple[int, float, float, str]:
    """Simulate the event's own TP/SL contract with adverse same-bar ordering."""
    highs = market["high"].to_list()
    lows = market["low"].to_list()
    closes = market["close"].to_list()
    end = min(event_market_index + max_holding_bars, len(market) - 1)

    for idx in range(event_market_index + 1, end + 1):
        high = float(highs[idx])
        low = float(lows[idx])
        if direction == "BUY":
            stop_hit = low <= stop_mid
            target_hit = high >= target_mid
        elif direction == "SELL":
            stop_hit = high >= stop_mid
            target_hit = low <= target_mid
        else:
            raise ValueError(f"unknown event direction: {direction}")

        if stop_hit:
            reason = "AMBIGUOUS_BAR_STOP_FIRST" if target_hit else "STOP_LOSS"
            pnl = cost_model.pnl_from_mid(
                side=direction,
                entry_mid=entry_mid,
                exit_mid=stop_mid,
                lot_size=lot,
            ).net_pnl
            return idx, stop_mid, float(pnl), reason
        if target_hit:
            pnl = cost_model.pnl_from_mid(
                side=direction,
                entry_mid=entry_mid,
                exit_mid=target_mid,
                lot_size=lot,
            ).net_pnl
            return idx, target_mid, float(pnl), "TAKE_PROFIT"

    exit_mid = float(closes[end])
    pnl = cost_model.pnl_from_mid(
        side=direction,
        entry_mid=entry_mid,
        exit_mid=exit_mid,
        lot_size=lot,
    ).net_pnl
    return end, exit_mid, float(pnl), "TIMEOUT"


def _event_matrix(events: pl.DataFrame, booster: xgb.Booster) -> xgb.DMatrix:
    feature_names = list(booster.feature_names or [])
    if not feature_names:
        raise ValueError("event booster does not contain feature names")
    missing = [name for name in feature_names if name not in events.columns]
    if missing:
        raise ValueError(f"event dataset missing model features: {missing[:5]}")
    X = events.select(feature_names).to_numpy()
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return xgb.DMatrix(X, feature_names=feature_names)


def _event_probabilities(events: pl.DataFrame, booster: xgb.Booster) -> np.ndarray:
    """Raw XGBoost logistic scores retained for diagnostics/backward tests only."""
    return booster.predict(_event_matrix(events, booster))


def _calibrated_event_probabilities(
    events: pl.DataFrame,
    booster: xgb.Booster,
    calibration_path: Path,
) -> np.ndarray:
    if not calibration_path.exists():
        raise FileNotFoundError(f"event calibration artifact missing: {calibration_path}")
    calibration = read_calibration(calibration_path)
    margins = booster.predict(_event_matrix(events, booster), output_margin=True)
    return apply_platt_calibrator(margins, calibration)


def _break_even_probability(
    *,
    direction: str,
    entry_mid: float,
    stop_mid: float,
    target_mid: float,
    cost_model: GoldmicroCostModel,
) -> float:
    """Return the success probability needed for zero expected value after cost.

    Uses one-lot broker-correct PnL to the setup's own TP/SL. If either side of
    the payoff contract is invalid after cost, fail closed with a break-even
    probability of 1.0.
    """
    win = float(
        cost_model.pnl_from_mid(
            side=direction,
            entry_mid=entry_mid,
            exit_mid=target_mid,
            lot_size=1.0,
        ).net_pnl
    )
    loss = float(
        cost_model.pnl_from_mid(
            side=direction,
            entry_mid=entry_mid,
            exit_mid=stop_mid,
            lot_size=1.0,
        ).net_pnl
    )
    loss_abs = abs(min(loss, 0.0))
    if win <= 0.0 or loss_abs <= 0.0:
        return 1.0
    return loss_abs / (win + loss_abs)


def evaluate_event_strategy_sample(
    sample: dict[str, Any],
    *,
    market_snapshot_path: Path,
    cost_profile: str,
    thresholds: StrategyOOSThresholds = StrategyOOSThresholds(),
) -> SampleStrategyResult:
    """Evaluate untouched OOS with the frozen economic probability gate."""
    model_id = str(sample.get("model_id") or "unknown")
    if "-event32-" not in model_id:
        raise ValueError(f"not an event-target model id: {model_id}")

    event_path = Path(str(sample.get("training_data_path") or ""))
    model_path = Path(str(sample.get("xgb_path") or ""))
    calibration_path = Path(str(sample.get("calibration_path") or ""))
    if not event_path.exists() or not model_path.exists():
        raise FileNotFoundError(f"event candidate artifact missing for {model_id}")
    if not calibration_path.exists():
        raise FileNotFoundError(
            f"calibrated OOS evaluation requires calibration artifact for {model_id}"
        )
    if not market_snapshot_path.exists():
        raise FileNotFoundError(f"market snapshot missing: {market_snapshot_path}")

    events = pl.read_parquet(event_path)
    market = pl.read_parquet(market_snapshot_path)
    split = sample.get("split") or {}
    test_first_event_index = int(split.get("test_first_event_index") or -1)
    if test_first_event_index < 0:
        raise ValueError(f"event split metadata missing for {model_id}")
    if "event_index" not in events.columns:
        raise ValueError(f"event dataset missing event_index for {model_id}")

    oos = events.filter(pl.col("event_index") >= test_first_event_index).sort("time")
    if oos.is_empty():
        raise ValueError(f"no held-out event rows for {model_id}")

    booster = xgb.Booster()
    booster.load_model(model_path)
    probabilities = _calibrated_event_probabilities(oos, booster, calibration_path)

    market_times = market["time"].to_list()
    market_index = {value: idx for idx, value in enumerate(market_times)}
    event_times = oos["time"].to_list()

    profile = default_goldmicro_profile()
    cost_cfg = cost_config_for_profile(cost_profile)
    cost_model = GoldmicroCostModel(profile, cost_cfg)

    balance = thresholds.initial_capital_thb
    peak = balance
    max_dd = 0.0
    profits: list[float] = []
    lots: list[float] = []
    risk_skips = 0
    model_blocks = 0
    accepted_candidates = 0
    last_exit_market_idx = -10**9

    for row_idx, probability in enumerate(probabilities):
        event_time = event_times[row_idx]
        if hasattr(event_time, "weekday") and event_time.weekday() >= 5:
            continue
        _, can_trade = _session(event_time)
        if not can_trade:
            continue

        idx = market_index.get(event_time)
        if idx is None:
            raise ValueError(f"event time not found in master snapshot: {event_time}")
        if idx - last_exit_market_idx < thresholds.cooldown_bars:
            continue

        row = oos.row(row_idx, named=True)
        direction = str(row["event_direction"])
        entry_mid = float(row["event_entry"])
        stop_mid = float(row["event_stop_loss"])
        target_mid = float(row["event_take_profit"])
        break_even_probability = _break_even_probability(
            direction=direction,
            entry_mid=entry_mid,
            stop_mid=stop_mid,
            target_mid=target_mid,
            cost_model=cost_model,
        )
        if float(probability) < break_even_probability:
            model_blocks += 1
            continue

        accepted_candidates += 1
        regime_name = str(row.get("regime_name") or "medium_volatility")
        regime_multiplier = 0.5 if regime_name == "high_volatility" else 1.0
        lot, _, _ = size_with_execution_cost(
            profile=profile,
            cost=cost_cfg,
            balance_thb=balance,
            entry_mid=entry_mid,
            stop_mid=stop_mid,
            risk_percent=thresholds.risk_per_trade_percent,
            regime_multiplier=regime_multiplier,
        )
        if lot <= 0:
            risk_skips += 1
            continue

        exit_idx, _, pnl, _ = _simulate_smc_exit(
            market=market,
            event_market_index=idx,
            direction=direction,
            entry_mid=entry_mid,
            stop_mid=stop_mid,
            target_mid=target_mid,
            lot=lot,
            cost_model=cost_model,
            max_holding_bars=thresholds.max_holding_bars,
        )
        profits.append(float(pnl))
        lots.append(lot)
        balance += float(pnl)
        peak = max(peak, balance)
        dd = (peak - balance) / peak * 100.0 if peak > 0 else 100.0
        max_dd = max(max_dd, dd)
        last_exit_market_idx = exit_idx

    wins = sum(1 for value in profits if value > 0)
    losses = sum(1 for value in profits if value < 0)
    gross_win = sum(value for value in profits if value > 0)
    gross_loss = abs(sum(value for value in profits if value < 0))
    pf = gross_win / gross_loss if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)
    expectancy = sum(profits) / len(profits) if profits else 0.0
    risk_skip_pct = risk_skips / accepted_candidates * 100.0 if accepted_candidates else 0.0

    reasons: list[str] = []
    if accepted_candidates == 0:
        reasons.append("no event candidates cleared calibrated cost break-even probability gate")
    if len(profits) < thresholds.min_trades:
        reasons.append(f"trades {len(profits)} < {thresholds.min_trades}")
    if pf < thresholds.min_profit_factor:
        reasons.append(f"PF {pf:.3f} < {thresholds.min_profit_factor:.2f}")
    if max_dd > thresholds.max_drawdown_percent:
        reasons.append(f"DD {max_dd:.2f}% > {thresholds.max_drawdown_percent:.2f}%")
    if risk_skip_pct > thresholds.max_risk_skip_percent:
        reasons.append(
            f"risk skip {risk_skip_pct:.2f}% > {thresholds.max_risk_skip_percent:.2f}%"
        )
    if expectancy <= 0:
        reasons.append(f"expectancy {expectancy:.2f} THB <= 0")

    return SampleStrategyResult(
        model_id=f"{model_id}::cost={cost_profile}::gate=break-even",
        sample_index=int(sample.get("sample_index") or 0),
        trades=len(profits),
        wins=wins,
        losses=losses,
        net_pnl_thb=float(sum(profits)),
        profit_factor=float(pf),
        max_drawdown_percent=float(max_dd),
        expectancy_thb=float(expectancy),
        risk_skips=risk_skips,
        risk_skip_percent=float(risk_skip_pct),
        model_blocks=model_blocks,
        average_lot=(sum(lots) / len(lots)) if lots else 0.0,
        status="STRATEGY_SAMPLE_PASS" if not reasons else "STRATEGY_SAMPLE_REJECT",
        reasons=tuple(reasons) if reasons else (
            "calibrated event-success PF/DD/cost sample gate passed with cost break-even entry policy",
        ),
    )
