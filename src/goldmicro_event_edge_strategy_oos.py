"""Research-only PF/DD/cost evaluator for GOLDmicro Event Economic-Edge V4.

The V4 model predicts calibrated expected gross R for a causal SMC setup.  SMC
remains the sole source of direction/entry/SL/TP.  Entry is accepted only when
calibrated predicted gross R minus deterministic broker-cost R is > 0.  No
return-driven threshold is tuned and no live model/order path is touched.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import xgboost as xgb

from backtests.goldmicro_cost_model import GoldmicroCostModel
from src.goldmicro_event_edge_calibration import apply_affine_calibrator, read_edge_calibration
from src.goldmicro_strategy_oos import (
    SampleStrategyResult,
    StrategyOOSThresholds,
    _session,
    cost_config_for_profile,
    default_goldmicro_profile,
    size_with_execution_cost,
)

EDGE_MODEL_SEMANTICS = "expected_realized_gross_r_of_causal_smc_setup"
EDGE_ECONOMIC_GATE = "calibrated_predicted_net_r_gt_0"


def _event_matrix(events: pl.DataFrame, booster: xgb.Booster) -> xgb.DMatrix:
    names = list(booster.feature_names or [])
    if not names:
        raise ValueError("event-edge booster does not contain feature names")
    missing = [name for name in names if name not in events.columns]
    if missing:
        raise ValueError(f"event-edge dataset missing model features: {missing[:5]}")
    X = events.select(names).to_numpy()
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return xgb.DMatrix(X, feature_names=names)


def calibrated_edge_predictions(events: pl.DataFrame, booster: xgb.Booster, calibration_path: Path) -> np.ndarray:
    calibration = read_edge_calibration(calibration_path)
    raw = booster.predict(_event_matrix(events, booster))
    return apply_affine_calibrator(raw, calibration)


def setup_cost_r(*, direction: str, entry_mid: float, risk_distance: float,
                 cost_model: GoldmicroCostModel) -> float:
    """Round-trip execution drag expressed in setup-risk units for one lot."""
    if risk_distance <= 0.0:
        return float("inf")
    flat = cost_model.pnl_from_mid(
        side=direction, entry_mid=entry_mid, exit_mid=entry_mid, lot_size=1.0
    ).net_pnl
    risk_cash = abs(cost_model.profile.cash_pnl_for_price_delta(risk_distance, 1.0))
    if risk_cash <= 0.0:
        return float("inf")
    return max(0.0, -float(flat) / risk_cash)


def _simulate_exit(*, market: pl.DataFrame, event_market_index: int, direction: str,
                   entry_mid: float, stop_mid: float, target_mid: float, lot: float,
                   cost_model: GoldmicroCostModel, max_holding_bars: int):
    highs = market["high"].to_list(); lows = market["low"].to_list(); closes = market["close"].to_list()
    end = min(event_market_index + max_holding_bars, len(market) - 1)
    for idx in range(event_market_index + 1, end + 1):
        high = float(highs[idx]); low = float(lows[idx])
        if direction == "BUY":
            stop_hit, target_hit = low <= stop_mid, high >= target_mid
        elif direction == "SELL":
            stop_hit, target_hit = high >= stop_mid, low <= target_mid
        else:
            raise ValueError(f"unknown event direction: {direction}")
        if stop_hit:
            pnl = cost_model.pnl_from_mid(side=direction, entry_mid=entry_mid, exit_mid=stop_mid, lot_size=lot).net_pnl
            return idx, float(pnl), "AMBIGUOUS_BAR_STOP_FIRST" if target_hit else "STOP_LOSS"
        if target_hit:
            pnl = cost_model.pnl_from_mid(side=direction, entry_mid=entry_mid, exit_mid=target_mid, lot_size=lot).net_pnl
            return idx, float(pnl), "TAKE_PROFIT"
    pnl = cost_model.pnl_from_mid(side=direction, entry_mid=entry_mid, exit_mid=float(closes[end]), lot_size=lot).net_pnl
    return end, float(pnl), "TIMEOUT"


def evaluate_event_edge_strategy_sample(
    sample: dict[str, Any], *, market_snapshot_path: Path, cost_profile: str,
    thresholds: StrategyOOSThresholds = StrategyOOSThresholds(),
) -> SampleStrategyResult:
    model_id = str(sample.get("model_id") or "unknown")
    if "-edgev4-" not in model_id:
        raise ValueError(f"not an event-edge V4 model id: {model_id}")
    data_path = Path(str(sample.get("training_data_path") or ""))
    model_path = Path(str(sample.get("xgb_path") or ""))
    calibration_path = Path(str(sample.get("calibration_path") or ""))
    for path, label in ((data_path, "data"), (model_path, "model"), (calibration_path, "calibration"),
                        (market_snapshot_path, "market snapshot")):
        if not path.exists():
            raise FileNotFoundError(f"event-edge {label} missing for {model_id}: {path}")

    events = pl.read_parquet(data_path)
    market = pl.read_parquet(market_snapshot_path)
    split = sample.get("split") or {}
    test_first = int(split.get("test_first_event_index") or -1)
    if test_first < 0:
        raise ValueError(f"event-edge split metadata missing for {model_id}")
    oos = events.filter(pl.col("event_index") >= test_first).sort("time")
    if oos.is_empty():
        raise ValueError(f"no untouched OOS event-edge rows for {model_id}")

    booster = xgb.Booster(); booster.load_model(model_path)
    predicted_gross_r = calibrated_edge_predictions(oos, booster, calibration_path)
    market_index = {value: idx for idx, value in enumerate(market["time"].to_list())}
    event_times = oos["time"].to_list()
    profile = default_goldmicro_profile(); cost_cfg = cost_config_for_profile(cost_profile)
    cost_model = GoldmicroCostModel(profile, cost_cfg)

    balance = thresholds.initial_capital_thb; peak = balance; max_dd = 0.0
    profits: list[float] = []; lots: list[float] = []
    risk_skips = 0; model_blocks = 0; accepted = 0; last_exit_idx = -10**9

    for row_idx, pred_r in enumerate(predicted_gross_r):
        event_time = event_times[row_idx]
        if hasattr(event_time, "weekday") and event_time.weekday() >= 5:
            continue
        _, can_trade = _session(event_time)
        if not can_trade:
            continue
        idx = market_index.get(event_time)
        if idx is None:
            raise ValueError(f"event time not found in master snapshot: {event_time}")
        if idx - last_exit_idx < thresholds.cooldown_bars:
            continue

        row = oos.row(row_idx, named=True)
        direction = str(row["event_direction"]); entry = float(row["event_entry"])
        stop = float(row["event_stop_loss"]); target = float(row["event_take_profit"])
        risk_distance = abs(entry - stop)
        cost_r = setup_cost_r(direction=direction, entry_mid=entry, risk_distance=risk_distance, cost_model=cost_model)
        if float(pred_r) - cost_r <= 0.0:
            model_blocks += 1
            continue

        accepted += 1
        regime_name = str(row.get("regime_name") or "medium_volatility")
        regime_multiplier = 0.5 if regime_name == "high_volatility" else 1.0
        lot, _, _ = size_with_execution_cost(
            profile=profile, cost=cost_cfg, balance_thb=balance, entry_mid=entry,
            stop_mid=stop, risk_percent=thresholds.risk_per_trade_percent,
            regime_multiplier=regime_multiplier,
        )
        if lot <= 0:
            risk_skips += 1
            continue
        exit_idx, pnl, _ = _simulate_exit(
            market=market, event_market_index=idx, direction=direction, entry_mid=entry,
            stop_mid=stop, target_mid=target, lot=lot, cost_model=cost_model,
            max_holding_bars=thresholds.max_holding_bars,
        )
        profits.append(pnl); lots.append(lot); balance += pnl; peak = max(peak, balance)
        dd = (peak - balance) / peak * 100.0 if peak > 0 else 100.0
        max_dd = max(max_dd, dd); last_exit_idx = exit_idx

    wins = sum(v > 0 for v in profits); losses = sum(v < 0 for v in profits)
    gross_win = sum(v for v in profits if v > 0); gross_loss = abs(sum(v for v in profits if v < 0))
    pf = gross_win / gross_loss if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)
    expectancy = sum(profits) / len(profits) if profits else 0.0
    skip_pct = risk_skips / accepted * 100.0 if accepted else 0.0
    reasons: list[str] = []
    if accepted == 0: reasons.append("no event candidates cleared calibrated predicted net R > 0")
    if len(profits) < thresholds.min_trades: reasons.append(f"trades {len(profits)} < {thresholds.min_trades}")
    if pf < thresholds.min_profit_factor: reasons.append(f"PF {pf:.3f} < {thresholds.min_profit_factor:.2f}")
    if max_dd > thresholds.max_drawdown_percent: reasons.append(f"DD {max_dd:.2f}% > {thresholds.max_drawdown_percent:.2f}%")
    if skip_pct > thresholds.max_risk_skip_percent: reasons.append(f"risk skip {skip_pct:.2f}% > {thresholds.max_risk_skip_percent:.2f}%")
    if expectancy <= 0: reasons.append(f"expectancy {expectancy:.2f} THB <= 0")

    return SampleStrategyResult(
        model_id=f"{model_id}::cost={cost_profile}::netR>0", sample_index=int(sample.get("sample_index") or 0),
        trades=len(profits), wins=wins, losses=losses, net_pnl_thb=float(sum(profits)),
        profit_factor=float(pf), max_drawdown_percent=float(max_dd), expectancy_thb=float(expectancy),
        risk_skips=risk_skips, risk_skip_percent=float(skip_pct), model_blocks=model_blocks,
        average_lot=(sum(lots)/len(lots)) if lots else 0.0,
        status="STRATEGY_SAMPLE_PASS" if not reasons else "STRATEGY_SAMPLE_REJECT",
        reasons=tuple(reasons) if reasons else ("event-edge V4 PF/DD/cost sample gate passed",),
    )
