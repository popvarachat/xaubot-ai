"""Dedicated GOLDmicro V4 event-edge regression trainer.

Chronology is FIT -> raw-bar embargo -> CALIBRATION -> raw-bar embargo ->
UNTOUCHED OOS.  The model predicts realized gross R of a causal SMC setup; it
never predicts BUY/SELL and never writes an active/live model path.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import json

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.challenger_batch import ChallengerSpec, assert_isolated_output
from src.goldmicro_candidate_trainer import TRAIN_RATIO, _build_model_features_after_hmm, xgb_params_for_profile
from src.goldmicro_causal_hmm import predict_causal_regimes
from src.goldmicro_event_edge_calibration import apply_affine_calibrator, fit_affine_calibrator, write_edge_calibration
from src.goldmicro_event_edge_target import EDGE_TARGET_COLUMN, build_event_edge_frame
from src.goldmicro_event_target import EventTargetConfig, split_events_by_raw_time
from src.goldmicro_target_trainer import _prepare_prefix_causal_data
from src.model_registry import ModelManifest, sha256_file, write_manifest
from src.regime_detector import MarketRegimeDetector

MIN_FIT_EVENTS = 300
MIN_CALIBRATION_EVENTS = 100
MIN_TEST_EVENTS = 100
CALIBRATION_SHARE_OF_PRE_OOS_RAW = 0.20


def _clean_xy(frame: pl.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    clean = frame.select(features + [EDGE_TARGET_COLUMN]).drop_nulls()
    X = clean.select(features).to_numpy()
    y = clean[EDGE_TARGET_COLUMN].to_numpy().astype(float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y


def _rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return 0.0
    xr = np.argsort(np.argsort(x)).astype(float)
    yr = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(xr, yr)[0, 1])


def _linear_slope(pred: np.ndarray, realized: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=float)
    realized = np.asarray(realized, dtype=float)
    if pred.size < 2 or np.var(pred) <= 1e-12:
        return 0.0
    return float(np.polyfit(pred, realized, 1)[0])


def _split_fit_calibration_oos(events: pl.DataFrame, *, raw_oos_split_index: int, label_horizon_bars: int):
    raw_fit_split_index = int(raw_oos_split_index * (1.0 - CALIBRATION_SHARE_OF_PRE_OOS_RAW))
    fit_events, after_fit_embargo = split_events_by_raw_time(
        events,
        raw_split_index=raw_fit_split_index,
        label_horizon_bars=label_horizon_bars,
    )
    calibration_last_decision = raw_oos_split_index - label_horizon_bars - 1
    calibration_events = after_fit_embargo.filter(pl.col("event_index") <= calibration_last_decision)
    oos_events = events.filter(pl.col("event_index") >= raw_oos_split_index + label_horizon_bars)
    return fit_events, calibration_events, oos_events, raw_fit_split_index


def _regression_params(profile: str, seed: int) -> dict[str, Any]:
    params = dict(xgb_params_for_profile(profile, seed))
    params["objective"] = "reg:squarederror"
    params["eval_metric"] = "rmse"
    params.pop("max_delta_step", None)
    return params


def train_event_edge_candidate(
    spec: ChallengerSpec,
    *,
    connector,
    symbol: str = "GOLDmicro",
    timeframe: str = "M15",
    git_sha: str = "unknown",
    raw_m15: pl.DataFrame | None = None,
    raw_h1: pl.DataFrame | None = None,
    data_fingerprint: str = "",
    event_config: EventTargetConfig = EventTargetConfig(),
) -> dict[str, Any]:
    assert_isolated_output(spec)
    out = Path(spec.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data_dir = out / "data"
    data_dir.mkdir(exist_ok=True)
    started = datetime.now()

    df, df_h1 = _prepare_prefix_causal_data(
        connector, symbol, timeframe, spec, raw_m15=raw_m15, raw_h1=raw_h1
    )
    raw_oos_split_idx = int(len(df) * TRAIN_RATIO)
    raw_fit_split_idx = int(raw_oos_split_idx * (1.0 - CALIBRATION_SHARE_OF_PRE_OOS_RAW))
    if raw_fit_split_idx < 1000:
        raise ValueError("raw fit partition too small")

    hmm_path = out / "hmm_regime.pkl"
    hmm = MarketRegimeDetector(n_regimes=3, lookback_periods=spec.hmm_lookback, model_path=str(hmm_path))
    hmm.fit(df.head(raw_fit_split_idx))
    if not hmm.fitted:
        raise RuntimeError("HMM candidate training failed")
    df = predict_causal_regimes(hmm, df)
    df, feature_cols = _build_model_features_after_hmm(df, spec=spec, df_h1=df_h1)

    events = build_event_edge_frame(df, config=event_config)
    fit_events, calibration_events, test_events, raw_fit_split_idx = _split_fit_calibration_oos(
        events,
        raw_oos_split_index=raw_oos_split_idx,
        label_horizon_bars=event_config.max_holding_bars,
    )
    if len(fit_events) < MIN_FIT_EVENTS:
        raise ValueError(f"fit events {len(fit_events)} < {MIN_FIT_EVENTS}")
    if len(calibration_events) < MIN_CALIBRATION_EVENTS:
        raise ValueError(f"calibration events {len(calibration_events)} < {MIN_CALIBRATION_EVENTS}")
    if len(test_events) < MIN_TEST_EVENTS:
        raise ValueError(f"test events {len(test_events)} < {MIN_TEST_EVENTS}")

    X_fit, y_fit = _clean_xy(fit_events, feature_cols)
    X_cal, y_cal = _clean_xy(calibration_events, feature_cols)
    X_test, y_test = _clean_xy(test_events, feature_cols)
    if len(X_fit) < MIN_FIT_EVENTS or len(X_cal) < MIN_CALIBRATION_EVENTS or len(X_test) < MIN_TEST_EVENTS:
        raise ValueError("usable fit/calibration/OOS event count below V4 minimum")

    rounds = {"conservative": 50, "balanced": 70, "responsive": 90}[spec.xgb_profile]
    dfit = xgb.DMatrix(X_fit, label=y_fit, feature_names=feature_cols)
    dcal = xgb.DMatrix(X_cal, label=y_cal, feature_names=feature_cols)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)
    booster = xgb.train(
        _regression_params(spec.xgb_profile, spec.seed),
        dfit,
        num_boost_round=rounds,
        evals=[(dfit, "fit")],
        verbose_eval=False,
    )

    fit_raw = booster.predict(dfit)
    cal_raw = booster.predict(dcal)
    test_raw = booster.predict(dtest)
    calibration = fit_affine_calibrator(cal_raw, y_cal)
    cal_pred = apply_affine_calibrator(cal_raw, calibration)
    test_pred = apply_affine_calibrator(test_raw, calibration)

    oos_mae = float(mean_absolute_error(y_test, test_pred))
    oos_rmse = float(np.sqrt(mean_squared_error(y_test, test_pred)))
    rank_corr = _rank_corr(test_pred, y_test)
    oos_slope = _linear_slope(test_pred, y_test)
    positive_tail = test_pred > 0.0
    tail_count = int(np.sum(positive_tail))
    tail_pred_mean = float(np.mean(test_pred[positive_tail])) if tail_count else None
    tail_realized_mean = float(np.mean(y_test[positive_tail])) if tail_count else None
    tail_sign_consistent = bool(tail_count and tail_realized_mean is not None and tail_realized_mean > 0.0)

    model_path = out / "event_edge_xgb.json"
    booster.save_model(model_path)
    calibration_path = out / "event_edge_calibration.json"
    write_edge_calibration(calibration, calibration_path)
    data_path = data_dir / "event_edge_data.parquet"
    events.write_parquet(data_path)
    finished = datetime.now()

    result = {
        "model_id": spec.model_id,
        "success": True,
        "model_semantics": "expected_realized_gross_r_of_causal_smc_setup",
        "symbol": symbol,
        "timeframe": timeframe,
        "train_bars": spec.train_bars,
        "feature_profile": spec.feature_profile,
        "xgb_profile": spec.xgb_profile,
        "hmm_lookback": spec.hmm_lookback,
        "seed": spec.seed,
        "features": len(feature_cols),
        "feature_names": feature_cols,
        "event_horizon_bars": event_config.max_holding_bars,
        "event_cooldown_bars": event_config.event_cooldown_bars,
        "event_count": len(events),
        "fit_event_count": len(X_fit),
        "calibration_event_count": len(X_cal),
        "test_event_count": len(X_test),
        "fit_target_mean_r": float(np.mean(y_fit)),
        "calibration_target_mean_r": float(np.mean(y_cal)),
        "test_target_mean_r": float(np.mean(y_test)),
        "metrics": {
            "fit_raw_mae": float(mean_absolute_error(y_fit, fit_raw)),
            "calibration_mae": float(mean_absolute_error(y_cal, cal_pred)),
            "oos_mae": oos_mae,
            "oos_rmse": oos_rmse,
            "oos_rank_correlation": rank_corr,
            "oos_calibration_slope": oos_slope,
            "oos_predicted_mean_r": float(np.mean(test_pred)),
            "oos_realized_mean_r": float(np.mean(y_test)),
            "positive_pred_tail_count": tail_count,
            "positive_pred_tail_mean_predicted_r": tail_pred_mean,
            "positive_pred_tail_mean_realized_r": tail_realized_mean,
            "positive_pred_tail_sign_consistent": tail_sign_consistent,
        },
        "calibration": calibration,
        "split": {
            "raw_fit_split_index": raw_fit_split_idx,
            "fit_last_event_index": raw_fit_split_idx - event_config.max_holding_bars - 1,
            "calibration_first_event_index": raw_fit_split_idx + event_config.max_holding_bars,
            "calibration_last_event_index": raw_oos_split_idx - event_config.max_holding_bars - 1,
            "raw_split_index": raw_oos_split_idx,
            "test_first_event_index": raw_oos_split_idx + event_config.max_holding_bars,
            "raw_embargo_bars_each_boundary": event_config.max_holding_bars,
        },
        "data_fingerprint": data_fingerprint,
        "xgb_path": str(model_path),
        "calibration_path": str(calibration_path),
        "hmm_path": str(hmm_path),
        "training_data_path": str(data_path),
        "xgb_sha256": sha256_file(model_path),
        "calibration_sha256": sha256_file(calibration_path),
        "hmm_sha256": sha256_file(hmm_path),
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_seconds": (finished - started).total_seconds(),
    }
    (out / "training_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    manifest = ModelManifest(
        model_id=spec.model_id,
        status="EVENT_EDGE_V4_RESEARCH_ONLY_NOT_VALIDATED",
        git_sha=git_sha,
        created_at=finished.isoformat(),
        training_start=started.isoformat(),
        training_end=finished.isoformat(),
        feature_set=f"{spec.feature_profile}:smc_realized_gross_r_32",
        config_hash=(f"event-edge-v4:{spec.xgb_profile}:{spec.hmm_lookback}:{spec.seed}:"
                     f"{spec.feature_profile}:{data_fingerprint[:16]}"),
        random_seed=spec.seed,
        xgb_path=str(model_path),
        hmm_path=str(hmm_path),
        xgb_sha256=result["xgb_sha256"],
        hmm_sha256=result["hmm_sha256"],
    )
    write_manifest(manifest, out / "manifest.json")
    return result
