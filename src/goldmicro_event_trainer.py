"""Dedicated GOLDmicro event-success trainer.

This trainer intentionally does NOT use TradingModelV2.fit() because that helper
splits after row filtering and applies its embargo in cleaned-row units. Event
research requires train/OOS separation in raw market-bar time.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import json

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.challenger_batch import ChallengerSpec, assert_isolated_output
from src.goldmicro_candidate_trainer import (
    TRAIN_RATIO,
    _build_model_features_after_hmm,
    xgb_params_for_profile,
)
from src.goldmicro_causal_hmm import predict_causal_regimes
from src.goldmicro_event_target import EventTargetConfig, build_event_target_frame, split_events_by_raw_time
from src.goldmicro_target_trainer import _prepare_prefix_causal_data
from src.model_registry import ModelManifest, sha256_file, write_manifest
from src.regime_detector import MarketRegimeDetector


MIN_TRAIN_EVENTS = 300
MIN_TEST_EVENTS = 100


def _clean_xy(frame: pl.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    clean = frame.select(features + ["event_target"]).drop_nulls()
    X = clean.select(features).to_numpy()
    y = clean["event_target"].to_numpy()
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y


def train_event_candidate(
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
        connector,
        symbol,
        timeframe,
        spec,
        raw_m15=raw_m15,
        raw_h1=raw_h1,
    )

    raw_split_idx = int(len(df) * TRAIN_RATIO)
    if raw_split_idx < 1000:
        raise ValueError("raw training partition too small")

    hmm_path = out / "hmm_regime.pkl"
    hmm = MarketRegimeDetector(
        n_regimes=3,
        lookback_periods=spec.hmm_lookback,
        model_path=str(hmm_path),
    )
    hmm.fit(df.head(raw_split_idx))
    if not hmm.fitted:
        raise RuntimeError("HMM candidate training failed")
    df = predict_causal_regimes(hmm, df)
    df, feature_cols = _build_model_features_after_hmm(df, spec=spec, df_h1=df_h1)

    events = build_event_target_frame(df, config=event_config)
    train_events, test_events = split_events_by_raw_time(
        events,
        raw_split_index=raw_split_idx,
        label_horizon_bars=event_config.max_holding_bars,
    )
    if len(train_events) < MIN_TRAIN_EVENTS:
        raise ValueError(f"train events {len(train_events)} < {MIN_TRAIN_EVENTS}")
    if len(test_events) < MIN_TEST_EVENTS:
        raise ValueError(f"test events {len(test_events)} < {MIN_TEST_EVENTS}")

    X_train, y_train = _clean_xy(train_events, feature_cols)
    X_test, y_test = _clean_xy(test_events, feature_cols)
    if len(X_train) < MIN_TRAIN_EVENTS:
        raise ValueError(f"usable train events {len(X_train)} < {MIN_TRAIN_EVENTS}")
    if len(X_test) < MIN_TEST_EVENTS:
        raise ValueError(f"usable test events {len(X_test)} < {MIN_TEST_EVENTS}")
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        raise ValueError("event target has only one class in train or test")

    params = xgb_params_for_profile(spec.xgb_profile, spec.seed)
    rounds = {"conservative": 50, "balanced": 70, "responsive": 90}[spec.xgb_profile]
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)
    evals_result: dict[str, Any] = {}
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=rounds,
        evals=[(dtrain, "train"), (dtest, "eval")],
        early_stopping_rounds=5,
        evals_result=evals_result,
        verbose_eval=False,
    )
    p_train = booster.predict(dtrain)
    p_test = booster.predict(dtest)
    train_auc = float(roc_auc_score(y_train, p_train))
    test_auc = float(roc_auc_score(y_test, p_test))
    pr_auc = float(average_precision_score(y_test, p_test))
    brier = float(brier_score_loss(y_test, p_test))

    model_path = out / "event_xgb.json"
    booster.save_model(model_path)
    event_data_path = data_dir / "event_data.parquet"
    events.write_parquet(event_data_path)
    finished = datetime.now()

    result = {
        "model_id": spec.model_id,
        "success": True,
        "model_semantics": "p_smc_setup_tp_before_sl_within_32_bars",
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
        "train_event_count": len(X_train),
        "test_event_count": len(X_test),
        "train_positive_rate": float(np.mean(y_train)),
        "test_positive_rate": float(np.mean(y_test)),
        "train_metrics": {
            "xgb_train_score": train_auc,
            "xgb_test_score": test_auc,
            "test_samples": len(X_test),
            "pr_auc": pr_auc,
            "brier": brier,
            "generalization_gap": train_auc - test_auc,
        },
        "split": {
            "raw_split_index": raw_split_idx,
            "train_last_event_index": raw_split_idx - event_config.max_holding_bars - 1,
            "test_first_event_index": raw_split_idx + event_config.max_holding_bars,
            "raw_embargo_bars_each_side": event_config.max_holding_bars,
        },
        "data_fingerprint": data_fingerprint,
        "event_model_path": str(model_path),
        "hmm_path": str(hmm_path),
        "event_data_path": str(event_data_path),
        "event_model_sha256": sha256_file(model_path),
        "hmm_sha256": sha256_file(hmm_path),
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_seconds": (finished - started).total_seconds(),
    }
    (out / "training_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    manifest = ModelManifest(
        model_id=spec.model_id,
        status="EVENT_TARGET_RESEARCH_ONLY_NOT_VALIDATED",
        git_sha=git_sha,
        created_at=finished.isoformat(),
        training_start=started.isoformat(),
        training_end=finished.isoformat(),
        feature_set=f"{spec.feature_profile}:smc_event_success_32",
        config_hash=(
            f"event32:{spec.xgb_profile}:{spec.hmm_lookback}:{spec.seed}:"
            f"{spec.feature_profile}:{data_fingerprint[:16]}"
        ),
        random_seed=spec.seed,
        xgb_path=str(model_path),
        hmm_path=str(hmm_path),
        xgb_sha256=result["event_model_sha256"],
        hmm_sha256=result["hmm_sha256"],
    )
    write_manifest(manifest, out / "manifest.json")
    return result
