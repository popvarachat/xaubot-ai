"""Isolated GOLDmicro challenger trainer.

Candidates are trained only under models/candidates/<model_id>. This module never
writes models/xgboost_model.pkl, models/hmm_regime.pkl, or models/active/*.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import json
import polars as pl

from backtests.ml_v2.ml_v2_feature_eng import MLV2FeatureEngineer
from backtests.ml_v2.ml_v2_model import TradingModelV2, ModelType
from src.challenger_batch import ChallengerSpec, assert_isolated_output
from src.feature_eng import FeatureEngineer
from src.model_registry import ModelManifest, sha256_file, write_manifest
from src.regime_detector import MarketRegimeDetector
from src.smc_polars import SMCAnalyzer
from src.ml_model import get_default_feature_columns


def xgb_params_for_profile(profile: str, seed: int) -> dict[str, Any]:
    base = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "device": "cpu",
        "learning_rate": 0.05,
        "max_delta_step": 1,
        "seed": int(seed),
    }
    if profile == "conservative":
        base.update(max_depth=2, min_child_weight=15, subsample=0.65,
                    colsample_bytree=0.55, reg_alpha=1.5, reg_lambda=7.0, gamma=1.5)
    elif profile == "responsive":
        base.update(max_depth=4, min_child_weight=7, subsample=0.80,
                    colsample_bytree=0.75, reg_alpha=0.5, reg_lambda=3.0, gamma=0.5)
    elif profile == "balanced":
        base.update(max_depth=3, min_child_weight=10, subsample=0.70,
                    colsample_bytree=0.60, reg_alpha=1.0, reg_lambda=5.0, gamma=1.0)
    else:
        raise ValueError(f"unknown xgb profile: {profile}")
    return base


def _numeric_features(df: pl.DataFrame) -> list[str]:
    exclude = {
        "time", "open", "high", "low", "close", "volume", "target",
        "tick_volume", "spread", "real_volume", "multi_bar_target",
    }
    numeric = {pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8, pl.UInt64, pl.UInt32, pl.UInt16, pl.UInt8, pl.Boolean}
    return [c for c in df.columns if c not in exclude and df[c].dtype in numeric]


def _prepare_candidate_data(connector, symbol: str, timeframe: str, spec: ChallengerSpec):
    df = connector.get_market_data(symbol, timeframe, spec.train_bars)
    if len(df) < 1000:
        raise ValueError(f"insufficient training data: {len(df)} bars")

    fe = FeatureEngineer()
    smc = SMCAnalyzer(swing_length=5)
    df = fe.calculate_all(df, include_ml_features=True)
    df = smc.calculate_all(df)
    df = fe.create_target(df, lookahead=1)

    if spec.feature_profile == "core_plus_v2":
        df_h1 = None
        try:
            df_h1 = connector.get_market_data(symbol, "H1", min(spec.train_bars // 4, 3000))
            if len(df_h1) > 30:
                df_h1 = fe.calculate_all(df_h1, include_ml_features=False)
                df_h1 = smc.calculate_all(df_h1)
            else:
                df_h1 = None
        except Exception:
            df_h1 = None
        df = MLV2FeatureEngineer().add_all_v2_features(df, df_h1)
        feature_cols = _numeric_features(df)
    elif spec.feature_profile == "core":
        defaults = get_default_feature_columns()
        feature_cols = [c for c in defaults if c in df.columns]
    else:
        raise ValueError(f"unknown feature profile: {spec.feature_profile}")

    if not feature_cols:
        raise ValueError("candidate has no usable features")
    return df, feature_cols


def train_candidate(
    spec: ChallengerSpec,
    *,
    connector,
    symbol: str = "GOLDmicro",
    timeframe: str = "M15",
    git_sha: str = "unknown",
) -> dict[str, Any]:
    """Train one isolated candidate and return auditable metadata."""
    assert_isolated_output(spec)
    out = Path(spec.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data_dir = out / "data"
    data_dir.mkdir(exist_ok=True)

    started = datetime.now()
    df, feature_cols = _prepare_candidate_data(connector, symbol, timeframe, spec)

    hmm_path = out / "hmm_regime.pkl"
    hmm = MarketRegimeDetector(
        n_regimes=3,
        lookback_periods=spec.hmm_lookback,
        model_path=str(hmm_path),
    )
    hmm.fit(df)
    if not hmm.fitted:
        raise RuntimeError("HMM candidate training failed")
    df = hmm.predict(df)

    xgb_path = out / "xgboost_model.pkl"
    rounds = {"conservative": 50, "balanced": 70, "responsive": 90}[spec.xgb_profile]
    model = TradingModelV2(
        model_type=ModelType.XGBOOST_BINARY,
        confidence_threshold=spec.confidence_threshold,
        model_path=str(xgb_path),
        xgb_params=xgb_params_for_profile(spec.xgb_profile, spec.seed),
    )
    model.fit(
        df,
        feature_cols,
        target_col="target",
        train_ratio=0.70,
        num_boost_round=rounds,
        early_stopping_rounds=5,
    )
    if not model.fitted or not xgb_path.exists():
        raise RuntimeError("XGBoost candidate training failed")

    training_data = data_dir / "training_data.parquet"
    df.write_parquet(training_data)
    finished = datetime.now()

    metrics = dict(model._train_metrics or {})
    result = {
        "model_id": spec.model_id,
        "success": True,
        "symbol": symbol,
        "timeframe": timeframe,
        "train_bars": spec.train_bars,
        "features": len(feature_cols),
        "feature_profile": spec.feature_profile,
        "xgb_profile": spec.xgb_profile,
        "hmm_lookback": spec.hmm_lookback,
        "confidence_threshold": spec.confidence_threshold,
        "seed": spec.seed,
        "cost_profile": spec.cost_profile,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_seconds": (finished - started).total_seconds(),
        "train_metrics": metrics,
        "xgb_path": str(xgb_path),
        "hmm_path": str(hmm_path),
        "xgb_sha256": sha256_file(xgb_path),
        "hmm_sha256": sha256_file(hmm_path),
    }
    (out / "training_result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    manifest = ModelManifest(
        model_id=spec.model_id,
        status="CHALLENGER_TRAINED_NOT_VALIDATED",
        git_sha=git_sha,
        created_at=finished.isoformat(),
        training_start=started.isoformat(),
        training_end=finished.isoformat(),
        feature_set=spec.feature_profile,
        config_hash=f"{spec.xgb_profile}:{spec.hmm_lookback}:{spec.confidence_threshold}:{spec.cost_profile}",
        random_seed=spec.seed,
        xgb_path=str(xgb_path),
        hmm_path=str(hmm_path),
        xgb_sha256=result["xgb_sha256"],
        hmm_sha256=result["hmm_sha256"],
    )
    write_manifest(manifest, out / "manifest.json")
    return result
