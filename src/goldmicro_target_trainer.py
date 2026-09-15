"""Research-only GOLDmicro trainer with an explicit prediction horizon.

This module reuses the causal GOLDmicro candidate feature pipeline but replaces
its one-bar classification target with a caller-selected forward close horizon.
Artifacts remain isolated under models/candidates and are never promoted.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import json

import polars as pl

from backtests.ml_v2.ml_v2_model import ModelType, TradingModelV2
from src.challenger_batch import ChallengerSpec, assert_isolated_output
from src.feature_eng import FeatureEngineer
import src.goldmicro_candidate_trainer as candidate_trainer
from src.goldmicro_candidate_trainer import (
    OOS_GAP_BARS,
    TRAIN_RATIO,
    _build_model_features_after_hmm,
    xgb_params_for_profile,
)
from src.goldmicro_causal_hmm import predict_causal_regimes
from src.goldmicro_causal_smc import GoldmicroCausalSMCAnalyzer
from src.model_registry import ModelManifest, sha256_file, write_manifest
from src.regime_detector import MarketRegimeDetector


def _prepare_prefix_causal_data(*args, **kwargs):
    """Call the shared preparation path with the research-only causal SMC wrapper.

    The shared candidate trainer intentionally keeps its legacy SMC import for
    compatibility.  Target-alignment research must never fall back to retroactive
    order-block annotations, so the module-global analyzer is replaced only for
    this call and restored immediately afterwards.
    """
    original = candidate_trainer.SMCAnalyzer
    candidate_trainer.SMCAnalyzer = GoldmicroCausalSMCAnalyzer
    try:
        return candidate_trainer._prepare_candidate_data(*args, **kwargs)
    finally:
        candidate_trainer.SMCAnalyzer = original


def train_target_aligned_candidate(
    spec: ChallengerSpec,
    *,
    target_lookahead_bars: int,
    connector,
    symbol: str = "GOLDmicro",
    timeframe: str = "M15",
    git_sha: str = "unknown",
    raw_m15: pl.DataFrame | None = None,
    raw_h1: pl.DataFrame | None = None,
    data_fingerprint: str = "",
) -> dict[str, Any]:
    """Train one isolated candidate against a causal forward-close target.

    The target horizon must remain smaller than the fixed OOS leakage gap.  This
    keeps labels at the end of the training partition from reaching into the OOS
    feature partition.
    """
    target_lookahead_bars = int(target_lookahead_bars)
    if target_lookahead_bars < 1:
        raise ValueError("target_lookahead_bars must be >= 1")
    if target_lookahead_bars >= OOS_GAP_BARS:
        raise ValueError(
            f"target lookahead {target_lookahead_bars} must be < OOS gap {OOS_GAP_BARS}"
        )

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

    # Shared preparation creates the legacy one-bar target. Overwrite it here so
    # only the prediction horizon changes while every feature remains causal.
    df = FeatureEngineer().create_target(df, lookahead=target_lookahead_bars)

    split_idx = int(len(df) * TRAIN_RATIO)
    if split_idx < 500:
        raise ValueError("training partition too small")
    oos_start_idx = min(split_idx + OOS_GAP_BARS, len(df) - 1)
    if oos_start_idx >= len(df) - 100:
        raise ValueError("OOS partition too small after leakage gap")

    hmm_path = out / "hmm_regime.pkl"
    hmm = MarketRegimeDetector(
        n_regimes=3,
        lookback_periods=spec.hmm_lookback,
        model_path=str(hmm_path),
    )
    hmm.fit(df.head(split_idx))
    if not hmm.fitted:
        raise RuntimeError("HMM candidate training failed")
    df = predict_causal_regimes(hmm, df)

    df, feature_cols = _build_model_features_after_hmm(
        df,
        spec=spec,
        df_h1=df_h1,
    )

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
        train_ratio=TRAIN_RATIO,
        num_boost_round=rounds,
        early_stopping_rounds=5,
    )
    if not model.fitted or not xgb_path.exists():
        raise RuntimeError("XGBoost candidate training failed")

    training_data = data_dir / "training_data.parquet"
    df.write_parquet(training_data)
    finished = datetime.now()

    times = df["time"].to_list() if "time" in df.columns else []
    metrics = dict(model._train_metrics or {})
    target_non_null = df.select(pl.col("target").is_not_null().sum()).item()
    target_positive = df.filter(pl.col("target").is_not_null()).select(
        pl.col("target").mean()
    ).item()

    result = {
        "model_id": spec.model_id,
        "success": True,
        "symbol": symbol,
        "timeframe": timeframe,
        "target_lookahead_bars": target_lookahead_bars,
        "target_non_null_rows": int(target_non_null),
        "target_positive_rate": float(target_positive) if target_positive is not None else None,
        "train_bars": spec.train_bars,
        "features": len(feature_cols),
        "feature_profile": spec.feature_profile,
        "xgb_profile": spec.xgb_profile,
        "hmm_lookback": spec.hmm_lookback,
        "hmm_inference": "causal_forward_filter_with_confirmation",
        "smc_history": "prefix_causal_research_wrapper",
        "confidence_threshold": spec.confidence_threshold,
        "seed": spec.seed,
        "cost_profile": spec.cost_profile,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_seconds": (finished - started).total_seconds(),
        "train_metrics": metrics,
        "split": {
            "train_ratio": TRAIN_RATIO,
            "gap_bars": OOS_GAP_BARS,
            "target_lookahead_bars": target_lookahead_bars,
            "split_index": split_idx,
            "oos_start_index": oos_start_idx,
            "data_start": str(times[0]) if times else None,
            "train_end": str(times[split_idx - 1]) if times else None,
            "oos_start": str(times[oos_start_idx]) if times else None,
            "oos_end": str(times[-1]) if times else None,
        },
        "data_fingerprint": data_fingerprint,
        "xgb_path": str(xgb_path),
        "hmm_path": str(hmm_path),
        "training_data_path": str(training_data),
        "xgb_sha256": sha256_file(xgb_path),
        "hmm_sha256": sha256_file(hmm_path),
    }
    (out / "training_result.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )

    manifest = ModelManifest(
        model_id=spec.model_id,
        status="TARGET_ALIGNMENT_RESEARCH_ONLY",
        git_sha=git_sha,
        created_at=finished.isoformat(),
        training_start=started.isoformat(),
        training_end=finished.isoformat(),
        feature_set=f"{spec.feature_profile}:target_t{target_lookahead_bars}",
        config_hash=(
            f"target={target_lookahead_bars}:{spec.xgb_profile}:{spec.hmm_lookback}:"
            f"{spec.confidence_threshold}:{spec.cost_profile}:{data_fingerprint[:16]}"
        ),
        random_seed=spec.seed,
        xgb_path=str(xgb_path),
        hmm_path=str(hmm_path),
        xgb_sha256=result["xgb_sha256"],
        hmm_sha256=result["hmm_sha256"],
    )
    write_manifest(manifest, out / "manifest.json")
    return result
