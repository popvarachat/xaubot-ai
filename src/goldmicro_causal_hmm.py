"""Prefix-causal HMM inference for GOLDmicro challenger research.

The generic regime detector's ``predict`` method decodes an entire sequence and
then smooths complete segments.  That is useful for retrospective diagnostics,
but historical labels can change when future observations are appended.  Such
labels must not be used as model features in an out-of-sample research pipeline.

This module performs a forward-only HMM filter with a causal confirmation delay.
At timestamp t it uses observations <= t only.  It is research-only and never
places orders, changes active models, or performs promotion.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from src.regime_detector import MarketRegime


_EPS = 1e-300


def _logsumexp(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Small NumPy-only logsumexp helper to avoid another runtime dependency."""
    values = np.asarray(values, dtype=float)
    maximum = np.max(values, axis=axis, keepdims=True)
    maximum = np.where(np.isfinite(maximum), maximum, 0.0)
    summed = np.sum(np.exp(values - maximum), axis=axis, keepdims=True)
    out = maximum + np.log(np.maximum(summed, _EPS))
    if axis is not None:
        out = np.squeeze(out, axis=axis)
    return out


def _emission_log_likelihood(model: Any, features: np.ndarray) -> np.ndarray:
    """Return per-row, per-state Gaussian log likelihoods.

    hmmlearn exposes ``_compute_log_likelihood`` for exactly this calculation.
    A diagonal-Gaussian fallback keeps the causal algorithm explicit and makes
    the helper testable with a small fake model.
    """
    compute = getattr(model, "_compute_log_likelihood", None)
    if callable(compute):
        return np.asarray(compute(features), dtype=float)

    means = np.asarray(model.means_, dtype=float)
    covars = getattr(model, "_covars_", None)
    if covars is None:
        covars = np.asarray(model.covars_, dtype=float)
    else:
        covars = np.asarray(covars, dtype=float)

    if covars.ndim == 3:
        covars = np.diagonal(covars, axis1=1, axis2=2)
    if covars.ndim != 2:
        raise ValueError("causal HMM fallback supports diagonal covariance only")

    covars = np.maximum(covars, 1e-12)
    diff = features[:, None, :] - means[None, :, :]
    log_det = np.sum(np.log(2.0 * np.pi * covars), axis=1)
    mahal = np.sum((diff * diff) / covars[None, :, :], axis=2)
    return -0.5 * (mahal + log_det[None, :])


def causal_filter_probabilities(detector: Any, df: pl.DataFrame) -> tuple[np.ndarray, int]:
    """Compute filtered state probabilities using observations up to each row only.

    Returns ``(probabilities, leading_padding_rows)``.  ``prepare_features`` in the
    detector uses trailing/rolling calculations, so its dropped rows are warm-up
    rows at the beginning of the frame.
    """
    if not getattr(detector, "fitted", False) or getattr(detector, "model", None) is None:
        raise ValueError("HMM detector must be fitted before causal inference")

    raw = np.asarray(detector.prepare_features(df), dtype=float)
    if raw.ndim != 2 or len(raw) == 0:
        return np.empty((0, int(getattr(detector, "n_regimes", 0))), dtype=float), len(df)

    scaler = getattr(detector, "scaler", None)
    features = np.asarray(scaler.transform(raw) if scaler is not None else raw, dtype=float)
    model = detector.model
    emission = _emission_log_likelihood(model, features)

    start = np.maximum(np.asarray(model.startprob_, dtype=float), _EPS)
    trans = np.maximum(np.asarray(model.transmat_, dtype=float), _EPS)
    n_states = len(start)
    if emission.shape[1] != n_states or trans.shape != (n_states, n_states):
        raise ValueError("HMM parameter dimensions are inconsistent")

    filtered = np.empty((len(features), n_states), dtype=float)
    log_alpha = np.log(start) + emission[0]
    log_alpha = log_alpha - _logsumexp(log_alpha)
    filtered[0] = np.exp(log_alpha)

    log_trans = np.log(trans)
    for idx in range(1, len(features)):
        # p(z_t | x_<=t) ∝ p(x_t | z_t) * Σ p(z_t | z_t-1) p(z_t-1 | x_<=t-1)
        predicted = _logsumexp(log_alpha[:, None] + log_trans, axis=0)
        log_alpha = predicted + emission[idx]
        log_alpha = log_alpha - _logsumexp(log_alpha)
        filtered[idx] = np.exp(log_alpha)

    leading_padding = len(df) - len(features)
    if leading_padding < 0:
        raise ValueError("HMM feature preparation returned more rows than input")
    return filtered, leading_padding


def causal_confirm_states(
    raw_states: np.ndarray,
    *,
    min_duration: int,
) -> np.ndarray:
    """Apply a causal minimum-duration confirmation rule.

    A new state must persist for ``min_duration`` observations before it becomes
    the confirmed state.  Earlier outputs are never rewritten when future rows
    arrive, unlike retrospective segment smoothing.
    """
    states = np.asarray(raw_states, dtype=int)
    if len(states) == 0 or min_duration <= 1:
        return states.copy()

    out = np.empty_like(states)
    confirmed = int(states[0])
    candidate: int | None = None
    candidate_count = 0
    out[0] = confirmed

    for idx in range(1, len(states)):
        observed = int(states[idx])
        if observed == confirmed:
            candidate = None
            candidate_count = 0
        else:
            if candidate == observed:
                candidate_count += 1
            else:
                candidate = observed
                candidate_count = 1
            if candidate_count >= min_duration:
                confirmed = observed
                candidate = None
                candidate_count = 0
        out[idx] = confirmed
    return out


def predict_causal_regimes(detector: Any, df: pl.DataFrame) -> pl.DataFrame:
    """Attach prefix-causal ``regime`` columns to a historical frame."""
    probabilities, leading_padding = causal_filter_probabilities(detector, df)
    if len(probabilities) == 0:
        return df.with_columns(
            pl.lit(None, dtype=pl.Int64).alias("regime"),
            pl.lit(None, dtype=pl.Utf8).alias("regime_name"),
            pl.lit(None, dtype=pl.Float64).alias("regime_confidence"),
        )

    raw_states = np.argmax(probabilities, axis=1).astype(int)
    smoothing_enabled = bool(getattr(detector, "smoothing_enabled", False))
    min_duration = int(getattr(detector, "smoothing_min_duration", 1)) if smoothing_enabled else 1
    states = causal_confirm_states(raw_states, min_duration=min_duration)

    names = [
        getattr(detector, "regime_mapping", {}).get(int(state), MarketRegime.MEDIUM_VOLATILITY).value
        for state in states
    ]
    confidences = [float(probabilities[idx, int(state)]) for idx, state in enumerate(states)]

    pad_int = [None] * leading_padding + [int(x) for x in states]
    pad_name = [None] * leading_padding + names
    pad_conf = [None] * leading_padding + confidences
    return df.with_columns(
        pl.Series("regime", pad_int, dtype=pl.Int64),
        pl.Series("regime_name", pad_name, dtype=pl.Utf8),
        pl.Series("regime_confidence", pad_conf, dtype=pl.Float64),
    )
