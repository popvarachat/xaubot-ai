"""Frozen affine calibration for GOLDmicro V4 event-edge regression.

Fit only on the pre-OOS calibration partition.  The untouched OOS partition is
never used to estimate slope/intercept or choose thresholds.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CALIBRATION_METHOD = "affine_least_squares_pre_oos"


def fit_affine_calibrator(raw_pred: np.ndarray, realized_r: np.ndarray) -> dict:
    x = np.asarray(raw_pred, dtype=float)
    y = np.asarray(realized_r, dtype=float)
    if x.size != y.size or x.size < 20:
        raise ValueError("affine calibration requires >=20 paired samples")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("affine calibration inputs must be finite")
    variance = float(np.var(x))
    if variance <= 1e-12:
        raise ValueError("affine calibration failed: degenerate prediction variance")

    slope, intercept = np.polyfit(x, y, 1)
    slope = float(slope)
    intercept = float(intercept)
    if not np.isfinite(slope) or not np.isfinite(intercept):
        raise ValueError("affine calibration produced non-finite coefficients")
    if slope <= 0.0:
        raise ValueError(f"affine calibration failed closed: non-positive slope {slope:.6f}")

    calibrated = slope * x + intercept
    return {
        "method": CALIBRATION_METHOD,
        "slope": slope,
        "intercept": intercept,
        "sample_count": int(x.size),
        "raw_mean": float(np.mean(x)),
        "realized_mean": float(np.mean(y)),
        "calibrated_mean": float(np.mean(calibrated)),
    }


def apply_affine_calibrator(raw_pred: np.ndarray, calibration: dict) -> np.ndarray:
    if calibration.get("method") != CALIBRATION_METHOD:
        raise ValueError(f"unexpected calibration method: {calibration.get('method')}")
    slope = float(calibration["slope"])
    intercept = float(calibration["intercept"])
    return slope * np.asarray(raw_pred, dtype=float) + intercept


def write_edge_calibration(calibration: dict, path: Path) -> None:
    path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")


def read_edge_calibration(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
