"""Probability calibration helpers for GOLDmicro event-success research.

Calibration is fit only on a chronological pre-OOS calibration partition.  The
final OOS partition must never be used to fit or select the calibrator.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

CALIBRATION_METHOD = "platt_logistic_on_xgb_margin"


def fit_platt_calibrator(margins: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    margins = np.asarray(margins, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=int)
    if len(margins) != len(y) or len(y) < 2:
        raise ValueError("invalid calibration arrays")
    if len(np.unique(y)) < 2:
        raise ValueError("calibration target has only one class")
    model = LogisticRegression(solver="lbfgs", max_iter=1000)
    model.fit(margins, y)
    return {
        "method": CALIBRATION_METHOD,
        "coef": float(model.coef_[0][0]),
        "intercept": float(model.intercept_[0]),
        "sample_count": int(len(y)),
        "positive_rate": float(np.mean(y)),
    }


def apply_platt_calibrator(margins: np.ndarray, calibration: dict[str, Any]) -> np.ndarray:
    if calibration.get("method") != CALIBRATION_METHOD:
        raise ValueError(f"unsupported calibration method: {calibration.get('method')}")
    margins = np.asarray(margins, dtype=float)
    z = float(calibration["coef"]) * margins + float(calibration["intercept"])
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def write_calibration(calibration: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")


def read_calibration(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
