"""Batch planning for many GOLDmicro challengers.

This module only creates candidate specifications and isolated output paths.
Training engines can consume these specs without ever writing to active Champion paths.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path
from typing import Iterable
import json

from .model_registry import candidate_dir


@dataclass(frozen=True)
class ChallengerSpec:
    model_id: str
    train_bars: int
    seed: int
    xgb_profile: str
    hmm_lookback: int
    confidence_threshold: float
    feature_profile: str
    cost_profile: str
    output_dir: str

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_XGB_PROFILES = ("conservative", "balanced", "responsive")
DEFAULT_FEATURE_PROFILES = ("core", "core_plus_v2")
DEFAULT_COST_PROFILES = ("normal", "conservative")


def build_challenger_specs(
    *,
    train_bars: Iterable[int] = (10000, 15000, 20000),
    seeds: Iterable[int] = (11, 29, 47),
    xgb_profiles: Iterable[str] = DEFAULT_XGB_PROFILES,
    hmm_lookbacks: Iterable[int] = (400, 500, 750),
    confidence_thresholds: Iterable[float] = (0.58, 0.60, 0.62),
    feature_profiles: Iterable[str] = DEFAULT_FEATURE_PROFILES,
    cost_profiles: Iterable[str] = DEFAULT_COST_PROFILES,
    root: str | Path = "models",
    limit: int | None = 96,
) -> list[ChallengerSpec]:
    """Create many deterministic challenger specs in one batch.

    The full Cartesian grid can be very large, so default limit keeps one batch
    bounded while still exploring many independent variants.
    """
    specs: list[ChallengerSpec] = []
    idx = 1
    for vals in product(
        train_bars,
        seeds,
        xgb_profiles,
        hmm_lookbacks,
        confidence_thresholds,
        feature_profiles,
        cost_profiles,
    ):
        bars, seed, xgb, hmm, conf, feat, cost = vals
        model_id = f"gold-ch-{idx:03d}-b{bars}-s{seed}-{xgb}-h{hmm}-c{int(round(conf*100))}-{feat}-{cost}"
        out = candidate_dir(model_id, root)
        specs.append(
            ChallengerSpec(
                model_id=model_id,
                train_bars=int(bars),
                seed=int(seed),
                xgb_profile=str(xgb),
                hmm_lookback=int(hmm),
                confidence_threshold=float(conf),
                feature_profile=str(feat),
                cost_profile=str(cost),
                output_dir=str(out),
            )
        )
        idx += 1
        if limit is not None and len(specs) >= limit:
            break
    return specs


def write_batch_plan(specs: Iterable[ChallengerSpec], path: str | Path) -> Path:
    rows = [s.to_dict() for s in specs]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"count": len(rows), "challengers": rows}, indent=2), encoding="utf-8")
    return path


def assert_isolated_output(spec: ChallengerSpec, root: str | Path = "models") -> None:
    root = Path(root).resolve()
    active = (root / "active").resolve()
    out = Path(spec.output_dir).resolve()
    if active == out or active in out.parents:
        raise ValueError("challenger output may not be inside active Champion directory")
    expected = (root / "candidates").resolve()
    if expected != out.parent and expected not in out.parents:
        raise ValueError("challenger output must live under models/candidates")
