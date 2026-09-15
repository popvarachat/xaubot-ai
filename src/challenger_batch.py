"""Batch planning for many GOLDmicro challengers.

This module only creates candidate specifications and isolated output paths.
Training engines can consume these specs without ever writing to active Champion paths.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from hashlib import sha256
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
    batch_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_XGB_PROFILES = ("conservative", "balanced", "responsive")
DEFAULT_FEATURE_PROFILES = ("core", "core_plus_v2")
DEFAULT_COST_PROFILES = ("normal", "conservative")


def _safe_batch_id(batch_id: str | None) -> str:
    if not batch_id:
        return ""
    cleaned = "".join(ch for ch in str(batch_id) if ch.isalnum() or ch in ("-", "_"))
    if not cleaned or cleaned != str(batch_id):
        raise ValueError("batch_id contains unsafe characters")
    return cleaned


def _diverse_combinations(values: list[tuple], limit: int | None) -> list[tuple]:
    """Deterministically spread a bounded batch across the full Cartesian grid.

    Taking the first N values from itertools.product badly biases small batches
    toward the earliest train-window/seed/profile values. Hash ordering gives a
    reproducible pseudo-random sample that covers the full design space without
    relying on process-global RNG state.
    """
    if limit is None or limit >= len(values):
        return values
    ranked = sorted(
        values,
        key=lambda vals: sha256(repr(vals).encode("utf-8")).hexdigest(),
    )
    return ranked[:limit]


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
    batch_id: str | None = None,
) -> list[ChallengerSpec]:
    """Create many deterministic, diverse challenger specs in one batch."""
    safe_batch = _safe_batch_id(batch_id)
    full_grid = list(product(
        tuple(train_bars),
        tuple(seeds),
        tuple(xgb_profiles),
        tuple(hmm_lookbacks),
        tuple(confidence_thresholds),
        tuple(feature_profiles),
        tuple(cost_profiles),
    ))
    chosen = _diverse_combinations(full_grid, limit)

    specs: list[ChallengerSpec] = []
    for idx, vals in enumerate(chosen, start=1):
        bars, seed, xgb, hmm, conf, feat, cost = vals
        suffix = f"ch-{idx:03d}-b{bars}-s{seed}-{xgb}-h{hmm}-c{int(round(conf*100))}-{feat}-{cost}"
        model_id = f"gold-{safe_batch}-{suffix}" if safe_batch else f"gold-{suffix}"
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
                batch_id=safe_batch,
            )
        )
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
