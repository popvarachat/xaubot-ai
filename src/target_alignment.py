"""Aggregate GOLDmicro target-horizon research without relaxing quality gates."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import median
from typing import Any, Iterable

from src.candidate_screening import ScreeningThresholds
from src.multisample_screening import MultiSampleSummary, summarize_multisample


@dataclass(frozen=True)
class HorizonSummary:
    target_lookahead_bars: int
    status: str
    configuration_count: int
    stable_pass_count: int
    stable_pass_rate: float
    median_of_config_median_auc: float
    max_config_median_auc: float
    max_single_sample_auc: float
    median_of_config_min_auc: float
    best_base_model_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _horizon_by_base(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for row in rows:
        base_id = str(row.get("base_model_id") or row.get("model_id") or "unknown")
        horizon = int(row.get("target_lookahead_bars") or 1)
        previous = mapping.get(base_id)
        if previous is not None and previous != horizon:
            raise ValueError(f"base_model_id {base_id} mixes target horizons")
        mapping[base_id] = horizon
    return mapping


def summarize_target_alignment(
    rows: Iterable[dict[str, Any]],
    *,
    expected_samples: int,
    thresholds: ScreeningThresholds = ScreeningThresholds(),
    min_pass_rate: float = 0.80,
) -> tuple[list[MultiSampleSummary], list[HorizonSummary]]:
    rows = list(rows)
    by_base = _horizon_by_base(rows)
    config_summaries = summarize_multisample(
        rows,
        thresholds=thresholds,
        expected_samples=expected_samples,
        min_pass_rate=min_pass_rate,
    )

    horizons: dict[int, list[MultiSampleSummary]] = {}
    for item in config_summaries:
        horizon = by_base.get(item.base_model_id, 1)
        horizons.setdefault(horizon, []).append(item)

    horizon_summaries: list[HorizonSummary] = []
    for horizon, configs in sorted(horizons.items()):
        stable = [c for c in configs if c.status == "STABLE_SCREEN_PASS"]
        best = max(
            configs,
            key=lambda c: (c.median_test_auc, c.min_test_auc, -c.max_gap),
        )
        horizon_summaries.append(
            HorizonSummary(
                target_lookahead_bars=horizon,
                status="HORIZON_GATE_PASS" if stable else "HORIZON_GATE_REJECT",
                configuration_count=len(configs),
                stable_pass_count=len(stable),
                stable_pass_rate=(len(stable) / len(configs)) if configs else 0.0,
                median_of_config_median_auc=median(c.median_test_auc for c in configs),
                max_config_median_auc=max(c.median_test_auc for c in configs),
                max_single_sample_auc=max(c.max_test_auc for c in configs),
                median_of_config_min_auc=median(c.min_test_auc for c in configs),
                best_base_model_id=best.base_model_id,
            )
        )

    return config_summaries, horizon_summaries


def select_target_shortlist(
    config_summaries: Iterable[MultiSampleSummary],
    *,
    horizon_by_base: dict[str, int],
    top_k: int = 6,
) -> list[MultiSampleSummary]:
    """Select only hard-gate-passing configs while preserving horizon/profile diversity."""
    passed = [c for c in config_summaries if c.status == "STABLE_SCREEN_PASS"]
    passed.sort(
        key=lambda c: (
            c.pass_rate,
            c.median_adjusted_score,
            c.median_test_auc,
            c.min_test_auc,
            -c.max_gap,
        ),
        reverse=True,
    )

    chosen: list[MultiSampleSummary] = []
    used_ids: set[str] = set()
    used_signatures: set[tuple[int, str, str]] = set()
    for item in passed:
        sig = (
            int(horizon_by_base.get(item.base_model_id, 1)),
            item.xgb_profile,
            item.feature_profile,
        )
        if sig in used_signatures:
            continue
        chosen.append(item)
        used_ids.add(item.base_model_id)
        used_signatures.add(sig)
        if len(chosen) >= top_k:
            return chosen

    for item in passed:
        if item.base_model_id in used_ids:
            continue
        chosen.append(item)
        if len(chosen) >= top_k:
            break
    return chosen
