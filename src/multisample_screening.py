"""Aggregate cheap AUC/generalization screening across chronological samples.

This module answers a different question from single-run screening: is the same
configuration consistently trainable and reasonably generalizable at several
historical cutoffs?  It is still only a pre-screen.  Strategy PF/DD, GOLDmicro
costs, walk-forward stability, shadow observation and Human Gate remain mandatory.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from statistics import median
from typing import Any, Iterable

from .candidate_screening import ScreeningThresholds, screen_candidate


@dataclass(frozen=True)
class MultiSampleSummary:
    base_model_id: str
    status: str
    sample_count: int
    trained_count: int
    pass_count: int
    pass_rate: float
    median_test_auc: float
    min_test_auc: float
    max_test_auc: float
    median_gap: float
    max_gap: float
    median_adjusted_score: float
    xgb_profile: str
    feature_profile: str
    cost_profile: str
    reasons: tuple[str, ...]
    samples: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["reasons"] = list(self.reasons)
        data["samples"] = list(self.samples)
        return data


def _group_rows(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get("base_model_id") or row.get("model_id") or "unknown")
        groups.setdefault(key, []).append(row)
    return groups


def summarize_multisample(
    rows: Iterable[dict[str, Any]],
    *,
    thresholds: ScreeningThresholds = ScreeningThresholds(),
    expected_samples: int,
    min_pass_rate: float = 0.80,
) -> list[MultiSampleSummary]:
    if expected_samples < 1:
        raise ValueError("expected_samples must be >= 1")
    if not (0.0 < min_pass_rate <= 1.0):
        raise ValueError("min_pass_rate must be in (0, 1]")

    required_passes = ceil(expected_samples * min_pass_rate)
    summaries: list[MultiSampleSummary] = []

    for base_model_id, group in _group_rows(rows).items():
        group = sorted(group, key=lambda r: int(r.get("sample_index") or 0))
        screened = [screen_candidate(row, thresholds) for row in group]
        trained = [row for row in group if row.get("success")]
        passed = [item for item in screened if item.status == "SCREEN_PASS"]
        aucs = [item.test_auc for item in screened if item.test_samples > 0]
        gaps = [item.generalization_gap for item in screened if item.test_samples > 0]
        adjusted = [item.adjusted_score for item in screened if item.test_samples > 0]

        reasons: list[str] = []
        if len(group) != expected_samples:
            reasons.append(f"sample records {len(group)} != expected {expected_samples}")
        if len(trained) != expected_samples:
            reasons.append(f"trained {len(trained)}/{expected_samples}")
        if len(passed) < required_passes:
            reasons.append(
                f"screen pass {len(passed)}/{expected_samples} < required {required_passes}/{expected_samples}"
            )

        first = group[0] if group else {}
        sample_records = tuple(
            {
                "model_id": row.get("model_id"),
                "sample_index": row.get("sample_index"),
                "sample_cutoff": row.get("sample_cutoff"),
                "success": bool(row.get("success")),
                "screen_status": item.status,
                "test_auc": item.test_auc,
                "generalization_gap": item.generalization_gap,
                "adjusted_score": item.adjusted_score,
                "xgb_path": row.get("xgb_path"),
                "hmm_path": row.get("hmm_path"),
                "training_data_path": row.get("training_data_path"),
                "split": row.get("split"),
                "data_fingerprint": row.get("data_fingerprint"),
                "reasons": list(item.reasons),
            }
            for row, item in zip(group, screened)
        )

        summaries.append(
            MultiSampleSummary(
                base_model_id=base_model_id,
                status="STABLE_SCREEN_PASS" if not reasons else "STABLE_SCREEN_REJECT",
                sample_count=len(group),
                trained_count=len(trained),
                pass_count=len(passed),
                pass_rate=(len(passed) / expected_samples),
                median_test_auc=median(aucs) if aucs else 0.0,
                min_test_auc=min(aucs) if aucs else 0.0,
                max_test_auc=max(aucs) if aucs else 0.0,
                median_gap=median(gaps) if gaps else 1.0,
                max_gap=max(gaps) if gaps else 1.0,
                median_adjusted_score=median(adjusted) if adjusted else -1.0,
                xgb_profile=str(first.get("xgb_profile", "")),
                feature_profile=str(first.get("feature_profile", "")),
                cost_profile=str(first.get("cost_profile", "")),
                reasons=tuple(reasons) if reasons else (
                    f"passed >= {required_passes}/{expected_samples} chronological AUC screens",
                ),
                samples=sample_records,
            )
        )

    return sorted(
        summaries,
        key=lambda x: (
            x.status == "STABLE_SCREEN_PASS",
            x.pass_rate,
            x.median_adjusted_score,
            x.median_test_auc,
            x.min_test_auc,
            -x.max_gap,
        ),
        reverse=True,
    )


def select_stable_shortlist(
    summaries: Iterable[MultiSampleSummary],
    *,
    top_k: int = 6,
) -> list[MultiSampleSummary]:
    """Choose stable configurations while preserving profile diversity."""
    passed = [x for x in summaries if x.status == "STABLE_SCREEN_PASS"]
    chosen: list[MultiSampleSummary] = []
    used_ids: set[str] = set()
    used_signatures: set[tuple[str, str, str]] = set()

    for item in passed:
        sig = (item.xgb_profile, item.feature_profile, item.cost_profile)
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
