"""Research-only screening for many GOLDmicro challenger training results.

This stage is deliberately *not* a PF/DD promotion gate.  It removes obviously
weak or overfit model candidates before expensive strategy/OOS/shadow validation.
Only later lifecycle stages may use strategy PF/DD for promotion eligibility.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Any


@dataclass(frozen=True)
class ScreeningThresholds:
    min_test_auc: float = 0.55
    max_generalization_gap: float = 0.12
    min_test_samples: int = 500


@dataclass(frozen=True)
class CandidateScreen:
    model_id: str
    status: str
    train_auc: float
    test_auc: float
    generalization_gap: float
    test_samples: int
    adjusted_score: float
    xgb_profile: str
    feature_profile: str
    cost_profile: str
    reasons: tuple[str, ...]
    source: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["reasons"] = list(self.reasons)
        return data


def screen_candidate(
    row: dict[str, Any],
    thresholds: ScreeningThresholds = ScreeningThresholds(),
) -> CandidateScreen:
    model_id = str(row.get("model_id", "unknown"))
    reasons: list[str] = []

    if not row.get("success"):
        reasons.append(str(row.get("error") or "training failed"))
        return CandidateScreen(
            model_id=model_id,
            status="SCREEN_REJECT",
            train_auc=0.0,
            test_auc=0.0,
            generalization_gap=1.0,
            test_samples=0,
            adjusted_score=-1.0,
            xgb_profile=str(row.get("xgb_profile", "")),
            feature_profile=str(row.get("feature_profile", "")),
            cost_profile=str(row.get("cost_profile", "")),
            reasons=tuple(reasons),
            source=row,
        )

    metrics = row.get("train_metrics") or {}
    train_auc = float(metrics.get("xgb_train_score") or 0.0)
    test_auc = float(metrics.get("xgb_test_score") or 0.0)
    test_samples = int(metrics.get("test_samples") or 0)
    gap = max(0.0, train_auc - test_auc)

    if test_auc < thresholds.min_test_auc:
        reasons.append(f"test AUC {test_auc:.4f} < {thresholds.min_test_auc:.4f}")
    if gap > thresholds.max_generalization_gap:
        reasons.append(
            f"train-test AUC gap {gap:.4f} > {thresholds.max_generalization_gap:.4f}"
        )
    if test_samples < thresholds.min_test_samples:
        reasons.append(f"test samples {test_samples} < {thresholds.min_test_samples}")

    # AUC is only a cheap first-stage screen. Penalize apparent overfit but do
    # not pretend this score is strategy PF, DD, or expected profitability.
    adjusted = test_auc - 0.50 * gap
    return CandidateScreen(
        model_id=model_id,
        status="SCREEN_PASS" if not reasons else "SCREEN_REJECT",
        train_auc=train_auc,
        test_auc=test_auc,
        generalization_gap=gap,
        test_samples=test_samples,
        adjusted_score=adjusted,
        xgb_profile=str(row.get("xgb_profile", "")),
        feature_profile=str(row.get("feature_profile", "")),
        cost_profile=str(row.get("cost_profile", "")),
        reasons=tuple(reasons) if reasons else ("training-quality screen passed",),
        source=row,
    )


def screen_many(
    rows: Iterable[dict[str, Any]],
    thresholds: ScreeningThresholds = ScreeningThresholds(),
) -> list[CandidateScreen]:
    screened = [screen_candidate(row, thresholds) for row in rows]
    return sorted(
        screened,
        key=lambda x: (
            x.status == "SCREEN_PASS",
            x.adjusted_score,
            x.test_auc,
            -x.generalization_gap,
        ),
        reverse=True,
    )


def select_diverse_shortlist(
    screened: Iterable[CandidateScreen],
    *,
    top_k: int = 6,
) -> list[CandidateScreen]:
    """Select strong candidates while preserving profile diversity.

    First pass takes at most one candidate for each
    (xgb_profile, feature_profile, cost_profile) signature, then fills any
    remaining slots by score.  This reduces the chance that one lucky profile
    monopolizes the shadow queue.
    """
    passed = [x for x in screened if x.status == "SCREEN_PASS"]
    chosen: list[CandidateScreen] = []
    used_ids: set[str] = set()
    used_signatures: set[tuple[str, str, str]] = set()

    for item in passed:
        sig = (item.xgb_profile, item.feature_profile, item.cost_profile)
        if sig in used_signatures:
            continue
        chosen.append(item)
        used_ids.add(item.model_id)
        used_signatures.add(sig)
        if len(chosen) >= top_k:
            return chosen

    for item in passed:
        if item.model_id in used_ids:
            continue
        chosen.append(item)
        if len(chosen) >= top_k:
            break
    return chosen
