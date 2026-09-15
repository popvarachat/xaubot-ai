"""Pure policy evaluator for GOLDmicro model lifecycle decisions.

This module is intentionally side-effect free.  It does not train models,
replace model files, place orders, or promote a challenger.  It turns measured
metrics into auditable lifecycle decisions that an orchestrator can act on.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable


@dataclass(frozen=True)
class LifecycleThresholds:
    pf_hard_min: float = 1.30
    pf_watch_min: float = 1.40
    dd_hard_max_pct: float = 10.0
    dd_protect_pct: float = 8.0
    min_trades_micro: int = 30
    min_trades_promotion: int = 100
    min_shadow_trades: int = 50
    min_shadow_days: int = 3
    required_stable_windows: int = 3
    max_execution_cost_points: float = 90.0
    target_execution_cost_points: float = 85.0
    max_skip_pct: float = 20.0
    challenger_pf_margin: float = 0.03
    challenger_dd_tolerance_pct: float = 0.50


@dataclass(frozen=True)
class ModelWindowMetrics:
    model_id: str
    trades: int
    profit_factor: float
    max_drawdown_pct: float
    expectancy: float
    skip_pct: float
    execution_cost_points: float
    stable_windows: int = 1
    shadow_trades: int = 0
    shadow_days: int = 0


@dataclass(frozen=True)
class LifecycleDecision:
    model_id: str
    state: str
    allow_new_entries: bool
    retrain_challenger: bool
    eligible_for_promotion: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["reasons"] = list(self.reasons)
        return data


def evaluate_champion(
    metrics: ModelWindowMetrics,
    thresholds: LifecycleThresholds = LifecycleThresholds(),
) -> LifecycleDecision:
    """Evaluate the active champion without mutating anything.

    HOLD is reserved for hard safety/economic failures.  PROTECT means trading
    can continue but a challenger review should be scheduled.  WATCH is an
    early warning.  HEALTHY means the measured window remains inside policy.
    Small samples never trigger promotion-like conclusions; they only produce
    an INSUFFICIENT_SAMPLE state unless a hard execution/DD gate is breached.
    """
    reasons: list[str] = []

    if metrics.execution_cost_points > thresholds.max_execution_cost_points:
        reasons.append(
            f"execution cost {metrics.execution_cost_points:.1f} > hard {thresholds.max_execution_cost_points:.1f}"
        )
        return LifecycleDecision(metrics.model_id, "HOLD", False, True, False, tuple(reasons))

    if metrics.max_drawdown_pct >= thresholds.dd_hard_max_pct:
        reasons.append(
            f"drawdown {metrics.max_drawdown_pct:.2f}% >= hard {thresholds.dd_hard_max_pct:.2f}%"
        )
        return LifecycleDecision(metrics.model_id, "HOLD", False, True, False, tuple(reasons))

    if metrics.trades < thresholds.min_trades_micro:
        reasons.append(
            f"only {metrics.trades} trades; need {thresholds.min_trades_micro} for rolling PF/DD review"
        )
        return LifecycleDecision(metrics.model_id, "INSUFFICIENT_SAMPLE", True, False, False, tuple(reasons))

    if metrics.profit_factor < thresholds.pf_hard_min:
        reasons.append(
            f"PF {metrics.profit_factor:.3f} < hard {thresholds.pf_hard_min:.3f}"
        )
        return LifecycleDecision(metrics.model_id, "PROTECT", True, True, False, tuple(reasons))

    if metrics.max_drawdown_pct >= thresholds.dd_protect_pct:
        reasons.append(
            f"drawdown {metrics.max_drawdown_pct:.2f}% >= protect {thresholds.dd_protect_pct:.2f}%"
        )
        return LifecycleDecision(metrics.model_id, "PROTECT", True, True, False, tuple(reasons))

    watch = False
    if metrics.profit_factor < thresholds.pf_watch_min:
        reasons.append(
            f"PF {metrics.profit_factor:.3f} below watch target {thresholds.pf_watch_min:.3f}"
        )
        watch = True
    if metrics.execution_cost_points > thresholds.target_execution_cost_points:
        reasons.append(
            f"execution cost {metrics.execution_cost_points:.1f} above target {thresholds.target_execution_cost_points:.1f}"
        )
        watch = True
    if metrics.skip_pct > thresholds.max_skip_pct:
        reasons.append(f"skip {metrics.skip_pct:.2f}% > {thresholds.max_skip_pct:.2f}%")
        watch = True

    if watch:
        return LifecycleDecision(metrics.model_id, "WATCH", True, True, False, tuple(reasons))

    return LifecycleDecision(metrics.model_id, "HEALTHY", True, False, False, ("all monitored gates pass",))


def evaluate_challenger(
    challenger: ModelWindowMetrics,
    champion: ModelWindowMetrics,
    thresholds: LifecycleThresholds = LifecycleThresholds(),
) -> LifecycleDecision:
    """Decide whether a challenger is eligible to request Human-Gate promotion.

    Eligibility is deliberately conservative.  This function never promotes a
    model.  It only marks a challenger as ELIGIBLE_FOR_HUMAN_GATE when sample,
    PF, DD, cost, stability, skip-rate and shadow requirements all pass.
    """
    reasons: list[str] = []

    hard_checks = [
        (challenger.trades >= thresholds.min_trades_promotion,
         f"promotion sample {challenger.trades} < {thresholds.min_trades_promotion}"),
        (challenger.profit_factor >= thresholds.pf_hard_min,
         f"PF {challenger.profit_factor:.3f} < {thresholds.pf_hard_min:.3f}"),
        (challenger.max_drawdown_pct < thresholds.dd_hard_max_pct,
         f"DD {challenger.max_drawdown_pct:.2f}% >= {thresholds.dd_hard_max_pct:.2f}%"),
        (challenger.skip_pct <= thresholds.max_skip_pct,
         f"skip {challenger.skip_pct:.2f}% > {thresholds.max_skip_pct:.2f}%"),
        (challenger.execution_cost_points <= thresholds.max_execution_cost_points,
         f"cost {challenger.execution_cost_points:.1f} > {thresholds.max_execution_cost_points:.1f}"),
        (challenger.stable_windows >= thresholds.required_stable_windows,
         f"stable windows {challenger.stable_windows} < {thresholds.required_stable_windows}"),
        (challenger.shadow_trades >= thresholds.min_shadow_trades,
         f"shadow trades {challenger.shadow_trades} < {thresholds.min_shadow_trades}"),
        (challenger.shadow_days >= thresholds.min_shadow_days,
         f"shadow days {challenger.shadow_days} < {thresholds.min_shadow_days}"),
    ]
    for passed, reason in hard_checks:
        if not passed:
            reasons.append(reason)

    relative_pf_floor = max(
        thresholds.pf_hard_min,
        champion.profit_factor - thresholds.challenger_pf_margin,
    )
    if challenger.profit_factor < relative_pf_floor:
        reasons.append(
            f"PF {challenger.profit_factor:.3f} < relative floor {relative_pf_floor:.3f}"
        )

    dd_ceiling = min(
        thresholds.dd_hard_max_pct,
        champion.max_drawdown_pct + thresholds.challenger_dd_tolerance_pct,
    )
    if challenger.max_drawdown_pct > dd_ceiling:
        reasons.append(
            f"DD {challenger.max_drawdown_pct:.2f}% > relative ceiling {dd_ceiling:.2f}%"
        )

    if challenger.expectancy <= 0:
        reasons.append(f"expectancy {challenger.expectancy:.4f} must be positive")

    if reasons:
        return LifecycleDecision(
            challenger.model_id,
            "CHALLENGER_REJECT",
            False,
            False,
            False,
            tuple(reasons),
        )

    return LifecycleDecision(
        challenger.model_id,
        "ELIGIBLE_FOR_HUMAN_GATE",
        False,
        False,
        True,
        ("all absolute, relative, stability and shadow gates pass",),
    )


def rank_challengers(
    challengers: Iterable[ModelWindowMetrics],
    champion: ModelWindowMetrics,
    thresholds: LifecycleThresholds = LifecycleThresholds(),
) -> list[tuple[ModelWindowMetrics, LifecycleDecision]]:
    """Evaluate many challengers in one batch and sort strongest first."""
    evaluated = [(m, evaluate_challenger(m, champion, thresholds)) for m in challengers]
    return sorted(
        evaluated,
        key=lambda item: (
            item[1].eligible_for_promotion,
            item[0].profit_factor,
            -item[0].max_drawdown_pct,
            item[0].expectancy,
        ),
        reverse=True,
    )
