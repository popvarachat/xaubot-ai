"""GOLDmicro position sizing built on broker-reported symbol specifications.

This module is intentionally separate from the legacy RiskEngine while the
fork is being validated. It does not place orders and does not enable live
trading. The goal is to produce a conservative, auditable lot size before the
legacy engine is migrated.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .broker_profile import BrokerSymbolProfile


DEFAULT_STRATEGY_MIN_LOT = 0.10
DEFAULT_STRATEGY_LOT_STEP = 0.10


@dataclass(frozen=True)
class GoldmicroSizingResult:
    approved: bool
    lot_size: float
    requested_risk_amount: float
    actual_risk_amount: float
    actual_risk_percent: float
    raw_lot: float
    reason: str


def half_kelly_fraction(win_rate: float, reward_risk_ratio: float) -> float:
    """Return a non-negative half-Kelly fraction capped at 12.5%.

    The project will normally cap this again by ``risk_per_trade_percent``;
    the Kelly value is never allowed to increase risk above the configured
    account-risk ceiling.
    """
    if not 0 <= win_rate <= 1:
        raise ValueError("win_rate must be between 0 and 1")
    if reward_risk_ratio <= 0:
        return 0.0
    q = 1.0 - win_rate
    full_kelly = (win_rate * reward_risk_ratio - q) / reward_risk_ratio
    return max(0.0, min(full_kelly * 0.5, 0.125))


def _is_step_aligned(value: float, step: float, *, tol: float = 1e-9) -> bool:
    ratio = value / step
    return abs(ratio - round(ratio)) <= tol


def _normalize_strategy_volume_down(
    *,
    profile: BrokerSymbolProfile,
    raw_lot: float,
    strategy_min_lot: float,
    strategy_lot_step: float,
) -> float:
    """Floor to the strategy lot ladder while remaining broker-valid.

    GOLDmicro research intentionally uses a coarser 0.10-lot ladder even when
    the broker advertises a finer 0.01 volume step. The strategy ladder is a
    model constraint, not a claim about broker capability.
    """
    if strategy_min_lot <= 0 or strategy_lot_step <= 0:
        raise ValueError("strategy_min_lot and strategy_lot_step must be positive")
    if strategy_min_lot < profile.volume_min - 1e-9:
        raise ValueError("strategy_min_lot cannot be below broker volume_min")
    if strategy_lot_step < profile.volume_step - 1e-9:
        raise ValueError("strategy_lot_step cannot be below broker volume_step")
    if not _is_step_aligned(strategy_min_lot, profile.volume_step):
        raise ValueError("strategy_min_lot must align to broker volume_step")
    if not _is_step_aligned(strategy_lot_step, profile.volume_step):
        raise ValueError("strategy_lot_step must align to broker volume_step")

    effective_min = max(profile.volume_min, strategy_min_lot)
    if raw_lot < effective_min - 1e-12:
        return 0.0

    capped = min(raw_lot, profile.volume_max)
    steps = math.floor((capped + 1e-12) / strategy_lot_step)
    lot = round(steps * strategy_lot_step, 10)
    if lot < effective_min - 1e-12:
        return 0.0

    # Final broker-grid validation/floor. With a 0.10 strategy step and 0.01
    # broker step this should be unchanged, but keeping the broker check makes
    # the invariant explicit.
    broker_lot = profile.normalize_volume_down(lot)
    if broker_lot <= 0:
        return 0.0
    return round(broker_lot, 10)


def size_goldmicro_position(
    *,
    profile: BrokerSymbolProfile,
    account_balance: float,
    entry_price: float,
    stop_price: float,
    risk_per_trade_percent: float,
    win_rate: float = 0.5,
    reward_risk_ratio: float = 2.0,
    regime_multiplier: float = 1.0,
    strategy_min_lot: float = DEFAULT_STRATEGY_MIN_LOT,
    strategy_lot_step: float = DEFAULT_STRATEGY_LOT_STEP,
) -> GoldmicroSizingResult:
    """Calculate a conservative broker-valid GOLDmicro volume.

    Research sizing rules:
    - Strategy lot ladder defaults to 0.10, 0.20, 0.30, ... lots.
    - Broker metadata stays canonical; the broker may support a finer step.
    - Risk is capped by the configured risk-per-trade percentage.
    - Half-Kelly can reduce risk but never increase the configured cap.
    - Volume is always floored, never rounded upward.
    - If 0.10 lot exceeds the risk budget, the trade is rejected.
    """
    profile.validate()
    if account_balance <= 0:
        raise ValueError("account_balance must be positive")
    if risk_per_trade_percent <= 0:
        raise ValueError("risk_per_trade_percent must be positive")
    if regime_multiplier < 0:
        raise ValueError("regime_multiplier cannot be negative")

    risk_cap_fraction = risk_per_trade_percent / 100.0
    kelly = half_kelly_fraction(win_rate, reward_risk_ratio)
    effective_fraction = min(risk_cap_fraction, kelly * regime_multiplier)

    if effective_fraction <= 0:
        return GoldmicroSizingResult(
            approved=False,
            lot_size=0.0,
            requested_risk_amount=0.0,
            actual_risk_amount=0.0,
            actual_risk_percent=0.0,
            raw_lot=0.0,
            reason="No positive statistical risk budget (Kelly <= 0)",
        )

    requested_risk = account_balance * effective_fraction
    raw_lot = profile.raw_lot_for_risk(requested_risk, entry_price, stop_price)
    lot = _normalize_strategy_volume_down(
        profile=profile,
        raw_lot=raw_lot,
        strategy_min_lot=strategy_min_lot,
        strategy_lot_step=strategy_lot_step,
    )

    if lot <= 0:
        effective_min = max(profile.volume_min, strategy_min_lot)
        min_lot_risk = profile.cash_risk_per_lot(entry_price, stop_price) * effective_min
        return GoldmicroSizingResult(
            approved=False,
            lot_size=0.0,
            requested_risk_amount=requested_risk,
            actual_risk_amount=0.0,
            actual_risk_percent=0.0,
            raw_lot=raw_lot,
            reason=(
                f"Strategy minimum {effective_min:g} lot would risk "
                f"{min_lot_risk:.2f}, above the allowed risk budget "
                f"{requested_risk:.2f}"
            ),
        )

    actual_risk = profile.cash_risk_per_lot(entry_price, stop_price) * lot
    actual_pct = actual_risk / account_balance * 100.0

    # Defensive invariant: strategy/broker flooring must never increase risk
    # above the requested monetary budget (small tolerance for floating point).
    if actual_risk > requested_risk + 1e-9:
        return GoldmicroSizingResult(
            approved=False,
            lot_size=0.0,
            requested_risk_amount=requested_risk,
            actual_risk_amount=actual_risk,
            actual_risk_percent=actual_pct,
            raw_lot=raw_lot,
            reason="Normalized lot unexpectedly exceeds risk budget",
        )

    return GoldmicroSizingResult(
        approved=True,
        lot_size=lot,
        requested_risk_amount=requested_risk,
        actual_risk_amount=actual_risk,
        actual_risk_percent=actual_pct,
        raw_lot=raw_lot,
        reason="OK",
    )
