"""GOLDmicro position sizing built on broker-reported symbol specifications.

This module is intentionally separate from the legacy RiskEngine while the
fork is being validated. It does not place orders and does not enable live
trading. The goal is to produce a conservative, auditable lot size before the
legacy engine is migrated.
"""

from __future__ import annotations

from dataclasses import dataclass

from .broker_profile import BrokerSymbolProfile


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
) -> GoldmicroSizingResult:
    """Calculate a conservative broker-valid GOLDmicro volume.

    Rules:
    - Risk is capped by the configured risk-per-trade percentage.
    - Half-Kelly can reduce risk but never increase the configured cap.
    - Volume is always floored to ``volume_step`` so rounding cannot increase
      the intended risk.
    - If the calculated volume is below ``volume_min`` the trade is rejected;
      it is never rounded up to the broker minimum.
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
    lot = profile.normalize_volume_down(raw_lot)

    if lot <= 0:
        min_lot_risk = profile.cash_risk_per_lot(entry_price, stop_price) * profile.volume_min
        return GoldmicroSizingResult(
            approved=False,
            lot_size=0.0,
            requested_risk_amount=requested_risk,
            actual_risk_amount=0.0,
            actual_risk_percent=0.0,
            raw_lot=raw_lot,
            reason=(
                f"Broker minimum {profile.volume_min:g} lot would risk "
                f"{min_lot_risk:.2f}, above the allowed risk budget "
                f"{requested_risk:.2f}"
            ),
        )

    actual_risk = profile.cash_risk_per_lot(entry_price, stop_price) * lot
    actual_pct = actual_risk / account_balance * 100.0

    # Defensive invariant: broker-step normalization must never increase risk
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
