"""Research-only GOLDmicro strategy OOS evaluator.

This gate consumes candidate artifacts already produced by the 24 x N training
matrix.  It evaluates only each candidate's held-out partition and never places
orders or changes active models.

Strategy proxy V1 preserves the project architecture: SMC creates entries,
XGBoost acts as confirmation/reversal assistance, causal HMM regime labels are
used as context, and broker-correct GOLDmicro sizing/costs are applied.  It is a
validation gate, not a claim of live profitability and not an exact replacement
for the legacy #24 simulator.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import floor
from pathlib import Path
from statistics import median
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl

from backtests.goldmicro_cost_model import BacktestCostConfig, GoldmicroCostModel
from backtests.ml_v2.ml_v2_model import TradingModelV2
from src.broker_profile import BrokerSymbolProfile
from src.goldmicro_causal_smc import GoldmicroCausalSMCAnalyzer


WIB = ZoneInfo("Asia/Jakarta")


@dataclass(frozen=True)
class StrategyOOSThresholds:
    initial_capital_thb: float = 20000.0
    risk_per_trade_percent: float = 1.0
    min_profit_factor: float = 1.30
    max_drawdown_percent: float = 10.0
    max_risk_skip_percent: float = 20.0
    min_trades: int = 30
    min_sample_pass_rate: float = 0.80
    max_holding_bars: int = 32
    cooldown_bars: int = 10
    strong_ml_reversal_confidence: float = 0.75


@dataclass(frozen=True)
class SampleStrategyResult:
    model_id: str
    sample_index: int
    trades: int
    wins: int
    losses: int
    net_pnl_thb: float
    profit_factor: float
    max_drawdown_percent: float
    expectancy_thb: float
    risk_skips: int
    risk_skip_percent: float
    model_blocks: int
    average_lot: float
    status: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["reasons"] = list(self.reasons)
        return data


def default_goldmicro_profile() -> BrokerSymbolProfile:
    """XM GOLDmicro profile calibrated to the inspected THB account."""
    return BrokerSymbolProfile(
        symbol="GOLDmicro",
        point=0.01,
        tick_size=0.01,
        tick_value=0.01,
        contract_size=1.0,
        volume_min=0.10,
        volume_max=100.0,
        volume_step=0.01,
        cash_per_price_unit_per_lot=33.29,
        cash_currency="THB",
    )


def cost_config_for_profile(name: str) -> BacktestCostConfig:
    if name == "normal":
        return BacktestCostConfig(spread_points=50.0, slippage_points=1.0)
    if name == "conservative":
        return BacktestCostConfig(spread_points=55.0, slippage_points=12.0)
    raise ValueError(f"unknown GOLDmicro cost profile: {name}")


def infer_cost_profile(model_id: str) -> str:
    stem = model_id.rsplit("-s", 1)[0]
    if stem.endswith("-normal"):
        return "normal"
    if stem.endswith("-conservative"):
        return "conservative"
    raise ValueError(f"cannot infer cost profile from model id: {model_id}")


def _strategy_lot_down(raw_lot: float) -> float:
    if raw_lot < 0.10 - 1e-12:
        return 0.0
    return round(floor((raw_lot + 1e-12) / 0.10) * 0.10, 10)


def size_with_execution_cost(
    *,
    profile: BrokerSymbolProfile,
    cost: BacktestCostConfig,
    balance_thb: float,
    entry_mid: float,
    stop_mid: float,
    risk_percent: float,
    regime_multiplier: float = 1.0,
) -> tuple[float, float, float]:
    """Return (lot, budget, modeled stop loss) with cost included in sizing."""
    budget = balance_thb * (risk_percent / 100.0) * max(0.0, regime_multiplier)
    if budget <= 0:
        return 0.0, budget, 0.0
    adverse_distance = (
        abs(entry_mid - stop_mid)
        + cost.spread_points * profile.point
        + 2.0 * cost.slippage_points * profile.point
    )
    loss_per_lot = abs(profile.cash_pnl_for_price_delta(adverse_distance, 1.0))
    if loss_per_lot <= 0:
        return 0.0, budget, 0.0
    lot = _strategy_lot_down(budget / loss_per_lot)
    if lot <= 0:
        return 0.0, budget, loss_per_lot * 0.10
    modeled = loss_per_lot * lot
    if modeled > budget + 1e-9:
        raise AssertionError("cost-adjusted strategy lot exceeds risk budget")
    return lot, budget, modeled


def _session(dt) -> tuple[str, bool]:
    if getattr(dt, "tzinfo", None) is None:
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))
    local = dt.astimezone(WIB)
    h = local.hour
    if 6 <= h < 15:
        return "Sydney-Tokyo", True
    if 15 <= h < 16:
        return "Tokyo-London Overlap", False
    if 16 <= h < 19:
        return "London Early", True
    if 19 <= h < 24:
        return "London-NY Overlap (Golden)", True
    if 0 <= h < 4:
        return "NY Session", True
    return "Off Hours", False


def _opposite(signal: str) -> str:
    return "SELL" if signal == "BUY" else "BUY"


def _simulate_exit(
    *,
    df: pl.DataFrame,
    entry_idx: int,
    direction: str,
    entry_mid: float,
    stop_mid: float,
    target_mid: float,
    lot: float,
    model: TradingModelV2,
    cost_model: GoldmicroCostModel,
    thresholds: StrategyOOSThresholds,
) -> tuple[int, float, float, str]:
    highs = df["high"].to_list()
    lows = df["low"].to_list()
    closes = df["close"].to_list()
    end = min(entry_idx + thresholds.max_holding_bars, len(df) - 1)

    for idx in range(entry_idx + 1, end + 1):
        high = float(highs[idx])
        low = float(lows[idx])

        if direction == "BUY":
            stop_hit = low <= stop_mid
            target_hit = high >= target_mid
        else:
            stop_hit = high >= stop_mid
            target_hit = low <= target_mid

        # Bar-level OHLC cannot reveal touch order.  If both levels are touched
        # in one bar, choose the adverse outcome rather than optimistic ordering.
        if stop_hit:
            exit_mid = stop_mid
            reason = "STOP_LOSS" if not target_hit else "AMBIGUOUS_BAR_STOP_FIRST"
            pnl = cost_model.pnl_from_mid(
                side=direction, entry_mid=entry_mid, exit_mid=exit_mid, lot_size=lot
            ).net_pnl
            return idx, exit_mid, pnl, reason
        if target_hit:
            exit_mid = target_mid
            pnl = cost_model.pnl_from_mid(
                side=direction, entry_mid=entry_mid, exit_mid=exit_mid, lot_size=lot
            ).net_pnl
            return idx, exit_mid, pnl, "TAKE_PROFIT"

        if (idx - entry_idx) % 4 == 0:
            pred = model.predict(df.slice(0, idx + 1), model.feature_names)
            if (
                pred.signal == _opposite(direction)
                and pred.confidence >= thresholds.strong_ml_reversal_confidence
            ):
                exit_mid = float(closes[idx])
                pnl = cost_model.pnl_from_mid(
                    side=direction, entry_mid=entry_mid, exit_mid=exit_mid, lot_size=lot
                ).net_pnl
                return idx, exit_mid, pnl, "ML_REVERSAL"

    exit_mid = float(closes[end])
    pnl = cost_model.pnl_from_mid(
        side=direction, entry_mid=entry_mid, exit_mid=exit_mid, lot_size=lot
    ).net_pnl
    return end, exit_mid, pnl, "TIMEOUT"


def evaluate_strategy_sample(
    sample: dict[str, Any],
    *,
    thresholds: StrategyOOSThresholds = StrategyOOSThresholds(),
) -> SampleStrategyResult:
    data_path = Path(str(sample["training_data_path"]))
    xgb_path = Path(str(sample["xgb_path"]))
    if not data_path.exists() or not xgb_path.exists():
        raise FileNotFoundError(f"candidate artifact missing for {sample.get('model_id')}")

    df = pl.read_parquet(data_path)
    split = sample.get("split") or {}
    start_idx = int(split.get("oos_start_index") or 0)
    if start_idx < 100 or start_idx >= len(df) - thresholds.max_holding_bars - 1:
        raise ValueError(f"invalid OOS start index for {sample.get('model_id')}: {start_idx}")

    model = TradingModelV2(model_path=str(xgb_path)).load()
    if not model.fitted:
        raise RuntimeError(f"candidate model failed to load: {xgb_path}")

    profile = default_goldmicro_profile()
    cost_name = infer_cost_profile(str(sample["model_id"]))
    cost_cfg = cost_config_for_profile(cost_name)
    cost_model = GoldmicroCostModel(profile, cost_cfg)
    smc = GoldmicroCausalSMCAnalyzer(swing_length=5)

    balance = thresholds.initial_capital_thb
    peak = balance
    max_dd = 0.0
    profits: list[float] = []
    lots: list[float] = []
    risk_skips = 0
    model_blocks = 0
    smc_candidates = 0
    last_exit = start_idx - thresholds.cooldown_bars
    times = df["time"].to_list()

    for idx in range(start_idx, len(df) - thresholds.max_holding_bars):
        if idx - last_exit < thresholds.cooldown_bars:
            continue
        current_time = times[idx]
        if hasattr(current_time, "weekday") and current_time.weekday() >= 5:
            continue
        _, can_trade = _session(current_time)
        if not can_trade:
            continue

        # All features were precomputed causally.  The slice prevents the signal
        # and prediction APIs from ever receiving rows after the decision time.
        prefix = df.slice(0, idx + 1)
        signal = smc.generate_signal(prefix)
        if signal is None:
            continue
        smc_candidates += 1

        pred = model.predict(prefix, model.feature_names)
        if (
            pred.signal == _opposite(signal.signal_type)
            and pred.confidence >= thresholds.strong_ml_reversal_confidence
        ):
            model_blocks += 1
            continue

        regime_name = "medium_volatility"
        if "regime_name" in df.columns:
            value = df["regime_name"][idx]
            if value:
                regime_name = str(value)
        regime_multiplier = 0.5 if regime_name == "high_volatility" else 1.0

        lot, _, _ = size_with_execution_cost(
            profile=profile,
            cost=cost_cfg,
            balance_thb=balance,
            entry_mid=float(signal.entry_price),
            stop_mid=float(signal.stop_loss),
            risk_percent=thresholds.risk_per_trade_percent,
            regime_multiplier=regime_multiplier,
        )
        if lot <= 0:
            risk_skips += 1
            continue

        exit_idx, _, pnl, _ = _simulate_exit(
            df=df,
            entry_idx=idx,
            direction=signal.signal_type,
            entry_mid=float(signal.entry_price),
            stop_mid=float(signal.stop_loss),
            target_mid=float(signal.take_profit),
            lot=lot,
            model=model,
            cost_model=cost_model,
            thresholds=thresholds,
        )
        profits.append(float(pnl))
        lots.append(lot)
        balance += float(pnl)
        peak = max(peak, balance)
        dd = (peak - balance) / peak * 100.0 if peak > 0 else 100.0
        max_dd = max(max_dd, dd)
        last_exit = exit_idx

    wins = sum(1 for x in profits if x > 0)
    losses = sum(1 for x in profits if x < 0)
    gross_win = sum(x for x in profits if x > 0)
    gross_loss = abs(sum(x for x in profits if x < 0))
    pf = gross_win / gross_loss if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)
    expectancy = sum(profits) / len(profits) if profits else 0.0
    risk_skip_pct = risk_skips / smc_candidates * 100.0 if smc_candidates else 100.0

    reasons: list[str] = []
    if len(profits) < thresholds.min_trades:
        reasons.append(f"trades {len(profits)} < {thresholds.min_trades}")
    if pf < thresholds.min_profit_factor:
        reasons.append(f"PF {pf:.3f} < {thresholds.min_profit_factor:.2f}")
    if max_dd > thresholds.max_drawdown_percent:
        reasons.append(f"DD {max_dd:.2f}% > {thresholds.max_drawdown_percent:.2f}%")
    if risk_skip_pct > thresholds.max_risk_skip_percent:
        reasons.append(
            f"risk skip {risk_skip_pct:.2f}% > {thresholds.max_risk_skip_percent:.2f}%"
        )
    if expectancy <= 0:
        reasons.append(f"expectancy {expectancy:.2f} THB <= 0")

    return SampleStrategyResult(
        model_id=str(sample.get("model_id")),
        sample_index=int(sample.get("sample_index") or 0),
        trades=len(profits),
        wins=wins,
        losses=losses,
        net_pnl_thb=sum(profits),
        profit_factor=pf,
        max_drawdown_percent=max_dd,
        expectancy_thb=expectancy,
        risk_skips=risk_skips,
        risk_skip_percent=risk_skip_pct,
        model_blocks=model_blocks,
        average_lot=(sum(lots) / len(lots)) if lots else 0.0,
        status="STRATEGY_SAMPLE_PASS" if not reasons else "STRATEGY_SAMPLE_REJECT",
        reasons=tuple(reasons) if reasons else ("PF/DD/cost sample gate passed",),
    )


def summarize_configuration(
    base_model_id: str,
    sample_results: list[SampleStrategyResult],
    *,
    thresholds: StrategyOOSThresholds = StrategyOOSThresholds(),
) -> dict[str, Any]:
    expected = len(sample_results)
    passed = [x for x in sample_results if x.status == "STRATEGY_SAMPLE_PASS"]
    pass_rate = len(passed) / expected if expected else 0.0
    pfs = [x.profit_factor for x in sample_results]
    dds = [x.max_drawdown_percent for x in sample_results]
    expects = [x.expectancy_thb for x in sample_results]
    reasons: list[str] = []
    if pass_rate < thresholds.min_sample_pass_rate:
        reasons.append(
            f"strategy sample pass rate {pass_rate:.0%} < {thresholds.min_sample_pass_rate:.0%}"
        )
    if dds and max(dds) > thresholds.max_drawdown_percent:
        reasons.append(
            f"hard DD breach: worst sample {max(dds):.2f}% > {thresholds.max_drawdown_percent:.2f}%"
        )
    if pfs and median(pfs) < thresholds.min_profit_factor:
        reasons.append(
            f"median PF {median(pfs):.3f} < {thresholds.min_profit_factor:.2f}"
        )
    if expects and median(expects) <= 0:
        reasons.append(f"median expectancy {median(expects):.2f} THB <= 0")
    return {
        "base_model_id": base_model_id,
        "status": "STRATEGY_OOS_PASS" if not reasons else "STRATEGY_OOS_REJECT",
        "sample_count": expected,
        "sample_pass_count": len(passed),
        "sample_pass_rate": pass_rate,
        "median_pf": median(pfs) if pfs else 0.0,
        "min_pf": min(pfs) if pfs else 0.0,
        "worst_dd_percent": max(dds) if dds else 100.0,
        "median_expectancy_thb": median(expects) if expects else 0.0,
        "reasons": reasons if reasons else ["multi-sample PF/DD/cost OOS gate passed"],
        "samples": [x.to_dict() for x in sample_results],
    }
