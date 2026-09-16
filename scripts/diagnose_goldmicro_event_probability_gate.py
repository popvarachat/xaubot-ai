"""Diagnose GOLDmicro event-model score scale before any strategy-threshold change.

This script intentionally does NOT calculate PF/DD, optimize a threshold, or
promote a model.  It inspects held-out event probabilities, observed event base
rates and broker-cost break-even probabilities so that any later gate policy can
be declared from model semantics/economics rather than chosen after seeing
strategy returns.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtests.goldmicro_cost_model import GoldmicroCostModel
from src.goldmicro_event_strategy_oos import _event_probabilities
from src.goldmicro_strategy_oos import _session, cost_config_for_profile, default_goldmicro_profile

COST_PROFILES = ("normal", "conservative")


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {k: None for k in ("min", "p10", "p25", "p50", "p75", "p90", "p95", "max")}
    return {
        "min": float(np.min(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p25": float(np.quantile(values, 0.25)),
        "p50": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def _eligible_session_mask(times: list) -> np.ndarray:
    keep = []
    for value in times:
        if hasattr(value, "weekday") and value.weekday() >= 5:
            keep.append(False)
            continue
        _, can_trade = _session(value)
        keep.append(bool(can_trade))
    return np.asarray(keep, dtype=bool)


def _break_even_probabilities(events: pl.DataFrame, cost_profile: str) -> np.ndarray:
    profile = default_goldmicro_profile()
    cost_model = GoldmicroCostModel(profile, cost_config_for_profile(cost_profile))
    values: list[float] = []
    for row in events.iter_rows(named=True):
        side = str(row["event_direction"])
        entry = float(row["event_entry"])
        stop = float(row["event_stop_loss"])
        target = float(row["event_take_profit"])
        win = float(cost_model.pnl_from_mid(side=side, entry_mid=entry, exit_mid=target, lot_size=1.0).net_pnl)
        loss = float(cost_model.pnl_from_mid(side=side, entry_mid=entry, exit_mid=stop, lot_size=1.0).net_pnl)
        loss_abs = abs(min(loss, 0.0))
        if win <= 0.0 or loss_abs <= 0.0:
            values.append(1.0)
            continue
        values.append(loss_abs / (win + loss_abs))
    return np.asarray(values, dtype=float)


def run(queue_path: Path) -> Path:
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    configs = queue.get("configurations") or []
    if not configs:
        raise ValueError("event strategy queue is empty")

    rows: list[dict] = []
    for config in configs:
        base_id = str(config.get("base_model_id") or "unknown")
        for sample in config.get("samples") or []:
            event_path = Path(str(sample.get("training_data_path") or ""))
            model_path = Path(str(sample.get("xgb_path") or ""))
            split = sample.get("split") or {}
            test_first = int(split.get("test_first_event_index") or -1)
            if not event_path.exists() or not model_path.exists() or test_first < 0:
                raise ValueError(f"missing event artifact/split for {sample.get('model_id')}")

            events = pl.read_parquet(event_path)
            oos = events.filter(pl.col("event_index") >= test_first).sort("time")
            booster = xgb.Booster()
            booster.load_model(model_path)
            probs = np.asarray(_event_probabilities(oos, booster), dtype=float)
            y = np.asarray(oos["event_target"].to_numpy(), dtype=int)
            mask = _eligible_session_mask(oos["time"].to_list())
            session_oos = oos.filter(pl.Series(mask))
            session_probs = probs[mask]
            session_y = y[mask]

            record = {
                "base_model_id": base_id,
                "model_id": sample.get("model_id"),
                "sample_index": sample.get("sample_index"),
                "oos_events": int(len(oos)),
                "session_events": int(len(session_oos)),
                "oos_positive_rate": float(np.mean(y)) if len(y) else None,
                "session_positive_rate": float(np.mean(session_y)) if len(session_y) else None,
                "score_quantiles": _quantiles(session_probs),
                "count_p_ge_050": int(np.sum(session_probs >= 0.50)),
                "fraction_p_ge_050": float(np.mean(session_probs >= 0.50)) if len(session_probs) else 0.0,
                "brier": float(brier_score_loss(session_y, session_probs)) if len(session_y) else None,
                "auc": (
                    float(roc_auc_score(session_y, session_probs))
                    if len(session_y) and len(np.unique(session_y)) >= 2
                    else None
                ),
                "cost_profiles": {},
            }
            for cost_name in COST_PROFILES:
                p_be = _break_even_probabilities(session_oos, cost_name)
                clears = session_probs >= p_be if len(session_probs) else np.asarray([], dtype=bool)
                record["cost_profiles"][cost_name] = {
                    "break_even_probability_quantiles": _quantiles(p_be),
                    "count_score_ge_break_even": int(np.sum(clears)),
                    "fraction_score_ge_break_even": float(np.mean(clears)) if len(clears) else 0.0,
                }
            rows.append(record)

    output = {
        "generated_at": datetime.now().isoformat(),
        "batch_id": queue.get("batch_id"),
        "state": "EVENT_PROBABILITY_DIAGNOSTIC_ONLY",
        "strategy_pf_dd_evaluated": False,
        "threshold_optimized": False,
        "promotion_performed": False,
        "interpretation": (
            "Use this report only to understand event-score scale, base rate and broker-cost break-even "
            "probabilities. Do not select a probability threshold from PF/DD because PF/DD is not computed here."
        ),
        "samples": rows,
    }
    out = queue_path.parent / "event_probability_gate_diagnostic.json"
    out.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")

    print("=== GOLDmicro Event Probability Gate Diagnostic ===")
    print(f"Queue          : {queue_path}")
    print(f"Samples        : {len(rows)}")
    print("PF/DD          : NOT EVALUATED")
    print("Threshold tune : DISABLED")
    print("Promotion      : DISABLED")
    for item in rows:
        q = item["score_quantiles"]
        normal = item["cost_profiles"]["normal"]
        conservative = item["cost_profiles"]["conservative"]
        print(
            f"s{int(item['sample_index'] or 0):02d} "
            f"events={item['session_events']} pos={item['session_positive_rate']:.3f} "
            f"p50={q['p50']:.3f} p90={q['p90']:.3f} max={q['max']:.3f} "
            f">=0.50={item['count_p_ge_050']} "
            f">=BE(n/c)={normal['count_score_ge_break_even']}/{conservative['count_score_ge_break_even']} "
            f"{item['base_model_id']}"
        )
    print(f"Report         : {out}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True, type=Path)
    args = ap.parse_args()
    run(args.queue)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
