"""Diagnose calibrated event probabilities specifically in the economic tail.

This diagnostic does not run PF/DD, tune thresholds, or promote models. It checks
whether events whose calibrated success probability clears their broker-cost
break-even probability actually realize enough TP-before-SL successes on untouched
OOS probes to support the economic-gate assumption.

The five chronological probes overlap in market time; results are therefore
probe-weighted diagnostics, not independent trials.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import polars as pl
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtests.goldmicro_cost_model import GoldmicroCostModel
from src.goldmicro_event_strategy_oos import (
    _break_even_probability,
    _calibrated_event_probabilities,
)
from src.goldmicro_strategy_oos import _session, cost_config_for_profile, default_goldmicro_profile

COST_PROFILES = ("normal", "conservative")
MARGIN_BINS = (
    (float("-inf"), 0.0, "<=0"),
    (0.0, 0.02, "0-2pp"),
    (0.02, 0.05, "2-5pp"),
    (0.05, 0.10, "5-10pp"),
    (0.10, float("inf"), ">10pp"),
)


def _session_mask(times: list) -> np.ndarray:
    values: list[bool] = []
    for value in times:
        if hasattr(value, "weekday") and value.weekday() >= 5:
            values.append(False)
            continue
        _, can_trade = _session(value)
        values.append(bool(can_trade))
    return np.asarray(values, dtype=bool)


def _break_even_array(events: pl.DataFrame, cost_profile: str) -> np.ndarray:
    profile = default_goldmicro_profile()
    cost_model = GoldmicroCostModel(profile, cost_config_for_profile(cost_profile))
    values: list[float] = []
    for row in events.iter_rows(named=True):
        values.append(
            _break_even_probability(
                direction=str(row["event_direction"]),
                entry_mid=float(row["event_entry"]),
                stop_mid=float(row["event_stop_loss"]),
                target_mid=float(row["event_take_profit"]),
                cost_model=cost_model,
            )
        )
    return np.asarray(values, dtype=float)


def _bucket_rows(prob: np.ndarray, p_be: np.ndarray, y: np.ndarray) -> list[dict]:
    margin = prob - p_be
    rows: list[dict] = []
    for lo, hi, name in MARGIN_BINS:
        if lo == float("-inf"):
            mask = margin <= hi
        elif hi == float("inf"):
            mask = margin > lo
        else:
            mask = (margin > lo) & (margin <= hi)
        count = int(np.sum(mask))
        if count:
            rows.append({
                "bin": name,
                "count": count,
                "mean_probability": float(np.mean(prob[mask])),
                "mean_break_even_probability": float(np.mean(p_be[mask])),
                "observed_success_rate": float(np.mean(y[mask])),
                "predicted_minus_break_even": float(np.mean(prob[mask] - p_be[mask])),
                "observed_minus_break_even": float(np.mean(y[mask]) - np.mean(p_be[mask])),
                "calibration_gap_observed_minus_predicted": float(np.mean(y[mask]) - np.mean(prob[mask])),
            })
        else:
            rows.append({
                "bin": name,
                "count": 0,
                "mean_probability": None,
                "mean_break_even_probability": None,
                "observed_success_rate": None,
                "predicted_minus_break_even": None,
                "observed_minus_break_even": None,
                "calibration_gap_observed_minus_predicted": None,
            })
    return rows


def run(queue_path: Path) -> Path:
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    configs = queue.get("configurations") or []
    if not configs:
        raise ValueError("event strategy queue is empty")

    samples: list[dict] = []
    aggregate: dict[tuple[str, str], dict[str, list[float] | list[int]]] = {}

    for config in configs:
        base_id = str(config.get("base_model_id") or "unknown")
        for sample in config.get("samples") or []:
            event_path = Path(str(sample.get("training_data_path") or ""))
            model_path = Path(str(sample.get("xgb_path") or ""))
            calibration_path = Path(str(sample.get("calibration_path") or ""))
            split = sample.get("split") or {}
            test_first = int(split.get("test_first_event_index") or -1)
            if not event_path.exists() or not model_path.exists() or not calibration_path.exists() or test_first < 0:
                raise ValueError(f"missing event artifact/split for {sample.get('model_id')}")

            events = pl.read_parquet(event_path)
            oos = events.filter(pl.col("event_index") >= test_first).sort("time")
            booster = xgb.Booster()
            booster.load_model(model_path)
            prob_all = np.asarray(
                _calibrated_event_probabilities(oos, booster, calibration_path),
                dtype=float,
            )
            y_all = np.asarray(oos["event_target"].to_numpy(), dtype=int)
            mask = _session_mask(oos["time"].to_list())
            session_oos = oos.filter(pl.Series(mask))
            prob = prob_all[mask]
            y = y_all[mask]

            for cost_profile in COST_PROFILES:
                p_be = _break_even_array(session_oos, cost_profile)
                selected = prob >= p_be
                count = int(np.sum(selected))
                record = {
                    "base_model_id": base_id,
                    "model_id": sample.get("model_id"),
                    "sample_index": int(sample.get("sample_index") or 0),
                    "cost_profile": cost_profile,
                    "session_events": int(len(session_oos)),
                    "selected_events": count,
                    "selected_fraction": float(np.mean(selected)) if len(selected) else 0.0,
                    "selected_mean_probability": float(np.mean(prob[selected])) if count else None,
                    "selected_mean_break_even_probability": float(np.mean(p_be[selected])) if count else None,
                    "selected_observed_success_rate": float(np.mean(y[selected])) if count else None,
                    "selected_predicted_minus_break_even": (
                        float(np.mean(prob[selected] - p_be[selected])) if count else None
                    ),
                    "selected_observed_minus_break_even": (
                        float(np.mean(y[selected]) - np.mean(p_be[selected])) if count else None
                    ),
                    "selected_calibration_gap_observed_minus_predicted": (
                        float(np.mean(y[selected]) - np.mean(prob[selected])) if count else None
                    ),
                    "margin_bins": _bucket_rows(prob, p_be, y),
                }
                samples.append(record)

                key = (base_id, cost_profile)
                slot = aggregate.setdefault(key, {
                    "selected_events": [],
                    "observed_minus_be": [],
                    "calibration_gap": [],
                    "observed_success": [],
                    "mean_probability": [],
                    "mean_break_even": [],
                })
                slot["selected_events"].append(count)
                if count:
                    slot["observed_minus_be"].append(record["selected_observed_minus_break_even"])
                    slot["calibration_gap"].append(record["selected_calibration_gap_observed_minus_predicted"])
                    slot["observed_success"].append(record["selected_observed_success_rate"])
                    slot["mean_probability"].append(record["selected_mean_probability"])
                    slot["mean_break_even"].append(record["selected_mean_break_even_probability"])

    summaries: list[dict] = []
    for (base_id, cost_profile), slot in aggregate.items():
        selected_counts = np.asarray(slot["selected_events"], dtype=float)
        obs_minus = np.asarray(slot["observed_minus_be"], dtype=float)
        cal_gap = np.asarray(slot["calibration_gap"], dtype=float)
        observed = np.asarray(slot["observed_success"], dtype=float)
        mean_p = np.asarray(slot["mean_probability"], dtype=float)
        mean_be = np.asarray(slot["mean_break_even"], dtype=float)
        summaries.append({
            "base_model_id": base_id,
            "cost_profile": cost_profile,
            "probe_count": int(len(selected_counts)),
            "probes_with_selected_events": int(np.sum(selected_counts > 0)),
            "selected_events_median": float(np.median(selected_counts)) if len(selected_counts) else 0.0,
            "selected_events_min": int(np.min(selected_counts)) if len(selected_counts) else 0,
            "selected_events_max": int(np.max(selected_counts)) if len(selected_counts) else 0,
            "median_observed_success_rate": float(np.median(observed)) if len(observed) else None,
            "median_mean_probability": float(np.median(mean_p)) if len(mean_p) else None,
            "median_mean_break_even_probability": float(np.median(mean_be)) if len(mean_be) else None,
            "median_observed_minus_break_even": float(np.median(obs_minus)) if len(obs_minus) else None,
            "median_calibration_gap_observed_minus_predicted": float(np.median(cal_gap)) if len(cal_gap) else None,
            "probes_observed_above_break_even": int(np.sum(obs_minus > 0)) if len(obs_minus) else 0,
        })

    summaries.sort(key=lambda x: (x["base_model_id"], x["cost_profile"]))
    output = {
        "batch_id": queue.get("batch_id"),
        "state": "EVENT_ECONOMIC_TAIL_CALIBRATION_DIAGNOSTIC_ONLY",
        "pf_dd_evaluated": False,
        "threshold_tuned": False,
        "promotion_performed": False,
        "overlap_warning": "Chronological probes overlap; do not treat them as independent trials.",
        "samples": samples,
        "summaries": summaries,
    }
    out = queue_path.parent / "event_economic_tail_calibration_diagnostic.json"
    out.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")

    print("=== GOLDmicro Event Economic-Tail Calibration Diagnostic ===")
    print(f"Queue          : {queue_path}")
    print(f"Configurations : {len(configs)}")
    print(f"Sample-cost rows: {len(samples)}")
    print("PF/DD          : NOT EVALUATED")
    print("Threshold tune : DISABLED")
    print("Promotion      : DISABLED")
    print("Overlap note   : probes overlap; diagnostic only")
    print()
    for item in summaries:
        obs = item["median_observed_success_rate"]
        pred = item["median_mean_probability"]
        be = item["median_mean_break_even_probability"]
        obs_edge = item["median_observed_minus_break_even"]
        cal_gap = item["median_calibration_gap_observed_minus_predicted"]
        print(
            f"{item['cost_profile']:12s} sel med/min/max="
            f"{item['selected_events_median']:.0f}/{item['selected_events_min']}/{item['selected_events_max']} "
            f"selectedProbes={item['probes_with_selected_events']}/{item['probe_count']} "
            f"obs={obs if obs is not None else float('nan'):.3f} "
            f"pred={pred if pred is not None else float('nan'):.3f} "
            f"BE={be if be is not None else float('nan'):.3f} "
            f"obs-BE={obs_edge if obs_edge is not None else float('nan'):+.3f} "
            f"obs-p={cal_gap if cal_gap is not None else float('nan'):+.3f} "
            f"aboveBE={item['probes_observed_above_break_even']}/"
            f"{item['probes_with_selected_events']} {item['base_model_id']}"
        )
    print(f"\nReport         : {out}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True, type=Path)
    args = ap.parse_args()
    run(args.queue)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
