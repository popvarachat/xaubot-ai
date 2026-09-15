from __future__ import annotations

import pytest

from scripts.run_goldmicro_target_alignment_study import _parse_horizons, _target_base_id
from src.candidate_screening import ScreeningThresholds
from src.target_alignment import select_target_shortlist, summarize_target_alignment


def _row(base: str, horizon: int, sample: int, auc: float) -> dict:
    return {
        "model_id": f"{base}-s{sample:02d}",
        "base_model_id": base,
        "success": True,
        "target_lookahead_bars": horizon,
        "xgb_profile": "balanced",
        "feature_profile": "core",
        "cost_profile": "normal",
        "train_metrics": {
            "xgb_train_score": auc + 0.05,
            "xgb_test_score": auc,
            "test_samples": 1000,
        },
        "sample_index": sample,
        "sample_cutoff": f"2026-0{sample}-01",
        "xgb_path": f"models/{base}/xgb.pkl",
        "hmm_path": f"models/{base}/hmm.pkl",
        "training_data_path": f"models/{base}/data.parquet",
        "split": {"oos_start_index": 700},
        "data_fingerprint": f"fp-{base}-{sample}",
    }


def test_target_alignment_does_not_relax_auc_gate() -> None:
    rows = []
    base1 = "gold-target-ch001-core-t01-normal"
    base4 = "gold-target-ch002-core-t04-normal"
    for sample in range(1, 6):
        rows.append(_row(base1, 1, sample, 0.53))
    for sample, auc in enumerate((0.56, 0.57, 0.56, 0.58, 0.54), start=1):
        rows.append(_row(base4, 4, sample, auc))

    configs, horizons = summarize_target_alignment(
        rows,
        expected_samples=5,
        thresholds=ScreeningThresholds(min_test_auc=0.55, max_generalization_gap=0.12),
        min_pass_rate=0.80,
    )
    by_horizon = {h.target_lookahead_bars: h for h in horizons}
    assert by_horizon[1].status == "HORIZON_GATE_REJECT"
    assert by_horizon[1].stable_pass_count == 0
    assert by_horizon[4].status == "HORIZON_GATE_PASS"
    assert by_horizon[4].stable_pass_count == 1

    mapping = {base1: 1, base4: 4}
    shortlist = select_target_shortlist(configs, horizon_by_base=mapping, top_k=6)
    assert [x.base_model_id for x in shortlist] == [base4]


def test_target_id_keeps_cost_suffix_for_cross_cost_oos() -> None:
    base = "gold-batch-ch-001-b20000-s47-balanced-h500-c60-core-normal"
    aligned = _target_base_id(base, 8)
    assert aligned.endswith("-core-t08-normal")


def test_target_horizons_must_fit_inside_oos_gap() -> None:
    assert _parse_horizons("1,4,8,16") == (1, 4, 8, 16)
    with pytest.raises(ValueError):
        _parse_horizons("1,50")
