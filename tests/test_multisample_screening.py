import ast
from pathlib import Path

from src.candidate_screening import ScreeningThresholds
from src.multisample_screening import summarize_multisample, select_stable_shortlist
from src.target_alignment import select_target_shortlist, summarize_target_alignment


def _row(base: str, idx: int, auc: float, *, train_auc: float | None = None, success: bool = True):
    train_auc = auc + 0.04 if train_auc is None else train_auc
    return {
        "model_id": f"{base}-s{idx:02d}",
        "base_model_id": base,
        "sample_index": idx,
        "sample_cutoff": f"2026-0{idx}-01",
        "success": success,
        "xgb_profile": "balanced",
        "feature_profile": "core_plus_v2",
        "cost_profile": "normal",
        "train_metrics": {
            "xgb_train_score": train_auc,
            "xgb_test_score": auc,
            "test_samples": 900,
        },
        "xgb_path": f"models/candidates/{base}-s{idx:02d}/xgboost_model.pkl",
        "hmm_path": f"models/candidates/{base}-s{idx:02d}/hmm_regime.pkl",
    }


def test_four_of_five_samples_is_stable_pass_at_default_rate():
    rows = [_row("cfg-a", i, auc) for i, auc in enumerate([0.61, 0.60, 0.59, 0.58, 0.54], start=1)]
    out = summarize_multisample(rows, expected_samples=5)

    assert len(out) == 1
    assert out[0].status == "STABLE_SCREEN_PASS"
    assert out[0].pass_count == 4
    assert out[0].pass_rate == 0.8
    assert len(out[0].samples) == 5


def test_three_of_five_samples_is_rejected():
    rows = [_row("cfg-a", i, auc) for i, auc in enumerate([0.61, 0.60, 0.59, 0.54, 0.53], start=1)]
    out = summarize_multisample(rows, expected_samples=5)

    assert out[0].status == "STABLE_SCREEN_REJECT"
    assert out[0].pass_count == 3


def test_missing_training_job_is_rejected_even_when_remaining_samples_pass():
    rows = [_row("cfg-a", i, 0.60) for i in range(1, 5)]
    out = summarize_multisample(rows, expected_samples=5)

    assert out[0].status == "STABLE_SCREEN_REJECT"
    assert "sample records 4 != expected 5" in out[0].reasons


def test_generalization_gap_still_applies_per_sample():
    thresholds = ScreeningThresholds(min_test_auc=0.55, max_generalization_gap=0.12, min_test_samples=500)
    rows = [_row("cfg-a", i, 0.60, train_auc=(0.80 if i in (4, 5) else 0.64)) for i in range(1, 6)]
    out = summarize_multisample(rows, thresholds=thresholds, expected_samples=5)

    assert out[0].pass_count == 3
    assert out[0].status == "STABLE_SCREEN_REJECT"


def test_shortlist_contains_configurations_not_duplicate_sample_models():
    rows = []
    for base, auc in [("cfg-a", 0.62), ("cfg-b", 0.60)]:
        rows.extend(_row(base, i, auc) for i in range(1, 6))

    summaries = summarize_multisample(rows, expected_samples=5)
    shortlist = select_stable_shortlist(summaries, top_k=2)

    assert {x.base_model_id for x in shortlist} == {"cfg-a", "cfg-b"}
    assert all(len(x.samples) == 5 for x in shortlist)


def test_target_alignment_keeps_auc_gate_and_separates_horizons():
    rows = []
    base1 = "gold-target-a-t01-normal"
    base4 = "gold-target-b-t04-normal"
    for i in range(1, 6):
        row = _row(base1, i, 0.53)
        row["target_lookahead_bars"] = 1
        rows.append(row)
    for i, auc in enumerate([0.56, 0.57, 0.56, 0.58, 0.54], start=1):
        row = _row(base4, i, auc)
        row["target_lookahead_bars"] = 4
        rows.append(row)

    configs, horizons = summarize_target_alignment(
        rows,
        expected_samples=5,
        thresholds=ScreeningThresholds(min_test_auc=0.55, max_generalization_gap=0.12, min_test_samples=500),
        min_pass_rate=0.80,
    )
    by_horizon = {h.target_lookahead_bars: h for h in horizons}
    assert by_horizon[1].status == "HORIZON_GATE_REJECT"
    assert by_horizon[4].status == "HORIZON_GATE_PASS"

    shortlist = select_target_shortlist(
        configs,
        horizon_by_base={base1: 1, base4: 4},
        top_k=6,
    )
    assert [x.base_model_id for x in shortlist] == [base4]


def test_target_alignment_runner_is_syntax_valid():
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts" / "run_goldmicro_target_alignment_study.py").read_text(encoding="utf-8")
    ast.parse(source)
