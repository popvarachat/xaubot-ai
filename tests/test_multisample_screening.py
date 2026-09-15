from src.candidate_screening import ScreeningThresholds
from src.multisample_screening import summarize_multisample, select_stable_shortlist


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
