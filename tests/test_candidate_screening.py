from src.candidate_screening import (
    ScreeningThresholds,
    screen_candidate,
    screen_many,
    select_diverse_shortlist,
)


def _row(model_id, train_auc, test_auc, *, xgb="balanced", feat="core", cost="normal", samples=1000):
    return {
        "model_id": model_id,
        "success": True,
        "xgb_profile": xgb,
        "feature_profile": feat,
        "cost_profile": cost,
        "train_metrics": {
            "xgb_train_score": train_auc,
            "xgb_test_score": test_auc,
            "test_samples": samples,
        },
    }


def test_screen_rejects_large_generalization_gap():
    row = _row("m1", 0.90, 0.70)
    result = screen_candidate(row, ScreeningThresholds(max_generalization_gap=0.12))
    assert result.status == "SCREEN_REJECT"
    assert any("gap" in reason for reason in result.reasons)


def test_screen_passes_reasonable_candidate():
    row = _row("m2", 0.72, 0.67)
    result = screen_candidate(row)
    assert result.status == "SCREEN_PASS"
    assert result.adjusted_score > 0


def test_shortlist_preserves_profile_diversity_before_fill():
    rows = [
        _row("a1", 0.72, 0.69, xgb="balanced", feat="core", cost="normal"),
        _row("a2", 0.73, 0.68, xgb="balanced", feat="core", cost="normal"),
        _row("b1", 0.71, 0.67, xgb="conservative", feat="core", cost="normal"),
        _row("c1", 0.70, 0.66, xgb="responsive", feat="core_plus_v2", cost="conservative"),
    ]
    screened = screen_many(rows)
    shortlist = select_diverse_shortlist(screened, top_k=3)
    signatures = {(x.xgb_profile, x.feature_profile, x.cost_profile) for x in shortlist}
    assert len(shortlist) == 3
    assert len(signatures) == 3
