from pathlib import Path

from src.challenger_batch import build_challenger_specs, assert_isolated_output
from src.model_registry import ModelManifest, build_activation_plan
from src.shadow_evaluator import ShadowTrade, summarize_shadow


def test_challenger_batch_generates_many_isolated_models(tmp_path):
    specs = build_challenger_specs(root=tmp_path / "models", limit=24)
    assert len(specs) == 24
    assert len({s.model_id for s in specs}) == 24
    for spec in specs:
        assert_isolated_output(spec, tmp_path / "models")
        assert "candidates" in Path(spec.output_dir).parts


def test_activation_plan_requires_human_flat_boundary_and_rollback():
    manifest = ModelManifest(
        model_id="gold-ch-001",
        status="ELIGIBLE_FOR_HUMAN_GATE",
        git_sha="abc",
        created_at="2026-09-15T00:00:00",
        training_start="2026-01-01",
        training_end="2026-09-01",
        feature_set="core",
        config_hash="cfg",
        random_seed=11,
        xgb_path="models/candidates/gold-ch-001/xgboost.pkl",
        hmm_path="models/candidates/gold-ch-001/hmm.pkl",
        rollback_model_id="gold-v1-champion",
    )
    blocked = build_activation_plan(
        manifest,
        current_champion_id="gold-v1-champion",
        human_approved=False,
        positions_open=1,
        closed_candle_boundary=False,
    )
    assert blocked["ready"] is False
    assert len(blocked["blockers"]) >= 3

    ready = build_activation_plan(
        manifest,
        current_champion_id="gold-v1-champion",
        human_approved=True,
        positions_open=0,
        closed_candle_boundary=True,
    )
    assert ready["ready"] is True
    assert ready["action"] == "HUMAN_GATED_ATOMIC_ACTIVATION"


def test_shadow_summary_builds_pf_dd_and_skip():
    rows = [
        ShadowTrade("m1", "2026-09-01", 100.0, False, 70.0),
        ShadowTrade("m1", "2026-09-01", -50.0, False, 80.0),
        ShadowTrade("m1", "2026-09-02", 40.0, False, 75.0),
        ShadowTrade("m1", "2026-09-03", 0.0, True, 0.0),
    ]
    s = summarize_shadow(rows, starting_equity=20000.0)
    assert s.trades == 3
    assert s.days == 3
    assert abs(s.profit_factor - 2.8) < 1e-9
    assert abs(s.expectancy - 30.0) < 1e-9
    assert abs(s.skip_pct - 25.0) < 1e-9
    assert s.max_drawdown_pct > 0
