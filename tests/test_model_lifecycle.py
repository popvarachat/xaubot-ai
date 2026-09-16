from src.model_lifecycle import (
    LifecycleThresholds,
    ModelWindowMetrics,
    evaluate_champion,
    evaluate_challenger,
    rank_challengers,
)


def m(**kw):
    base = dict(
        model_id="m",
        trades=120,
        profit_factor=1.50,
        max_drawdown_pct=6.0,
        expectancy=1.0,
        skip_pct=5.0,
        execution_cost_points=80.0,
        stable_windows=3,
        shadow_trades=60,
        shadow_days=4,
    )
    base.update(kw)
    return ModelWindowMetrics(**base)


def test_champion_healthy():
    d = evaluate_champion(m(model_id="champion"))
    assert d.state == "HEALTHY"
    assert d.allow_new_entries is True
    assert d.retrain_challenger is False


def test_small_sample_does_not_replace_model_from_pf_noise():
    d = evaluate_champion(m(trades=10, profit_factor=0.7))
    assert d.state == "INSUFFICIENT_SAMPLE"
    assert d.allow_new_entries is True


def test_hard_execution_cost_holds_new_entries_even_with_small_sample():
    d = evaluate_champion(m(trades=5, execution_cost_points=91.0))
    assert d.state == "HOLD"
    assert d.allow_new_entries is False


def test_hard_drawdown_holds_new_entries():
    d = evaluate_champion(m(max_drawdown_pct=10.0))
    assert d.state == "HOLD"
    assert d.allow_new_entries is False


def test_pf_breach_enters_protect_and_requests_challenger():
    d = evaluate_champion(m(profit_factor=1.20))
    assert d.state == "PROTECT"
    assert d.retrain_challenger is True


def test_watch_zone_does_not_hold():
    d = evaluate_champion(m(profit_factor=1.35))
    assert d.state == "WATCH"
    assert d.allow_new_entries is True


def test_challenger_requires_shadow_evidence():
    champion = m(model_id="champion", profit_factor=1.45, max_drawdown_pct=6.0)
    challenger = m(model_id="c1", profit_factor=1.55, max_drawdown_pct=5.5, shadow_trades=20)
    d = evaluate_challenger(challenger, champion)
    assert d.eligible_for_promotion is False
    assert d.state == "CHALLENGER_REJECT"


def test_strong_challenger_only_becomes_human_gate_eligible():
    champion = m(model_id="champion", profit_factor=1.45, max_drawdown_pct=6.0)
    challenger = m(model_id="c1", profit_factor=1.55, max_drawdown_pct=5.5)
    d = evaluate_challenger(challenger, champion)
    assert d.eligible_for_promotion is True
    assert d.state == "ELIGIBLE_FOR_HUMAN_GATE"
    assert d.allow_new_entries is False


def test_rank_many_challengers_puts_eligible_strong_candidate_first():
    champion = m(model_id="champion", profit_factor=1.45, max_drawdown_pct=6.0)
    candidates = [
        m(model_id="weak", profit_factor=1.31, max_drawdown_pct=8.0),
        m(model_id="strong", profit_factor=1.60, max_drawdown_pct=5.0),
        m(model_id="no-shadow", profit_factor=1.70, max_drawdown_pct=4.0, shadow_trades=0),
    ]
    ranked = rank_challengers(candidates, champion)
    assert ranked[0][0].model_id == "strong"
    assert ranked[0][1].eligible_for_promotion is True
