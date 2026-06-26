"""Tests for the sector-level count model + dynamic startup ranker."""

import numpy as np

from hawkes_calibration import (
    backtest_synthetic_pipeline,
    candidate_set,
    evaluate_ranker,
    fit_sector_count_model,
    fit_startup_ranker,
    ranker_predict_proba,
    simulate_synthetic_startup_market,
)


def test_synthetic_market_shapes_and_dynamic_risk_sets():
    data = simulate_synthetic_startup_market(
        T=70,
        n_sectors=4,
        startups_per_sector=10,
        n_lags=2,
        cooldown_weeks=8,
        seed=1,
    )
    assert data.sector_counts.shape == (70, 4)
    assert data.startup_counts.shape == (70, 40)
    assert data.startup_features.shape[:2] == (70, 40)
    assert data.events.shape[1] == 3
    assert data.startup_sector.shape == (40,)
    assert data.active.shape == (70, 40)

    # Some startups enter late, so the live sector risk set changes over time.
    early = candidate_set(data.startup_sector, data.active, sector=0, week=0).size
    late = candidate_set(data.startup_sector, data.active, sector=0, week=69).size
    assert late >= early


def test_sector_model_and_ranker_fit_on_synthetic_data():
    data = simulate_synthetic_startup_market(
        T=90,
        n_sectors=4,
        startups_per_sector=12,
        n_lags=2,
        cooldown_weeks=8,
        seed=2,
    )
    train_end = 60
    sector_fit = fit_sector_count_model(
        data.sector_counts,
        data.covariates,
        n_lags=2,
        train_end=train_end,
        l2=1e-3,
        max_iter=150,
    )
    assert sector_fit.success or np.isfinite(sector_fit.loss)
    assert np.all(sector_fit.excitation >= -1e-12)
    assert sector_fit.excitation.shape == (4, 4, 2)

    ranker_fit = fit_startup_ranker(
        data.events,
        data.startup_features,
        data.startup_sector,
        data.active,
        data.startup_counts,
        train_end=train_end,
        cooldown_weeks=8,
        max_iter=150,
    )
    assert ranker_fit.success or np.isfinite(ranker_fit.loss)
    assert np.all(ranker_fit.cooldown_coef <= 1e-12)

    # Probability vector normalises over the current sector risk set, not all startups.
    test_event = data.events[data.events[:, 0] >= train_end][0]
    cand, prob = ranker_predict_proba(
        ranker_fit,
        data.startup_features,
        data.startup_sector,
        data.active,
        data.startup_counts,
        week=int(test_event[0]),
        sector=int(test_event[1]),
    )
    assert cand.size == prob.size
    assert np.isclose(prob.sum(), 1.0)

    metrics = evaluate_ranker(
        ranker_fit,
        data.events,
        data.startup_features,
        data.startup_sector,
        data.active,
        data.startup_counts,
        start_week=train_end,
        end_week=90,
    )
    assert metrics["n_events"] > 0
    assert np.isfinite(metrics["nll"])
    assert metrics["nll"] < metrics["random_nll"]


def test_end_to_end_backtest_runs_and_beats_simple_baselines():
    out = backtest_synthetic_pipeline(
        seed=4,
        T=100,
        train_end=65,
        n_sectors=5,
        startups_per_sector=12,
        n_lags=2,
        cooldown_weeks=10,
        n_paths=10,
    )
    m = out["metrics"]
    assert m["n_events_total"] > m["n_events_train"] > 0
    assert np.isfinite(m["sector_model_nll_per_cell"])
    assert np.isfinite(m["sim_sector_mae"])
    # Synthetic data are generated from this family, so these should beat naive baselines.
    assert m["sector_model_nll_per_cell"] < m["sector_baseline_nll_per_cell"]
    assert m["ranker"]["nll"] < m["ranker"]["random_nll"]
