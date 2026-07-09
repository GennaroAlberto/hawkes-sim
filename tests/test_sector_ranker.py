"""Tests for the sector-level count model + dynamic startup second stages."""

import numpy as np

from hawkes_calibration import (
    backtest_synthetic_pipeline,
    candidate_set,
    discrete_hazard_predict_proba,
    evaluate_discrete_hazard,
    evaluate_ranker,
    evaluate_survival,
    excitation_spectral_radius,
    fit_discrete_hazard,
    fit_sector_count_model,
    fit_startup_ranker,
    fit_startup_survival,
    make_tracked_mask,
    ranker_predict_proba,
    simulate_synthetic_startup_market,
    survival_predict_proba,
)


def test_public_api_binds_canonical_implementations():
    # Names that exist in more than one module must resolve to the canonical
    # implementation regardless of star-import order in __init__ (an
    # import-sorter once flipped the shadowing silently -- see the explicit
    # assignment block in hawkes_calibration/__init__.py).
    import hawkes_calibration as h

    assert h.fit_sector_count_model.__module__ == "hawkes_calibration.sector_stability"
    assert h.sector_rate_at.__module__ == "hawkes_calibration.sector_stability"
    assert h.backtest_synthetic_pipeline.__module__ == "hawkes_calibration.sector_backtest"
    assert len(h.__all__) == len(set(h.__all__))


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
        max_excitation_radius=0.95,
        max_iter=150,
    )
    assert sector_fit.success or np.isfinite(sector_fit.loss)
    assert np.all(sector_fit.excitation >= -1e-12)
    assert sector_fit.excitation.shape == (4, 4, 2)
    assert excitation_spectral_radius(sector_fit.excitation) <= 0.95000001
    assert sector_fit.spectral_radius <= 0.95000001

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


def test_sector_one_step_rates_bounded_under_covariate_extrapolation():
    # Held-out covariates outside the training support must not blow up the
    # exp baseline (the historical failure mode: exp(a + beta'x) hitting the
    # rate clip ~4.85e8 and producing million-scale one-step NLL).
    data = simulate_synthetic_startup_market(
        T=90,
        n_sectors=4,
        startups_per_sector=12,
        n_lags=2,
        cooldown_weeks=8,
        seed=5,
    )
    fit = fit_sector_count_model(
        data.sector_counts,
        data.covariates,
        n_lags=2,
        train_end=60,
        max_excitation_radius=0.95,
        max_iter=150,
    )
    # Score at an artificially extreme covariate vector (way outside training).
    from hawkes_calibration.sector_stability import sector_rate_at

    extreme = np.array(data.covariates, copy=True)
    extreme[70] = 50.0
    rates = sector_rate_at(fit, data.sector_counts, extreme, 70)
    margin = fit.baseline_clip_margin
    max_excitation = fit.excitation.sum(axis=(1, 2)) * data.sector_counts.max()
    assert np.all(rates <= fit.baseline_high * margin + max_excitation + 1e-9)
    assert np.all(np.isfinite(rates))


def test_survival_and_hazard_second_stages_fit_and_score():
    data = simulate_synthetic_startup_market(
        T=90,
        n_sectors=4,
        startups_per_sector=12,
        n_lags=2,
        cooldown_weeks=8,
        seed=3,
    )
    train_end = 60
    test_event = data.events[data.events[:, 0] >= train_end][0]
    tracked = make_tracked_mask(data.active, fraction=0.75, seed=10)

    surv = fit_startup_survival(
        data.events,
        data.startup_features,
        data.startup_sector,
        data.active,
        data.startup_counts,
        tracked=tracked,
        train_end=train_end,
        cooldown_weeks=8,
        max_iter=150,
    )
    assert surv.success or np.isfinite(surv.loss)
    assert np.all(surv.cooldown_coef <= 1e-12)
    cand, prob, p_out = survival_predict_proba(
        surv,
        data.startup_features,
        data.startup_sector,
        data.active,
        data.startup_counts,
        week=int(test_event[0]),
        sector=int(test_event[1]),
        tracked=tracked,
    )
    assert cand.size == prob.size
    assert np.isclose(prob.sum() + p_out, 1.0)

    hazard = fit_discrete_hazard(
        data.events,
        data.startup_features,
        data.startup_sector,
        data.active,
        data.startup_counts,
        train_end=train_end,
        cooldown_weeks=8,
        negative_sampling_ratio=10,
        seed=11,
        max_iter=150,
    )
    assert hazard.success or np.isfinite(hazard.loss)
    assert np.all(hazard.cooldown_coef <= 1e-12)
    cand, prob = discrete_hazard_predict_proba(
        hazard,
        data.startup_features,
        data.startup_sector,
        data.active,
        data.startup_counts,
        week=int(test_event[0]),
        sector=int(test_event[1]),
    )
    assert cand.size == prob.size
    assert np.isclose(prob.sum(), 1.0)

    surv_metrics = evaluate_survival(
        surv,
        data.events,
        data.startup_features,
        data.startup_sector,
        data.active,
        data.startup_counts,
        tracked=tracked,
        start_week=train_end,
        end_week=90,
    )
    hazard_metrics = evaluate_discrete_hazard(
        hazard,
        data.events,
        data.startup_features,
        data.startup_sector,
        data.active,
        data.startup_counts,
        start_week=train_end,
        end_week=90,
    )
    assert surv_metrics["n_events"] > 0
    assert hazard_metrics["n_events"] > 0
    assert np.isfinite(surv_metrics["nll"])
    assert np.isfinite(hazard_metrics["nll"])


def test_end_to_end_backtest_runs_with_noncritical_sector_layer():
    # Realistic market (enough candidates per pick that ranking is meaningful); the
    # stability-constrained sector fit + first-K cap keep the simulation bounded.
    out = backtest_synthetic_pipeline(
        seed=0,
        T=150,
        train_end=100,
        n_sectors=10,
        startups_per_sector=30,
        n_lags=4,
        cooldown_weeks=20,
        n_paths=8,
        max_events_per_week=2,
    )
    m = out["metrics"]
    assert m["n_events_total"] > m["n_events_train"] > 0
    assert np.isfinite(m["sector_model_nll_per_cell"])
    assert np.isfinite(m["sim_sector_mae"])
    # Sector layer is now stability-constrained (noncritical fit).
    assert m["sector_spectral_radius"] <= 0.95000001
    assert m["sector_max_row_sum"] <= 0.95000001
    # The risk-set ranker clearly beats a random pick over the same live risk set.
    assert m["ranker"]["nll"] < m["ranker"]["random_nll"]
    assert m["ranker"]["top5"] > m["ranker"]["random_top5"]
    # Second-stage survival / hazard models score on held-out events.
    assert m["survival"]["n_events"] > 0
    assert m["discrete_hazard"]["n_events"] > 0
    # NOTE: we do not assert the sector COUNT model beats the historical-mean
    # baseline -- even stability-constrained it does not yet (weak sector signal in
    # this synthetic DGP). See docs/investment_market_architecture.md.
