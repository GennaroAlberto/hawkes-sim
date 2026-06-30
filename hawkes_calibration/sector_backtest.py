"""Clean end-to-end synthetic backtest for the sector-ranker prototype."""

from __future__ import annotations

import numpy as np

from .sector_ranker import (
    evaluate_ranker,
    fit_sector_count_model,
    fit_startup_ranker,
    poisson_nll,
    sector_baseline_rates,
    simulate_marked_paths,
    simulate_synthetic_startup_market,
)


def backtest_synthetic_pipeline(
    *,
    seed=0,
    T=180,
    train_end=120,
    n_sectors=11,
    startups_per_sector=35,
    n_lags=4,
    cooldown_weeks=26,
    n_paths=100,
    max_events_per_week=None,
):
    """Run an end-to-end synthetic backtest for the two-layer architecture.

    This wrapper deliberately zeros all post-``train_end`` event histories before
    simulation.  The sector and ranker one-step evaluations may condition on
    observed lagged history, but simulated paths must not see future startup events
    when building cooldown covariates.

    ``max_events_per_week`` caps the simulated funding events per sector-week to the
    first ``K`` (forwarded to :func:`simulate_marked_paths`) -- both the business
    question ("next K firms to raise") and a guard against a near/over-critical fitted
    sector model exploding the simulation.
    """
    data = simulate_synthetic_startup_market(
        T=T,
        n_sectors=n_sectors,
        startups_per_sector=startups_per_sector,
        n_lags=n_lags,
        cooldown_weeks=cooldown_weeks,
        seed=seed,
    )
    sector_fit = fit_sector_count_model(
        data.sector_counts,
        data.covariates,
        n_lags=n_lags,
        train_end=train_end,
        l2=1e-3,
    )
    ranker_fit = fit_startup_ranker(
        data.events,
        data.startup_features,
        data.startup_sector,
        data.active,
        data.startup_counts,
        train_end=train_end,
        cooldown_weeks=cooldown_weeks,
        l2_global=1e-3,
        l2_sector=5e-2,
    )

    # Sector held-out one-step scores, using actual lag history.
    rates_all = sector_fit.rates(data.sector_counts, data.covariates)
    model_rates = rates_all[train_end:T]
    baseline_rates = sector_baseline_rates(data.sector_counts, train_end, train_end, T)
    observed = data.sector_counts[train_end:T]
    sector_model_nll = poisson_nll(observed, model_rates) / observed.size
    sector_baseline_nll = poisson_nll(observed, baseline_rates) / observed.size

    rank_metrics = evaluate_ranker(
        ranker_fit,
        data.events,
        data.startup_features,
        data.startup_sector,
        data.active,
        data.startup_counts,
        start_week=train_end,
        end_week=T,
        topk=(1, 5, 10),
    )

    sector_history = data.sector_counts.copy()
    startup_history = data.startup_counts.copy()
    sector_history[train_end:T] = 0
    startup_history[train_end:T] = 0
    paths = simulate_marked_paths(
        sector_fit,
        ranker_fit,
        data.startup_features,
        data.startup_sector,
        data.active,
        sector_history,
        startup_history,
        data.covariates,
        start_week=train_end,
        end_week=T,
        n_paths=n_paths,
        seed=seed + 123,
        max_events_per_week=max_events_per_week,
    )
    sim_mean_sector = paths["sector_counts"].mean(axis=0)
    sim_sector_mae = float(np.mean(np.abs(sim_mean_sector - observed)))
    base_sector_mae = float(np.mean(np.abs(baseline_rates - observed)))

    return {
        "data": data,
        "sector_fit": sector_fit,
        "ranker_fit": ranker_fit,
        "metrics": {
            "n_events_total": int(data.events.shape[0]),
            "n_events_train": int(np.sum(data.events[:, 0] < train_end)),
            "n_events_test": int(np.sum(data.events[:, 0] >= train_end)),
            "sector_model_nll_per_cell": float(sector_model_nll),
            "sector_baseline_nll_per_cell": float(sector_baseline_nll),
            "sector_nll_improvement": float(sector_baseline_nll - sector_model_nll),
            "sim_sector_mae": sim_sector_mae,
            "baseline_sector_mae": base_sector_mae,
            "ranker": rank_metrics,
        },
    }


__all__ = ["backtest_synthetic_pipeline"]
