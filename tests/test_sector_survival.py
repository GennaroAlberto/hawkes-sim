r"""
Tests for the within-sector survival model with an outside ("not in dataset") option.

Run:  PYTHONPATH=. python tests/test_sector_survival.py
"""

import numpy as np

from hawkes_calibration import (
    evaluate_survival,
    fit_startup_survival,
    make_tracked_mask,
    simulate_synthetic_startup_market,
    survival_predict_proba,
)
from hawkes_calibration.sector_survival import _objective, _prepare, sector_week_counts


def _data(seed=3):
    return simulate_synthetic_startup_market(
        T=80, n_sectors=4, startups_per_sector=12, n_lags=2, cooldown_weeks=8, seed=seed
    )


def _tracked(data):
    return make_tracked_mask(data.active, fraction=0.7, seed=1)


def test_objective_gradient_matches_finite_differences():
    data = _data()
    tracked = _tracked(data)
    Z = data.startup_features.astype(float)
    M = 4
    p = Z.shape[2]
    secw = sector_week_counts(data.startup_counts, data.startup_sector, M)
    prep = _prepare(
        data.events,
        Z,
        data.startup_sector,
        data.active.astype(bool),
        data.startup_counts.astype(float),
        tracked,
        secw,
        50,
        8,
        8,
    )
    dims = (p, M, 2)
    n = p + M * p + M + 1 + 2
    th = np.random.default_rng(0).normal(0, 0.3, n)
    _, g = _objective(th, prep, dims, 1e-3, 1e-2, 1e-2)
    gn = np.zeros(n)
    e = 1e-6
    for i in range(n):
        a = th.copy()
        a[i] += e
        b = th.copy()
        b[i] -= e
        gn[i] = (
            _objective(a, prep, dims, 1e-3, 1e-2, 1e-2)[0]
            - _objective(b, prep, dims, 1e-3, 1e-2, 1e-2)[0]
        ) / (2 * e)
    assert np.max(np.abs(g - gn)) < 1e-5  # analytic gradient is correct


def test_fit_respects_cooldown_sign_and_probabilities_normalise():
    data = _data()
    tracked = _tracked(data)
    fit = fit_startup_survival(
        data.events,
        data.startup_features,
        data.startup_sector,
        data.active,
        data.startup_counts,
        tracked=tracked,
        train_end=55,
        cooldown_weeks=8,
        max_iter=120,
    )
    assert np.all(fit.cooldown_coef <= 1e-9)  # self-inhibition (recency)
    cand, pr, p_out = survival_predict_proba(
        fit,
        data.startup_features,
        data.startup_sector,
        data.active,
        data.startup_counts,
        week=60,
        sector=0,
        tracked=tracked,
    )
    assert 0.0 <= p_out <= 1.0
    assert abs(pr.sum() + p_out - 1.0) < 1e-6  # candidates + outside sum to 1


def test_outside_option_discriminates_and_ranks_above_chance():
    data = _data()
    tracked = _tracked(data)
    fit = fit_startup_survival(
        data.events,
        data.startup_features,
        data.startup_sector,
        data.active,
        data.startup_counts,
        tracked=tracked,
        train_end=55,
        cooldown_weeks=8,
        max_iter=120,
    )
    ev = evaluate_survival(
        fit,
        data.events,
        data.startup_features,
        data.startup_sector,
        data.active,
        data.startup_counts,
        tracked=tracked,
        start_week=55,
        end_week=80,
    )
    assert ev["outside_auc"] > 0.5  # the outside option carries signal
    assert ev["top5"] > 0.3  # ranker beats chance over the risk set


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} survival tests passed.")
