r"""
Tests for the within-sector 1-D Hawkes rankers (linear + non-linear), with the
last-funded startup dropped from the risk set.

Run:  PYTHONPATH=. python tests/test_hawkes_ranker.py
"""

import numpy as np

from hawkes_calibration import (
    simulate_synthetic_startup_market, fit_hawkes_ranker, evaluate_hawkes_ranker,
    hawkes_ranker_predict_proba,
)
from hawkes_calibration.models.hawkes_ranker import (
    _recency_fields, _prepare, _objective, _last_funded_before,
)


def _market(seed=3):
    return simulate_synthetic_startup_market(
        T=120, n_sectors=6, startups_per_sector=20, n_lags=2, cooldown_weeks=8, seed=seed)


def test_gradients_match_finite_differences_both_links():
    d = _market()
    Z = d.startup_features.astype(float); M = 6; p = Z.shape[2]
    own, peer = _recency_fields(d.startup_counts, d.startup_sector, M, 0.6, 0.6)
    prep = _prepare(d.events, Z, d.startup_sector, d.active.astype(bool), own, peer, 80, True)
    n = p + 3 * M
    for link in ("linear", "exp"):
        th = np.random.default_rng(0).normal(0, 0.3, n); th[p + M:] = np.abs(th[p + M:])
        _, g = _objective(th, prep, p, M, link, 1e-2)
        gn = np.zeros(n); e = 1e-6
        for i in range(n):
            a = th.copy(); a[i] += e; b = th.copy(); b[i] -= e
            gn[i] = (_objective(a, prep, p, M, link, 1e-2)[0]
                     - _objective(b, prep, p, M, link, 1e-2)[0]) / (2 * e)
        assert np.max(np.abs(g - gn)) < 1e-5, link


def test_fit_constraints_and_probabilities_normalise():
    d = _market()
    for link in ("linear", "exp"):
        fit = fit_hawkes_ranker(d.events, d.startup_features, d.startup_sector, d.active,
                                d.startup_counts, link=link, train_end=80, max_iter=200)
        assert np.all(fit.sector_inhibit >= -1e-9)      # cooldown enters with rho >= 0
        assert np.all(fit.sector_excite >= -1e-9)       # peer excitation a >= 0
        cand, pr = hawkes_ranker_predict_proba(
            fit, d.startup_features, d.startup_sector, d.active, d.startup_counts,
            week=90, sector=0, events=d.events)
        assert cand.size > 0 and abs(pr.sum() - 1.0) < 1e-9


def test_both_links_rank_above_chance():
    d = _market()
    for link in ("linear", "exp"):
        fit = fit_hawkes_ranker(d.events, d.startup_features, d.startup_sector, d.active,
                                d.startup_counts, link=link, train_end=80, max_iter=200)
        ev = evaluate_hawkes_ranker(fit, d.events, d.startup_features, d.startup_sector,
                                    d.active, d.startup_counts, start_week=80, end_week=120)
        # ~20 candidates per sector -> chance top-5 ~0.25; require a clear beat
        assert ev["top5"] > 0.4, link
        assert ev["mrr"] > 0.15, link


def test_last_funded_is_dropped_from_risk_set():
    d = _market()
    fit = fit_hawkes_ranker(d.events, d.startup_features, d.startup_sector, d.active,
                            d.startup_counts, link="exp", train_end=80,
                            drop_last_funded=True, max_iter=120)
    # find a (week, sector) with a prior event in that sector and an at-risk last-funded firm
    checked = False
    for week, s, _ in d.events[d.events[:, 0] >= 40]:
        lf = _last_funded_before(d.events, int(s), int(week))
        if lf is None or not d.active[int(week), lf]:
            continue
        cand, _ = hawkes_ranker_predict_proba(
            fit, d.startup_features, d.startup_sector, d.active, d.startup_counts,
            week=int(week), sector=int(s), events=d.events)
        assert lf not in set(cand.tolist())             # the just-funded firm is excluded
        checked = True
        break
    assert checked


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} Hawkes-ranker tests passed.")
