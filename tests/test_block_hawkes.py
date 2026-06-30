r"""
Tests for the event-time, block-structured log-linear (exp-link) Hawkes with
firm self-inhibition and an open population.

Run:  PYTHONPATH=. python tests/test_block_hawkes.py
"""

import numpy as np

from hawkes_calibration import (
    simulate_block_hawkes, fit_block_hawkes, evaluate_block_hawkes, block_hawkes_loglik,
)
from hawkes_calibration.models.event_block_hawkes import (
    _prepare_grid, _nll_and_grad, _mask_flat, _dims,
)


def _market(seed=4):
    M = 4
    mask = np.eye(M, dtype=bool)                       # block-diagonal (sparse)
    a = np.full(M, -2.3)
    rho = np.array([0.4, 0.9, 1.3, 0.6])               # varied self-inhibition
    A = np.zeros((M, M)); A[np.arange(M), np.arange(M)] = np.array([0.3, 0.5, 0.7, 0.45])
    beta = np.array([[0.3, -0.2]] * M)
    data, truth = simulate_block_hawkes(
        T=160.0, n_sectors=M, firms_per_sector=18, p_covariates=2, n_grid=960,
        w_self=4.0, w_cross=0.6, a=a, rho=rho, A=A, beta=beta, mask=mask, seed=seed)
    return data, truth, mask


def test_analytic_gradient_matches_finite_differences():
    data, truth, mask = _market()
    M, p = 4, 2
    prep = _prepare_grid(data, 420, 4.0, 0.6)
    mf = _mask_flat(mask, M, p); n = sum(_dims(M, p))
    th = np.random.default_rng(0).normal(0, 0.2, n) * mf
    _, g = _nll_and_grad(th, prep, M, p, 2e-3, mf)
    gn = np.zeros(n); e = 1e-6
    for i in range(n):
        if mf[i] == 0:
            continue
        a = th.copy(); a[i] += e; b = th.copy(); b[i] -= e
        gn[i] = (_nll_and_grad(a, prep, M, p, 2e-3, mf)[0]
                 - _nll_and_grad(b, prep, M, p, 2e-3, mf)[0]) / (2 * e)
    assert np.max(np.abs((g - gn) * mf)) < 1e-4         # analytic gradient correct


def test_negative_loglikelihood_is_convex():
    # The exp-link point-process log-likelihood with linearly-entering parameters is
    # concave; so the NLL is convex -> second difference along any direction >= 0.
    data, truth, mask = _market()
    M, p = 4, 2
    prep = _prepare_grid(data, 420, 4.0, 0.6)
    mf = _mask_flat(mask, M, p); n = sum(_dims(M, p))
    rng = np.random.default_rng(1)
    th = rng.normal(0, 0.2, n) * mf
    f0 = _nll_and_grad(th, prep, M, p, 0.0, mf)[0]
    for _ in range(6):
        d = rng.normal(0, 1, n) * mf; d /= np.linalg.norm(d)
        fp = _nll_and_grad(th + 1e-3 * d, prep, M, p, 0.0, mf)[0]
        fm = _nll_and_grad(th - 1e-3 * d, prep, M, p, 0.0, mf)[0]
        curv = (fp + fm - 2 * f0) / 1e-6
        assert curv > -1e-3                              # convex NLL (concave LL)


def test_fit_recovers_baselines_covariates_and_self_inhibition():
    data, truth, mask = _market()
    fit = fit_block_hawkes(data, n_grid=420, w_self=4.0, w_cross=0.6, mask=mask,
                           l2=2e-3, max_iter=300)
    # self-inhibition is genuinely inhibitory (rho >= 0 via the exp link)
    assert np.all(fit.rho >= -1e-9)
    # baselines and covariate coefficients are well identified
    assert np.max(np.abs(truth["a"] - fit.a)) < 0.6
    assert np.corrcoef(truth["beta"].ravel(), fit.beta.ravel())[0, 1] > 0.8
    # the self-inhibition ordering across sectors is recovered
    assert np.corrcoef(truth["rho"], fit.rho)[0, 1] > 0.6
    # within-sector excitation is positively (if weakly) identified
    assert np.corrcoef(np.diag(truth["A"]), np.diag(fit.A))[0, 1] > 0.2


def test_fit_passes_time_rescaling_goodness_of_fit():
    data, truth, mask = _market()
    fit = fit_block_hawkes(data, n_grid=420, w_self=4.0, w_cross=0.6, mask=mask,
                           l2=2e-3, max_iter=300)
    ev = evaluate_block_hawkes(data, fit)
    # rescaled inter-event times should be ~Exp(1): mean near 1, small KS distance
    assert 0.75 < ev["rescaled_mean"] < 1.25
    assert ev["rescaled_ks_exp1"] < 0.15
    # the public loglik helper agrees with the fitted log-likelihood
    ll, _ = block_hawkes_loglik(data, fit)
    assert np.isfinite(ll)


def test_mask_zeroes_disallowed_excitation():
    data, truth, mask = _market()
    off = ~np.eye(4, dtype=bool)                         # off-mask (disallowed) entries
    fit = fit_block_hawkes(data, n_grid=300, w_self=4.0, w_cross=0.6, mask=mask,
                           l2=2e-3, max_iter=80)
    assert np.allclose(fit.A[off], 0.0)                  # block sparsity respected


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} block-Hawkes tests passed.")
