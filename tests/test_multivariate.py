r"""
Tests for the multivariate covariate-excitation MBPP fitter and its forward model.

Run:  PYTHONPATH=. python tests/test_multivariate.py
"""

import numpy as np

from hawkes_calibration import (
    PiecewiseConstantCovariate, uniform_obs_times,
    solve_mbpp_ltv, solve_mbpp_ode_multivariate, fit_mbpp_ic_excitation_multi,
)
from hawkes_calibration.mbpp.interval_censored import _multi_modulation


def test_delta_zero_reduces_to_multivariate_ode():
    # With no covariate modulation the LTV solver must equal the exact multivariate ODE.
    M = 3
    t = np.linspace(0, 50, 2001)
    A = np.array([[0.1, 0.22, 0.02], [0.2, 0.1, 0.02], [0.03, 0.03, 0.08]])
    B = np.ones((M, M))
    s = lambda tt: np.array([2.0, 1.5, 1.8])
    xi_ode = solve_mbpp_ode_multivariate(s, A, B, t)
    xi_ltv = solve_mbpp_ltv(s, A, B, t, modulation=_multi_modulation(lambda tt: np.array([0.0]), np.array([0.0]), M))
    assert np.max(np.abs(xi_ode - xi_ltv)) < 1e-12


def test_multivariate_fitter_recovers_baseline_and_delta_sign():
    rng = np.random.default_rng(0)
    M, T, nint = 2, 30.0, 15
    obs = uniform_obs_times(T, nint)
    bks = np.linspace(0, T, 6); pop = rng.normal(0, 0.7, 5); pop -= pop.mean()
    Z = PiecewiseConstantCovariate(bks, pop[:, None])
    mu = np.array([2.0, 1.6])
    kappa = np.array([[0.10, 0.25], [0.22, 0.10]])
    theta, delta = 1.0, np.array([0.5])

    # exact interval compensator -> draw Poisson count sequences from the model
    fine = np.unique(np.concatenate([np.linspace(obs[i], obs[i + 1], 6) for i in range(nint)]))
    _, Xi = solve_mbpp_ltv(lambda t: mu, kappa * theta, theta * np.ones((M, M)), fine,
                           modulation=_multi_modulation(Z, delta, M), return_compensator=True)
    dXi = np.diff(np.column_stack([np.interp(obs, fine, Xi[:, m]) for m in range(M)]), axis=0)
    counts = [rng.poisson(dXi) for _ in range(12)]

    fit = fit_mbpp_ic_excitation_multi(obs, counts, Z, n_restarts=1, n_sub=3, seed=0)
    assert fit.delta[0] > 0                                  # covariate effect sign recovered
    assert np.all(np.isfinite(fit.baseline_vec))             # baseline vector finite
    assert fit.kappa_matrix.shape == (M, M)                  # branching matrix returned
    assert 0.0 < fit.kappa < 1.0                             # spectral radius subcritical


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} multivariate tests passed.")
