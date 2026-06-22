r"""
Tests for the deep additions: general Volterra solver, covariate-modulated
excitation simulator, sum-of-exponentials fitting, goodness-of-fit, and IC
standard errors.

Run:  PYTHONPATH=. python tests/test_deep.py   (or pytest)
"""

import numpy as np

from hawkes_calibration import (
    MBPP, ExponentialKernel, Constant, PiecewiseConstantCovariate,
    solve_mbpp_volterra, solve_mbpp_ltv,
    simulate_hawkes_excitation, interval_censor,
    fit_mbpp_ic_sumexp, fit_mbpp_ic,
    simulate_separable_hawkes, uniform_obs_times, MultiImpulse,
    time_rescaling_residuals, ks_test_exp1, poisson_pearson_residuals, dispersion,
)
from hawkes_calibration.operators import solve_mbpp_ode
from hawkes_calibration.mbpp.interval_censored import _sumexp_compensator_const, _excitation_modulation


def test_volterra_matches_closed_form_convolution():
    kappa, theta = 0.6, 0.8
    t = np.linspace(0, 30, 1201)
    exo = Constant(2.0, 40)
    xi_cf = MBPP(ExponentialKernel(kappa, theta), exo, method="closed").intensity(t)
    K = lambda tt, uu: kappa * theta * np.exp(-theta * (tt - uu))
    xi_v = solve_mbpp_volterra(exo.intensity, K, t, M=1)[:, 0]
    assert np.max(np.abs(xi_v - xi_cf)) < 0.02       # O(dt) Nystrom


def test_volterra_matches_ltv_excitation():
    kappa, theta, mu = 0.6, 0.8, 2.0
    m = 40
    obs = np.arange(m + 1, dtype=float)
    Z = PiecewiseConstantCovariate(obs, np.array([[((i // 8) % 2) - 0.5] for i in range(m)], float))
    fine = np.linspace(0, m, 1601)
    mod = _excitation_modulation(Z, [1.0])
    xi_ltv = solve_mbpp_ltv(lambda t: np.array([mu]), np.array([[kappa * theta]]),
                            np.array([[theta]]), fine, modulation=mod)[:, 0]
    Kx = lambda tt, uu: kappa * theta * float(mod(tt)[0, 0]) * np.exp(-theta * (tt - uu))
    xi_v = solve_mbpp_volterra(lambda t: np.array([mu]), Kx, fine, M=1)[:, 0]
    msk = fine > 1.0
    assert np.max(np.abs(xi_v - xi_ltv)[msk]) < 0.2  # two discretisations agree


def test_excitation_simulator_mean_matches_compensator():
    # mean event count over many true-Hawkes sims = MBPP (LTV) compensator
    from hawkes_calibration.mbpp.interval_censored import _excitation_compensator_fast
    rng = np.random.default_rng(0)
    T, m = 40.0, 40
    obs = np.arange(m + 1, dtype=float)
    Z = PiecewiseConstantCovariate(obs, np.array([[((i // 8) % 2) - 0.5] for i in range(m)], float))
    kappa, theta, mu, delta = 0.4, 1.0, 2.0, 1.0
    dXi = np.diff(_excitation_compensator_fast(mu, kappa, theta, Z, [delta], obs))
    N = 600
    acc = np.zeros(m)
    for _ in range(N):
        ev = simulate_hawkes_excitation(Constant(mu, T), kappa, theta, Z, [delta], T,
                                        seed=int(rng.integers(1e9)))
        acc += interval_censor(ev, obs)
    mean_counts = acc / N
    assert abs(mean_counts.sum() - dXi.sum()) / dXi.sum() < 0.06   # within MC error


def test_sumexp_closed_form_compensator():
    mu = 1.5
    thetas = np.array([2.0, 0.4]); kappas = np.array([0.15, 0.45]); a = kappas * thetas
    obs = np.arange(61, dtype=float)
    Xi_cf = _sumexp_compensator_const(mu, a, thetas, obs)
    fine = np.linspace(0, 60, 6001)
    _, Xi_ode = solve_mbpp_ode(lambda t: mu, a, thetas, fine, return_compensator=True)
    assert np.max(np.abs(Xi_cf - np.interp(obs, fine, Xi_ode))) < 1e-3


def test_sumexp_fit_recovers_total_branching():
    rng = np.random.default_rng(0)
    mu = 1.5
    thetas = np.array([2.0, 0.4]); kappas = np.array([0.15, 0.45])
    obs = np.arange(61, dtype=float)
    dXi = np.diff(_sumexp_compensator_const(mu, kappas * thetas, thetas, obs))
    counts = [rng.poisson(np.maximum(dXi, 0)).astype(float) for _ in range(20)]
    res = fit_mbpp_ic_sumexp(obs, counts, np.array([0.4, 1.0, 2.0]), n_restarts=3)
    assert abs(float(np.sum(res.kappas)) - 0.60) < 0.12   # total branching recovered


def test_gof_diagnostics():
    rng = np.random.default_rng(0)
    assert ks_test_exp1(rng.exponential(1.0, 2000))[1] > 0.05     # Exp(1) not rejected
    assert ks_test_exp1(rng.exponential(1 / 0.6, 2000))[1] < 0.05  # wrong scale rejected
    # time rescaling of a homogeneous Poisson -> Exp(1) gaps
    lam = 3.0; T = 300.0
    ev = np.cumsum(rng.exponential(1 / lam, int(lam * T * 1.2))); ev = ev[ev < T]
    tau = time_rescaling_residuals(ev, lambda t: lam * t)
    assert ks_test_exp1(tau)[1] > 0.05
    # dispersion: Poisson ~1, over-dispersed >1
    Xi = rng.uniform(2, 8, 300)
    assert abs(dispersion(rng.poisson(Xi).astype(float), Xi) - 1.0) < 0.3
    assert dispersion(rng.poisson(Xi * rng.gamma(2, 0.5, 300)).astype(float), Xi) > 1.5


def test_ic_fitter_reports_standard_errors():
    rng = np.random.default_rng(0)
    kappa, theta, T = 0.5, 1.0, 30.0
    obs = uniform_obs_times(T, 30)
    imm, off = simulate_separable_hawkes(Constant(5.0, T), kappa, theta, T, seed=1)
    res = fit_mbpp_ic(obs, interval_censor(off, obs), MultiImpulse(imm),
                      loss="ic-ll", endogenous=True, n_restarts=1)
    assert res.se_kappa is not None and res.se_kappa > 0 and np.isfinite(res.se_kappa)
    assert res.se_theta is not None and res.se_theta > 0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} deep-addition tests passed.")
