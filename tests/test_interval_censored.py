r"""
Verification tests for the interval-censored / MBPP additions.

These assert the mathematical properties that the implementation must satisfy and
double as the end-to-end verification of the paper's constructions:

  * parameter conversions (kappa,theta) <-> (alpha,beta);
  * the closed-form MBPP equals the Monte-Carlo mean Hawkes intensity (Fig. 2);
  * the closed-form and numerical MBPP solvers agree (O(dt));
  * the analytic endogenous compensator equals the quadrature of the intensity;
  * a single impulse's total endogenous mass equals the mean total progeny
    kappa/(1-kappa);
  * interval-censoring and the LHPP rate (Prop. 13) are correct;
  * IC-LL recovers the branching ratio from interval-censored counts;
  * the forecasting recursion is unbiased at the true parameters.

Run with:  python -m pytest tests/ -q     (or: python tests/test_interval_censored.py)
"""

import numpy as np

from hawkes_calibration import (
    MBPP, ExponentialKernel, kappa_theta_to_alpha_beta, alpha_beta_to_kappa_theta,
    Constant, Rectangle, LHPP, MultiImpulse,
    simulate_separable_hawkes, interval_censor, uniform_obs_times,
    fit_mbpp_ic_multi, forecast_counts, ic_ll, sse_loss,
    PiecewiseConstantCovariate, CovariateExogenous, fit_mbpp_ic_covariates,
)


def test_param_conversion_roundtrip():
    for kappa, theta in [(0.6, 0.8), (0.3, 2.0), (0.95, 1.15)]:
        a, b = kappa_theta_to_alpha_beta(kappa, theta)
        k2, t2 = alpha_beta_to_kappa_theta(a, b)
        assert abs(k2 - kappa) < 1e-12 and abs(t2 - theta) < 1e-12
        # kappa is the branching ratio = integral of the kernel
        assert abs(ExponentialKernel(kappa, theta).branching_ratio - kappa) < 1e-12


def test_closed_vs_numeric_agreement():
    kappa, theta = 0.6, 0.8
    t = np.linspace(0.5, 30, 120)
    for exo in [Constant(2.0, 35), Rectangle(5, 15, 1.5), LHPP([0, 5, 10, 15], [7, 6, 8])]:
        mc = MBPP(ExponentialKernel(kappa, theta), exo, method="closed")
        mn = MBPP(ExponentialKernel(kappa, theta), exo, method="numeric", grid_dt=0.0025, grid_T=40)
        # O(dt) approximation -> loose but meaningful tolerance
        assert np.max(np.abs(mc.intensity(t) - mn.intensity(t))) < 0.02
        assert np.max(np.abs(mc.compensator(t) - mn.compensator(t))) < 0.5


def test_endogenous_compensator_matches_quadrature():
    kappa, theta = 0.6, 0.8
    exo = LHPP([0, 5, 10, 15], [7, 6, 8])
    m = MBPP(ExponentialKernel(kappa, theta), exo, method="closed")
    tg = np.linspace(0, 30, 60001)
    ei = m.endogenous_intensity(tg)
    quad = np.concatenate([[0], np.cumsum(0.5 * (ei[1:] + ei[:-1]) * np.diff(tg))])
    ec = m.endogenous_compensator(tg)
    assert np.max(np.abs(ec - quad)) < 1e-6


def test_total_progeny_equals_branching_sum():
    for kappa, theta in [(0.4, 1.0), (0.6, 0.8), (0.8, 1.2)]:
        m = MBPP(ExponentialKernel(kappa, theta), MultiImpulse([1.0]), method="closed")
        total = m.endogenous_compensator([1e6])[0]
        assert abs(total - kappa / (1 - kappa)) < 1e-6


def test_mbpp_equals_mean_hawkes_intensity():
    # Fig. 2: closed-form MBPP tracks the Monte-Carlo mean Hawkes intensity.
    kappa, theta, a, T = 0.6, 0.8, 2.0, 30.0
    grid = np.linspace(0, T, 400)
    rng = np.random.default_rng(0)
    acc = np.zeros_like(grid)
    N = 3000
    for _ in range(N):
        imm, off = simulate_separable_hawkes(MultiImpulse([a]), kappa, theta, T, seed=int(rng.integers(1e9)))
        for ti in np.concatenate([imm, off]):
            mm = grid > ti
            acc[mm] += kappa * theta * np.exp(-theta * (grid[mm] - ti))
    mean = acc / N
    xi = MBPP(ExponentialKernel(kappa, theta), MultiImpulse([a]), method="closed").intensity(grid)
    mask = grid > a + 0.2
    # within Monte-Carlo error (peak intensity ~0.5)
    assert np.max(np.abs(mean[mask] - xi[mask])) < 0.03


def test_interval_censor_and_lhpp_rate():
    obs = np.array([0.0, 1.0, 2.0, 3.0])
    events = np.array([0.5, 0.7, 1.5, 2.9, 2.95])
    counts = interval_censor(events, obs)
    assert list(counts) == [2, 1, 2]
    # LHPP rate = counts / width (Prop. 13)
    lhpp = LHPP(obs, counts)
    assert np.allclose(lhpp.rates, counts / np.diff(obs))


def test_ic_ll_recovers_branching_ratio():
    rng = np.random.default_rng(0)
    kappa_true, theta_true, T = 0.6, 0.8, 30.0
    exo_gen = Constant(5.0, T)
    obs = uniform_obs_times(T, 30)
    counts_list, exo_list = [], []
    for _ in range(20):
        imm, off = simulate_separable_hawkes(exo_gen, kappa_true, theta_true, T, seed=int(rng.integers(1e9)))
        counts_list.append(interval_censor(off, obs))
        exo_list.append(LHPP(obs, interval_censor(imm, obs)))
    res = fit_mbpp_ic_multi(obs, counts_list, exo_list, loss="ic-ll", endogenous=True, n_restarts=2)
    assert abs(res.kappa - kappa_true) < 0.05   # branching ratio recovered tightly


def test_forecast_unbiased_at_truth():
    rng = np.random.default_rng(1)
    kappa, theta, mu = 0.5, 1.0, 3.0
    Tfull = 120.0
    exo = Constant(mu, Tfull)
    obs_all = uniform_obs_times(Tfull, 120)
    rels = []
    for _ in range(25):
        imm, off = simulate_separable_hawkes(exo, kappa, theta, Tfull, seed=int(rng.integers(1e9)))
        counts = interval_censor(np.sort(np.concatenate([imm, off])), obs_all)
        pred = forecast_counts(kappa, theta, Constant(mu, Tfull), obs_all[:91], counts[:90], obs_all[90:])
        rels.append((pred.sum() - counts[90:].sum()) / counts[90:].sum())
    assert abs(np.mean(rels)) < 0.07   # mass-exact recursion -> <~MC noise


def test_covariate_exogenous_loglinear_rates():
    # s(t) = exp(gamma0 + gamma^T X(t)); with piecewise-constant X this is a
    # PiecewiseConstant whose rates are exp(gamma0 + gamma^T X_k).
    breaks = np.array([0.0, 1.0, 2.0, 3.0])
    values = np.array([[0.0], [1.0], [0.0]])  # regime indicator
    cov = PiecewiseConstantCovariate(breaks, values)
    g0, g1 = -0.3, 0.8
    exo = CovariateExogenous(cov, g0, [g1])
    expected = np.exp(g0 + g1 * values[:, 0])
    assert np.allclose(exo.rates, expected)
    # zero covariate slope reduces to a constant baseline exp(gamma0)
    exo0 = CovariateExogenous(cov, g0, [0.0])
    assert np.allclose(exo0.rates, np.exp(g0))


def test_covariate_fit_recovers_effect():
    # The covariate coefficient gamma1 is recovered tightly from counts alone.
    rng = np.random.default_rng(0)
    T, period = 200.0, 20.0
    breaks = np.arange(0.0, T + period, period)
    values = np.array([[i % 2] for i in range(breaks.size - 1)], dtype=float)
    cov = PiecewiseConstantCovariate(breaks, values)
    g0, g1, kappa, theta = -0.2, 0.9, 0.5, 1.0
    obs = uniform_obs_times(T, 200)
    counts = []
    for _ in range(15):
        exo = CovariateExogenous(cov, g0, [g1])
        imm, off = simulate_separable_hawkes(exo, kappa, theta, T, seed=int(rng.integers(1e9)))
        counts.append(interval_censor(np.sort(np.concatenate([imm, off])), obs))
    res = fit_mbpp_ic_covariates(obs, counts, cov, loss="ic-ll", endogenous=False, n_restarts=3)
    assert abs(res.gamma[0] - g1) < 0.2          # covariate effect recovered
    assert 0.25 < res.kappa < 0.85               # branching ratio in a sane range


def test_losses_basic():
    counts = np.array([3.0, 1.0, 4.0])
    Xi = np.array([2.5, 1.2, 3.8])
    # IC-LL minimized exactly when Xi == counts (Poisson MLE)
    assert ic_ll(counts, counts) <= ic_ll(counts, Xi) + 1e-9
    # SSE zero at perfect match
    assert sse_loss(counts, counts) == 0.0
    assert sse_loss(counts, Xi) > 0.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} verification tests passed.")
