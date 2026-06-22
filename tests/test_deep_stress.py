r"""
Deeper / larger-scale stress tests for hawkes_calibration.

These go beyond the small unit tests in `test_interval_censored.py` and
`test_operators.py`: bigger horizons, more events, higher dimension, near-critical
branching, and -- crucially -- direct correctness checks of the analytic
machinery that every event-time fit relies on but that the existing suite only
exercises indirectly:

  * the two independent log-likelihood implementations agree
    (`log_likelihood` used for the Hessian/SE path vs. `log_likelihood_and_grad`
    used in the BFGS inner loop);
  * the closed-form analytic gradient matches central finite differences
    (univariate, multivariate, and with covariates);
  * large-T univariate / multivariate MLE recovery + finite, sane standard errors;
  * stability near the critical branching ratio (spectral radius ~0.85);
  * high-dimensional lasso edge recovery (M = 10);
  * interval-censored kappa recovery at larger scale;
  * the multivariate MBPP ODE solver hits the stationary law at M = 15.

Run:  PYTHONPATH=. python tests/test_deep_stress.py
  or: PYTHONPATH=. python -m pytest tests/test_deep_stress.py -q
"""

import time
import numpy as np

from hawkes_calibration import (
    simulate_multivariate_hawkes, log_likelihood,
    fit_univariate, fit_multivariate, fit_multivariate_with_covariates,
    fit_multivariate_lasso, PiecewiseConstantCovariate,
    ExponentialKernel, Constant, LHPP,
    simulate_separable_hawkes, interval_censor, uniform_obs_times,
    fit_mbpp_ic_multi, solve_mbpp_ode_multivariate,
)
from hawkes_calibration.eventtime.likelihood import log_likelihood_and_grad


# --------------------------------------------------------------------------
# 1. The two likelihood implementations must agree to machine precision.
# --------------------------------------------------------------------------
def test_two_loglik_implementations_agree():
    rng = np.random.default_rng(0)
    max_diff = 0.0
    for trial in range(8):
        M = int(rng.integers(1, 4))
        gamma0 = rng.uniform(-1.5, 0.5, M)
        alpha = rng.uniform(0.0, 0.25, (M, M))
        beta = rng.uniform(0.8, 2.0, (M, M))
        # keep subcritical
        G = alpha / beta
        sr = np.max(np.abs(np.linalg.eigvals(G)))
        if sr >= 0.95:
            alpha *= 0.9 / sr
        events = simulate_multivariate_hawkes(gamma0, alpha, beta, T=300.0,
                                              seed=int(rng.integers(1e9)))
        ll1 = log_likelihood(events, 300.0, gamma0, alpha, beta)
        ll2, _ = log_likelihood_and_grad(events, 300.0, gamma0, alpha, beta)
        max_diff = max(max_diff, abs(ll1 - ll2))
    print(f"  max |ll_scalar - ll_grad| over 8 random configs = {max_diff:.2e}")
    assert max_diff < 1e-8


# --------------------------------------------------------------------------
# 2. Analytic gradient vs central finite differences.
# --------------------------------------------------------------------------
def _fd_grad_gamma0_alpha(events, T, gamma0, alpha, beta, gamma=None, cov=None, eps=1e-6):
    M = len(events)
    g_g0 = np.zeros(M)
    for m in range(M):
        gp = gamma0.copy(); gp[m] += eps
        gm = gamma0.copy(); gm[m] -= eps
        g_g0[m] = (log_likelihood(events, T, gp, alpha, beta, gamma, cov)
                   - log_likelihood(events, T, gm, alpha, beta, gamma, cov)) / (2 * eps)
    g_al = np.zeros((M, M))
    for m in range(M):
        for j in range(M):
            ap = alpha.copy(); ap[m, j] += eps
            am = alpha.copy(); am[m, j] -= eps
            g_al[m, j] = (log_likelihood(events, T, gamma0, ap, beta, gamma, cov)
                          - log_likelihood(events, T, gamma0, am, beta, gamma, cov)) / (2 * eps)
    return g_g0, g_al


def test_analytic_gradient_matches_fd():
    rng = np.random.default_rng(1)
    worst = 0.0
    # (a) univariate, (b) 3D dense-ish, both subcritical
    configs = []
    configs.append((np.array([0.0]), np.array([[0.3]]), np.array([[1.0]])))
    M = 3
    alpha = np.array([[0.20, 0.10, 0.05],
                      [0.00, 0.25, 0.10],
                      [0.08, 0.00, 0.18]])
    beta = np.full((M, M), 1.3)
    configs.append((np.array([-0.2, -0.4, -0.1]), alpha, beta))
    for gamma0, alpha, beta in configs:
        events = simulate_multivariate_hawkes(gamma0, alpha, beta, T=400.0, seed=7)
        _, (ag0, _, aalpha) = log_likelihood_and_grad(events, 400.0, gamma0, alpha, beta)
        fg0, falpha = _fd_grad_gamma0_alpha(events, 400.0, gamma0, alpha, beta)
        d0 = np.max(np.abs(ag0 - fg0))
        da = np.max(np.abs(aalpha - falpha))
        worst = max(worst, d0, da)
        print(f"  M={len(events)}: max|grad-fd| gamma0={d0:.2e} alpha={da:.2e}")
    assert worst < 1e-4


def test_analytic_gradient_with_covariates():
    rng = np.random.default_rng(2)
    # 2D with a single regime-switching covariate in the baseline
    T = 300.0
    breaks = np.arange(0.0, T + 25.0, 25.0)
    values = np.array([[i % 2] for i in range(breaks.size - 1)], dtype=float)
    cov = PiecewiseConstantCovariate(breaks, values)
    gamma0 = np.array([-0.3, -0.5])
    gamma = np.array([[0.7], [-0.4]])
    alpha = np.array([[0.2, 0.05], [0.1, 0.22]])
    beta = np.full((2, 2), 1.2)
    events = simulate_multivariate_hawkes(gamma0, alpha, beta, T=T,
                                          gamma=gamma, covariate=cov, seed=3)
    _, (ag0, agam, aalpha) = log_likelihood_and_grad(
        events, T, gamma0, alpha, beta, gamma=gamma, covariate=cov)
    fg0, falpha = _fd_grad_gamma0_alpha(events, T, gamma0, alpha, beta, gamma=gamma, cov=cov)
    # finite-diff for gamma
    eps = 1e-6
    fgam = np.zeros_like(gamma)
    for m in range(2):
        for k in range(gamma.shape[1]):
            gp = gamma.copy(); gp[m, k] += eps
            gm = gamma.copy(); gm[m, k] -= eps
            fgam[m, k] = (log_likelihood(events, T, gamma0, alpha, beta, gp, cov)
                          - log_likelihood(events, T, gamma0, alpha, beta, gm, cov)) / (2 * eps)
    d0 = np.max(np.abs(ag0 - fg0)); da = np.max(np.abs(aalpha - falpha)); dg = np.max(np.abs(agam - fgam))
    print(f"  covariate model: max|grad-fd| gamma0={d0:.2e} gamma={dg:.2e} alpha={da:.2e}")
    assert max(d0, da, dg) < 1e-4


# --------------------------------------------------------------------------
# 3. Large-T univariate MLE: bias ~ 0, SE finite, stationary-rate sanity.
# --------------------------------------------------------------------------
def test_univariate_mle_large_T():
    rng = np.random.default_rng(10)
    mu, alpha, beta, T = 0.5, 0.4, 1.0, 8000.0
    mu_hat, al_hat = [], []
    n_ev = []
    for _ in range(6):
        ev = simulate_multivariate_hawkes([np.log(mu)], [[alpha]], [[beta]], T=T,
                                          seed=int(rng.integers(1e9)))
        res = fit_univariate(ev[0], T, beta=beta, se=True)
        mu_hat.append(np.exp(res.gamma0[0])); al_hat.append(res.alpha[0, 0])
        n_ev.append(ev[0].size)
        assert res.se_alpha is not None and np.all(np.isfinite(res.se_alpha))
    mu_hat, al_hat = np.array(mu_hat), np.array(al_hat)
    # theoretical stationary rate mu/(1 - alpha/beta)
    rate_theo = mu / (1 - alpha / beta)
    rate_emp = np.mean(n_ev) / T
    print(f"  T={T:.0f}  <N>={np.mean(n_ev):.0f}  mu_hat={mu_hat.mean():.3f}±{mu_hat.std():.3f}"
          f"  alpha_hat={al_hat.mean():.3f}±{al_hat.std():.3f}"
          f"  rate emp/theo={rate_emp:.3f}/{rate_theo:.3f}")
    assert abs(mu_hat.mean() - mu) < 0.03
    assert abs(al_hat.mean() - alpha) < 0.03
    assert abs(rate_emp - rate_theo) / rate_theo < 0.1


# --------------------------------------------------------------------------
# 4. Higher-dim multivariate MLE (M = 6, sparse network).
# --------------------------------------------------------------------------
def _sparse_alpha(M, density, hi, rng):
    A = np.zeros((M, M))
    mask = rng.uniform(size=(M, M)) < density
    A[mask] = rng.uniform(0.1, hi, mask.sum())
    return A


def test_multivariate_mle_M6():
    rng = np.random.default_rng(20)
    M, T = 6, 2500.0
    A = _sparse_alpha(M, density=0.3, hi=0.3, rng=rng)
    beta = np.full((M, M), 1.5)
    G = A / beta
    sr = np.max(np.abs(np.linalg.eigvals(G)))
    if sr >= 0.9:
        A *= 0.85 / sr
    gamma0 = np.log(np.full(M, 0.3))
    ev = simulate_multivariate_hawkes(gamma0, A, beta, T=T, seed=42)
    t0 = time.time()
    res = fit_multivariate(ev, T, beta=beta, se=True)
    dt = time.time() - t0
    err = np.max(np.abs(res.alpha - A))
    # zeros recovered near zero
    zero_mae = np.mean(np.abs(res.alpha[A == 0]))
    print(f"  M={M} T={T:.0f} N={sum(e.size for e in ev)} fit {dt:.1f}s | "
          f"max|alpha-hat - alpha|={err:.3f}  zero-entry MAE={zero_mae:.3f}  "
          f"SE finite={res.se_alpha is not None and np.all(np.isfinite(res.se_alpha))}")
    assert res.success or res.message in ("ftol satisfied", "gtol satisfied")
    assert err < 0.12
    assert zero_mae < 0.06
    assert res.se_alpha is not None and np.all(np.isfinite(res.se_alpha))


# --------------------------------------------------------------------------
# 5. Near-critical branching: must not blow up, and must stay recoverable.
# --------------------------------------------------------------------------
def test_near_critical_stability():
    rng = np.random.default_rng(30)
    # univariate with alpha/beta = 0.85 (spectral radius 0.85)
    mu, alpha, beta, T = 0.3, 0.85, 1.0, 4000.0
    ev = simulate_multivariate_hawkes([np.log(mu)], [[alpha]], [[beta]], T=T, seed=5)
    N = ev[0].size
    rate_theo = mu / (1 - alpha / beta)
    res = fit_univariate(ev[0], T, beta=beta, se=True)
    print(f"  near-critical alpha/beta=0.85  N={N}  rate emp/theo={N/T:.3f}/{rate_theo:.3f}  "
          f"alpha_hat={res.alpha[0,0]:.3f}  mu_hat={np.exp(res.gamma0[0]):.3f}")
    assert np.isfinite(res.loglik)
    assert abs(res.alpha[0, 0] - alpha) < 0.05
    assert abs(N / T - rate_theo) / rate_theo < 0.15


# --------------------------------------------------------------------------
# 6. High-dimensional lasso (M = 10): sparsity recovery beats raw MLE.
# --------------------------------------------------------------------------
def _edge_pr(A_hat, A_true, tol):
    pred = np.abs(A_hat) > tol
    true = A_true > 0
    tp = np.sum(pred & true)
    prec = tp / max(pred.sum(), 1)
    rec = tp / max(true.sum(), 1)
    return prec, rec


def test_lasso_highdim_M10():
    rng = np.random.default_rng(40)
    M, T = 10, 1500.0
    A = _sparse_alpha(M, density=0.25, hi=0.3, rng=rng)
    beta = np.full((M, M), 1.5)
    G = A / beta
    sr = np.max(np.abs(np.linalg.eigvals(G)))
    if sr >= 0.85:
        A *= 0.8 / sr
    gamma0 = np.log(np.full(M, 0.4))
    ev = simulate_multivariate_hawkes(gamma0, A, beta, T=T, seed=99)
    N = sum(e.size for e in ev)
    mle = fit_multivariate(ev, T, beta=beta, se=False)
    las = fit_multivariate_lasso(ev, T, beta=beta, lam=40.0)
    p_mle, r_mle = _edge_pr(mle.alpha, A, tol=0.05)
    p_las, r_las = _edge_pr(las.alpha, A, tol=1e-6)
    zmae_mle = np.mean(np.abs(mle.alpha[A == 0]))
    zmae_las = np.mean(np.abs(las.alpha[A == 0]))
    print(f"  M={M} T={T:.0f} N={N} | MLE prec/rec={p_mle:.2f}/{r_mle:.2f} zMAE={zmae_mle:.3f}"
          f" | LASSO prec/rec={p_las:.2f}/{r_las:.2f} zMAE={zmae_las:.3f}")
    # lasso should clean up the true zeros relative to the raw MLE
    assert zmae_las < zmae_mle
    assert p_las >= p_mle - 1e-9


# --------------------------------------------------------------------------
# 7. Event-time covariate recovery at scale.
# --------------------------------------------------------------------------
def test_covariate_event_time_recovery():
    T = 3000.0
    breaks = np.arange(0.0, T + 200.0, 200.0)
    values = np.array([[i % 2] for i in range(breaks.size - 1)], dtype=float)
    cov = PiecewiseConstantCovariate(breaks, values)
    gamma0 = np.array([0.0, -0.2])
    gamma = np.array([[0.8], [-0.5]])
    alpha = np.array([[0.25, 0.05], [0.10, 0.20]])
    beta = np.full((2, 2), 1.0)
    ev = simulate_multivariate_hawkes(gamma0, alpha, beta, T=T,
                                      gamma=gamma, covariate=cov, seed=11)
    res = fit_multivariate_with_covariates(ev, T, beta=beta, covariate=cov, se=True)
    print(f"  N={sum(e.size for e in ev)}  gamma_hat={res.gamma.ravel()}  (true {gamma.ravel()})"
          f"  max|alpha err|={np.max(np.abs(res.alpha - alpha)):.3f}")
    assert abs(res.gamma[0, 0] - 0.8) < 0.15
    assert abs(res.gamma[1, 0] - (-0.5)) < 0.15
    assert np.max(np.abs(res.alpha - alpha)) < 0.1


# --------------------------------------------------------------------------
# 8. Interval-censored kappa recovery at larger scale.
# --------------------------------------------------------------------------
def test_ic_large_scale():
    rng = np.random.default_rng(50)
    kappa, theta, T = 0.65, 0.9, 120.0
    exo_gen = Constant(4.0, T)
    obs = uniform_obs_times(T, 240)   # finer grid, longer horizon
    counts_list, exo_list = [], []
    for _ in range(40):              # many i.i.d. sequences
        imm, off = simulate_separable_hawkes(exo_gen, kappa, theta, T,
                                             seed=int(rng.integers(1e9)))
        counts_list.append(interval_censor(off, obs))
        exo_list.append(LHPP(obs, interval_censor(imm, obs)))
    res = fit_mbpp_ic_multi(obs, counts_list, exo_list, loss="ic-ll",
                            endogenous=True, n_restarts=3)
    print(f"  {len(counts_list)} seqs, {obs.size-1} intervals | kappa_hat={res.kappa:.3f}"
          f" (true {kappa})  theta_hat={res.theta:.3f} (true {theta})")
    assert abs(res.kappa - kappa) < 0.05


# --------------------------------------------------------------------------
# 9. Multivariate MBPP ODE solver at M = 15 -> stationary law (I-G)^{-1} mu.
# --------------------------------------------------------------------------
def test_operator_multivariate_M15():
    rng = np.random.default_rng(60)
    M = 15
    A = _sparse_alpha(M, density=0.2, hi=0.25, rng=rng)
    B = np.full((M, M), 1.4)
    G = A / B
    sr = np.max(np.abs(np.linalg.eigvals(G)))
    if sr >= 0.85:
        A *= 0.8 / sr; G = A / B
    mu = rng.uniform(0.2, 0.6, M)
    xi_star = np.linalg.solve(np.eye(M) - G, mu)
    t = np.linspace(0, 200, 8001)
    xi = solve_mbpp_ode_multivariate(lambda tt: mu, A, B, t)
    err = np.max(np.abs(xi[-1] - xi_star))
    print(f"  M={M} spectral radius={np.max(np.abs(np.linalg.eigvals(G))):.3f} | "
          f"max|xi(T) - xi*|={err:.2e}")
    assert np.all(np.isfinite(xi))
    assert err < 1e-4


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    t0 = time.time()
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns)-failed}/{len(fns)} passed in {time.time()-t0:.1f}s")
