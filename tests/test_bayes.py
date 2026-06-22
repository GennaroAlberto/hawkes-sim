r"""
Tests for the Bayesian calibration module.

Validates the value proposition: kappa is recovered tightly, theta honestly
broadly, an informative theta-prior tightens it, the chains converge (R-hat),
and Laplace agrees with MCMC.

Run:  PYTHONPATH=. python tests/test_bayes.py
"""

import numpy as np

from hawkes_calibration import (
    Constant, simulate_separable_hawkes, interval_censor, uniform_obs_times, MultiImpulse,
    fit_mbpp_bayes, default_prior, GaussianPrior, adaptive_metropolis, rhat,
)
from hawkes_calibration.mbpp.interval_censored import _logit


def _data(seed=1):
    kappa, theta, T = 0.5, 1.0, 30.0
    obs = uniform_obs_times(T, 30)
    imm, off = simulate_separable_hawkes(Constant(5.0, T), kappa, theta, T, seed=seed)
    return obs, interval_censor(off, obs), MultiImpulse(imm), kappa, theta


def test_mcmc_recovers_kappa_and_is_honest_about_theta():
    obs, counts, Z, kappa, theta = _data()
    res = fit_mbpp_bayes(obs, counts, Z, endogenous=True, method="mcmc",
                         n_samples=2500, burn=800, n_chains=2, seed=0)
    klo, khi = np.percentile(res.samples["kappa"], [2.5, 97.5])
    tlo, thi = np.percentile(res.samples["theta"], [2.5, 97.5])
    assert klo < kappa < khi                          # kappa credible interval covers truth
    assert (khi - klo) < 0.25                          # and is reasonably tight
    assert (thi - tlo) > (khi - klo)                   # theta is the wide (weakly-identified) one
    assert res.rhat["kappa"] < 1.1 and res.rhat["theta"] < 1.1   # converged


def test_informative_theta_prior_tightens_theta():
    obs, counts, Z, kappa, theta = _data()
    flat = fit_mbpp_bayes(obs, counts, Z, endogenous=True, method="mcmc",
                          n_samples=2500, burn=800, n_chains=2, seed=0)
    tight_prior = GaussianPrior(means=[default_prior().means[0], np.log(1.0)], sds=[1.5, 0.12])
    tight = fit_mbpp_bayes(obs, counts, Z, endogenous=True, method="mcmc",
                           n_samples=2500, burn=800, n_chains=2, prior=tight_prior, seed=0)
    w_flat = np.diff(np.percentile(flat.samples["theta"], [2.5, 97.5]))[0]
    w_tight = np.diff(np.percentile(tight.samples["theta"], [2.5, 97.5]))[0]
    assert w_tight < 0.6 * w_flat                      # prior on theta narrows it


def test_laplace_agrees_with_mcmc_on_kappa():
    obs, counts, Z, kappa, theta = _data()
    mcmc = fit_mbpp_bayes(obs, counts, Z, endogenous=True, method="mcmc",
                          n_samples=2500, burn=800, n_chains=2, seed=0)
    lap = fit_mbpp_bayes(obs, counts, Z, endogenous=True, method="laplace",
                         n_samples=3000, seed=0)
    assert abs(mcmc.samples["kappa"].mean() - lap.samples["kappa"].mean()) < 0.05


def test_sampler_targets_a_gaussian():
    # the adaptive sampler reproduces the mean/cov of a known 2-D Gaussian target
    mu = np.array([0.5, -1.0]); cov = np.array([[1.0, 0.6], [0.6, 1.5]])
    P = np.linalg.inv(cov)
    log_post = lambda x: -0.5 * (x - mu) @ P @ (x - mu)
    chains, acc = adaptive_metropolis(log_post, np.zeros(2), n_samples=6000, burn=2000,
                                      n_chains=2, seed=0)
    X = chains.reshape(-1, 2)
    assert np.allclose(X.mean(0), mu, atol=0.15)
    assert np.allclose(np.cov(X.T), cov, atol=0.35)
    assert np.all(rhat(chains) < 1.1)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} Bayesian tests passed.")
