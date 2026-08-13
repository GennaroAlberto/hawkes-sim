r"""
Bayesian calibration of the MBPP from interval-censored counts.

Why Bayesian here?  The recurring obstacle in this package is *weak
identifiability*: the branching ratio kappa is pinned down by counts, but the
decay theta is not (a bucket integrates away the within-bucket timing).  A
frequentist point estimate plus a single standard error hides this; the Bayesian
posterior handles it correctly and usefully:

* it stays **wide along the weakly-identified direction** and tight along the
  informative one, respecting uncertainty, by construction;
* it lets you **inject prior knowledge on theta** (the weak direction) to tighten
  it legitimately, advice is to run a covaraite free regression on the two moments
  of the process and use the estimated theta to set up a sensible prior;
* it exposes the kappa--theta **ridge as a posterior correlation** rather than a
  scalar SE;
* with **hierarchical pooling** across many series it borrows strength and
  genuinely sharpens per-series estimates.

It will *not* invent information the counts do not contain -- if theta is
unidentified, the posterior (under a flat prior) is simply broad.  That honesty
is the point.

The likelihood is the interval-censored Poisson log-likelihood of the MBPP
(``interval_censored.ic_ll``); the posterior is
``p(theta | C) ∝ exp(-IC-LL) * prior``.  Everything here is numpy-only:

* :class:`GaussianPrior` -- independent Gaussian priors on the *unconstrained*
  parameters (a Normal on ``logit(kappa)`` is a logit-normal on kappa in (0,1); a
  Normal on ``log(theta)`` is a log-normal on theta>0).  No Jacobian needed.
* :func:`adaptive_metropolis` -- Haario adaptive random-walk MCMC.
* :func:`laplace_posterior` -- Gaussian posterior at the MAP (reuses the Hessian).
* :func:`fit_mbpp_bayes` -- high-level inference over (kappa, theta) for a fixed
  exogenous, returning posterior samples, credible intervals and the ridge
  correlation.
* :func:`fit_mbpp_bayes_hierarchical` -- partial pooling of kappa across sequences.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .core import MBPP, ExponentialKernel
from .interval_censored import _logit, _sigmoid, ic_ll


# ===========================================================================
# Priors (independent Gaussians on the unconstrained scale).
# ===========================================================================
class GaussianPrior:
    r"""
    Independent Gaussian priors on the unconstrained parameters ``u``.

    For the (kappa, theta) model, ``u = [logit(kappa), log(theta)]`` (optionally
    ``+[log(mu)]``).  ``means`` and ``sds`` are on that scale, so a Normal on
    ``log(theta)`` is a log-normal prior on theta.  Defaults are weakly
    informative; set a small ``sd`` on the theta component to encode strong prior
    knowledge of the timescale.
    """

    def __init__(self, means, sds):
        self.means = np.asarray(means, float)
        self.sds = np.asarray(sds, float)

    def log_prob(self, u):
        z = (np.asarray(u, float) - self.means) / self.sds
        return float(
            -0.5 * np.sum(z**2) - np.sum(np.log(self.sds)) - 0.5 * len(z) * np.log(2 * np.pi)
        )

    def sample(self, rng, n=1):
        return self.means + self.sds * rng.standard_normal((n, self.means.size))


def default_prior(
    kappa0=0.5, theta0=1.0, with_mu=False, mu0=1.0, sd_kappa=1.5, sd_theta=1.0, sd_mu=1.0
):
    r"""Weakly-informative default prior centred at ``(kappa0, theta0[, mu0])``."""
    means = [_logit(kappa0), np.log(theta0)]
    sds = [sd_kappa, sd_theta]
    if with_mu:
        means.append(np.log(mu0))
        sds.append(sd_mu)
    return GaussianPrior(means, sds)


# ===========================================================================
# Samplers and the Laplace approximation.
# ===========================================================================
def adaptive_metropolis(
    log_post, x0, n_samples=8000, burn=2000, n_chains=4, init_scale=0.1, seed=0
):
    r"""
    Haario adaptive random-walk Metropolis.

    Runs ``n_chains`` chains; after ``burn`` steps the proposal covariance is the
    empirical chain covariance scaled by 2.38^2/d (the optimal RWM scaling), with a
    small ridge for stability.  Returns the post-burn samples stacked per chain.

    Returns
    -------
    chains : (n_chains, n_keep, d) array of unconstrained samples.
    accept : mean acceptance rate.
    """
    x0 = np.asarray(x0, float)
    d = x0.size
    rng = np.random.default_rng(seed)
    keep = n_samples - burn
    chains = np.empty((n_chains, keep, d))
    accepts = []
    for c in range(n_chains):
        x = x0 + init_scale * rng.standard_normal(d)
        lp = log_post(x)
        cov = (init_scale**2) * np.eye(d)
        hist = np.empty((n_samples, d))
        n_acc = 0
        for s in range(n_samples):
            if s > burn and s % 50 == 0:  # adapt from the chain so far
                emp = np.cov(hist[:s].T) + 1e-8 * np.eye(d)
                cov = (2.38**2 / d) * emp
            prop = x + rng.multivariate_normal(np.zeros(d), cov)
            lp_prop = log_post(prop)
            if np.log(rng.uniform()) < lp_prop - lp:
                x, lp = prop, lp_prop
                n_acc += 1
            hist[s] = x
        chains[c] = hist[burn:]
        accepts.append(n_acc / n_samples)
    return chains, float(np.mean(accepts))


def laplace_posterior(neg_log_post, x_map, eps=1e-4):
    r"""
    Gaussian (Laplace) posterior at the MAP: mean ``x_map`` and covariance the
    inverse Hessian of the negative log-posterior.  Cheap, and exact in the
    large-sample limit (Bernstein--von Mises).
    """
    x = np.asarray(x_map, float)
    d = x.size
    H = np.zeros((d, d))
    for i in range(d):
        for j in range(i, d):
            xpp = x.copy()
            xpp[i] += eps
            xpp[j] += eps
            xpm = x.copy()
            xpm[i] += eps
            xpm[j] -= eps
            xmp = x.copy()
            xmp[i] -= eps
            xmp[j] += eps
            xmm = x.copy()
            xmm[i] -= eps
            xmm[j] -= eps
            H[i, j] = H[j, i] = (
                neg_log_post(xpp) - neg_log_post(xpm) - neg_log_post(xmp) + neg_log_post(xmm)
            ) / (4 * eps * eps)
    cov = np.linalg.pinv(H)
    return x, cov


# ---------------------------------------------------------------------------
# Convergence diagnostics.
# ---------------------------------------------------------------------------
def rhat(chains):
    """Gelman--Rubin R-hat per dimension; ~1.0 indicates convergence."""
    m, n, d = chains.shape
    means = chains.mean(axis=1)  # (m, d)
    B = n * means.var(axis=0, ddof=1)
    W = chains.var(axis=1, ddof=1).mean(axis=0)
    var = (n - 1) / n * W + B / n
    return np.sqrt(np.maximum(var, 1e-12) / np.maximum(W, 1e-12))


# ===========================================================================
# Posterior summary.
# ===========================================================================
@dataclass
class BayesResult:
    samples: dict  # natural-parameter posterior samples
    method: str
    accept: float = None
    rhat: dict = field(default_factory=dict)

    def summary(self):
        lines = [
            f"  posterior via {self.method}"
            + (f"   accept={self.accept:.2f}" if self.accept is not None else "")
        ]
        for name, x in self.samples.items():
            lo, hi = np.percentile(x, [2.5, 97.5])
            rh = f"  R-hat={self.rhat[name]:.3f}" if name in self.rhat else ""
            lines.append(
                f"  {name:7s}: mean={x.mean():.4f}  median={np.median(x):.4f}"
                f"  95% CrI=[{lo:.4f}, {hi:.4f}]{rh}"
            )
        if "kappa" in self.samples and "theta" in self.samples:
            r = np.corrcoef(self.samples["kappa"], self.samples["theta"])[0, 1]
            lines.append(f"  posterior corr(kappa, theta) = {r:+.3f}   (the identifiability ridge)")
        return "\n".join(lines)


# ===========================================================================
# High-level Bayesian fit over (kappa, theta) for a fixed exogenous.
# ===========================================================================
def fit_mbpp_bayes(
    obs_times,
    counts,
    exogenous,
    *,
    endogenous=True,
    method="mcmc",
    prior=None,
    n_samples=8000,
    burn=2000,
    n_chains=4,
    seed=0,
):
    r"""
    Posterior over (kappa, theta) for the MBPP given interval-censored ``counts``.

    Parameters
    ----------
    obs_times, counts, exogenous, endogenous : as in ``fit_mbpp_ic``.  ``counts``
        may be a single vector or a list of vectors sharing the model.
    method : {"mcmc", "laplace"}.
    prior : a :class:`GaussianPrior` on ``[logit(kappa), log(theta)]``; default
        weakly-informative.  Pass a tight ``sd`` on the theta component to encode
        prior knowledge of the timescale.

    Returns
    -------
    BayesResult with posterior samples of ``kappa`` and ``theta``.
    """
    obs = np.asarray(obs_times, float)
    counts_list = counts if isinstance(counts, (list, tuple)) else [counts]
    counts_list = [np.asarray(c, float) for c in counts_list]
    if prior is None:
        prior = default_prior()

    def neg_log_lik(u):
        kappa = float(np.clip(_sigmoid(u[0]), 1e-4, 1 - 1e-4))
        theta = float(np.clip(np.exp(np.clip(u[1], -20, 20)), 1e-4, 1e4))
        mbpp = MBPP(ExponentialKernel(kappa, theta), exogenous, method="closed")
        Xi = mbpp.compensator_interval(obs, endogenous=endogenous)
        if not np.all(np.isfinite(Xi)) or np.any(Xi <= 0):
            return 1e12
        return sum(ic_ll(c, Xi) for c in counts_list)

    def log_post(u):
        return -neg_log_lik(u) + prior.log_prob(u)

    # MAP start (cheap optimisation of -log_post)
    from ..optim import minimize_bfgs

    res = minimize_bfgs(
        lambda u: -log_post(u), np.array(prior.means, float), max_iter=200, ftol=1e-10, gtol=1e-6
    )
    u_map = res.x

    if method == "laplace":
        mean, cov = laplace_posterior(lambda u: -log_post(u), u_map)
        rng = np.random.default_rng(seed)
        U = rng.multivariate_normal(mean, cov, size=n_samples)
        chains = U[None]
        accept = None
        rh = {}
    elif method == "mcmc":
        chains, accept = adaptive_metropolis(
            log_post, u_map, n_samples=n_samples, burn=burn, n_chains=n_chains, seed=seed
        )
        rkv = rhat(chains)
        rh = {"kappa": float(rkv[0]), "theta": float(rkv[1])}
    else:
        raise ValueError(f"unknown method {method!r}")

    U = chains.reshape(-1, chains.shape[-1])
    samples = {"kappa": _sigmoid(U[:, 0]), "theta": np.exp(np.clip(U[:, 1], -20, 20))}
    return BayesResult(samples=samples, method=method, accept=accept, rhat=rh)


# ===========================================================================
# Hierarchical (partial pooling) over many sequences.
# ===========================================================================
def fit_mbpp_bayes_hierarchical(
    obs_times,
    counts_list,
    exogenous_list,
    *,
    endogenous=True,
    theta=1.0,
    n_samples=6000,
    burn=2000,
    n_chains=3,
    pop_mean0=0.0,
    pop_sd_prior=1.0,
    seed=0,
):
    r"""
    Partial pooling of the branching ratio across ``J`` sequences.

    Model (theta fixed for clarity; it is the weakly-identified nuisance):
    ``logit(kappa_j) ~ Normal(mu_pop, tau)``,
    ``mu_pop ~ Normal(pop_mean0, 1.5)``, ``log(tau) ~ Normal(0, pop_sd_prior)``,
    and the per-sequence IC-LL likelihood.  Pooling lets short/uninformative
    sequences borrow strength from the population, sharpening every kappa_j --- a
    genuine identifiability gain that a per-sequence fit cannot achieve.

    Returns
    -------
    BayesResult with posterior samples of the population mean kappa
    (``kappa_pop``) and the spread (``tau``).
    """
    obs = np.asarray(obs_times, float)
    counts_list = [np.asarray(c, float) for c in counts_list]
    J = len(counts_list)

    # parameter vector u = [mu_pop, log_tau, uk_1, ..., uk_J]
    def log_post(u):
        mu_pop, log_tau = u[0], u[1]
        tau = np.exp(np.clip(log_tau, -10, 5))
        uk = u[2:]
        lp = -0.5 * ((mu_pop - pop_mean0) / 1.5) ** 2 - 0.5 * (log_tau / pop_sd_prior) ** 2
        lp += np.sum(-0.5 * ((uk - mu_pop) / tau) ** 2 - np.log(tau))  # hierarchical prior
        ll = 0.0
        for j in range(J):
            kappa = float(np.clip(_sigmoid(uk[j]), 1e-4, 1 - 1e-4))
            mbpp = MBPP(ExponentialKernel(kappa, theta), exogenous_list[j], method="closed")
            Xi = mbpp.compensator_interval(obs, endogenous=endogenous)
            if not np.all(np.isfinite(Xi)) or np.any(Xi <= 0):
                return -1e12
            ll -= ic_ll(counts_list[j], Xi)
        return lp + ll

    u0 = np.concatenate([[pop_mean0, 0.0], np.full(J, pop_mean0)])
    chains, accept = adaptive_metropolis(
        log_post, u0, n_samples=n_samples, burn=burn, n_chains=n_chains, init_scale=0.2, seed=seed
    )
    U = chains.reshape(-1, chains.shape[-1])
    samples = {"kappa_pop": _sigmoid(U[:, 0]), "tau": np.exp(np.clip(U[:, 1], -10, 5))}
    return BayesResult(samples=samples, method="mcmc-hierarchical", accept=accept)
