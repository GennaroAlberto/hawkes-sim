r"""
Simulation utilities for the interval-censored / MBPP machinery.

Two things are provided:

1. ``simulate_separable_hawkes`` -- a cluster-based (Hawkes-Oakes branching)
   simulator that returns the immigrant and offspring event times *separately*.
   The separability is what makes the six observation scenarios of Table 2
   (Rizoiu et al., 2022) reproducible: by selectively hiding labels and
   interval-censoring one or both streams we can construct scenarios A-F from a
   single realization.

2. ``interval_censor`` -- collapse an event-time stream into the per-interval
   counts C(o_{i-1}, o_i] of the interval-censored setting (Section 4.1).

The branching construction (Hawkes and Oakes, 1974): immigrants arrive from the
exogenous intensity s(t); each event independently spawns a Poisson(kappa)
number of direct offspring whose delays are drawn from the *normalized* kernel
phi(t)/kappa (a probability density), recursively, until the family dies out
(which happens almost surely when kappa < 1).
"""

from __future__ import annotations

import numpy as np

from .exogenous import PiecewiseConstant, MultiImpulse


# ---------------------------------------------------------------------------
# Immigrant sampling from an exogenous intensity s(t).
# ---------------------------------------------------------------------------
def sample_immigrants(exogenous, T, rng):
    r"""
    Sample immigrant (exogenous) event times on (0, T] from the exogenous
    intensity s(t) by thinning against a piecewise upper bound.

    Supports the piecewise-constant family (Constant, Rectangle, LHPP, HP-pc)
    exactly, and falls back to thinning with a numeric sup for generic s(t).
    """
    if isinstance(exogenous, PiecewiseConstant):
        times = []
        breaks, rates = exogenous.breaks, exogenous.rates
        for i in range(rates.size):
            a, b, lam = breaks[i], min(breaks[i + 1], T), rates[i]
            if b <= a or lam <= 0:
                continue
            n = rng.poisson(lam * (b - a))
            if n:
                times.extend(rng.uniform(a, b, size=n).tolist())
        return np.sort(np.array(times, dtype=float))

    # generic: thinning with a numeric supremum on a fine grid
    grid = np.linspace(0, T, max(1000, int(T * 50)))
    lam_sup = float(np.max(exogenous.intensity(grid))) * 1.05 + 1e-9
    times = []
    t = 0.0
    while True:
        t += rng.exponential(1.0 / lam_sup)
        if t > T:
            break
        if rng.uniform() < float(exogenous.intensity(np.array([t]))[0]) / lam_sup:
            times.append(t)
    return np.sort(np.array(times, dtype=float))


# ---------------------------------------------------------------------------
# Offspring sampling for one cascade (exponential kernel).
# ---------------------------------------------------------------------------
def _sample_offspring_exp(parents, kappa, theta, T, rng):
    r"""
    Given an array of parent times, generate all direct+indirect offspring on
    (parent, T] for an exponential kernel phi(t)=kappa*theta*e^{-theta t}.  Each
    parent spawns Poisson(kappa) children with i.i.d. Exp(theta) delays.
    """
    offspring = []
    current = list(np.asarray(parents, dtype=float))
    while current:
        nxt = []
        for p in current:
            n = rng.poisson(kappa)
            if n == 0:
                continue
            delays = rng.exponential(1.0 / theta, size=n)
            child = p + delays
            child = child[child <= T]
            for ct in child:
                offspring.append(ct)
                nxt.append(ct)
        current = nxt
    return np.sort(np.array(offspring, dtype=float))


def simulate_separable_hawkes(exogenous, kappa, theta, T, seed=None):
    r"""
    Simulate a univariate Hawkes process by the cluster/branching method,
    returning immigrants and offspring separately.

    Parameters
    ----------
    exogenous : Exogenous
        Exogenous intensity s(t) generating the immigrants.  For ``MultiImpulse``
        the immigrant times are taken as given (no sampling).
    kappa : float in (0,1)
        Branching ratio (mean direct offspring per event).
    theta : float > 0
        Exponential kernel decay rate.
    T : float
        Horizon.
    seed : int or None.

    Returns
    -------
    immigrants : 1D array of immigrant (exogenous) event times.
    offspring  : 1D array of offspring (endogenous) event times.
    """
    rng = np.random.default_rng(seed)
    if isinstance(exogenous, MultiImpulse):
        immigrants = np.asarray(exogenous.times, dtype=float)
        immigrants = immigrants[immigrants <= T]
    else:
        immigrants = sample_immigrants(exogenous, T, rng)
    offspring = _sample_offspring_exp(immigrants, kappa, theta, T, rng)
    return immigrants, offspring


# ---------------------------------------------------------------------------
# Covariate-modulated excitation (true Hawkes with time-varying triggering).
# ---------------------------------------------------------------------------
def simulate_hawkes_excitation(exogenous, kappa, theta, Z, delta, T, seed=None,
                               return_labels=False):
    r"""
    Simulate a *true* univariate Hawkes process whose excitation is modulated by
    covariates, ``alpha(t) = kappa*theta * exp(delta^T Z(t))`` (the receiving-time
    convention of :func:`hawkes_calibration.solve_mbpp_ltv`).  Each event at time
    ``u`` spawns offspring with intensity
    ``mu_off(t) = kappa*theta * exp(delta^T Z(t)) * exp(-theta (t-u))`` on ``t>u``.

    Sampling is by exact thinning against the constant-modulation envelope
    (branching ``kappa*B`` with ``B = max_t exp(delta^T Z(t))``), recursively over
    generations.  Returns the full event stream (immigrants + offspring); this is
    the generator used to validate ``fit_mbpp_ic_excitation`` against genuine
    Hawkes data rather than MBPP-Poisson data.

    Parameters
    ----------
    exogenous : Exogenous (the immigrant baseline s(t)).
    kappa, theta : base branching ratio and decay.
    Z : callable t -> covariate vector.
    delta : (p,) excitation-covariate coefficients.
    T : horizon.  seed : int or None.
    return_labels : if True also return (immigrants, offspring) separately.
    """
    rng = np.random.default_rng(seed)
    delta = np.atleast_1d(np.asarray(delta, dtype=float))
    immigrants = sample_immigrants(exogenous, T, rng)

    # envelope bound B = max over [0,T] of exp(delta^T Z(t)); grid-sampled.
    grid = np.linspace(0.0, T, max(400, int(T * 8)))
    dz_grid = np.array([float(delta @ np.atleast_1d(np.asarray(Z(g), float)).reshape(-1)) for g in grid])
    dz_max = float(dz_grid.max())
    B = np.exp(dz_max)

    offspring = []
    current = list(np.asarray(immigrants, dtype=float))
    while current:
        nxt = []
        for p in current:
            n = rng.poisson(kappa * B)               # envelope offspring count
            if n == 0:
                continue
            times = p + rng.exponential(1.0 / theta, size=n)
            times = times[times <= T]
            for t in times:
                dzt = float(delta @ np.atleast_1d(np.asarray(Z(t), float)).reshape(-1))
                if rng.uniform() < np.exp(dzt - dz_max):   # thinning acceptance
                    offspring.append(t)
                    nxt.append(t)
        current = nxt

    immigrants = np.sort(immigrants)
    offspring = np.sort(np.array(offspring, dtype=float))
    if return_labels:
        return immigrants, offspring
    return np.sort(np.concatenate([immigrants, offspring]))


# ---------------------------------------------------------------------------
# Interval censoring.
# ---------------------------------------------------------------------------
def interval_censor(event_times, obs_times):
    r"""
    Collapse an event-time stream into per-interval counts.

    Parameters
    ----------
    event_times : 1D array of event times.
    obs_times : 1D array [o_0, ..., o_m] of observation endpoints partitioning
        (o_0, o_m] into m half-open intervals (o_{i-1}, o_i].

    Returns
    -------
    counts : (m,) integer array, counts[i] = #{ event in (o_{i-1}, o_i] }.
    """
    event_times = np.asarray(event_times, dtype=float)
    obs_times = np.asarray(obs_times, dtype=float)
    idx = np.searchsorted(obs_times, event_times, side="left")
    counts = np.zeros(obs_times.size - 1, dtype=int)
    valid = (idx >= 1) & (idx <= obs_times.size - 1)
    np.add.at(counts, idx[valid] - 1, 1)
    return counts


def uniform_obs_times(T, n_intervals):
    """Convenience: equally spaced observation endpoints [0, ..., T]."""
    return np.linspace(0.0, T, n_intervals + 1)
