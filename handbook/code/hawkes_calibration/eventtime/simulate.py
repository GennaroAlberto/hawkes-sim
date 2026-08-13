"""
Exact simulation of multivariate Hawkes processes with exponential kernel,
optionally with time-varying baseline driven by piecewise-constant covariates.

We use Ogata's modified thinning algorithm. The key observation is that for
exponential kernels, between two consecutive events the intensity is strictly
decreasing (towards the baseline value at the current time). So the value of
lambda just after the most recent event upper-bounds lambda over the next
inter-event interval -- provided the baseline doesn't jump up inside that
interval. We handle baseline jumps by adding the covariate breakpoints to the
set of "events" that need to refresh the upper bound.
"""

from __future__ import annotations

import numpy as np

from .covariates import baseline_value


def _intensity_vector(t, events_per_comp, alpha, beta, gamma0, gamma, covariate):
    """Naive O(N*M^2) intensity computation. Used to verify the recursive code."""
    M = alpha.shape[0]
    lam = np.zeros(M)
    for m in range(M):
        if covariate is None:
            lam[m] = float(np.exp(gamma0[m]))
        else:
            lam[m] = baseline_value(covariate, t, gamma0[m], gamma[m])
        for j in range(M):
            ev = events_per_comp[j]
            if len(ev) == 0:
                continue
            past = ev[ev < t]
            if past.size:
                lam[m] += alpha[m, j] * np.exp(-beta[m, j] * (t - past)).sum()
    return lam


def simulate_multivariate_hawkes(
    gamma0,
    alpha,
    beta,
    T,
    gamma=None,
    covariate=None,
    seed=None,
    max_events=None,
):
    """
    Simulate a multivariate Hawkes process on [0, T].

    Parameters
    ----------
    gamma0 : (M,) array. Baseline log-intensity intercepts:
             mu_m(t) = exp(gamma0[m] + gamma[m] @ X(t)).
    alpha : (M, M) array, alpha[m, j] = elicitation of m by events of j.
    beta : (M, M) array, beta[m, j] = decay rate.
    T : float, horizon.
    gamma : (M, p) array of covariate coefficients (one per component).
            If covariate is None this is ignored.
    covariate : PiecewiseConstantCovariate or None.
    seed : optional random seed.
    max_events : optional cap on total number of events.

    Returns
    -------
    events : list of length M of 1D float arrays of jump times.
    """
    rng = np.random.default_rng(seed)
    gamma0 = np.asarray(gamma0, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    M = gamma0.size
    if gamma is None:
        gamma = np.zeros((M, 1))
    else:
        gamma = np.asarray(gamma, dtype=float)

    # Per-component intensity bookkeeping. For each m, we keep a "kernel sum"
    # S[m, j] = sum_{t_{j,l} < current_time} exp(-beta[m,j] * (current_time - t_{j,l})).
    # After a time advance from t to t', S[m, j] decays by exp(-beta[m,j]*(t'-t)).
    # When event of type j occurs at t', S[m, j] += 1 for all m.
    S = np.zeros((M, M))
    events = [[] for _ in range(M)]

    t = 0.0
    last_t = 0.0
    total_events = 0

    # We need to handle covariate jumps so the upper bound is correct.
    if covariate is not None:
        cov_breaks = list(covariate.breakpoints[1:])  # exclude 0
    else:
        cov_breaks = []
    cov_breaks = [b for b in cov_breaks if b < T]

    while t < T:
        # Decay S from last_t to t (we keep S consistent at time = t)
        if t > last_t:
            S *= np.exp(-beta * (t - last_t))
            last_t = t

        # Intensity at time t (just-after any event that just happened)
        baseline_now = np.array(
            [baseline_value(covariate, t, gamma0[m], gamma[m]) for m in range(M)]
        )
        lam = baseline_now + (alpha * S).sum(axis=1)
        lam = np.maximum(lam, 0.0)
        lam_bar = float(lam.sum())
        if lam_bar <= 0.0:
            # No events possible until baseline changes; jump to next break
            if cov_breaks and cov_breaks[0] < T:
                next_bp = cov_breaks.pop(0)
                # Decay S to next_bp
                S *= np.exp(-beta * (next_bp - t))
                t = next_bp
                last_t = t
                continue
            else:
                break

        # Propose next time using upper bound lam_bar (valid until covariate jumps)
        # because intensities decay between events.
        next_bp = T
        if cov_breaks:
            next_bp = min(next_bp, cov_breaks[0])

        u = rng.exponential(1.0 / lam_bar)
        t_candidate = t + u

        if t_candidate >= next_bp:
            # No event before the next breakpoint or T; advance to it and continue
            if next_bp >= T:
                break
            # Decay S to next_bp
            S *= np.exp(-beta * (next_bp - t))
            t = next_bp
            last_t = t
            if cov_breaks and cov_breaks[0] == t:
                cov_breaks.pop(0)
            continue

        # Decay S to t_candidate
        S_new = S * np.exp(-beta * (t_candidate - t))
        baseline_cand = np.array(
            [baseline_value(covariate, t_candidate, gamma0[m], gamma[m]) for m in range(M)]
        )
        lam_new = baseline_cand + (alpha * S_new).sum(axis=1)
        lam_new = np.maximum(lam_new, 0.0)
        lam_new_total = float(lam_new.sum())

        # Acceptance probability (thinning)
        if rng.uniform() < lam_new_total / lam_bar:
            # Accept and pick which component fired
            probs = lam_new / lam_new_total
            j = int(rng.choice(M, p=probs))
            S = S_new.copy()
            # Event of type j increments column j of S by 1
            S[:, j] += 1.0
            events[j].append(t_candidate)
            total_events += 1
            t = t_candidate
            last_t = t
            if max_events is not None and total_events >= max_events:
                break
        else:
            # Reject: just advance time, S decays
            S = S_new
            t = t_candidate
            last_t = t

    return [np.array(sorted(e)) for e in events]
