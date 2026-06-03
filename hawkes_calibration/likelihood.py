"""
Log-likelihood for a multivariate Hawkes process with exponential kernel and
(optional) piecewise-constant covariates in the baseline.

We use the O(N*M) recursion described in NOTES.md, §2.1, so this scales fine
to tens of thousands of events.
"""

from __future__ import annotations
import numpy as np
from .covariates import baseline_value


def _merge_sorted(arrays, labels):
    """
    Given a list of sorted 1D arrays and corresponding integer labels, return
    a single sorted (times, labels) pair representing the merged stream.
    """
    n = sum(a.size for a in arrays)
    times = np.empty(n)
    lbls = np.empty(n, dtype=np.int64)
    pointers = [0] * len(arrays)
    next_vals = [arr[0] if arr.size else np.inf for arr in arrays]
    for i in range(n):
        k = int(np.argmin(next_vals))
        times[i] = next_vals[k]
        lbls[i] = labels[k]
        pointers[k] += 1
        if pointers[k] < arrays[k].size:
            next_vals[k] = arrays[k][pointers[k]]
        else:
            next_vals[k] = np.inf
    return times, lbls


def log_likelihood(
    events,
    T,
    gamma0,
    alpha,
    beta,
    gamma=None,
    covariate=None,
):
    """
    Compute the log-likelihood of `events` under the Hawkes specification.

    Parameters as in `simulate.simulate_multivariate_hawkes`, except `events`
    is a list of M arrays of sorted jump times.
    """
    M = len(events)
    gamma0 = np.asarray(gamma0, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    if gamma is None:
        gamma = np.zeros((M, 1))
    else:
        gamma = np.asarray(gamma, dtype=float)

    # 1) Baseline integral on [0, T]
    if covariate is None:
        baseline_int = np.array([np.exp(gamma0[m]) * T for m in range(M)])
    else:
        baseline_int = np.array(
            [
                covariate.integrate_exp_gamma(gamma0[m], gamma[m], T)
                for m in range(M)
            ]
        )

    # 2) Excitation compensator: sum_j (alpha/beta) * sum_k (1 - exp(-beta(T - t_jk)))
    excit_int = np.zeros(M)
    for j in range(M):
        ev = events[j]
        if ev.size == 0:
            continue
        for m in range(M):
            decay = 1.0 - np.exp(-beta[m, j] * (T - ev))
            excit_int[m] += (alpha[m, j] / beta[m, j]) * decay.sum()

    # 3) Sum over events of log lambda_{m}(t_{m,k}).
    # We merge all events into one stream sorted in time, and maintain running
    # kernel sums S[m, j] valid just before the current event.
    arr_list = [np.asarray(e, dtype=float) for e in events]
    times, labels = _merge_sorted(arr_list, list(range(M)))

    S = np.zeros((M, M))  # S[m, j] kernel sum, decayed to current time
    log_lam_sum = 0.0
    t_prev = 0.0
    for i in range(times.size):
        t_i = times[i]
        m = int(labels[i])
        # Decay S from t_prev to t_i (just before event)
        S *= np.exp(-beta * (t_i - t_prev))
        # Baseline at t_i (just before event = just at event for left-continuous lambda)
        if covariate is None:
            mu_m = float(np.exp(gamma0[m]))
        else:
            mu_m = baseline_value(covariate, t_i, gamma0[m], gamma[m])
        # lambda_m just before t_i
        lam_m = mu_m + (alpha[m] * S[m]).sum()
        if lam_m <= 0.0:
            return -np.inf
        log_lam_sum += np.log(lam_m)
        # Then event of type m occurs: column m of S increments by 1
        S[:, m] += 1.0
        t_prev = t_i

    ll = log_lam_sum - baseline_int.sum() - excit_int.sum()
    return float(ll)


def _K_vectorized(times, beta, max_exp=400.0):
    """
    For a sorted 1D array of event times, return K_l = sum_{l'<=l} exp(-beta(t_l - t_{l'})).
    Uses block-shifted cumulative sums; numerically stable for arbitrary horizon.
    """
    N = times.size
    if N == 0:
        return np.empty(0)
    if beta == 0:
        return np.arange(1, N + 1, dtype=float)
    K = np.empty(N)
    start = 0
    while start < N:
        end = start
        # Find the longest run where beta*(t_l - t_start) stays in safe exp range.
        # times is sorted, so we can use searchsorted.
        max_t = times[start] + max_exp / beta
        end = int(np.searchsorted(times, max_t, side="right"))
        end = max(end, start + 1)
        ts = times[start:end]
        u = np.exp(beta * (ts - times[start]))
        cum = np.cumsum(u)
        decay_self = np.exp(-beta * (ts - times[start]))
        block_part = decay_self * cum
        if start == 0:
            K[start:end] = block_part
        else:
            # Carry contribution from previous chunks via K[start-1]
            carry = K[start - 1] * np.exp(-beta * (ts - times[start - 1]))
            K[start:end] = carry + block_part
        start = end
    return K


def log_likelihood_and_grad(
    events,
    T,
    gamma0,
    alpha,
    beta,
    gamma=None,
    covariate=None,
):
    """
    Like `log_likelihood`, but also returns the analytical gradient. For the
    exponential-kernel Hawkes model the gradient has a closed form that we
    accumulate alongside the likelihood in a single forward pass, which is the
    key to scaling to large M (numerical gradients would cost O(M^2) extra
    likelihood evaluations per BFGS step).

    Returns
    -------
    ll : float
    (g_gamma0, g_gamma, g_alpha) : ndarrays of shapes (M,), (M, p), (M, M)
        such that ll has gradient g_gamma0 w.r.t. gamma0, g_gamma w.r.t. gamma,
        g_alpha w.r.t. alpha (not log alpha).
    """
    M = len(events)
    gamma0 = np.asarray(gamma0, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    if gamma is None:
        gamma = np.zeros((M, 0))
        p = 0
    else:
        gamma = np.asarray(gamma, dtype=float)
        p = gamma.shape[1]

    # 1) Baseline integral + its gradients
    baseline_int = np.zeros(M)
    d_base_gamma0 = np.zeros(M)  # = baseline_int by chain rule
    d_base_gamma = np.zeros((M, p))
    if covariate is None:
        baseline_int[:] = np.exp(gamma0) * T
        d_base_gamma0[:] = baseline_int
    else:
        K = covariate.values.shape[0]
        for k in range(K):
            a = covariate.breakpoints[k]
            b = min(covariate.breakpoints[k + 1], T)
            if b <= a:
                break
            dt = b - a
            X_k = covariate.values[k]  # (p,)
            mu_k = np.exp(gamma0 + gamma @ X_k)  # (M,)
            baseline_int += mu_k * dt
            d_base_gamma0 += mu_k * dt
            if p > 0:
                d_base_gamma += np.outer(mu_k * dt, X_k)
            if b >= T:
                break

    # 2) Excitation compensator and gradient
    # decay_sums[m, j] = sum_l (1 - exp(-beta[m,j] (T - t_{j,l})))
    decay_sums = np.zeros((M, M))
    for j in range(M):
        ev = events[j]
        if ev.size == 0:
            continue
        # vectorized over m
        # broadcasting: beta[:, j] (M,) times (T - ev) (N_j,) gives (M, N_j)
        decay_sums[:, j] = (1.0 - np.exp(-np.outer(beta[:, j], T - ev))).sum(axis=1)
    excit_int = alpha * decay_sums / beta  # (M, M)
    d_excit_alpha = decay_sums / beta  # (M, M)

    # 3) Vectorized per-pair kernel-sum computation.
    # For each (m, j), we want R_k = sum_{t_{j,l} < t_{m,k}} exp(-beta_{m,j}(t_{m,k} - t_{j,l}))
    # at every t_{m,k}. We use the recursion-via-cumsum approach.
    log_lam_sum = 0.0
    d_loglam_gamma0 = np.zeros(M)
    d_loglam_gamma = np.zeros((M, p))
    d_loglam_alpha = np.zeros((M, M))

    arr_list = [np.asarray(e, dtype=float) for e in events]

    for m in range(M):
        ev_m = arr_list[m]
        if ev_m.size == 0:
            continue
        # R_at_m: (N_m, M)
        R_at_m = np.zeros((ev_m.size, M))
        for j in range(M):
            ev_j = arr_list[j]
            if ev_j.size == 0:
                continue
            b = beta[m, j]
            K_j = _K_vectorized(ev_j, b)
            # For each event of m, find the largest index L with ev_j[L] < ev_m
            L = np.searchsorted(ev_j, ev_m, side="left") - 1
            valid = L >= 0
            if valid.any():
                dt = ev_m[valid] - ev_j[L[valid]]
                R_at_m[valid, j] = K_j[L[valid]] * np.exp(-b * dt)

        # Baseline at each event of m
        if covariate is None:
            mu_at = np.full(ev_m.size, float(np.exp(gamma0[m])))
            X_at = np.zeros((ev_m.size, p))
        else:
            X_at = covariate(ev_m)  # (N_m, p)
            mu_at = np.exp(gamma0[m] + X_at @ gamma[m])

        lam_at = mu_at + R_at_m @ alpha[m]  # (N_m,)
        if (lam_at <= 0).any():
            return -np.inf, (
                np.zeros(M),
                np.zeros((M, p)),
                np.zeros((M, M)),
            )
        log_lam_sum += float(np.log(lam_at).sum())
        d_loglam_gamma0[m] = float((mu_at / lam_at).sum())
        if p > 0:
            d_loglam_gamma[m] = ((mu_at / lam_at)[:, None] * X_at).sum(axis=0)
        d_loglam_alpha[m] = (R_at_m / lam_at[:, None]).sum(axis=0)

    ll = log_lam_sum - baseline_int.sum() - excit_int.sum()
    g_gamma0 = d_loglam_gamma0 - d_base_gamma0
    g_gamma = d_loglam_gamma - d_base_gamma
    g_alpha = d_loglam_alpha - d_excit_alpha
    return float(ll), (g_gamma0, g_gamma, g_alpha)


def neg_log_likelihood(theta_packed, packer, events, T, beta, covariate=None):
    """
    Wrapper for the optimizer (negative log-likelihood, scalar).
    """
    gamma0, gamma, alpha = packer.unpack(theta_packed)
    ll = log_likelihood(
        events, T, gamma0=gamma0, alpha=alpha, beta=beta,
        gamma=gamma, covariate=covariate,
    )
    if not np.isfinite(ll):
        return 1e12
    return -ll


def neg_log_likelihood_and_grad(
    theta_packed, packer, events, T, beta, covariate=None
):
    """
    Negative log-likelihood and packed gradient for the optimizer.

    The packer stores alpha as a = log(alpha) internally, so the gradient is
    transformed by the chain rule: dL/da_{m,j} = alpha_{m,j} * dL/dalpha_{m,j}.
    """
    gamma0, gamma, alpha = packer.unpack(theta_packed)
    ll, (g_g0, g_gam, g_alpha) = log_likelihood_and_grad(
        events, T, gamma0=gamma0, alpha=alpha, beta=beta,
        gamma=gamma, covariate=covariate,
    )
    if not np.isfinite(ll):
        return 1e12, np.zeros_like(theta_packed)
    g_a = alpha * g_alpha  # since alpha = exp(a)
    # Pack gradient in the same layout as the parameter vector
    grad = np.concatenate([
        g_g0.ravel(),
        g_gam.ravel(),
        g_a.ravel(),
    ])
    return -ll, -grad
