"""
L1-regularized MLE for the elicitation matrix via accelerated proximal
gradient (FISTA). The objective is

    F(theta) = -loglik(theta) + lambda_l1 * sum_{m,j} |alpha_{m,j}|

with alpha_{m,j} >= 0 enforced through the proximal operator

    prox_{tau, L1+R_+}(z) = max(z - tau * lambda_l1, 0).

We parametrize alpha directly (rather than as exp(a)) so that the L1 penalty
is genuinely sparsifying and zeros are exact.

The smooth part is the negative log-likelihood (with gradient from
likelihood.log_likelihood_and_grad). gamma0 and gamma are unpenalized; their
gradient step is the usual one.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .likelihood import log_likelihood_and_grad


@dataclass
class LassoResult:
    gamma0: np.ndarray
    gamma: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    loglik: float
    objective: float
    nnz: int  # number of non-zero entries in alpha
    nit: int
    n_events: np.ndarray


def _objective(events, T, gamma0, gamma, alpha, beta, covariate, lam):
    ll, grads = log_likelihood_and_grad(
        events, T, gamma0=gamma0, alpha=alpha, beta=beta, gamma=gamma, covariate=covariate
    )
    g_gamma0, g_gamma, g_alpha = grads
    obj = -ll + lam * np.abs(alpha).sum()
    return obj, ll, (-g_gamma0, -g_gamma, -g_alpha)  # gradient of -ll


def _prox(alpha, tau, lam):
    """Proximal step: soft-threshold + non-negativity."""
    return np.maximum(alpha - tau * lam, 0.0)


def fit_multivariate_lasso(
    events,
    T,
    beta,
    lam,
    covariate=None,
    init=None,
    max_iter=500,
    tol=1e-5,
    L0=1.0,
    verbose=False,
):
    """
    L1-penalized MLE for (gamma0, gamma, alpha).  Gamma0 and gamma are not
    penalized.

    Parameters
    ----------
    events : list of M arrays of jump times
    T : horizon
    beta : (M, M) fixed decay matrix
    lam : float, L1 penalty strength (per entry of alpha)
    covariate : optional PiecewiseConstantCovariate
    init : optional dict with keys 'gamma0', 'gamma', 'alpha'
    L0 : initial Lipschitz estimate (line search will increase if needed)
    """
    M = len(events)
    p_cov = covariate.p if covariate is not None else 0
    n_events = np.array([len(e) for e in events])
    beta = np.asarray(beta, dtype=float)

    if init is None:
        rate_est = np.maximum(n_events / max(T, 1.0), 1e-3)
        gamma0 = np.log(0.5 * rate_est)
        gamma = np.zeros((M, p_cov)) if p_cov > 0 else np.zeros((M, 0))
        alpha = 0.1 * np.ones((M, M))
        np.fill_diagonal(alpha, 0.2)
    else:
        gamma0 = init["gamma0"].copy()
        gamma = init.get("gamma", np.zeros((M, p_cov))).copy()
        alpha = init["alpha"].copy()

    # Momentum variables
    y_g0 = gamma0.copy()
    y_gam = gamma.copy()
    y_a = alpha.copy()
    t_k = 1.0
    L = L0
    prev_obj = np.inf

    for it in range(max_iter):
        # Gradient at y
        f_y, ll_y, (gg0, ggam, ga) = _objective(events, T, y_g0, y_gam, y_a, beta, covariate, lam)

        # Backtracking line search on L
        while True:
            step = 1.0 / L
            new_g0 = y_g0 - step * gg0
            new_gam = y_gam - step * ggam if p_cov > 0 else y_gam
            new_a = _prox(y_a - step * ga, step, lam)
            f_new, ll_new, _ = _objective(events, T, new_g0, new_gam, new_a, beta, covariate, lam)
            # Sufficient decrease condition: f_new <= Q(new, y) where Q is the quadratic model
            dg0 = new_g0 - y_g0
            dgam = new_gam - y_gam
            da = new_a - y_a
            inner = (gg0 * dg0).sum() + (ggam * dgam).sum() + (ga * da).sum()
            sqnorm = (dg0 ** 2).sum() + (dgam ** 2).sum() + (da ** 2).sum()
            # Compare smooth part of objective (-ll); penalty is identical in both
            smooth_new = -ll_new
            smooth_model = (-ll_y) + inner + 0.5 * L * sqnorm
            if smooth_new <= smooth_model + 1e-9 or L > 1e12:
                break
            L *= 2.0

        # Nesterov momentum update
        t_kp1 = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_k ** 2))
        mom = (t_k - 1.0) / t_kp1
        y_g0 = new_g0 + mom * (new_g0 - gamma0)
        y_gam = new_gam + mom * (new_gam - gamma) if p_cov > 0 else y_gam
        y_a = new_a + mom * (new_a - alpha)

        gamma0 = new_g0
        gamma = new_gam
        alpha = new_a
        t_k = t_kp1

        if verbose and (it % 20 == 0):
            nnz = int((alpha > 0).sum())
            print(f"  iter {it:3d}: F = {f_new:.4f}  L = {L:.1f}  nnz(alpha) = {nnz}")

        if abs(prev_obj - f_new) < tol * (1 + abs(f_new)):
            break
        prev_obj = f_new

        # Gently decrease L on accepted step
        L = max(L * 0.9, 1.0)

    # Final logging
    f_final, ll_final, _ = _objective(events, T, gamma0, gamma, alpha, beta, covariate, lam)
    nnz = int((alpha > 0).sum())

    return LassoResult(
        gamma0=gamma0,
        gamma=gamma,
        alpha=alpha,
        beta=beta,
        loglik=ll_final,
        objective=f_final,
        nnz=nnz,
        nit=it + 1,
        n_events=n_events,
    )
