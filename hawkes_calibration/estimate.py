"""
Maximum-likelihood estimation for the Hawkes models.

We treat `beta` as fixed (a standard simplification because (alpha, beta) are
only weakly jointly identified on a single sample path; analyst typically
fixes the decay scale from prior knowledge or grid search).

Internally we parametrize alpha = exp(a) and gamma0 free, so that the
optimizer is unconstrained. We use scipy.optimize.minimize with L-BFGS-B.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .likelihood import (
    log_likelihood,
    neg_log_likelihood,
    neg_log_likelihood_and_grad,
)
from .optim import minimize_bfgs


@dataclass
class FitResult:
    gamma0: np.ndarray
    gamma: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    loglik: float
    success: bool
    message: str
    n_events: np.ndarray
    se_gamma0: np.ndarray | None = None
    se_alpha: np.ndarray | None = None

    def summary(self):
        lines = []
        lines.append(f"  log-likelihood = {self.loglik:.3f}")
        lines.append(f"  events per component: {self.n_events.tolist()}")
        lines.append(f"  gamma0 (baseline log-rate intercept):")
        for m, g in enumerate(self.gamma0):
            se = (
                f" ± {self.se_gamma0[m]:.4f}"
                if self.se_gamma0 is not None
                else ""
            )
            lines.append(f"    [{m}] {g:.4f}{se}    (mu={np.exp(g):.4f})")
        if self.gamma.size and self.gamma.shape[1] > 0:
            lines.append("  gamma (covariate effects):")
            for m, row in enumerate(self.gamma):
                lines.append(f"    [{m}] {row.tolist()}")
        lines.append("  alpha (elicitation matrix):")
        for m in range(self.alpha.shape[0]):
            row = "  ".join(f"{a:.4f}" for a in self.alpha[m])
            lines.append(f"    [{m}] {row}")
        if self.se_alpha is not None:
            lines.append("  alpha standard errors:")
            for m in range(self.alpha.shape[0]):
                row = "  ".join(f"{s:.4f}" for s in self.se_alpha[m])
                lines.append(f"    [{m}] {row}")
        return "\n".join(lines)


class _Packer:
    """Pack/unpack model parameters as a flat real vector for the optimizer."""

    def __init__(self, M, p_cov=0):
        self.M = M
        self.p_cov = p_cov
        self.n = M + M * p_cov + M * M

    def pack(self, gamma0, gamma, alpha):
        # alpha enters as log(alpha) for positivity
        parts = [
            np.asarray(gamma0, dtype=float).ravel(),
            np.asarray(gamma, dtype=float).ravel(),
            np.log(np.maximum(np.asarray(alpha, dtype=float), 1e-12)).ravel(),
        ]
        return np.concatenate(parts)

    def unpack(self, theta):
        M, p = self.M, self.p_cov
        gamma0 = theta[:M]
        gamma = theta[M : M + M * p].reshape(M, p) if p > 0 else np.zeros((M, 0))
        a = theta[M + M * p :].reshape(M, M)
        alpha = np.exp(a)
        return gamma0, gamma, alpha


def _numerical_hessian(f, x, eps=1e-4, fg=None):
    """
    Central-difference Hessian. If a gradient callable `fg(x) -> (f, grad)` is
    provided, we differentiate the gradient (O(n) grad evals), otherwise we
    differentiate f twice (O(n^2) function evals).
    """
    n = x.size
    H = np.zeros((n, n))
    if fg is not None:
        for i in range(n):
            xp = x.copy(); xp[i] += eps
            xm = x.copy(); xm[i] -= eps
            _, gp = fg(xp)
            _, gm = fg(xm)
            H[:, i] = (gp - gm) / (2 * eps)
        # symmetrize
        H = 0.5 * (H + H.T)
        return H
    for i in range(n):
        for j in range(i, n):
            xpp = x.copy(); xpp[i] += eps; xpp[j] += eps
            xpm = x.copy(); xpm[i] += eps; xpm[j] -= eps
            xmp = x.copy(); xmp[i] -= eps; xmp[j] += eps
            xmm = x.copy(); xmm[i] -= eps; xmm[j] -= eps
            H[i, j] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * eps * eps)
            H[j, i] = H[i, j]
    return H


def _fit_general(events, T, beta, covariate=None, p_cov=0, init=None, se=True):
    M = len(events)
    n_events = np.array([len(e) for e in events])
    packer = _Packer(M, p_cov=p_cov)

    # Sensible initial values
    if init is None:
        rate_est = np.maximum(n_events / max(T, 1.0), 1e-3)
        gamma0_init = np.log(0.5 * rate_est)  # half of empirical rate is baseline
        alpha_init = np.full((M, M), 0.1)
        # tiny self-excitation seed
        for m in range(M):
            alpha_init[m, m] = 0.2
        gamma_init = np.zeros((M, p_cov)) if p_cov > 0 else np.zeros((M, 0))
        x0 = packer.pack(gamma0_init, gamma_init, alpha_init)
    else:
        x0 = init

    def f(theta):
        return neg_log_likelihood(theta, packer, events, T, beta, covariate=covariate)

    def fg(theta):
        return neg_log_likelihood_and_grad(theta, packer, events, T, beta, covariate=covariate)

    res = minimize_bfgs(f, x0, fg=fg, max_iter=400, ftol=1e-9, gtol=1e-5)
    gamma0_hat, gamma_hat, alpha_hat = packer.unpack(res.x)

    se_gamma0 = None
    se_alpha = None
    if se:
        try:
            H = _numerical_hessian(f, res.x, eps=1e-3, fg=fg)
            # Inverse of observed information (= Hessian of negative log-lik)
            Cov = np.linalg.pinv(H)
            std = np.sqrt(np.clip(np.diag(Cov), 0, None))
            se_gamma0 = std[: M]
            # alpha is exp(a) so SE of alpha = alpha * SE(a) by delta method
            a_part_std = std[M + M * p_cov :].reshape(M, M)
            se_alpha = alpha_hat * a_part_std
        except Exception:
            pass

    return FitResult(
        gamma0=gamma0_hat,
        gamma=gamma_hat,
        alpha=alpha_hat,
        beta=np.asarray(beta, dtype=float),
        loglik=-res.fun,
        success=bool(res.success),
        message=str(res.message),

        n_events=n_events,
        se_gamma0=se_gamma0,
        se_alpha=se_alpha,
    )


def fit_univariate(events_1d, T, beta, se=True):
    """
    Fit a 1D Hawkes process with constant baseline.

    events_1d : 1D array of event times.
    beta : scalar decay rate (fixed).
    """
    events = [np.asarray(events_1d, dtype=float)]
    return _fit_general(events, T, beta=np.array([[beta]]), se=se)


def fit_multivariate(events, T, beta, se=True):
    """
    Fit an M-dimensional Hawkes process with constant baselines.

    events : list of M arrays of jump times.
    beta : (M, M) array of decay rates (fixed).
    """
    return _fit_general(events, T, beta=np.asarray(beta, dtype=float), se=se)


def fit_multivariate_with_covariates(events, T, beta, covariate, se=True):
    """
    Fit an M-dimensional Hawkes process with piecewise-constant covariates
    in the baseline: mu_m(t) = exp(gamma0_m + gamma_m^T X(t)).
    """
    p_cov = covariate.p
    return _fit_general(
        events, T, beta=np.asarray(beta, dtype=float),
        covariate=covariate, p_cov=p_cov, se=se,
    )
