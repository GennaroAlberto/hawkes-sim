"""Stability-constrained sector count fitting.

The first sector model used nonnegative lag excitation but did not constrain the
aggregate excitation matrix to be subcritical.  In forward simulation this can make
rates explode.  This module provides a drop-in replacement for
``fit_sector_count_model`` that enforces a noncritical aggregate excitation.

The stabilized model uses an additive Hawkes-style weekly rate,

    Lambda[s,t] = exp(a[s] + beta[s]' X[t]) + sum_{r,l} b[s,r,l] Y[r,t-l],

rather than putting the lagged counts inside a log link.  With ``b >= 0`` the
summed-lag matrix ``G = sum_l B_l`` has the usual branching interpretation.  We
enforce ``rho(G) < 1`` through a row-sum sufficient condition: for nonnegative
matrices, Perron--Frobenius/Gershgorin gives ``rho(G) <= max_s sum_r G[s,r]``.
"""

from __future__ import annotations

import numpy as np

try:  # SciPy is optional; constrained fitting uses SLSQP when available.
    from scipy.optimize import minimize

    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover - exercised only without SciPy
    from ._opt_fallback import minimize

    _HAVE_SCIPY = False

from .sector_ranker import SectorCountResult


def aggregate_excitation(excitation):
    """Return the summed-over-lags sector excitation matrix ``G``.

    ``excitation`` may be either ``(M, M, L)`` or already ``(M, M)``.
    """
    b = np.asarray(excitation, dtype=float)
    if b.ndim == 2:
        return b
    if b.ndim != 3:
        raise ValueError("excitation must have shape (M, M, L) or (M, M)")
    return b.sum(axis=2)


def excitation_spectral_radius(excitation):
    """Spectral radius of the summed lag excitation matrix."""
    g = aggregate_excitation(excitation)
    if g.size == 0:
        return 0.0
    vals = np.linalg.eigvals(g)
    return float(np.max(np.abs(vals)))


def _project_to_spectral_radius(b, max_radius):
    """Scale a nonnegative lag tensor so ``rho(sum_l b_l) <= max_radius``."""
    b = np.asarray(b, dtype=float).copy()
    rho = excitation_spectral_radius(b)
    if max_radius is not None and rho > float(max_radius) and rho > 0:
        b *= float(max_radius) / rho
    return b


def sector_rate_at(result: SectorCountResult, counts_history, covariates, week: int):
    """One-step sector rates, supporting the stabilized additive sector model."""
    counts_history = np.asarray(counts_history, dtype=float)
    covariates = np.asarray(covariates, dtype=float)
    x = covariates[int(week)] if covariates.size else np.zeros(0)
    link = getattr(result, "rate_link", "log")

    if link == "additive":
        baseline_eta = result.intercept + result.beta @ x
        baseline = np.exp(np.clip(baseline_eta, -30.0, 20.0))
        # Guard against exp-baseline extrapolation: a held-out covariate vector
        # outside the training support can otherwise drive exp(a + beta'x) to the
        # rate clip (~4.85e8) and produce catastrophic one-step NLL.  Predictions
        # are winsorized to the envelope of fitted *training* baselines (times a
        # margin), which is the most the training data can justify.
        lo = getattr(result, "baseline_low", None)
        hi = getattr(result, "baseline_high", None)
        if lo is not None and hi is not None:
            margin = float(getattr(result, "baseline_clip_margin", 2.0))
            baseline = np.clip(baseline, np.asarray(lo) / margin, np.asarray(hi) * margin)
        lam = baseline.copy()
        for lag in range(1, result.n_lags + 1):
            lam += result.excitation[:, :, lag - 1] @ counts_history[int(week) - lag]
        return np.maximum(lam, 1e-12)

    eta = result.intercept + result.beta @ x
    for lag in range(1, result.n_lags + 1):
        eta += result.excitation[:, :, lag - 1] @ counts_history[int(week) - lag]
    return np.exp(np.clip(eta, -30.0, 20.0))


def fit_sector_count_model(
    sector_counts,
    covariates=None,
    *,
    n_lags=4,
    train_end=None,
    l2=1e-3,
    l2_beta=None,
    l1_excitation=0.0,
    adjacency=None,
    nonnegative_excitation=True,
    max_excitation_radius=0.95,
    stability_method="row_sum",
    project_if_unstable=True,
    baseline_clip_margin=2.0,
    max_iter=500,
):
    r"""Fit a stable additive sector Hawkes/count model.

    Model:

    .. math::

        Y_{s,t} \sim \mathrm{Poisson}(\Lambda_{s,t}),\qquad
        \Lambda_{s,t} = \exp(a_s + \beta_s^\top X_t)
          + \sum_{r=1}^M\sum_{\ell=1}^L b_{sr\ell}Y_{r,t-\ell}.

    By default ``b >= 0`` and the aggregate excitation

    .. math::

        G_{sr} = \sum_{\ell=1}^L b_{sr\ell}

    is constrained to be noncritical.  The implemented constraint is the row-sum
    sufficient condition

    .. math::

        \sum_{r,\ell} b_{sr\ell} \le \rho_{\max} < 1 \quad\text{for every receiver }s,

    which implies ``spectral_radius(G) <= rho_max`` for nonnegative ``G``.
    ``adjacency`` may be an ``(M, M)`` hard mask; entries with ``False`` are forced
    to zero for every lag.

    ``l2_beta`` is the ridge on the covariate coefficients only (defaults to
    ``max(l2, 0.05)``): with sparse weekly counts, a nearly unpenalized ``beta``
    can grow large enough that ``exp(a + beta'x)`` explodes on held-out
    covariates outside the training support.  For the same reason, one-step
    *predicted* baselines are winsorized to the envelope of the fitted training
    baselines times ``baseline_clip_margin`` (set ``baseline_clip_margin=None``
    to disable).

    ``l1_excitation`` adds an L1 penalty ``lambda * sum(b)`` on the (nonnegative)
    lag coefficients.  In sparse weekly data the additive likelihood is happy to
    attribute any two events in adjacent weeks to excitation, so the unpenalized
    fit drives row sums to the stability bound and overfits badly out of sample;
    because ``b >= 0`` the penalty is linear and smooth, and it produces exact
    zeros.  A value around ``1.0`` is a good default for sparse (<~0.5
    events/sector-week) data; ``0.0`` reproduces the old behaviour.

    If SciPy is unavailable, the function falls back to the unconstrained box fit
    and then projects the aggregate excitation back to ``max_excitation_radius``.
    """
    y = np.asarray(sector_counts, dtype=float)
    if y.ndim != 2:
        raise ValueError("sector_counts must be a (T, M) array")
    T, M = y.shape
    if covariates is None:
        X = np.zeros((T, 0), dtype=float)
    else:
        X = np.asarray(covariates, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
    p = X.shape[1]
    L = int(n_lags)
    if train_end is None:
        train_end = T
    train_end = int(train_end)
    if train_end <= L:
        raise ValueError("train_end must be larger than n_lags")

    if max_excitation_radius is not None:
        max_excitation_radius = float(max_excitation_radius)
        if max_excitation_radius <= 0:
            raise ValueError("max_excitation_radius must be positive")
        if not nonnegative_excitation:
            raise ValueError("the row-sum stability constraint requires nonnegative excitation")
    if l1_excitation and not nonnegative_excitation:
        raise ValueError("l1_excitation requires nonnegative_excitation=True")

    if adjacency is None:
        mask = np.ones((M, M), dtype=bool)
    else:
        mask = np.asarray(adjacency, dtype=bool)
        if mask.shape != (M, M):
            raise ValueError("adjacency must have shape (M, M)")

    # Sparse weekly counts leave beta nearly unidentified along some directions;
    # without a real ridge, exp(a + beta'x) can explode on held-out covariates.
    l2_beta = max(float(l2), 0.05) if l2_beta is None else float(l2_beta)

    n_intercept = M
    n_beta = M * p
    n_b = M * M * L
    n = n_intercept + n_beta + n_b
    start_b = n_intercept + n_beta

    mean_rate = np.maximum(y[L:train_end].mean(axis=0), 1e-3)
    x0 = np.zeros(n, dtype=float)
    x0[:M] = np.log(mean_rate)
    b0 = np.zeros((M, M, L), dtype=float)
    b0[mask, :] = 0.01 / max(L, 1)
    if max_excitation_radius is not None:
        b0 = _project_to_spectral_radius(b0, 0.5 * max_excitation_radius)
    x0[start_b:] = b0.ravel()

    bounds = [(None, None)] * n
    for idx in range(n_b):
        s = idx // (M * L)
        rem = idx % (M * L)
        r = rem // L
        allowed = bool(mask[s, r])
        if not allowed:
            bounds[start_b + idx] = (0.0, 0.0)
        elif nonnegative_excitation:
            bounds[start_b + idx] = (0.0, None)

    weeks = np.arange(L, train_end)

    def unpack(theta):
        intercept = theta[:M]
        beta = theta[M : M + n_beta].reshape(M, p) if p else np.zeros((M, 0))
        b = theta[start_b:].reshape(M, M, L)
        return intercept, beta, b

    def pack(intercept, beta, b):
        return np.concatenate([intercept, beta.ravel(), b.ravel()])

    def objective(theta):
        intercept, beta, b = unpack(theta)
        loss = 0.0
        grad_intercept = np.zeros(M)
        grad_beta = np.zeros((M, p))
        grad_b = np.zeros((M, M, L))

        for t in weeks:
            base_eta = intercept.copy()
            if p:
                base_eta += beta @ X[t]
            base = np.exp(np.clip(base_eta, -30.0, 20.0))
            lam = base.copy()
            for lag in range(1, L + 1):
                lam += b[:, :, lag - 1] @ y[t - lag]
            lam = np.maximum(lam, 1e-12)
            err = 1.0 - y[t] / lam
            loss += float(np.sum(lam - y[t] * np.log(lam)))
            grad_intercept += err * base
            if p:
                grad_beta += (err * base)[:, None] * X[t][None, :]
            for lag in range(1, L + 1):
                grad_b[:, :, lag - 1] += err[:, None] * y[t - lag][None, :]

        loss += 0.5 * l2_beta * np.sum(beta**2) + 0.5 * l2 * np.sum(b**2)
        grad_beta += l2_beta * beta
        grad_b += l2 * b
        if l1_excitation:
            # b >= 0 on allowed entries, so the L1 term is just a linear penalty.
            loss += float(l1_excitation) * float(np.sum(b[mask, :]))
            grad_b[mask, :] += float(l1_excitation)
        grad_b[~mask, :] = 0.0
        grad = np.concatenate([grad_intercept, grad_beta.ravel(), grad_b.ravel()])
        return loss, grad

    use_row_sum = (
        max_excitation_radius is not None
        and stability_method == "row_sum"
        and nonnegative_excitation
        and _HAVE_SCIPY
    )
    if use_row_sum:

        def row_margin(theta):
            _, _, b = unpack(theta)
            return max_excitation_radius - b.sum(axis=(1, 2))

        def row_margin_jac(theta):
            _ = theta
            jac = np.zeros((M, n), dtype=float)
            for s in range(M):
                row_start = start_b + s * M * L
                jac[s, row_start : row_start + M * L] = -1.0
            return jac

        opt = minimize(
            lambda th: objective(th),
            x0,
            jac=True,
            method="SLSQP",
            bounds=bounds,
            constraints=({"type": "ineq", "fun": row_margin, "jac": row_margin_jac},),
            options={"maxiter": int(max_iter), "ftol": 1e-9, "disp": False},
        )
    else:
        opt = minimize(
            lambda th: objective(th),
            x0,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(max_iter), "ftol": 1e-9},
        )

    intercept, beta, b = unpack(opt.x)
    b[~mask, :] = 0.0
    if nonnegative_excitation:
        b = np.maximum(b, 0.0)

    projected = False
    rho_before_projection = excitation_spectral_radius(b)
    if (
        project_if_unstable
        and max_excitation_radius is not None
        and rho_before_projection > max_excitation_radius
    ):
        b = _project_to_spectral_radius(b, max_excitation_radius)
        projected = True

    theta_final = pack(intercept, beta, b)
    final_loss = float(objective(theta_final)[0])
    rho = excitation_spectral_radius(b)
    row_sum_max = float(aggregate_excitation(b).sum(axis=1).max()) if M else 0.0
    message = str(opt.message)
    if projected:
        message += f"; excitation projected from rho={rho_before_projection:.6g} to rho={rho:.6g}"
    elif use_row_sum:
        message += f"; row-sum stability constraint <= {max_excitation_radius:g}"

    res = SectorCountResult(
        intercept=intercept,
        beta=beta,
        excitation=b,
        n_lags=L,
        train_end=train_end,
        loss=final_loss,
        success=bool(opt.success)
        and (max_excitation_radius is None or rho <= max_excitation_radius + 1e-8),
        message=message,
    )
    # Backward-compatible dynamic attributes; SectorCountResult is intentionally not
    # slotted, so existing code can ignore these while tests/docs can report them.
    res.spectral_radius = rho
    res.max_excitation_radius = max_excitation_radius
    res.max_row_sum = row_sum_max
    res.stability_method = stability_method
    res.rate_link = "additive"
    if baseline_clip_margin is not None:
        # Envelope of the fitted baselines over the training weeks: one-step
        # predictions are winsorized to this range (times the margin) so a
        # held-out covariate outside the training support cannot blow up the
        # exp baseline (see sector_rate_at).
        train_eta = intercept[None, :] + (X[weeks] @ beta.T if p else 0.0)
        train_base = np.exp(np.clip(train_eta, -30.0, 20.0))
        res.baseline_low = train_base.min(axis=0)
        res.baseline_high = train_base.max(axis=0)
        res.baseline_clip_margin = float(baseline_clip_margin)
    return res


# Patch the original module when this module is imported after sector_ranker.  This
# keeps ``from hawkes_calibration.sector_ranker import fit_sector_count_model`` and
# the simulator/backtest code on the stable additive rate path.
try:  # pragma: no cover - import side effect only
    import hawkes_calibration.sector_ranker as _sector_ranker

    _sector_ranker.fit_sector_count_model = fit_sector_count_model
    _sector_ranker.sector_rate_at = sector_rate_at
except Exception:
    pass


__all__ = [
    "aggregate_excitation",
    "excitation_spectral_radius",
    "fit_sector_count_model",
    "sector_rate_at",
]
