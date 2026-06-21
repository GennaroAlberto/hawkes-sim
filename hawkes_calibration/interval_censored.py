r"""
Interval-censored calibration of Hawkes processes via the MBPP.

This module implements the loss functions, fitting procedure and forecasting
scheme of Rizoiu et al. (2022), "Interval-censored Hawkes processes".

The workflow.  We never fit the Hawkes process directly on interval-censored
counts -- it has no independent-increments property and no tractable
interval-censored likelihood.  Instead we fit the *MBPP* (a genuine Poisson
process, ``mbpp.py``), whose interval-censored negative log-likelihood is a sum
of Poisson terms in the interval compensators Xi(o_{i-1}, o_i].  Because the MBPP
and the Hawkes process share parameters (kappa, theta), the fitted MBPP
parameters approximate the generating Hawkes parameters.

Loss functions (Sections 4-5)
-----------------------------
* IC-LL (Eq. 19) -- the negative Poisson log-likelihood,
      L = sum_i Xi(o_{i-1},o_i]  -  sum_i C(o_{i-1},o_i] * log Xi(o_{i-1},o_i].
  This is the generalized KL divergence between the counts and the compensators
  (Proposition 8), i.e. it assumes each interval is Poisson distributed.
* SSE (Eq. 36) -- the squared-error Bregman loss,
      L = sum_i ( C(o_{i-1},o_i] - Xi(o_{i-1},o_i] )^2,
  obtained by swapping the KL generator for the squared-Euclidean generator; it
  assumes each interval is Gaussian.  Up to using xi in place of Xi and unit
  intervals it generalizes/corrects the HIP loss (Theorems 10-11).

Endogenous vs. non-endogenous (Section 6.3)
-------------------------------------------
In separable scenarios the exogenous events are observed directly, so we only
fit the *endogenous* response: the compensator uses Xi^endo = \int (h*s), and the
counts are the *offspring* counts.  This is selected with ``endogenous=True``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mbpp import MBPP, ExponentialKernel, kappa_theta_to_alpha_beta
from .exogenous import PiecewiseConstant, CovariateExogenous
from .operators import solve_mbpp_ltv
from .optim import minimize_bfgs


# ---------------------------------------------------------------------------
# Loss functions over interval compensators.
# ---------------------------------------------------------------------------
def ic_ll(counts, Xi, eps=1e-12):
    r"""Interval-censored negative log-likelihood (Eq. 19), constant dropped."""
    Xi = np.maximum(np.asarray(Xi, dtype=float), eps)
    counts = np.asarray(counts, dtype=float)
    return float(np.sum(Xi) - np.sum(counts * np.log(Xi)))


def sse_loss(counts, Xi):
    r"""Sum of squared errors Bregman loss (Eq. 36)."""
    Xi = np.asarray(Xi, dtype=float)
    counts = np.asarray(counts, dtype=float)
    return float(np.sum((counts - Xi) ** 2))


_LOSSES = {"ic-ll": ic_ll, "ic_ll": ic_ll, "kl": ic_ll,
           "sse": sse_loss, "squared": sse_loss}


# ---------------------------------------------------------------------------
# Augmentation of a piecewise-constant exogenous (scale + constant background).
# Mirrors the ACTIVE augmentation s_hat = nu + mu * s(t)  (Eq. 54, simplified).
# ---------------------------------------------------------------------------
def _augment_piecewise(exo: PiecewiseConstant, scale, background):
    rates = scale * exo.rates + background
    return PiecewiseConstant(exo.breaks, np.maximum(rates, 0.0))


# ---------------------------------------------------------------------------
# Parameter transforms keeping kappa in (0,1) and theta>0 during optimization.
# ---------------------------------------------------------------------------
def _sigmoid(u):
    u = np.clip(u, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-u))


def _logit(k):
    k = min(max(k, 1e-6), 1 - 1e-6)
    return np.log(k / (1.0 - k))


@dataclass
class ICFitResult:
    kappa: float
    theta: float
    loss: float
    loss_name: str
    success: bool
    n_intervals: int
    scale: float = 1.0
    background: float = 0.0
    n_events: int = 0
    gamma0: float | None = None
    gamma: np.ndarray | None = None
    delta: np.ndarray | None = None        # excitation-covariate coefficients
    baseline: float | None = None          # fitted constant baseline (excitation model)

    @property
    def alpha_beta(self):
        """Return the (alpha, beta) of phi(t)=alpha e^{-beta t} (=kappa*theta, theta)."""
        return kappa_theta_to_alpha_beta(self.kappa, self.theta)

    @property
    def branching_ratio(self):
        return self.kappa

    def summary(self):
        a, b = self.alpha_beta
        lines = [
            f"  loss ({self.loss_name}) = {self.loss:.4f}   intervals = {self.n_intervals}",
            f"  kappa (branching ratio) = {self.kappa:.4f}",
            f"  theta (decay)           = {self.theta:.4f}",
            f"  -> exponential kernel phi(t) = {a:.4f} * exp(-{b:.4f} t)",
        ]
        if self.gamma0 is not None:
            lines.append(f"  gamma0 (log-baseline)   = {self.gamma0:.4f}   (baseline mu0 = {np.exp(self.gamma0):.4f})")
            lines.append(f"  gamma (covariate coefs) = {np.asarray(self.gamma).tolist()}")
        if self.delta is not None:
            lines.append(f"  baseline mu             = {self.baseline:.4f}")
            lines.append(f"  delta (excitation-covariate coefs) = {np.asarray(self.delta).tolist()}")
            lines.append(f"  -> branching ratio kappa(t) = {self.kappa:.4f} * exp(delta^T Z(t))")
        if self.scale != 1.0 or self.background != 0.0:
            lines.append(f"  exogenous scale mu = {self.scale:.4f}, background nu = {self.background:.4f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core fit: recover (kappa, theta) from interval-censored counts.
# ---------------------------------------------------------------------------
def fit_mbpp_ic(
    obs_times,
    counts,
    exogenous,
    *,
    loss="ic-ll",
    endogenous=True,
    method="closed",
    kappa0=0.5,
    theta0=1.0,
    fit_scale=False,
    fit_background=False,
    n_restarts=1,
    seed=0,
    grid_dt=0.01,
    grid_T=None,
):
    r"""
    Fit MBPP parameters (kappa, theta) to interval-censored ``counts`` over the
    observation grid ``obs_times`` = [o_0, ..., o_m], using the given exogenous
    function.

    Parameters
    ----------
    obs_times : (m+1,) observation endpoints.
    counts : (m,) per-interval event counts C(o_{i-1}, o_i] (offspring counts in
        the separable / endogenous case).
    exogenous : Exogenous
        The exogenous function (MultiImpulse / LHPP for separable scenarios; the
        true s(t) for non-separable scenarios A and D).
    loss : {"ic-ll", "sse"}.
    endogenous : bool
        If True use the endogenous compensator Xi^endo (separable scenarios).
    method : {"closed", "numeric", "auto"}
        MBPP solution method (closed form for the exponential kernel).
    fit_scale, fit_background : bool
        Also fit a multiplicative scale and/or additive background on a
        piecewise-constant exogenous (Eq. 54).  Requires a PiecewiseConstant exo.
    n_restarts : int
        Number of random restarts (the loss surface can be non-convex, App. D).

    Returns
    -------
    ICFitResult
    """
    obs_times = np.asarray(obs_times, dtype=float)
    counts = np.asarray(counts, dtype=float)
    loss_fn = _LOSSES[loss]
    augmenting = fit_scale or fit_background
    if augmenting and not isinstance(exogenous, PiecewiseConstant):
        raise ValueError("fit_scale/fit_background require a PiecewiseConstant exogenous")

    def build_params(vec):
        kappa = float(np.clip(_sigmoid(vec[0]), 1e-4, 1.0 - 1e-4))
        theta = float(np.clip(np.exp(np.clip(vec[1], -20, 20)), 1e-4, 1e4))
        i = 2
        scale = float(np.exp(np.clip(vec[i], -20, 20))) if fit_scale else 1.0
        i += 1 if fit_scale else 0
        background = float(np.exp(np.clip(vec[i], -20, 20))) if fit_background else 0.0
        return kappa, theta, scale, background

    def objective(vec):
        kappa, theta, scale, background = build_params(vec)
        exo = exogenous
        if augmenting:
            exo = _augment_piecewise(exogenous, scale, background)
        mbpp = MBPP(ExponentialKernel(kappa, theta), exo, method=method,
                    grid_dt=grid_dt, grid_T=grid_T)
        Xi = mbpp.compensator_interval(obs_times, endogenous=endogenous)
        if not np.all(np.isfinite(Xi)):
            return 1e12
        return loss_fn(counts, Xi)

    rng = np.random.default_rng(seed)
    best = None
    for r in range(max(1, n_restarts)):
        if r == 0:
            v0 = [_logit(kappa0), np.log(theta0)]
            if fit_scale:
                v0.append(0.0)
            if fit_background:
                v0.append(np.log(1e-3))
        else:
            v0 = [_logit(rng.uniform(0.1, 0.9)), np.log(rng.uniform(0.3, 3.0))]
            if fit_scale:
                v0.append(rng.normal(0, 0.5))
            if fit_background:
                v0.append(np.log(rng.uniform(1e-3, 1.0)))
        v0 = np.array(v0, dtype=float)
        res = minimize_bfgs(objective, v0, max_iter=300, ftol=1e-10, gtol=1e-6)
        if best is None or res.fun < best.fun:
            best = res

    kappa, theta, scale, background = build_params(best.x)
    return ICFitResult(
        kappa=float(kappa),
        theta=float(theta),
        loss=float(best.fun),
        loss_name=loss,
        success=bool(best.success),
        n_intervals=int(counts.size),
        scale=float(scale),
        background=float(background),
        n_events=int(counts.sum()),
    )


def fit_mbpp_ic_multi(
    obs_times,
    counts_list,
    exogenous_list,
    *,
    loss="ic-ll",
    endogenous=True,
    method="closed",
    kappa0=0.5,
    theta0=1.0,
    n_restarts=1,
    seed=0,
):
    r"""
    Jointly fit (kappa, theta) across several i.i.d. interval-censored sequences
    that share the observation grid ``obs_times`` but each have their own counts
    and own exogenous function (own observed immigrants).  The objective is the
    sum of the per-sequence losses -- this is how the paper aggregates sequences
    to reduce variance (Section 7.2).

    Parameters
    ----------
    obs_times : (m+1,) shared observation endpoints.
    counts_list : list of (m,) per-sequence count vectors.
    exogenous_list : list of Exogenous, one per sequence.
    """
    obs_times = np.asarray(obs_times, dtype=float)
    loss_fn = _LOSSES[loss]

    def build(vec):
        return (float(np.clip(_sigmoid(vec[0]), 1e-4, 1 - 1e-4)),
                float(np.clip(np.exp(np.clip(vec[1], -20, 20)), 1e-4, 1e4)))

    def objective(vec):
        kappa, theta = build(vec)
        total = 0.0
        for counts, exo in zip(counts_list, exogenous_list):
            mbpp = MBPP(ExponentialKernel(kappa, theta), exo, method=method)
            Xi = mbpp.compensator_interval(obs_times, endogenous=endogenous)
            if not np.all(np.isfinite(Xi)):
                return 1e12
            total += loss_fn(counts, Xi)
        return total

    rng = np.random.default_rng(seed)
    best = None
    for r in range(max(1, n_restarts)):
        if r == 0:
            v0 = np.array([_logit(kappa0), np.log(theta0)])
        else:
            v0 = np.array([_logit(rng.uniform(0.1, 0.9)), np.log(rng.uniform(0.3, 3.0))])
        res = minimize_bfgs(objective, v0, max_iter=300, ftol=1e-10, gtol=1e-6)
        if best is None or res.fun < best.fun:
            best = res

    kappa, theta = build(best.x)
    total_events = int(sum(int(np.sum(c)) for c in counts_list))
    return ICFitResult(
        kappa=float(kappa), theta=float(theta), loss=float(best.fun), loss_name=loss,
        success=bool(best.success), n_intervals=int(np.asarray(counts_list[0]).size),
        n_events=total_events,
    )


# ---------------------------------------------------------------------------
# Fitting with a log-linear covariate baseline s(t) = exp(gamma0 + gamma^T X(t)).
# ---------------------------------------------------------------------------
def fit_mbpp_ic_covariates(
    obs_times,
    counts,
    covariate,
    *,
    loss="ic-ll",
    endogenous=False,
    method="closed",
    kappa0=0.5,
    theta0=1.0,
    gamma0_init=None,
    n_restarts=1,
    seed=0,
):
    r"""
    Fit an interval-censored MBPP whose exogenous baseline is driven by
    time-varying covariates through a log-linear link,

        s(t) = exp( gamma0 + gamma^T X(t) ),

    recovering the covariate coefficients ``(gamma0, gamma)`` jointly with the
    kernel parameters ``(kappa, theta)`` from interval-censored counts alone.

    This is the interval-censored analogue of the event-time covariate model in
    ``estimate.fit_multivariate_with_covariates``.  Because a piecewise-constant
    covariate keeps ``s(t)`` piecewise-constant, the closed-form MBPP solver is
    used directly (no new approximation).

    Parameters
    ----------
    obs_times : (m+1,) observation endpoints.
    counts : (m,) array, OR a list of (m,) arrays for several i.i.d. sequences
        that share both ``obs_times`` and the covariate (their losses are summed).
    covariate : object with ``.breakpoints`` and ``.values`` (p covariates), e.g.
        :class:`hawkes_calibration.PiecewiseConstantCovariate`.
    loss : {"ic-ll", "sse"}.
    endogenous : bool
        If True, fit only the endogenous response (separable immigrants).
    method : {"closed", "numeric", "auto"}.
    n_restarts : int
        Random restarts (the loss can be non-convex).

    Returns
    -------
    ICFitResult  (with ``gamma0`` and ``gamma`` populated).
    """
    obs_times = np.asarray(obs_times, dtype=float)
    loss_fn = _LOSSES[loss]
    counts_list = counts if isinstance(counts, (list, tuple)) else [counts]
    counts_list = [np.asarray(c, dtype=float) for c in counts_list]
    p = np.atleast_2d(np.asarray(covariate.values, dtype=float)).shape[1]

    def build(vec):
        kappa = float(np.clip(_sigmoid(vec[0]), 1e-4, 1 - 1e-4))
        theta = float(np.clip(np.exp(np.clip(vec[1], -20, 20)), 1e-4, 1e4))
        gamma0 = float(vec[2])
        gamma = np.asarray(vec[3:3 + p], dtype=float)
        return kappa, theta, gamma0, gamma

    def objective(vec):
        kappa, theta, gamma0, gamma = build(vec)
        exo = CovariateExogenous(covariate, gamma0, gamma)
        mbpp = MBPP(ExponentialKernel(kappa, theta), exo, method=method)
        # All sequences share the same covariate, hence the same compensator;
        # compute it once and sum the per-sequence losses.
        Xi = mbpp.compensator_interval(obs_times, endogenous=endogenous)
        if not np.all(np.isfinite(Xi)):
            return 1e12
        return sum(loss_fn(counts_i, Xi) for counts_i in counts_list)

    # crude initial gamma0 from the average count level
    if gamma0_init is None:
        mean_rate = max(np.mean([c.sum() for c in counts_list]) / max(obs_times[-1], 1.0), 1e-3)
        gamma0_init = float(np.log(0.5 * mean_rate))

    rng = np.random.default_rng(seed)
    best = None
    for r in range(max(1, n_restarts)):
        if r == 0:
            v0 = np.concatenate([[_logit(kappa0), np.log(theta0), gamma0_init], np.zeros(p)])
        else:
            v0 = np.concatenate([
                [_logit(rng.uniform(0.1, 0.9)), np.log(rng.uniform(0.3, 3.0)),
                 gamma0_init + rng.normal(0, 0.5)],
                rng.normal(0, 0.5, size=p),
            ])
        res = minimize_bfgs(objective, v0, max_iter=400, ftol=1e-10, gtol=1e-6)
        if best is None or res.fun < best.fun:
            best = res

    kappa, theta, gamma0, gamma = build(best.x)
    total_events = int(sum(int(c.sum()) for c in counts_list))
    return ICFitResult(
        kappa=float(kappa), theta=float(theta), loss=float(best.fun), loss_name=loss,
        success=bool(best.success), n_intervals=int(counts_list[0].size),
        n_events=total_events, gamma0=float(gamma0), gamma=gamma,
    )


# ---------------------------------------------------------------------------
# Fitting with COVARIATE-MODULATED EXCITATION (the general/LTV Volterra case).
# ---------------------------------------------------------------------------
def _excitation_modulation(Z, delta):
    """Return a callable t -> (1,1) array exp(delta^T Z(t)) for the LTV solver."""
    delta = np.atleast_1d(np.asarray(delta, dtype=float))

    def mod(t):
        z = np.atleast_2d(np.asarray(Z(t), dtype=float))[0]   # (p,)
        return np.array([[np.exp(np.clip(float(delta @ z), -30.0, 30.0))]])

    return mod


def fit_mbpp_ic_excitation(
    obs_times,
    counts,
    Z,
    *,
    loss="ic-ll",
    kappa0=0.4,
    theta0=1.0,
    mu0=None,
    n_sub=10,
    n_restarts=4,
    seed=0,
):
    r"""
    Fit a univariate MBPP whose **excitation depends on covariates**,

        alpha(t) = kappa * theta * exp(delta^T Z(t)),

    i.e. the time-varying branching ratio is kappa(t) = kappa * exp(delta^T Z(t))
    (Section 6 of the LaTeX notes).  The kernel is no longer a convolution, so the
    compensator is obtained by integrating the linear time-varying ODE of
    :func:`hawkes_calibration.solve_mbpp_ltv` on a refined grid; the loss is the
    usual interval-censored log-likelihood.

    Parameters
    ----------
    obs_times : (m+1,) observation endpoints.
    counts : (m,) array OR a list of (m,) arrays (i.i.d. sequences sharing Z).
    Z : callable t -> covariate vector (e.g. a ``PiecewiseConstantCovariate``).
    loss : {"ic-ll", "sse"}.
    kappa0, theta0, mu0 : initial base branching ratio, decay, constant baseline.
    n_sub : sub-steps per observation interval for the ODE integration.
    n_restarts : random restarts (the excitation-covariate loss is non-convex).

    Returns
    -------
    ICFitResult  (with ``kappa``, ``theta``, ``baseline``, ``delta`` populated).

    Notes
    -----
    Univariate by design: multivariate excitation covariates add up to p*M^2
    parameters and a rougher landscape (see the LaTeX notes, Section 6). The
    forward solver :func:`solve_mbpp_ltv` is already multivariate.
    """
    obs_times = np.asarray(obs_times, dtype=float)
    counts_list = counts if isinstance(counts, (list, tuple)) else [counts]
    counts_list = [np.asarray(c, dtype=float) for c in counts_list]
    loss_fn = _LOSSES[loss]
    p = np.atleast_1d(np.asarray(Z(obs_times[0]), dtype=float)).reshape(-1).size
    T = float(obs_times[-1])

    # refined integration grid: obs endpoints + sub-steps (so Xi is read exactly
    # at the observation times, and discontinuities in Z are resolved).
    fine = np.unique(np.concatenate(
        [np.linspace(obs_times[i], obs_times[i + 1], n_sub + 1) for i in range(obs_times.size - 1)]))
    if mu0 is None:
        mu0 = max(np.mean([c.sum() for c in counts_list]) / max(T, 1.0), 1e-2) * 0.5

    def build(vec):
        kappa = float(np.clip(_sigmoid(vec[0]), 1e-4, 1 - 1e-4))
        theta = float(np.clip(np.exp(np.clip(vec[1], -20, 20)), 1e-4, 1e4))
        mu = float(np.exp(np.clip(vec[2], -20, 20)))
        delta = np.asarray(vec[3:3 + p], dtype=float)
        return kappa, theta, mu, delta

    def interval_compensators(kappa, theta, mu, delta):
        A0 = np.array([[kappa * theta]])
        B = np.array([[theta]])
        mod = _excitation_modulation(Z, delta)
        # the optimiser may transiently visit supercritical (kappa(t)>1) regions
        # where the LTV ODE grows large; those over/invalid warnings are benign
        # (the objective rejects non-finite Xi), so silence them.
        with np.errstate(over="ignore", invalid="ignore"):
            _, Xi = solve_mbpp_ltv(lambda t: np.array([mu]), A0, B, fine,
                                   modulation=mod, return_compensator=True)
        Xi_obs = np.interp(obs_times, fine, Xi[:, 0])
        return np.diff(Xi_obs)

    def objective(vec):
        kappa, theta, mu, delta = build(vec)
        Xi = interval_compensators(kappa, theta, mu, delta)
        if not np.all(np.isfinite(Xi)):
            return 1e12
        return sum(loss_fn(c, Xi) for c in counts_list)

    rng = np.random.default_rng(seed)
    best = None
    for r in range(max(1, n_restarts)):
        if r == 0:
            v0 = np.concatenate([[_logit(kappa0), np.log(theta0), np.log(mu0)], np.zeros(p)])
        else:
            v0 = np.concatenate([[_logit(rng.uniform(0.1, 0.8)), np.log(rng.uniform(0.4, 2.0)),
                                  np.log(mu0 * rng.uniform(0.5, 1.5))], rng.normal(0, 0.5, size=p)])
        res = minimize_bfgs(objective, v0, max_iter=300, ftol=1e-10, gtol=1e-6)
        if best is None or res.fun < best.fun:
            best = res

    kappa, theta, mu, delta = build(best.x)
    total_events = int(sum(int(c.sum()) for c in counts_list))
    return ICFitResult(
        kappa=float(kappa), theta=float(theta), loss=float(best.fun), loss_name=loss,
        success=bool(best.success), n_intervals=int(counts_list[0].size),
        n_events=total_events, baseline=float(mu), delta=delta,
    )


# ---------------------------------------------------------------------------
# Forecasting future interval counts (Eq. 55, via the Prop. 7 lower bound).
# ---------------------------------------------------------------------------
def forecast_counts(
    kappa,
    theta,
    exogenous,
    obs_times,
    observed_counts,
    horizon_obs_times,
    extra_source_times=None,
    extra_source_counts=None,
):
    r"""
    Forecast future per-interval counts using the numerical lower-bound
    compensator of Proposition 7, in the form used for the ACTIVE popularity
    experiment (Eq. 55).

    The observed count process has intensity xi(t) = s_hat(t) + sum_{e} phi(t-e),
    i.e. it is directly generated by the (augmented) exogenous intensity s_hat
    and amplified by self-excitation.  The forecast on a future interval is

        pred[i] = \int s_hat  +  sum_{past + predicted counts} c_z * \int_{lo-z}^{hi-z} phi.

    The trick (Prop. 7): the expected MBPP count equals its compensator, so we
    replace the expected counts on *past* intervals by the *observed* counts, and
    feed each fresh prediction back in as a new excitation source.

    Parameters
    ----------
    kappa, theta : fitted exponential-kernel parameters.
    exogenous : Exogenous
        The (augmented) exogenous function s_hat valid over past + future.
    obs_times : (m+1,) past observation endpoints (fitted window).
    observed_counts : (m,) observed counts on the past intervals.
    horizon_obs_times : (h+1,) future observation endpoints, contiguous with
        obs_times (horizon_obs_times[0] == obs_times[-1]).
    extra_source_times, extra_source_counts : optional
        Additional excitation sources (e.g. observed immigrant interval centers
        and counts) that excite the observed process via phi but are not part of
        the predicted count -- used for the purely-endogenous (offspring) framing.

    Returns
    -------
    pred : (h,) predicted counts on the future intervals.
    """
    mbpp = MBPP(ExponentialKernel(kappa, theta), exogenous, method="auto")

    obs_times = np.asarray(obs_times, dtype=float)
    observed_counts = np.asarray(observed_counts, dtype=float)
    future = np.asarray(horizon_obs_times, dtype=float)

    # Anchor each source interval's mass at its RIGHT edge (most recent instant).
    # For unit intervals this telescopes to exactly kappa of excitation per
    # source, i.e. it is mass-exact at stationarity; centre-anchoring would lose
    # the near-field mass int_0^{dt/2} phi.
    src_edges = list(obs_times[1:])
    src_counts = list(observed_counts)
    if extra_source_times is not None:
        src_edges += list(np.asarray(extra_source_times, dtype=float))
        src_counts += list(np.asarray(extra_source_counts, dtype=float))

    preds = []
    for h in range(future.size - 1):
        lo, hi = future[h], future[h + 1]
        exo_mass = float(mbpp._S_eval(np.array([hi]))[0] - mbpp._S_eval(np.array([lo]))[0])
        endo = 0.0
        for ez, nz in zip(src_edges, src_counts):
            a = max(lo - ez, 0.0)
            b = max(hi - ez, 0.0)
            endo += nz * mbpp._kernel_integral(a, b)
        pred = exo_mass + endo
        preds.append(pred)
        src_edges.append(hi)
        src_counts.append(pred)

    return np.array(preds)
