r"""
Exogenous intensity functions s(t) for the Mean Behavior Poisson process
(MBPP) and for interval-censored Hawkes calibration.

This module implements the exogenous (immigrant) intensity functions used in
Rizoiu et al. (2022), "Interval-censored Hawkes processes", JMLR 23(1).  In the
Hawkes / MBPP intensity

    lambda(t) = s(t) + (kernel * dN)(t)            (Hawkes)
    xi(t)     = s(t) + (h * s)(t)                  (MBPP, h = endogenous impulse
                                                    response, Eq. 9)

the function s(t) controls the arrival of *immigrants* (exogenous events), while
the kernel controls the *offspring* (endogenous self-excitation).

Each exogenous object exposes:

* ``intensity(t)``    -- s(t), the exogenous rate (NaN/inf at impulses);
* ``cumulative(t)``   -- S(t) = \int_0^t s(u) du, the expected immigrant count;
* for the *exponential* MBPP kernel phi(t) = kappa*theta*exp(-theta t), the two
  closed-form building blocks used to assemble the MBPP intensity/compensator:
      ``endo_response_exp(t, kappa, theta)``     -- (h * s)(t)
      ``endo_compensator_exp(t, kappa, theta)``  -- \int_0^t (h * s)(u) du
  where h(t) = kappa*theta*exp((kappa-1)*theta*t), t>0  (Eq. 14).

The two paper-specific exogenous functions are:

* ``MultiImpulse``  (Definition 12): exogenous events observed as *event times*;
  s(t) = sum_z delta(t - s_z).  Used for separable scenarios B and E.
* ``LHPP``          (Definition 14, Proposition 13): exogenous events observed
  *interval-censored*; the maximum-likelihood piecewise-constant rate over each
  observation interval is lambda_i = S(q_{i-1}, q_i] / (q_i - q_{i-1}).  Used for
  separable scenarios C and F (and the real-world ACTIVE experiment).

The remaining classes (Constant, Rectangle, PiecewiseConstant, Sine, Dassios)
reproduce the exogenous functions used in the impulse-response figures (Figs 2-3)
and the HP-pc / HP-sin synthetic datasets.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Numerically-stable helpers for the exponential-kernel closed forms.
# c := (kappa - 1) * theta < 0 for a subcritical process (0 < kappa < 1).
# ---------------------------------------------------------------------------
def _c(kappa, theta):
    return (kappa - 1.0) * theta


def _expm1_over_c(x, c):
    """(exp(c x) - 1) / c, stable as c -> 0 via the series limit x + c x^2/2 ..."""
    c = float(c)
    x = np.asarray(x, dtype=float)
    if abs(c) < 1e-12:
        return x * (1.0 + 0.5 * c * x)
    return np.expm1(c * x) / c


class Exogenous:
    """Base class for exogenous intensity functions s(t)."""

    is_impulse = False  # True for Dirac-comb functions (MultiImpulse)

    # -- s(t) and its integral S(t) -----------------------------------------
    def intensity(self, t):
        raise NotImplementedError

    def cumulative(self, t):
        raise NotImplementedError

    # -- exponential-kernel closed-form endogenous response ------------------
    # (h * s)(t), with h(t) = kappa*theta*exp((kappa-1)*theta t).  Subclasses
    # that have an analytical form override these; otherwise the MBPP falls back
    # to numerical convolution / quadrature.
    def endo_response_exp(self, t, kappa, theta):
        raise NotImplementedError

    def endo_compensator_exp(self, t, kappa, theta):
        raise NotImplementedError

    def has_closed_form_exp(self):
        return True


# ---------------------------------------------------------------------------
# Piecewise-constant exogenous (covers Constant, Rectangle, LHPP, HP-pc).
# ---------------------------------------------------------------------------
class PiecewiseConstant(Exogenous):
    r"""
    Piecewise-constant exogenous intensity.

    ``s(t) = rates[i]`` on the half-open interval ``(breaks[i], breaks[i+1]]``,
    and 0 outside ``(breaks[0], breaks[-1]]``.

    Parameters
    ----------
    breaks : (m+1,) array, strictly increasing, breaks[0] is the left edge.
    rates  : (m,) array of non-negative rates, one per interval.
    """

    def __init__(self, breaks, rates):
        self.breaks = np.asarray(breaks, dtype=float)
        self.rates = np.asarray(rates, dtype=float)
        assert self.breaks.ndim == 1 and self.rates.ndim == 1
        assert self.breaks.size == self.rates.size + 1
        assert np.all(np.diff(self.breaks) > 0)

    def intensity(self, t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        out = np.zeros_like(t)
        idx = np.searchsorted(self.breaks, t, side="left") - 1
        # event exactly at a left edge belongs to the interval starting there
        inside = (t > self.breaks[0]) & (t <= self.breaks[-1])
        idx = np.clip(idx, 0, self.rates.size - 1)
        out[inside] = self.rates[idx[inside]]
        return out

    def cumulative(self, t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        widths = np.diff(self.breaks)
        seg_mass = self.rates * widths
        cum_at_break = np.concatenate([[0.0], np.cumsum(seg_mass)])  # at each break
        out = np.empty_like(t)
        for n, tn in enumerate(t):
            if tn <= self.breaks[0]:
                out[n] = 0.0
            elif tn >= self.breaks[-1]:
                out[n] = cum_at_break[-1]
            else:
                i = int(np.searchsorted(self.breaks, tn, side="right") - 1)
                out[n] = cum_at_break[i] + self.rates[i] * (tn - self.breaks[i])
        return out

    # -- exponential-kernel response: superpose rectangle responses ----------
    def _rect_response(self, t, a, b, c, K):
        """(h * (1_{(a,b]}))(t) * K, with K = kappa/(1-kappa)."""
        t = np.asarray(t, dtype=float)
        out = np.zeros_like(t)
        mid = (t > a) & (t <= b)
        out[mid] = K * (1.0 - np.exp(c * (t[mid] - a)))
        post = t > b
        out[post] = K * (np.exp(c * (t[post] - b)) - np.exp(c * (t[post] - a)))
        return out

    def _rect_compensator(self, t, a, b, c, K):
        r"""\int_0^t of the rectangle response above."""
        t = np.asarray(t, dtype=float)
        out = np.zeros_like(t)
        mid = (t > a) & (t <= b)
        tm = t[mid]
        out[mid] = K * ((tm - a) - _expm1_over_c(tm - a, c))
        post = t > b
        tp = t[post]
        # value accumulated up to b plus the tail integral from b to t
        val_at_b = K * ((b - a) - _expm1_over_c(b - a, c))
        tail = K * (
            _expm1_over_c(tp - b, c)
            - (np.exp(c * (tp - a)) - np.exp(c * (b - a))) / (c if abs(c) > 1e-12 else 1e-12)
        )
        out[post] = val_at_b + tail
        return out

    def endo_response_exp(self, t, kappa, theta):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        c = _c(kappa, theta)
        K = kappa / (1.0 - kappa)
        out = np.zeros_like(t)
        for i in range(self.rates.size):
            if self.rates[i] == 0.0:
                continue
            out += self.rates[i] * self._rect_response(t, self.breaks[i], self.breaks[i + 1], c, K)
        return out

    def endo_compensator_exp(self, t, kappa, theta):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        c = _c(kappa, theta)
        K = kappa / (1.0 - kappa)
        out = np.zeros_like(t)
        for i in range(self.rates.size):
            if self.rates[i] == 0.0:
                continue
            out += self.rates[i] * self._rect_compensator(
                t, self.breaks[i], self.breaks[i + 1], c, K
            )
        return out


class Constant(PiecewiseConstant):
    """Constant exogenous intensity s(t) = nu on (0, horizon]."""

    def __init__(self, nu, horizon):
        super().__init__(breaks=[0.0, float(horizon)], rates=[float(nu)])


class Rectangle(PiecewiseConstant):
    """Rectangle exogenous intensity s(t) = height * 1_{(a,b]}."""

    def __init__(self, a, b, height=1.0):
        super().__init__(breaks=[float(a), float(b)], rates=[float(height)])


class LHPP(PiecewiseConstant):
    r"""
    Latent Homogeneous Poisson Process exogenous function (Definition 14).

    Given observation times Q = {q_0, ..., q_m} and the immigrant volume
    S(q_{i-1}, q_i] observed in each interval, the maximum-likelihood
    piecewise-constant exogenous rate is (Proposition 13)

        lambda_i = S(q_{i-1}, q_i] / (q_i - q_{i-1}).

    This is the exogenous function used when the *immigrants* are observed
    interval-censored (scenarios C and F).
    """

    def __init__(self, obs_times, immigrant_counts):
        obs_times = np.asarray(obs_times, dtype=float)
        counts = np.asarray(immigrant_counts, dtype=float)
        assert obs_times.size == counts.size + 1, "need len(obs_times) == len(counts)+1"
        widths = np.diff(obs_times)
        rates = counts / widths
        super().__init__(breaks=obs_times, rates=rates)
        self.immigrant_counts = counts


# ---------------------------------------------------------------------------
# Log-linear covariate baseline as an exogenous function.
# ---------------------------------------------------------------------------
class CovariateExogenous(PiecewiseConstant):
    r"""
    Exogenous intensity driven by time-varying covariates through a log-linear
    link, the interval-censored analogue of the event-time baseline in
    ``covariates.py``:

        s(t) = exp( gamma0 + gamma^T X(t) ).

    The key observation (and why no new MBPP machinery is needed): when the
    covariate ``X(t)`` is *piecewise-constant*, so is ``s(t)``, so this is just a
    ``PiecewiseConstant`` exogenous whose per-interval rates are
    ``exp(gamma0 + gamma^T X_k)``.  All the closed-form intensity/compensator
    formulas are therefore inherited unchanged; only the parameterisation of the
    rates differs.  This lets the interval-censored fitter recover the covariate
    coefficients ``gamma`` jointly with the kernel parameters ``(kappa, theta)``.

    Parameters
    ----------
    covariate : object with ``.breakpoints`` (K+1,) and ``.values`` (K, p)
        e.g. a :class:`hawkes_calibration.PiecewiseConstantCovariate`.
    gamma0 : float
        Log-baseline intercept.
    gamma : (p,) array_like
        Covariate coefficients.

    For a continuous covariate, sample it onto a fine grid and pass a
    piecewise-constant approximation (or use ``MBPP(method="numeric")``).
    """

    def __init__(self, covariate, gamma0, gamma):
        self.covariate = covariate
        self.gamma0 = float(gamma0)
        self.gamma = np.atleast_1d(np.asarray(gamma, dtype=float))
        breaks = np.asarray(covariate.breakpoints, dtype=float)
        values = np.atleast_2d(np.asarray(covariate.values, dtype=float))
        if values.shape[0] != breaks.size - 1 and values.shape[1] == breaks.size - 1:
            values = values.T
        rates = np.exp(np.clip(self.gamma0 + values @ self.gamma, -30.0, 30.0))
        super().__init__(breaks=breaks, rates=rates)


# ---------------------------------------------------------------------------
# Multi-impulse exogenous (Definition 12): exogenous events as event times.
# ---------------------------------------------------------------------------
class MultiImpulse(Exogenous):
    r"""
    Multi-impulse exogenous function (Definition 12):

        s(t) = sum_{z} w_z * delta(t - s_z),

    a Dirac comb at the observed immigrant event times ``s_z`` (weights w_z=1 by
    default).  s(t) is a generalized function and is not directly evaluable; we
    therefore only use it through its endogenous response and its cumulative
    count S(t) = #{z : s_z <= t}.  Used for separable scenarios B and E together
    with the *endogenous* loss functions of Section 6.3.
    """

    is_impulse = True

    def __init__(self, times, weights=None):
        self.times = np.sort(np.asarray(times, dtype=float))
        if weights is None:
            self.weights = np.ones_like(self.times)
        else:
            w = np.asarray(weights, dtype=float)
            self.weights = w[np.argsort(np.asarray(times, dtype=float))]

    def intensity(self, t):
        # Dirac comb: 0 almost everywhere, undefined at the impulses.
        t = np.atleast_1d(np.asarray(t, dtype=float))
        return np.zeros_like(t)

    def cumulative(self, t):
        """S(t) = sum of weights of impulses at or before t (a step function)."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        cw = np.concatenate([[0.0], np.cumsum(self.weights)])
        idx = np.searchsorted(self.times, t, side="right")
        return cw[idx]

    def endo_response_exp(self, t, kappa, theta):
        r"""(h * s)(t) = sum_z w_z * kappa*theta*exp((kappa-1)*theta*(t - s_z)) 1[t>s_z]."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        c = _c(kappa, theta)
        out = np.zeros_like(t)
        for sz, wz in zip(self.times, self.weights):
            m = t > sz
            out[m] += wz * kappa * theta * np.exp(c * (t[m] - sz))
        return out

    def endo_compensator_exp(self, t, kappa, theta):
        r"""\int_0^t (h*s) = sum_z w_z * (kappa/(1-kappa))
        * (1 - exp((kappa-1)theta(t-s_z))) 1[t>s_z]."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        c = _c(kappa, theta)
        K = kappa / (1.0 - kappa)
        out = np.zeros_like(t)
        for sz, wz in zip(self.times, self.weights):
            m = t > sz
            out[m] += wz * K * (1.0 - np.exp(c * (t[m] - sz)))
        return out


# ---------------------------------------------------------------------------
# Sine and Dassios exogenous functions (used in the impulse-response figures).
# Intensity has an exponential-kernel closed form (Table 10, rows V-VI); the
# compensator is obtained by accurate quadrature in mbpp.py when needed.
# ---------------------------------------------------------------------------
class Sine(Exogenous):
    r"""Sinusoidal exogenous intensity s(t) = sin(t) + alpha (Table 10, row VI)."""

    def __init__(self, alpha=2.0):
        self.alpha = float(alpha)

    def intensity(self, t):
        t = np.asarray(t, dtype=float)
        return np.sin(t) + self.alpha

    def cumulative(self, t):
        t = np.asarray(t, dtype=float)
        return (1.0 - np.cos(t)) + self.alpha * t

    def has_closed_form_exp(self):
        return True

    def endo_response_exp(self, t, kappa, theta):
        """(h * s)(t) = xi(t) - s(t), with xi the closed form of Table 10 row VI."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        a = self.alpha
        c = _c(kappa, theta)
        denom = 1.0 + theta**2 - 2.0 * kappa * theta**2 + kappa**2 * theta**2
        const = -a / (kappa - 1.0)
        expc = (
            kappa
            / (kappa - 1.0)
            * (
                a
                + a * theta**2
                - 2 * a * kappa * theta**2
                + a * kappa**2 * theta**2
                + theta * kappa
                - theta
            )
            / denom
        ) * np.exp(c * t)
        trig = (
            np.sin(t)
            + theta**2 * np.sin(t)
            - kappa * theta**2 * np.sin(t)
            - kappa * theta * np.cos(t)
        ) / denom
        xi = const + expc + trig
        return xi - self.intensity(t)

    def endo_compensator_exp(self, t, kappa, theta):
        # Use accurate quadrature of the closed-form response (smooth integrand).
        return _quadrature_compensator(self.endo_response_exp, t, kappa, theta)


class Dassios(Exogenous):
    r"""
    Dassios-Zhao exogenous intensity (Eq. 12):

        s(t) = kappa*theta + (u0 - kappa*theta) * exp(-theta t).

    Because this particular exogenous form is defined *in terms of* the kernel
    parameters, the kernel's (kappa, theta) are supplied at construction.  For
    the exponential kernel the MBPP intensity then has the closed form (Eq. 16)

        xi(t) = kappa*theta/(1-kappa) (1 - exp(-(1-kappa) theta t)) + u0 exp(-(1-kappa) theta t).
    """

    def __init__(self, u0=0.0, kappa=0.6, theta=0.8):
        self.u0 = float(u0)
        self.kappa = float(kappa)
        self.theta = float(theta)

    def s_of(self, t, kappa, theta):
        t = np.asarray(t, dtype=float)
        return kappa * theta + (self.u0 - kappa * theta) * np.exp(-theta * t)

    def intensity(self, t):
        return self.s_of(t, self.kappa, self.theta)

    def cumulative(self, t):
        t = np.asarray(t, dtype=float)
        ka, th, u0 = self.kappa, self.theta, self.u0
        return ka * th * t + (u0 - ka * th) * (1.0 - np.exp(-th * t)) / th

    def xi_closed_form(self, t, kappa, theta):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        d = (1.0 - kappa) * theta
        return kappa * theta / (1.0 - kappa) * (1.0 - np.exp(-d * t)) + self.u0 * np.exp(-d * t)

    def endo_response_exp(self, t, kappa, theta):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        return self.xi_closed_form(t, kappa, theta) - self.s_of(t, kappa, theta)

    def endo_compensator_exp(self, t, kappa, theta):
        return _quadrature_compensator(self.endo_response_exp, t, kappa, theta)


def _quadrature_compensator(response_fn, t, kappa, theta, n_grid=20000):
    r"""
    Accurate \int_0^t response_fn(u) du for each t, via a single fine trapezoid
    grid on [0, max(t)].  Used for smooth closed-form responses (Sine, Dassios)
    where a hand-derived antiderivative is not worth the risk.
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    tmax = float(t.max()) if t.size else 0.0
    if tmax <= 0:
        return np.zeros_like(t)
    grid = np.linspace(0.0, tmax, n_grid)
    vals = response_fn(grid, kappa, theta)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (vals[1:] + vals[:-1]) * np.diff(grid))])
    return np.interp(t, grid, cum)
