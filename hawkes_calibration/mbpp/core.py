r"""
The Mean Behavior Poisson process (MBPP).

This module implements the central object of Rizoiu et al. (2022),
"Interval-censored Hawkes processes" (JMLR 23(1), paper 21-0917): the Mean
Behavior Poisson process.

Motivation.  A Hawkes process has a *stochastic* intensity lambda(t) that
depends on the realized history, so it has neither the independent-increments
property nor a tractable likelihood when we only observe interval-censored
counts.  The MBPP is the non-homogeneous Poisson process whose *deterministic*
intensity equals the expected Hawkes intensity over all realizations,

    xi(t) = E_{H_t}[ lambda(t) ] = s(t) + (xi * phi)(t),              (Eq. 9)

a Volterra integral equation of the second kind.  Because the MBPP *is* a
Poisson process it has independent increments, so its interval-censored
log-likelihood (IC-LL, see ``interval_censored.py``) is a product of Poisson
terms.  There is a one-to-one parameter correspondence between a Hawkes process
and "its" MBPP, so fitting the MBPP on interval-censored counts recovers
(approximations of) the Hawkes parameters.

Two ways to solve Eq. (9) are provided:

1. Closed form (Theorem 1 / Theorem 2 + Corollary 3).  Eq. (9) defines a
   causal LTI system with impulse response E(t) = delta(t) + h(t), where
   h = sum_{n>=1} phi^{*n}.  For the exponential kernel phi(t)=kappa*theta*e^{-theta t}
   the infinite convolution sum collapses to

       h(t) = kappa*theta * exp((kappa-1)*theta*t),  t > 0            (Eq. 14)

   and xi = (E * s) = s + h*s has a closed form for every exogenous function in
   ``exogenous.py``.  The compensator obeys the *same* self-consistent equation
   (Eq. 10), Xi(t) = S(t) + (Xi * phi)(t), so Xi = S + h*S.

2. Numerical approximation, for kernels with no closed form (power-law,
   Rayleigh).  We approximate h ~ sum_{n=1}^{N} phi^{*n} by repeated discrete
   convolution, form xi = s + h*s on a grid, and integrate for Xi.  We also
   provide the lower/upper-bound compensator approximation of Proposition 7,
   which can reuse *observed* interval counts in place of expected counts -- the
   basis of the forecasting scheme in ``interval_censored.py``.

Parameterisation.  Following the paper we work with the univariate exponential
kernel phi(t) = kappa*theta*e^{-theta t}, where ``kappa`` in (0,1) is the
branching ratio (n* = \int phi = kappa, the mean number of direct offspring per
event) and ``theta`` > 0 is the decay rate.  The relationship to the
(alpha, beta) convention used elsewhere in this package (phi(t)=alpha e^{-beta t})
is alpha = kappa*theta and beta = theta, i.e. kappa = alpha/beta.  Helpers
``kappa_theta_to_alpha_beta`` and ``alpha_beta_to_kappa_theta`` convert between
the two.
"""

from __future__ import annotations

import numpy as np

from .exogenous import Dassios


# ---------------------------------------------------------------------------
# Parameter conversions between (kappa, theta) and (alpha, beta).
# ---------------------------------------------------------------------------
def kappa_theta_to_alpha_beta(kappa, theta):
    """phi(t) = kappa*theta*e^{-theta t}, i.e. alpha*e^{-beta t} with
    alpha=kappa*theta and beta=theta."""
    return kappa * theta, theta


def alpha_beta_to_kappa_theta(alpha, beta):
    """Inverse of the above: kappa = alpha/beta (the branching ratio), theta = beta."""
    return alpha / beta, beta


# ---------------------------------------------------------------------------
# Kernels.
# ---------------------------------------------------------------------------
class ExponentialKernel:
    r"""phi(t) = kappa*theta*exp(-theta t),  branching ratio n* = kappa."""

    has_closed_form = True

    def __init__(self, kappa, theta):
        self.kappa = float(kappa)
        self.theta = float(theta)

    @property
    def branching_ratio(self):
        return self.kappa

    def __call__(self, t):
        t = np.asarray(t, dtype=float)
        return np.where(t > 0, self.kappa * self.theta * np.exp(-self.theta * t), 0.0)

    def impulse_response_h(self, t):
        r"""h(t) = sum_{n>=1} phi^{*n}(t) = kappa*theta*exp((kappa-1)*theta t), t>0 (Eq. 14)."""
        t = np.asarray(t, dtype=float)
        c = (self.kappa - 1.0) * self.theta
        return np.where(t > 0, self.kappa * self.theta * np.exp(c * t), 0.0)

    def value_at_zero_plus(self):
        return self.kappa * self.theta


class PowerLawKernel:
    r"""
    Power-law kernel, normalised so that the branching ratio is ``kappa``:

        phi(t) = kappa * theta * c^theta * (t + c)^{-(1+theta)},   t > 0,

    giving \int_0^infty phi = kappa.  No closed-form MBPP solution exists; the
    MBPP must be solved numerically.  Heavy-tailed kernels of this form are the
    best performers on social-media data (Rizoiu et al., 2017b; Mishra et al., 2016).
    """

    has_closed_form = False

    def __init__(self, kappa, theta, c=1.0):
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.c = float(c)

    @property
    def branching_ratio(self):
        return self.kappa

    def __call__(self, t):
        t = np.asarray(t, dtype=float)
        c, th, ka = self.c, self.theta, self.kappa
        return np.where(t > 0, ka * th * c**th * (t + c) ** (-(1.0 + th)), 0.0)

    def value_at_zero_plus(self):
        return self.kappa * self.theta / self.c


# ---------------------------------------------------------------------------
# The MBPP.
# ---------------------------------------------------------------------------
class MBPP:
    r"""
    Mean Behavior Poisson process for a given kernel and exogenous function.

    Parameters
    ----------
    kernel : ExponentialKernel or PowerLawKernel
    exogenous : Exogenous
        One of the functions in ``exogenous.py``.
    method : {"auto", "closed", "numeric"}
        "closed" uses the exponential-kernel impulse-response solution; "numeric"
        uses the finite-convolution approximation; "auto" picks closed form when
        both the kernel and exogenous function support it, else numeric.
    grid_dt, grid_T : float
        Time step and horizon for the numerical solver.
    n_conv_terms : int
        Maximum number of convolution powers in h ~ sum_{n=1}^{N} phi^{*n}.
    """

    def __init__(
        self, kernel, exogenous, method="auto", grid_dt=0.01, grid_T=None, n_conv_terms=200
    ):
        self.kernel = kernel
        self.exo = exogenous
        self.grid_dt = float(grid_dt)
        self.grid_T = grid_T
        self.n_conv_terms = int(n_conv_terms)

        closed_ok = getattr(kernel, "has_closed_form", False) and exogenous.has_closed_form_exp()
        if method == "auto":
            method = "closed" if closed_ok else "numeric"
        if method == "closed" and not closed_ok:
            raise ValueError("closed form unavailable for this kernel/exogenous combination")
        self.method = method
        self._grid = None  # lazily built numeric solution cache

    # =====================================================================
    # Intensity xi(t) and compensator Xi(0, t].
    # =====================================================================
    def intensity(self, t):
        r"""
        MBPP intensity xi(t) = s(t) + (h * s)(t).

        For an impulse-type exogenous (MultiImpulse) the s(t) Dirac comb is
        omitted (it is 0 almost everywhere), so this returns the smooth
        endogenous response -- i.e. what is plotted as the MBPP curve.
        """
        t = np.atleast_1d(np.asarray(t, dtype=float))
        if self.method == "closed":
            endo = self._endo_response_closed(t)
            s = 0.0 if self.exo.is_impulse else self._s_eval(t)
            return s + endo
        else:
            return self._numeric_eval(t, which="intensity")

    def endogenous_intensity(self, t):
        r"""Endogenous part only, (h * s)(t) (Eq. 47)."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        if self.method == "closed":
            return self._endo_response_closed(t)
        return self._numeric_eval(t, which="endo_intensity")

    def compensator(self, t):
        r"""Cumulative compensator Xi(t) = Xi(0, t] = S(t) + (h * S)(t)."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        if self.method == "closed":
            endo = self._endo_compensator_closed(t)
            S = self._S_eval(t)
            return S + endo
        return self._numeric_eval(t, which="compensator")

    def endogenous_compensator(self, t):
        r"""Endogenous compensator Xi^endo(t) = \int_0^t (h*s)(u) du (Eq. 51)."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        if self.method == "closed":
            return self._endo_compensator_closed(t)
        return self._numeric_eval(t, which="endo_compensator")

    def compensator_interval(self, obs_times, endogenous=False):
        r"""
        Vector of interval compensators Xi(o_{i-1}, o_i] = Xi(o_i) - Xi(o_{i-1})
        for the observation grid ``obs_times`` = [o_0, ..., o_m] (Eq. 20).
        """
        obs_times = np.asarray(obs_times, dtype=float)
        comp = self.endogenous_compensator if endogenous else self.compensator
        Xi = comp(obs_times)
        return np.diff(Xi)

    # =====================================================================
    # Closed-form helpers (exponential kernel).
    # =====================================================================
    def _s_eval(self, t):
        if isinstance(self.exo, Dassios):
            return self.exo.s_of(t, self.kernel.kappa, self.kernel.theta)
        return self.exo.intensity(t)

    def _S_eval(self, t):
        if isinstance(self.exo, Dassios):
            # S(t) = \int_0^t [kappa*theta + (u0-kappa*theta)e^{-theta u}] du
            ka, th = self.kernel.kappa, self.kernel.theta
            u0 = self.exo.u0
            t = np.asarray(t, dtype=float)
            return ka * th * t + (u0 - ka * th) * (1.0 - np.exp(-th * t)) / th
        return self.exo.cumulative(t)

    def _endo_response_closed(self, t):
        return self.exo.endo_response_exp(t, self.kernel.kappa, self.kernel.theta)

    def _endo_compensator_closed(self, t):
        return self.exo.endo_compensator_exp(t, self.kernel.kappa, self.kernel.theta)

    # =====================================================================
    # Numerical solver (general kernel): h ~ sum_{n=1}^N phi^{*n}, xi = s + h*s.
    # =====================================================================
    def _build_numeric(self):
        if self._grid is not None:
            return self._grid
        T = self.grid_T
        if T is None:
            # default horizon: cover the exogenous support generously
            if hasattr(self.exo, "breaks"):
                T = float(self.exo.breaks[-1]) * 2.0
            elif hasattr(self.exo, "times") and self.exo.times.size:
                T = float(self.exo.times[-1]) * 2.0 + 10.0
            else:
                T = 60.0
        dt = self.grid_dt
        grid = np.arange(0.0, T + dt, dt)
        phi = self.kernel(grid)
        # Replace the zero-sample (the kernel is defined for t>0) by its right
        # limit phi(0+), so the dt-scaled discrete convolution carries the right
        # mass near the origin -- this removes the leading O(dt) bias.
        if hasattr(self.kernel, "value_at_zero_plus"):
            phi[0] = self.kernel.value_at_zero_plus()

        # h = sum_{n>=1} phi^{*n}  (discrete convolution, dt-scaled)
        h = phi.copy()
        term = phi.copy()
        for _ in range(2, self.n_conv_terms + 1):
            term = dt * np.convolve(term, phi)[: grid.size]
            h += term
            if term.sum() * dt < 1e-12:
                break
        h0 = h[0]

        # endogenous response (h * s), with a trapezoidal endpoint correction
        if self.exo.is_impulse:
            endo = np.zeros_like(grid)
            for sz, wz in zip(self.exo.times, self.exo.weights):
                shift = int(round(sz / dt))
                if shift < grid.size:
                    endo[shift:] += wz * h[: grid.size - shift]
            s_grid = np.zeros_like(grid)
        else:
            s_grid = self.exo.intensity(grid)
            endo = dt * np.convolve(h, s_grid)[: grid.size]
            # trapezoid correction: discrete convolution is a left Riemann sum;
            # remove half-weight of the two endpoints of each running integral.
            endo = endo - 0.5 * dt * (h[: grid.size] * s_grid[0] + h0 * s_grid)

        xi = s_grid + endo
        # cumulative compensators via trapezoid
        Xi_endo = np.concatenate([[0.0], np.cumsum(0.5 * (endo[1:] + endo[:-1]) * dt)])
        S_grid = self.exo.cumulative(grid) if not self.exo.is_impulse else self.exo.cumulative(grid)
        Xi = S_grid + Xi_endo

        self._grid = dict(
            grid=grid, phi=phi, h=h, s=s_grid, endo=endo, xi=xi, Xi=Xi, Xi_endo=Xi_endo, dt=dt
        )
        return self._grid

    def _numeric_eval(self, t, which):
        g = self._build_numeric()
        grid = g["grid"]
        key = dict(
            intensity="xi", endo_intensity="endo", compensator="Xi", endo_compensator="Xi_endo"
        )[which]
        return np.interp(t, grid, g[key])

    # =====================================================================
    # Proposition 6-7: lower-bound compensator from (expected or observed) counts.
    # =====================================================================
    def compensator_lower_bound(self, t, approx_times, interval_counts):
        r"""
        Numeric lower-bound compensator Xi^-(t) of Proposition 7, which reuses
        per-interval expected counts E[M(d_{j-1}, d_j]] = ``interval_counts`` over
        the approximation grid ``approx_times`` = [d_0, ..., d_D]:

            Xi^-(t) = S(t)
                + sum_{j: d_j < t} E[M(d_{j-1}, d_j]] * \int_{d_j}^{min(t,d_D)} phi(y) dy.

        When the expected counts are replaced by *observed* interval counts this
        is exactly the forecasting recursion used on the ACTIVE data (Eq. 55).
        """
        t = np.atleast_1d(np.asarray(t, dtype=float))
        d = np.asarray(approx_times, dtype=float)
        counts = np.asarray(interval_counts, dtype=float)
        dD = d[-1]
        S = self._S_eval(t) if self.method == "closed" else self.exo.cumulative(t)
        out = np.array(S, dtype=float)
        for n, tn in enumerate(t):
            acc = 0.0
            upper = min(tn, dD)
            for j in range(1, len(d)):
                if d[j] >= tn:
                    break
                # contribution of interval (d_{j-1}, d_j], anchored at d_j
                acc += counts[j - 1] * self._kernel_integral(d[j], upper)
            out[n] += acc
        return out

    def _kernel_integral(self, lo, hi):
        r"""\int_{lo}^{hi} phi(y) dy for the configured kernel (lo, hi >= 0)."""
        if hi <= lo:
            return 0.0
        k = self.kernel
        if isinstance(k, ExponentialKernel):
            # \int kappa*theta e^{-theta y} dy = kappa(e^{-theta lo} - e^{-theta hi})
            return k.kappa * (np.exp(-k.theta * lo) - np.exp(-k.theta * hi))
        if isinstance(k, PowerLawKernel):
            th, c, ka = k.theta, k.c, k.kappa

            def F(y):
                return -ka * c**th * (y + c) ** (-th)

            return F(hi) - F(lo)
        # generic fallback: fine quadrature
        ys = np.linspace(lo, hi, 256)
        return float(np.trapz(k(ys), ys))


# ---------------------------------------------------------------------------
# Convenience: an MBPP straight from (kappa, theta) and an exogenous function.
# ---------------------------------------------------------------------------
def make_mbpp(kappa, theta, exogenous, method="auto", **kw):
    """Build an MBPP with an exponential kernel of branching ratio ``kappa``."""
    return MBPP(ExponentialKernel(kappa, theta), exogenous, method=method, **kw)
