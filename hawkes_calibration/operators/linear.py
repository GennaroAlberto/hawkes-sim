r"""
Functional / operator views of the Mean Behavior Poisson process.

The MBPP solution map  G : s |-> xi  (forcing -> intensity) defined by the
Volterra equation  xi = s + xi * phi  (Eq. 9) is a *linear, translation-invariant
operator*.  This module exposes several interchangeable representations of that
operator, so the same MBPP can be solved/learned in whichever way suits the
problem.  All of them are selectable through :class:`FunctionalMBPP`.

Backends
--------
1. ``"ode"`` -- state-space reduction (this file, :func:`solve_mbpp_ode`).
   A convolution Volterra equation reduces to a *finite linear ODE* iff the
   kernel has a rational Laplace transform, i.e. iff it is a sum of exponentials
   phi(t) = sum_q a_q e^{-b_q t}.  Then with state u_q = (a_q e^{-b_q .} * xi),

       u'(t) = A u(t) + a s(t),   xi(t) = s(t) + 1^T u(t),
       A = a 1^T - diag(b),

   an LTI system *forced by the (covariate) baseline s(t)*.  Solving it by
   integration handles ANY functional form of the forcing and any sum-of-exp
   kernel.  Exact (to integrator tolerance); reduces to the single first-order
   ODE  y' = (kappa-1) theta y + kappa theta s  for the exponential kernel.

2. ``"spectral"`` -- Fourier neural operator (:class:`SpectralOperator`).
   In frequency space the operator is the pointwise multiplier
   xi_hat(w) = R(w) s_hat(w), R(w) = 1 / (1 - phi_hat(w)) (Theorem 1).  A single
   linear FNO layer with spectral weights R(w) *is* the exact operator; learning
   R(w) from (forcing, response) pairs is nonparametric kernel identification,
   recovering phi_hat = 1 - 1/R.

3. ``"deeponet"`` -- a small learned branch/trunk operator surrogate
   (:class:`DeepONetOperator`, see ``operators_nn.py``), for the nonlinear /
   no-closed-form regime.

The amortized inverse map counts -> parameters lives in
:class:`AmortizedInference` (``operators_nn.py``).
"""

from __future__ import annotations

import numpy as np

from ..mbpp.core import ExponentialKernel, PowerLawKernel


# ===========================================================================
# Kernel <-> sum-of-exponentials representation.
# ===========================================================================
def kernel_exponentials(kernel):
    r"""
    Return arrays (a, b) such that phi(t) = sum_q a_q exp(-b_q t).

    Exact for :class:`ExponentialKernel` (a=[kappa*theta], b=[theta]); for a
    :class:`PowerLawKernel` we return a sum-of-exponentials *approximation*
    (a rational-Laplace surrogate), fitted on a log-spaced grid of rates.
    """
    if isinstance(kernel, ExponentialKernel):
        return np.array([kernel.kappa * kernel.theta]), np.array([kernel.theta])
    if isinstance(kernel, PowerLawKernel):
        return _powerlaw_to_exponentials(kernel)
    raise TypeError("kernel must be ExponentialKernel or PowerLawKernel")


def _powerlaw_to_exponentials(kernel, Q=12, t_max=None):
    """Least-squares fit phi_pl(t) ~ sum_q a_q exp(-b_q t) on a time grid."""
    th, c, ka = kernel.theta, kernel.c, kernel.kappa
    t_max = t_max or 50.0 * c
    t = np.linspace(1e-3, t_max, 4000)
    target = kernel(t)
    b = np.logspace(np.log10(1.0 / t_max), np.log10(50.0 / c), Q)
    B = np.exp(-np.outer(t, b))               # (T, Q)
    a, *_ = np.linalg.lstsq(B, target, rcond=None)
    a = np.maximum(a, 0.0)                      # keep the kernel non-negative
    return a, b


def make_exp_sum_kernel(a, b):
    """A callable phi(t)=sum_q a_q e^{-b_q t} with branching ratio sum a_q/b_q."""
    a = np.asarray(a, float); b = np.asarray(b, float)

    def phi(t):
        t = np.asarray(t, float)
        return np.where(t > 0, (a[:, None] * np.exp(-np.outer(b, np.maximum(t, 0)))).sum(0).reshape(np.shape(t)), 0.0)

    phi.a, phi.b = a, b
    phi.branching_ratio = float(np.sum(a / b))
    return phi


# ===========================================================================
# (1) State-space ODE backend.
# ===========================================================================
def solve_mbpp_ode(forcing, a, b, t_grid, return_compensator=False):
    r"""
    Solve the MBPP intensity xi (and optionally compensator Xi) for a
    sum-of-exponentials kernel phi(t)=sum_q a_q e^{-b_q t} and an arbitrary
    forcing function ``forcing`` (a callable s(t)), via the linear state-space
    ODE  u' = A u + a s,  xi = s + 1^T u,  integrated with RK4.

    Parameters
    ----------
    forcing : callable
        Vectorised or scalar s(t), the exogenous baseline (the ODE forcing term).
    a, b : (Q,) arrays
        Kernel exponential weights and rates.
    t_grid : (N,) increasing array starting at 0.
    return_compensator : bool
        If True also return Xi(t) = \int_0^t xi (cumulative trapezoid).

    Returns
    -------
    xi : (N,) array   (and Xi : (N,) if return_compensator).
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    Q = a.size
    A = np.outer(a, np.ones(Q)) - np.diag(b)        # A_{qp} = a_q - b_q [q==p]
    t_grid = np.asarray(t_grid, float)
    N = t_grid.size

    def s_at(t):
        return float(np.asarray(forcing(np.atleast_1d(t))).reshape(-1)[0])

    def f(t, u):
        return a * s_at(t) + A @ u

    u = np.zeros(Q)
    xi = np.empty(N)
    xi[0] = s_at(t_grid[0]) + u.sum()
    for n in range(1, N):
        t0, t1 = t_grid[n - 1], t_grid[n]
        dt = t1 - t0
        k1 = f(t0, u)
        k2 = f(t0 + dt / 2, u + dt / 2 * k1)
        k3 = f(t0 + dt / 2, u + dt / 2 * k2)
        k4 = f(t1, u + dt * k3)
        u = u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        xi[n] = s_at(t1) + u.sum()
    if return_compensator:
        Xi = np.concatenate([[0.0], np.cumsum(0.5 * (xi[1:] + xi[:-1]) * np.diff(t_grid))])
        return xi, Xi
    return xi


def solve_mbpp_ode_multivariate(forcing, A, B, t_grid, return_compensator=False):
    r"""
    Multivariate MBPP intensity for an exponential kernel *matrix*.

    For an M-dimensional point process the MBPP intensity is a vector
    xi(t) in R^M solving  xi = s + Phi * xi  where Phi(u) is the M x M matrix of
    exponential kernels Phi_{m,j}(u) = A_{m,j} e^{-B_{m,j} u}.  Introducing the
    M^2 states y_{m,j} = (A_{m,j} e^{-B_{m,j} .}) * xi_j gives the linear ODE

        y'_{m,j} = A_{m,j} xi_j - B_{m,j} y_{m,j},   xi_m = s_m + sum_j y_{m,j},

    integrated here with RK4.  This is the high-dimensional ground-truth solver
    used to train the TensorFlow operators in ``operators_tf.py``.

    Parameters
    ----------
    forcing : callable t -> (M,) array   (the multivariate baseline s(t)).
    A, B : (M, M) arrays   (kernel weights a_{m,j} and decays b_{m,j}>0).
        Stationarity requires the branching matrix G=A/B to have spectral radius<1.
    t_grid : (N,) increasing grid starting at 0.

    Returns
    -------
    xi : (N, M) array of intensities  (and Xi : (N, M) compensators if requested).
    """
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    M = A.shape[0]
    t_grid = np.asarray(t_grid, float)
    N = t_grid.size

    def s_at(t):
        return np.asarray(forcing(t), float).reshape(M)

    def deriv(Y, s):
        xi = s + Y.sum(axis=1)               # (M,)
        return A * xi[None, :] - B * Y        # (M, M)

    Y = np.zeros((M, M))
    xi_out = np.empty((N, M))
    xi_out[0] = s_at(t_grid[0]) + Y.sum(axis=1)
    for n in range(1, N):
        t0, t1 = t_grid[n - 1], t_grid[n]
        dt = t1 - t0
        s0, smid, s1 = s_at(t0), s_at(0.5 * (t0 + t1)), s_at(t1)
        k1 = deriv(Y, s0)
        k2 = deriv(Y + dt / 2 * k1, smid)
        k3 = deriv(Y + dt / 2 * k2, smid)
        k4 = deriv(Y + dt * k3, s1)
        Y = Y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        xi_out[n] = s1 + Y.sum(axis=1)
    if return_compensator:
        Xi = np.concatenate([[np.zeros(M)],
                             np.cumsum(0.5 * (xi_out[1:] + xi_out[:-1]) * np.diff(t_grid)[:, None], axis=0)])
        return xi_out, Xi
    return xi_out


def solve_mbpp_ltv(forcing, A0, B, t_grid, modulation=None, return_compensator=False):
    r"""
    Multivariate MBPP with **covariate-modulated excitation** -- the case where the
    triggering strength itself depends on covariates,
    ``alpha_{m,j}(t) = A0_{m,j} * exp(delta_{m,j}^T Z(t))``.

    The mean-intensity equation then has a *non-convolution* kernel
    ``K_{m,j}(t,u) = A0_{m,j} m_{m,j}(t) g_{m,j}(t-u)`` (it depends on ``t`` and ``u``
    separately, not only on ``t-u``), so the system is no longer time-invariant: a
    general linear Volterra equation of the second kind.  For the exponential
    ``g_{m,j}(t-u)=e^{-B_{m,j}(t-u)}`` the kernel is separable, so it still reduces
    to a finite **linear time-varying (LTV)** ODE in the states
    ``y_{m,j}(t)=\int_0^t e^{-B_{m,j}(t-u)} xi_j(u) du``:

        y'_{m,j} = xi_j - B_{m,j} y_{m,j},
        xi_m     = s_m + sum_j A0_{m,j} m_{m,j}(t) y_{m,j},

    integrated here with RK4.  With ``modulation = 1`` this coincides exactly with
    :func:`solve_mbpp_ode_multivariate`; a time-varying ``modulation`` makes the
    excitation wax and wane with the covariates.

    Parameters
    ----------
    forcing : callable t -> (M,) baseline s(t).
    A0 : (M, M) base excitation weights (the covariate-free part).
    B : (M, M) decays.
    t_grid : (N,) grid from 0.
    modulation : callable t -> (M, M) giving m_{m,j}(t)=exp(delta^T Z(t)); default 1.

    Returns
    -------
    xi : (N, M)  (and Xi : (N, M) if requested).
    """
    A0 = np.asarray(A0, float); B = np.asarray(B, float)
    M = A0.shape[0]
    t_grid = np.asarray(t_grid, float)
    N = t_grid.size
    if modulation is None:
        modulation = lambda t: np.ones((M, M))

    def s_at(t):
        return np.asarray(forcing(t), float).reshape(M)

    def xi_of(Y, s, mod):
        return s + (A0 * mod * Y).sum(axis=1)          # (M,)

    def deriv(Y, s, mod):
        xi = xi_of(Y, s, mod)                          # (M,)
        return xi[None, :] - B * Y                     # (M,M): y'_{mj}=xi_j-B_{mj}y_{mj}

    Y = np.zeros((M, M))
    xi_out = np.empty((N, M))
    xi_out[0] = xi_of(Y, s_at(t_grid[0]), modulation(t_grid[0]))
    for n in range(1, N):
        t0, t1 = t_grid[n - 1], t_grid[n]
        dt = t1 - t0
        tm = 0.5 * (t0 + t1)
        s0, sm, s1 = s_at(t0), s_at(tm), s_at(t1)
        m0, mm, m1 = modulation(t0), modulation(tm), modulation(t1)
        k1 = deriv(Y, s0, m0)
        k2 = deriv(Y + dt / 2 * k1, sm, mm)
        k3 = deriv(Y + dt / 2 * k2, sm, mm)
        k4 = deriv(Y + dt * k3, s1, m1)
        Y = Y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        xi_out[n] = xi_of(Y, s1, m1)
    if return_compensator:
        Xi = np.concatenate([[np.zeros(M)],
                             np.cumsum(0.5 * (xi_out[1:] + xi_out[:-1]) * np.diff(t_grid)[:, None], axis=0)])
        return xi_out, Xi
    return xi_out


def solve_mbpp_volterra(forcing, kernel, t_grid, M=1, return_compensator=False):
    r"""
    General second-kind Volterra solver (trapezoid Nystrom) for the MBPP mean
    intensity with an **arbitrary** (possibly non-convolution, multivariate)
    kernel:

        xi(t) = s(t) + int_0^t K(t,u) xi(u) du.

    This is the fallback for cases with no closed form and no finite ODE: a
    power-law (or other non-separable) triggering shape, with or without
    covariate-modulated excitation (LaTeX notes, Appendix A.4(iii)).  It converges
    at first order, O(dt) (the startup corner of the MBPP intensity caps the
    trapezoid order, as usual for Volterra--Nystrom), and costs O(N^2 M^2); prefer
    the closed-form or LTV-ODE solvers whenever they apply.  Validated to agree
    with the closed-form MBPP (convolution), ``solve_mbpp_ltv`` (excitation) and
    the numeric power-law MBPP.

    Parameters
    ----------
    forcing : callable t -> (M,) baseline s(t)  (scalar accepted for M=1).
    kernel : callable (t, u) -> (M, M) kernel matrix K(t,u)  (scalar for M=1).
        For a convolution kernel pass ``lambda t, u: Phi(t-u)``; for
        covariate-modulated excitation pass ``lambda t, u: A0*mod(t)*g(t-u)``.
    t_grid : (N,) increasing grid starting at 0.
    M : int, dimension.

    Returns
    -------
    xi : (N, M)  (and Xi : (N, M) cumulative compensator if requested).
    """
    t = np.asarray(t_grid, float)
    N = t.size
    I = np.eye(M)

    def s_at(tt):
        return np.atleast_1d(np.asarray(forcing(tt), float)).reshape(M)

    def K_at(tt, uu):
        return np.asarray(kernel(tt, uu), float).reshape(M, M)

    xi = np.zeros((N, M))
    xi[0] = s_at(t[0])                       # integral over [0,0] is zero
    for i in range(1, N):
        ti = t[i]
        # trapezoid sum of K(ti, t_k) xi_k over panels [t_{k-1}, t_k], k=1..i,
        # leaving the unknown end term (h_i/2) K(ti,ti) xi_i implicit.
        acc = s_at(ti).copy()
        for k in range(1, i + 1):
            h = t[k] - t[k - 1]
            acc += 0.5 * h * (K_at(ti, t[k - 1]) @ xi[k - 1])
            if k < i:
                acc += 0.5 * h * (K_at(ti, t[k]) @ xi[k])
        h_i = t[i] - t[i - 1]
        Kii = K_at(ti, ti)
        xi[i] = np.linalg.solve(I - 0.5 * h_i * Kii, acc)
    if return_compensator:
        Xi = np.concatenate([[np.zeros(M)],
                             np.cumsum(0.5 * (xi[1:] + xi[:-1]) * np.diff(t)[:, None], axis=0)])
        return xi, Xi
    return xi


# ===========================================================================
# (2) Spectral (Fourier) neural operator.
# ===========================================================================
class SpectralOperator:
    r"""
    Linear Fourier neural operator for the MBPP:  xi = IFFT( R(w) * FFT(s) ).

    The operator is diagonal in frequency, with transfer function
    R(w) = 1 / (1 - phi_hat(w)).  It can be either *constructed exactly* from a
    known kernel, or *learned* from (forcing, response) example pairs by
    per-frequency complex least squares -- nonparametric operator/kernel
    identification, after which the kernel is recovered as phi_hat = 1 - 1/R.

    Parameters
    ----------
    t_grid : (N,) uniform grid starting at 0 (defines the frequencies).
    pad : int
        Zero-padding factor to suppress circular-convolution wraparound (>=2).
    """

    def __init__(self, t_grid, pad=4):
        self.t = np.asarray(t_grid, float)
        self.dt = float(self.t[1] - self.t[0])
        self.N = self.t.size
        self.M = int(pad) * self.N
        self.R = None           # learned/!constructed transfer function (length M//2+1)

    # -- exact construction from a known kernel --------------------------
    def from_kernel(self, kernel):
        a, b = kernel_exponentials(kernel)
        w = 2 * np.pi * np.fft.rfftfreq(self.M, d=self.dt)
        # phi_hat(w) for a one-sided sum of exponentials: sum a_q / (b_q + i w)
        phi_hat = np.sum(a[:, None] / (b[:, None] + 1j * w[None, :]), axis=0)
        self.R = 1.0 / (1.0 - phi_hat)
        self._phi_hat = phi_hat
        return self

    # -- learn the operator from data ------------------------------------
    def fit(self, S, XI, ridge=1e-6):
        r"""
        Learn R(w) from arrays of forcing/response pairs.

        Parameters
        ----------
        S, XI : (K, N) arrays
            K example forcings and the corresponding MBPP intensities, sampled on
            ``t_grid``.
        """
        Sf = np.fft.rfft(S, n=self.M, axis=1)        # (K, F)
        Xf = np.fft.rfft(XI, n=self.M, axis=1)
        num = np.sum(Xf * np.conj(Sf), axis=0)
        den = np.sum(np.abs(Sf) ** 2, axis=0) + ridge
        self.R = num / den
        self._phi_hat = 1.0 - 1.0 / self.R
        return self

    # -- apply the operator ----------------------------------------------
    def __call__(self, s_samples):
        """Apply xi = R * s in frequency space to a forcing sampled on t_grid."""
        if self.R is None:
            raise RuntimeError("operator not set; call from_kernel or fit first")
        sf = np.fft.rfft(np.asarray(s_samples, float), n=self.M)
        xi = np.fft.irfft(self.R * sf, n=self.M)[: self.N]
        return xi

    # -- recover the time-domain kernel ----------------------------------
    def recover_kernel(self):
        """Return (t, phi_recovered) by inverse-transforming phi_hat = 1 - 1/R."""
        phi_t = np.fft.irfft(self._phi_hat, n=self.M)[: self.N] / self.dt
        return self.t, phi_t


# ===========================================================================
# Unified interface: pick a backend for the same forcing -> intensity map.
# ===========================================================================
class FunctionalMBPP:
    r"""
    A single ``solve(forcing, t_grid)`` interface to the MBPP operator with a
    selectable backend, so the same problem can be handled by the exact ODE
    reduction or by a (neural) operator.

    Parameters
    ----------
    kernel : ExponentialKernel or PowerLawKernel (used by "ode"/"spectral").
    method : {"ode", "spectral", "deeponet"}.
        * "ode"      -- exact state-space integration (sum-of-exp kernel);
        * "spectral" -- exact Fourier operator R(w)=1/(1-phi_hat);
        * "deeponet" -- a trained :class:`~operators_nn.DeepONetOperator`
                        (pass it as ``operator=``).
    operator : optional pre-built/-trained operator (a SpectralOperator that was
        ``fit`` from data, or a DeepONetOperator).
    sensors : (n_sensors,) sensor times required by a DeepONet operator.
    """

    def __init__(self, kernel=None, method="ode", operator=None, sensors=None):
        self.kernel = kernel
        self.method = method
        self.operator = operator
        self.sensors = sensors
        if method in ("ode",) and kernel is not None:
            self.a, self.b = kernel_exponentials(kernel)

    def solve(self, forcing, t_grid, return_compensator=False):
        """Return xi(t) on ``t_grid`` for the forcing s(t) (a callable)."""
        t_grid = np.asarray(t_grid, float)
        s_samples = np.asarray(forcing(t_grid), float)
        if self.method == "ode":
            return solve_mbpp_ode(forcing, self.a, self.b, t_grid,
                                  return_compensator=return_compensator)
        if self.method == "spectral":
            op = self.operator
            if op is None:
                op = SpectralOperator(t_grid).from_kernel(self.kernel)
            xi = op(s_samples)
            if return_compensator:
                Xi = np.concatenate([[0.0], np.cumsum(0.5 * (xi[1:] + xi[:-1]) * np.diff(t_grid))])
                return xi, Xi
            return xi
        if self.method == "deeponet":
            if self.operator is None:
                raise ValueError("method='deeponet' needs a trained operator=")
            sens = self.sensors if self.sensors is not None else t_grid
            xi = self.operator.predict(forcing(sens)[None, :], t_grid)[0]
            if return_compensator:
                Xi = np.concatenate([[0.0], np.cumsum(0.5 * (xi[1:] + xi[:-1]) * np.diff(t_grid))])
                return xi, Xi
            return xi
        raise ValueError(f"unknown method {self.method!r}")
