r"""
Neural solvers for the MBPP equation -- a fast, differentiable, *model-agnostic*
replacement for the pure-Python ODE/Volterra solve inside the optimiser.

Why
---
Fitting a non-closed-form MBPP repeatedly evaluates the forward solve, and a
BFGS step adds a factor ``2 * n_params`` for finite-difference gradients:

    cost ~ (restarts) x (iters) x (1 + 2 n_params) x (one Python time-stepping solve).

A neural solver attacks both factors at once.  We train a *parametric* network

    N_phi(t, p)  ~  xi(t; p)            (the MBPP intensity as a function of time
                                         AND the model parameters p = (kappa, theta,
                                         delta, ...))

to satisfy the MBPP equation.  After a one-off training run:

* the forward solve is a single vectorised network evaluation (no Python loop);
* gradients w.r.t. ``p`` come from autodiff (no ``2 n_params`` finite differences);
* it runs on GPU.

So at fit time the optimiser calls a cheap, differentiable surrogate solver for
*any* ``p`` in the trained range -- without a closed form and without re-deriving
anything per model.

Two training objectives are supported:

* ``mode="pinn"`` -- *physics-informed*: minimise the MBPP residual
  ``R(t,p) = N(t,p) - s(t;p) - int_0^t K(t,u;p) N(u,p) du`` at collocation points,
  plus the initial condition ``N(0,p)=s(0;p)``.  Needs no ground-truth solver and
  works for any kernel ``K`` (the integral is evaluated by quadrature).  This is
  the heart of "a neural net that *solves* the PDE".
* ``mode="operator"`` -- *supervised*: regress on ``(p, xi(.;p))`` pairs produced
  by the exact numpy solver (:mod:`hawkes_calibration.operators`).  Easier to
  train; uses the reference solver only to make data.

Backends
--------
``make_neural_solver(backend=...)`` dispatches to:

* ``"jax"``        -- :mod:`hawkes_calibration.neural_solver_jax` (jit/grad/optax, GPU),
* ``"tensorflow"`` -- :mod:`hawkes_calibration.neural_solver_tf`  (Keras/GradientTape, GPU),
* ``"numpy"``      -- the exact reference solver (no NN; for validation/fallback).

The JAX and TensorFlow modules are optional: importing *this* module is
numpy-only, and ``hawkes_calibration.__init__`` does not import any of them.

The one piece that does not depend on a deep-learning framework -- and that both
neural backends are trained to drive to zero -- is the **residual**, implemented
and tested here in numpy.
"""

from __future__ import annotations

import numpy as np


# ===========================================================================
# The physics-informed residual (numpy; the training target of the NN solvers).
# ===========================================================================
def mbpp_volterra_residual(xi, t_grid, forcing, kernel):
    r"""
    Collocation residual of the general MBPP Volterra equation

        R(t_i) = xi(t_i) - s(t_i) - int_0^{t_i} K(t_i,u) xi(u) du,

    with the integral evaluated by the trapezoid rule on ``t_grid``.  At the true
    MBPP intensity this is zero (to quadrature error); the JAX/TF PINNs minimise
    ``sum_i R(t_i)^2``.

    Parameters
    ----------
    xi : (N,) array, candidate intensity sampled on ``t_grid``.
    t_grid : (N,) increasing grid starting at 0.
    forcing : callable t -> s(t) (scalar).
    kernel : callable (t, u) -> K(t,u) (scalar).

    Returns
    -------
    R : (N,) residual at each grid point.
    """
    t = np.asarray(t_grid, float)
    xi = np.asarray(xi, float)
    N = t.size
    R = np.empty(N)
    s = np.array([float(forcing(ti)) for ti in t])
    R[0] = xi[0] - s[0]  # integral over [0,0] = 0
    for i in range(1, N):
        ti = t[i]
        integ = 0.0
        for k in range(1, i + 1):
            h = t[k] - t[k - 1]
            integ += 0.5 * h * (kernel(ti, t[k - 1]) * xi[k - 1] + kernel(ti, t[k]) * xi[k])
        R[i] = xi[i] - s[i] - integ
    return R


def trapezoid_weight_matrix(t_grid):
    """Lower-triangular trapezoid weights W with (W g)_i ~ int_0^{t_i} g(u) du."""
    t = np.asarray(t_grid, float)
    M = t.size
    W = np.zeros((M, M))
    for i in range(1, M):
        for k in range(1, i + 1):
            h = t[k] - t[k - 1]
            W[i, k - 1] += 0.5 * h
            W[i, k] += 0.5 * h
    return W


def solve_mbpp_residual_linear(t_grid, forcing, kernel):
    r"""
    Solve the MBPP equation by *driving the collocation residual to zero* --- the
    exact objective the PINNs minimise --- exploiting that the residual is
    **linear** in the unknown intensity:

        R = (I - W*K) xi - s = 0   =>   xi = (I - W*K)^{-1} s,

    where ``W`` are the trapezoid weights and ``K_{ij}=K(t_i,t_j)``.  This proves
    the PINN's global minimum is the true MBPP solution (it equals the
    time-stepping Volterra solve), and is itself a valid linear solver.
    """
    t = np.asarray(t_grid, float)
    M = t.size
    W = trapezoid_weight_matrix(t)
    K = np.array([[kernel(t[i], t[j]) if j <= i else 0.0 for j in range(M)] for i in range(M)])
    s = np.array([float(forcing(ti)) for ti in t])
    A = np.eye(M) - W * K
    return np.linalg.solve(A, s)


def sample_covariate_paths(t_grid, n_samples, n_steps=5, low=-0.5, high=0.5, seed=0):
    r"""
    Draw ``n_samples`` random piecewise-constant covariate paths on ``t_grid``
    (a (n_samples, len(t_grid)) array).  These are the *functional* inputs a
    neural operator trains over, so it generalises across the whole family of
    covariate paths rather than a single fixed one.
    """
    rng = np.random.default_rng(seed)
    t = np.asarray(t_grid, float)
    T = t[-1]
    Z = np.empty((n_samples, t.size))
    for i in range(n_samples):
        edges = np.sort(rng.uniform(0, T, n_steps - 1))
        levels = rng.uniform(low, high, n_steps)
        Z[i] = levels[np.searchsorted(edges, t)]
    return Z


def excitation_kernel_matrix(t_grid, Z_grid, kappa, theta, delta):
    r"""
    The MBPP excitation kernel sampled on a grid: ``K_{ij} = kappa*theta *
    exp(delta*Z(t_i)) * exp(-theta(t_i-t_j))`` for ``j<=i`` (lower-triangular).
    Shared by the numpy family-residual check and (in spirit) the JAX/TF PINOs.
    """
    t = np.asarray(t_grid, float)
    Z = np.asarray(Z_grid, float)
    mod = np.exp(np.clip(delta * Z, -30, 30))  # (M,)
    dt = t[:, None] - t[None, :]
    K = kappa * theta * mod[:, None] * np.exp(-theta * dt)
    return np.tril(K)


def family_residual(xi_batch, t_grid, Z_batch, params_batch):
    r"""
    Batched MBPP residual over a *family* of functional inputs (the PINO training
    objective).  For each example ``b`` with covariate path ``Z_batch[b]`` and
    scalar parameters ``(kappa,theta,mu,delta)``, returns the collocation residual
    of ``xi_batch[b]``.  At the exact solutions this is ~0 for every example ---
    which is what makes a single operator able to solve the whole family.

    Parameters
    ----------
    xi_batch : (B, M) candidate intensities.
    t_grid : (M,) collocation grid.
    Z_batch : (B, M) covariate paths.
    params_batch : (B, 4) rows (kappa, theta, mu, delta).

    Returns
    -------
    R : (B, M) residuals.
    """
    t = np.asarray(t_grid, float)
    W = trapezoid_weight_matrix(t)
    B, M = xi_batch.shape
    R = np.empty((B, M))
    for b in range(B):
        kappa, theta, mu, delta = params_batch[b]
        K = excitation_kernel_matrix(t, Z_batch[b], kappa, theta, delta)
        integ = (W * K) @ xi_batch[b]
        R[b] = xi_batch[b] - mu - integ
    return R


def make_anchor_data(t_grid, params_array, Z_array):
    r"""
    Generate **exact** supervised anchors for hybrid PINN/PINO training.

    For each row ``(kappa, theta, mu, delta)`` of ``params_array`` and covariate
    path ``Z_array[k]`` (on ``t_grid``), solve the MBPP exactly (the linear
    residual solve) and return the intensities.  A handful of these anchors,
    added to the residual loss, pin the network down where the residual alone is
    weak (e.g. near criticality kappa->1), which is the practical fix for the
    "PINO is hard to train over a wide family" caveat.

    Parameters
    ----------
    t_grid : (M,) collocation grid.
    params_array : (K, 4) rows (kappa, theta, mu, delta).
    Z_array : (K, M) covariate paths (use zeros for the no-excitation case).

    Returns
    -------
    XI : (K, M) exact intensities.
    """
    t = np.asarray(t_grid, float)
    P = np.atleast_2d(np.asarray(params_array, float))
    Z = np.atleast_2d(np.asarray(Z_array, float))
    W = trapezoid_weight_matrix(t)
    XI = np.empty((P.shape[0], t.size))
    for k in range(P.shape[0]):
        kappa, theta, mu, delta = P[k]
        K = excitation_kernel_matrix(t, Z[k], kappa, theta, delta)
        XI[k] = np.linalg.solve(np.eye(t.size) - W * K, np.full(t.size, mu))
    return XI


def solver_accuracy_report(predict, t_grid, params_array, Z_array=None, plot_path=None):
    r"""
    Backend-agnostic training diagnostic: compare a (trained) solver's output to
    the exact MBPP solution across a grid of parameters, and report *where* it
    breaks --- typically the relative error grows as ``kappa -> 1`` (stiff regime).

    Parameters
    ----------
    predict : callable ``(params_row, Z_row) -> xi_on_t`` -- the solver under test
        (e.g. ``lambda p, Z: pino.solve(dict(...), t_grid, Z)``).  For the numpy
        reference solver this returns the exact solution (error ~0), which is how
        the diagnostic itself is unit-tested.
    t_grid : (M,) grid.  params_array : (K, 4).  Z_array : (K, M) or None.

    Returns
    -------
    dict with per-instance ``rel_l2`` errors, the parameter rows, and a
    ``by_kappa`` summary (mean error in kappa bins).  If ``plot_path`` is given,
    also writes an error-vs-kappa figure.
    """
    t = np.asarray(t_grid, float)
    P = np.atleast_2d(np.asarray(params_array, float))
    Z = (
        np.zeros((P.shape[0], t.size))
        if Z_array is None
        else np.atleast_2d(np.asarray(Z_array, float))
    )
    XI_exact = make_anchor_data(t, P, Z)
    rel = np.empty(P.shape[0])
    for k in range(P.shape[0]):
        xi_hat = np.asarray(predict(P[k], Z[k]), float)
        rel[k] = np.linalg.norm(xi_hat - XI_exact[k]) / (np.linalg.norm(XI_exact[k]) + 1e-12)
    # bin by kappa
    kap = P[:, 0]
    edges = np.linspace(kap.min(), kap.max(), 6)
    idx = np.clip(np.searchsorted(edges, kap) - 1, 0, len(edges) - 2)
    by_kappa = {
        f"[{edges[i]:.2f},{edges[i + 1]:.2f}]": float(rel[idx == i].mean())
        for i in range(len(edges) - 1)
        if np.any(idx == i)
    }
    if plot_path is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.scatter(kap, rel, s=12, alpha=0.6)
        ax.set_xlabel(r"$\kappa$ (branching ratio)")
        ax.set_ylabel("relative L2 error")
        ax.set_title("Neural-solver accuracy vs. the exact MBPP")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=140)
        plt.close(fig)
    return dict(
        rel_l2=rel, params=P, by_kappa=by_kappa, mean=float(rel.mean()), max=float(rel.max())
    )


def mbpp_ode_residual(xi, dxi, t_grid, forcing, kappa, theta):
    r"""
    Residual of the *exponential-kernel* MBPP in first-order ODE form
    (Appendix C of the LaTeX notes): with ``y = xi - s``,

        y'(t) - (kappa-1) theta y(t) - kappa theta s(t) = 0.

    Given samples of ``xi`` and its time-derivative ``dxi`` (the latter from the
    network's autodiff), returns the residual.  Cleaner than the Volterra residual
    for the exponential kernel (no integral), and the template a PINN uses when
    the model reduces to an ODE.
    """
    t = np.asarray(t_grid, float)
    s = np.array([float(forcing(ti)) for ti in t])
    ds = np.gradient(s, t)
    dy = np.asarray(dxi, float) - ds
    y = np.asarray(xi, float) - s
    return dy - (kappa - 1.0) * theta * y - kappa * theta * s


# ===========================================================================
# Backend dispatch.
# ===========================================================================
def make_neural_solver(backend="jax", mode="pinn", **kwargs):
    r"""
    Construct a neural MBPP solver on the chosen backend.

    Parameters
    ----------
    backend : {"jax", "tensorflow", "numpy"}.
    mode : {"pinn", "operator"} (ignored by the numpy reference backend).
    **kwargs : passed to the backend solver (network width/depth, parameter
        ranges, collocation grid, optimiser settings, ...).

    Returns
    -------
    A solver object exposing ``train(...)``, ``solve(params, t_grid) -> xi`` and
    ``compensator(params, t_grid) -> Xi``.  For the numpy backend these call the
    exact solvers directly (no training needed).
    """
    backend = backend.lower()
    if backend == "jax":
        if mode == "pino":
            from .neural_solver_jax import JAXDeepONetPINO

            return JAXDeepONetPINO(**kwargs)
        from .neural_solver_jax import JAXNeuralMBPP

        return JAXNeuralMBPP(mode=mode, **kwargs)
    if backend in ("tensorflow", "tf"):
        if mode == "pino":
            from .neural_solver_tf import TFDeepONetPINO

            return TFDeepONetPINO(**kwargs)
        from .neural_solver_tf import TFNeuralMBPP

        return TFNeuralMBPP(mode=mode, **kwargs)
    if backend == "numpy":
        return NumpyReferenceSolver(**kwargs)
    raise ValueError(f"unknown backend {backend!r}")


class NumpyReferenceSolver:
    r"""
    Exact (non-neural) reference solver behind the same interface, for validation
    and as a fallback when no deep-learning backend is available.  ``params`` is a
    dict with keys among ``mu`` (baseline), ``kappa``, ``theta``, ``delta`` and a
    covariate callable ``Z``; the kernel is the exponential excitation kernel.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def train(self, *a, **k):
        return self  # nothing to train

    def _kernel_forcing(self, params):
        mu = params.get("mu", 1.0)
        kappa, theta = params["kappa"], params["theta"]
        Z = params.get("Z", None)
        delta = np.atleast_1d(np.asarray(params.get("delta", [0.0]), float))

        def forcing(t):
            return mu

        def kernel(t, u):
            mod = 1.0
            if Z is not None:
                z = np.atleast_1d(np.asarray(Z(t), float)).reshape(-1)
                mod = float(np.exp(np.clip(delta @ z, -30, 30)))
            return kappa * theta * mod * np.exp(-theta * (t - u))

        return forcing, kernel

    def solve(self, params, t_grid):
        from .linear import solve_mbpp_volterra

        forcing, kernel = self._kernel_forcing(params)
        return solve_mbpp_volterra(
            lambda t: np.array([forcing(t)]),
            lambda t, u: kernel(t, u),
            np.asarray(t_grid, float),
            M=1,
        )[:, 0]

    def compensator(self, params, t_grid):
        t = np.asarray(t_grid, float)
        xi = self.solve(params, t)
        return np.concatenate([[0.0], np.cumsum(0.5 * (xi[1:] + xi[:-1]) * np.diff(t))])
