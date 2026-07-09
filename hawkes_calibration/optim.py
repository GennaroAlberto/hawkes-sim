"""
Minimal BFGS optimizer with backtracking (Armijo) line search and a numerical
gradient. We implement this ourselves because scipy may not be installed in
all environments, and the parameter dimension here is small enough that a
hand-rolled BFGS is plenty fast.

The interface mimics scipy.optimize.minimize's OptimizeResult superficially.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OptResult:
    x: np.ndarray
    fun: float
    success: bool
    message: str
    nit: int
    nfev: int


def _numgrad(f, x, fx=None, eps=1e-5):
    """Forward/backward differences (central) for the gradient of f at x."""
    n = x.size
    g = np.zeros(n)
    for i in range(n):
        xp = x.copy()
        xp[i] += eps
        xm = x.copy()
        xm[i] -= eps
        g[i] = (f(xp) - f(xm)) / (2 * eps)
    return g


def minimize_bfgs(
    f,
    x0,
    max_iter=200,
    ftol=1e-8,
    gtol=1e-5,
    grad_eps=1e-5,
    fg=None,
    verbose=False,
):
    """
    Unconstrained BFGS with backtracking line search.

    Parameters
    ----------
    f : callable, scalar objective
    x0 : starting point (1D array)
    max_iter : max number of outer iterations
    ftol : relative tolerance on f
    gtol : norm tolerance on gradient
    fg : optional callable returning (f, grad). If provided, used in place of
        finite differences — essential for high-dimensional problems.
    """
    x = np.asarray(x0, dtype=float).copy()
    n = x.size
    H = np.eye(n)
    if fg is not None:
        fx, g = fg(x)
        nfev = 1
    else:
        fx = f(x)
        g = _numgrad(f, x, fx=fx, eps=grad_eps)
        nfev = 1 + 2 * n
    message = "max_iter reached"
    success = False

    for it in range(max_iter):
        if np.linalg.norm(g) < gtol:
            message = "gtol satisfied"
            success = True
            break

        # Search direction
        p = -H @ g
        # Make sure p is a descent direction; otherwise reset H
        if g @ p > -1e-12:
            H = np.eye(n)
            p = -g

        # Backtracking line search (Armijo)
        alpha = 1.0
        c1 = 1e-4
        slope = g @ p
        for _ in range(60):
            x_new = x + alpha * p
            if fg is not None:
                fx_new, g_new_candidate = fg(x_new)
            else:
                fx_new = f(x_new)
                g_new_candidate = None
            nfev += 1
            if np.isfinite(fx_new) and fx_new <= fx + c1 * alpha * slope:
                break
            alpha *= 0.5
        else:
            # line search failed; bail out
            message = "line search failed"
            break

        # Convergence on function value
        if abs(fx - fx_new) < ftol * (1 + abs(fx)):
            x = x_new
            fx = fx_new
            message = "ftol satisfied"
            success = True
            break

        # BFGS update
        if g_new_candidate is not None:
            g_new = g_new_candidate
        else:
            g_new = _numgrad(f, x_new, fx=fx_new, eps=grad_eps)
            nfev += 2 * n
        s = (x_new - x).reshape(-1, 1)
        y = (g_new - g).reshape(-1, 1)
        sy = float((s.T @ y).item())
        if sy > 1e-12:
            rho = 1.0 / sy
            I = np.eye(n)
            H = (I - rho * s @ y.T) @ H @ (I - rho * y @ s.T) + rho * s @ s.T
        # else skip update (keep H)

        x = x_new
        fx = fx_new
        g = g_new

        if verbose and (it % 10 == 0):
            print(f"  iter {it:3d}: f={fx:.6f}  |g|={np.linalg.norm(g):.3e}")

    return OptResult(x=x, fun=fx, success=success, message=message, nit=it + 1, nfev=nfev)
