"""
Pure-numpy fallbacks for the two SciPy entry points used by :mod:`sector_ranker`,
so the package imports and runs without SciPy installed.

* :func:`logsumexp` -- numerically stable log-sum-exp.
* :func:`minimize`  -- a drop-in for ``scipy.optimize.minimize(..., jac=True,
  method="L-BFGS-B", bounds=...)`` covering the box-constrained, smooth, convex
  objectives in this package. It uses projected Adam with the supplied analytic
  gradient and keeps the best iterate. SciPy (L-BFGS-B) is used automatically when
  available and is faster and more accurate; this fallback exists only so the code
  has no hard SciPy dependency.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np


def logsumexp(a, axis=None):
    a = np.asarray(a, dtype=float)
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    if axis is None:
        return float(out.reshape(()))  # numpy >= 2 rejects float() on shape-(1,) arrays
    return np.squeeze(out, axis=axis)


def _bounds_arrays(bounds, n):
    if bounds is None:
        return np.full(n, -np.inf), np.full(n, np.inf)
    lo = np.array([(-np.inf if b[0] is None else b[0]) for b in bounds], dtype=float)
    hi = np.array([(np.inf if b[1] is None else b[1]) for b in bounds], dtype=float)
    return lo, hi


def minimize(fun, x0, jac=True, method=None, bounds=None, options=None):
    """Projected-Adam minimiser matching the SciPy call signature used here.

    ``fun`` must return ``(value, gradient)`` (``jac=True``). ``bounds`` is a list of
    ``(lo, hi)`` pairs with ``None`` for an open side. Returns an object exposing
    ``x``, ``fun``, ``success`` and ``message`` like a SciPy ``OptimizeResult``.
    """
    x = np.array(x0, dtype=float)
    n = x.size
    lo, hi = _bounds_arrays(bounds, n)
    x = np.minimum(np.maximum(x, lo), hi)
    opts = options or {}
    steps = max(4000, 8 * int(opts.get("maxiter", 500)))
    lr0 = 0.05
    b1, b2, eps = 0.9, 0.999, 1e-8
    m = np.zeros(n)
    v = np.zeros(n)

    f0, g = fun(x)
    best_x, best_f = x.copy(), float(f0)
    for t in range(1, steps + 1):
        f, g = fun(x)
        g = np.asarray(g, dtype=float)
        if np.isfinite(f) and f < best_f:
            best_f, best_x = float(f), x.copy()
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mhat = m / (1 - b1**t)
        vhat = v / (1 - b2**t)
        lr = lr0 * (0.1 ** (t / steps))  # decay lr0 -> 0.1*lr0
        x = x - lr * mhat / (np.sqrt(vhat) + eps)
        x = np.minimum(np.maximum(x, lo), hi)

    f, g = fun(x)
    if np.isfinite(f) and f < best_f:
        best_f, best_x = float(f), x.copy()
    return SimpleNamespace(
        x=best_x,
        fun=best_f,
        success=True,
        message="numpy projected-Adam fallback (SciPy not installed)",
    )
