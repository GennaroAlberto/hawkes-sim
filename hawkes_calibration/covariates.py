"""
Covariate utilities. We support piecewise-constant covariates, which cover
many natural cases (regime indicators, treatment/control, time-of-day buckets)
and have an exact closed-form integral for the baseline compensator.
"""

import numpy as np


class PiecewiseConstantCovariate:
    """
    A piecewise-constant function of time.

    The function takes value ``values[k]`` on the interval
    ``[breakpoints[k], breakpoints[k+1])``.  We require
    ``breakpoints[0] == 0`` and ``breakpoints[-1] >= T`` for any horizon T.

    Parameters
    ----------
    breakpoints : (K+1,) array_like, strictly increasing, first entry 0
    values : (K, p) array_like of values on each interval (p covariates)
    """

    def __init__(self, breakpoints, values):
        self.breakpoints = np.asarray(breakpoints, dtype=float)
        self.values = np.atleast_2d(np.asarray(values, dtype=float))
        if self.values.shape[0] == 1 and self.breakpoints.size > 2:
            # column vector mistakenly given as row -> transpose
            self.values = self.values.T
        assert self.breakpoints[0] == 0.0
        assert np.all(np.diff(self.breakpoints) > 0)
        assert self.values.shape[0] == self.breakpoints.size - 1
        self.p = self.values.shape[1]

    def __call__(self, t):
        """Return the value of the covariate at time t (or array t)."""
        t = np.atleast_1d(t)
        # piecewise: find which interval each t lies in
        idx = np.searchsorted(self.breakpoints, t, side="right") - 1
        idx = np.clip(idx, 0, self.values.shape[0] - 1)
        out = self.values[idx]
        return out

    def integrate_exp_gamma(self, gamma0, gamma, T):
        """
        Compute the integral

            \\int_0^T exp(gamma0 + gamma^T X(s)) ds

        exactly, using the piecewise-constant structure.
        """
        total = 0.0
        for k in range(self.values.shape[0]):
            a = self.breakpoints[k]
            b = min(self.breakpoints[k + 1], T)
            if b <= a:
                break
            val = float(np.exp(gamma0 + gamma @ self.values[k]))
            total += val * (b - a)
            if b >= T:
                break
        return total


def baseline_value(covariate, t, gamma0, gamma):
    """
    Evaluate exp(gamma0 + gamma^T X(t)) at scalar t.
    If covariate is None we return exp(gamma0) (a positive constant baseline).
    """
    if covariate is None:
        return float(np.exp(gamma0))
    x = covariate(t)[0]  # (p,)
    return float(np.exp(gamma0 + gamma @ x))
