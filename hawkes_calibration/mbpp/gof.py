r"""
Goodness-of-fit diagnostics.

Two complementary checks, matching the theory in the LaTeX notes (Appendix G):

* **Event-time --- time-rescaling theorem.** If the fitted compensator
  :math:`\Lambda` is correct, the rescaled inter-event gaps
  :math:`\Lambda(t_k)-\Lambda(t_{k-1})` are i.i.d.\ :math:`\mathrm{Exp}(1)`.
  :func:`time_rescaling_residuals` returns them and :func:`ks_test_exp1` tests
  them against :math:`\mathrm{Exp}(1)` with a (numpy-only) Kolmogorov--Smirnov
  test --- no SciPy dependency.

* **Interval-censored --- Poisson/Pearson residuals.** For MBPP bucket counts,
  :math:`r_i=(\mathsf C_i-\Xi_i)/\sqrt{\Xi_i}` are approximately standard normal,
  and :math:`\sum_i r_i^2/(m-p)` (the dispersion) should be near 1.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Event-time: time-rescaling residuals + KS test against Exp(1).
# ---------------------------------------------------------------------------
def time_rescaling_residuals(events, compensator):
    r"""
    Rescaled inter-event gaps under the time-rescaling theorem.

    Parameters
    ----------
    events : 1D array of event times (need not be pre-sorted).
    compensator : callable t -> Lambda(t) (cumulative compensator), or a 1D array
        of the same length as ``events`` giving Lambda at each event time.

    Returns
    -------
    tau : 1D array of gaps Lambda(t_k) - Lambda(t_{k-1}) (with t_0 = 0); under a
        correct model these are i.i.d. Exp(1).
    """
    t = np.sort(np.asarray(events, dtype=float))
    if callable(compensator):
        L = np.array([float(compensator(ti)) for ti in t])
    else:
        L = np.asarray(compensator, dtype=float)
        L = np.sort(L)
    return np.diff(np.concatenate([[0.0], L]))


def ks_test_exp1(x):
    r"""
    One-sample Kolmogorov--Smirnov test of ``x`` against the Exp(1) distribution.

    Returns
    -------
    D : the KS statistic sup_x |F_emp(x) - (1 - e^{-x})|.
    p : an asymptotic p-value (small p => reject the Exp(1) / model fit).
    """
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    if n == 0:
        return 0.0, 1.0
    F = 1.0 - np.exp(-np.maximum(x, 0.0))  # Exp(1) CDF
    emp_hi = np.arange(1, n + 1) / n
    emp_lo = np.arange(0, n) / n
    D = float(max(np.max(emp_hi - F), np.max(F - emp_lo)))
    en = np.sqrt(n)
    lam = (en + 0.12 + 0.11 / en) * D
    p = 2.0 * sum((-1) ** (k - 1) * np.exp(-2.0 * k * k * lam * lam) for k in range(1, 101))
    return D, float(min(max(p, 0.0), 1.0))


def qq_exp1(x):
    r"""
    QQ-plot coordinates against Exp(1): returns (theoretical_quantiles,
    sorted_sample). Plot ``sample`` vs ``theoretical``; points on the diagonal
    indicate a good fit.
    """
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    probs = (np.arange(1, n + 1) - 0.5) / n
    theo = -np.log(1.0 - probs)  # Exp(1) inverse CDF
    return theo, x


# ---------------------------------------------------------------------------
# Interval-censored: Poisson/Pearson residuals + dispersion.
# ---------------------------------------------------------------------------
def poisson_pearson_residuals(counts, Xi):
    r"""Pearson residuals r_i = (C_i - Xi_i)/sqrt(Xi_i); approx N(0,1) if the fit is good."""
    counts = np.asarray(counts, dtype=float)
    Xi = np.maximum(np.asarray(Xi, dtype=float), 1e-12)
    return (counts - Xi) / np.sqrt(Xi)


def dispersion(counts, Xi, n_params=0):
    r"""
    Pearson dispersion statistic sum_i r_i^2 / (m - n_params).  Values near 1
    indicate Poisson-consistent counts; >1 over-dispersion (e.g. unmodelled
    self-excitation clustering), <1 under-dispersion.
    """
    r = poisson_pearson_residuals(counts, Xi)
    dof = max(r.size - n_params, 1)
    return float(np.sum(r**2) / dof)
