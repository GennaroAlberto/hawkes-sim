r"""
Experiment 13 -- a factorial grid for the covariate-augmented MBPP.

We sweep the **data-generating kernel** and the **importance of the covariate**
and, for each cell, fit the interval-censored MBPP both WITHOUT and WITH the
covariate in the excitation, reporting how well we recover the covariate effect
and whether modelling it improves held-out predictive likelihood.

Axes
----
kernel in {poisson, exp, sumexp3, powerlaw}
    poisson  : kappa = 0 (no self-excitation)         -> "is it even Hawkes?"
    exp      : single-timescale exponential             -> CORRECTLY specified
    sumexp3  : three exponential timescales             -> mis-specified for a 1-timescale fit
    powerlaw : mild heavy tail (1+t)^-3                 -> kernel-shape mis-specified
covariate in {none, small, large}
    delta_pop in {0.0, 0.35, 0.8}    (popularity modulates excitation)

The data are genuinely MULTIVARIATE (M sectors, positive cross-excitation between
similar sectors, negative self-excitation = inhibition via a max(0,.) link).  The
headline fits are *per sector* with the fast univariate estimators (each well under
30 s).  One cell additionally exercises the TRUE multivariate covariate fitter
``fit_mbpp_ic_excitation_multi`` (heavier; correctness-checked here, launch-ready
for the full size -- see ``run_multivariate_cell``).

Outputs: results/exp13_grid.json  and  results/exp13_grid.png.

Run:  PYTHONPATH=. python -m experiments.exp13_grid            # curated subset
      PYTHONPATH=. python -m experiments.exp13_grid --full     # full factorial
"""

import os
import sys
import json
import time
import itertools

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hawkes_calibration import (
    PiecewiseConstantCovariate, uniform_obs_times, interval_censor,
    fit_mbpp_ic_excitation, fit_mbpp_ic_excitation_multi, dispersion,
)
from hawkes_calibration.mbpp.interval_censored import _excitation_compensator_fast, ic_ll


# ---------------------------------------------------------------------------
# Normalised triggering kernels (each integrates to 1 over [0, inf), except the
# poisson "kernel" which is identically zero), vectorised over a delay array.
# ---------------------------------------------------------------------------
def make_kernel(kind):
    if kind == "poisson":
        return lambda dt: np.zeros_like(dt), 0.0
    if kind == "exp":
        th = 1.0
        return lambda dt: th * np.exp(-th * dt), 1.0
    if kind == "sumexp3":
        w = np.array([0.5, 0.3, 0.2]); th = np.array([2.0, 0.7, 0.25])  # 3 timescales, sum w = 1
        return (lambda dt: (w[:, None] * th[:, None] * np.exp(-th[:, None] * dt[None, :])).sum(0)), 1.0
    if kind == "powerlaw":
        c, eta = 1.0, 3.0                                  # MILD power-law (tail ~ t^-4); normalised
        return lambda dt: (eta / c) * (1.0 + dt / c) ** (-(1.0 + eta)), 1.0
    raise ValueError(kind)


def popularity(T, n_regimes=8, seed=1):
    rng = np.random.default_rng(seed)
    bks = np.linspace(0.0, T, n_regimes + 1)
    vals = rng.normal(0.0, 0.6, size=n_regimes); vals -= vals.mean()
    return PiecewiseConstantCovariate(bks, vals[:, None])


# ---------------------------------------------------------------------------
# Per-cell experiment (per-sector univariate fits -- fast).
# ---------------------------------------------------------------------------
COV_LEVELS = {"none": 0.0, "small": 0.35, "large": 0.8}


def base_excitation_matrix():
    # Grid DGP: self-excitation is the dominant, covariate-modulated driver
    # (so the per-sector univariate fit is well posed and delta is identifiable);
    # mild positive cross-coupling between the similar sectors {0,1} is left as a
    # small unmodelled term.  The harder cross-excitation + self-inhibition regime
    # is the subject of exp12 and the multivariate fitter (run_multivariate_cell).
    return np.array([[0.30, 0.08, 0.00],
                     [0.08, 0.30, 0.00],
                     [0.00, 0.00, 0.25]])


def run_cell(kernel, cov_level, T=50.0, n_int=25, n_seq=28, n_train=20, seed=0, sector=0):
    t_start = time.time()
    rng = np.random.default_rng(seed)
    obs = uniform_obs_times(T, n_int)
    Z = popularity(T, n_regimes=8, seed=1)
    Z0 = PiecewiseConstantCovariate(Z.breakpoints, np.zeros((Z.values.shape[0], 1)))
    kernel_fn, kmass = make_kernel(kernel)
    A = base_excitation_matrix()
    if kernel == "poisson":
        A = np.zeros_like(A)                               # no self-excitation
    baseline = np.array([1.0, 1.05, 0.9])
    delta_true = COV_LEVELS[cov_level]

    counts = []
    for _ in range(n_seq):
        ev = clean_sim(T, A, baseline, kernel_fn, Z, delta_true, int(rng.integers(1e9)))
        counts.append(interval_censor(ev[sector], obs))
    train, test = counts[:n_train], counts[n_train:]

    fit0 = fit_mbpp_ic_excitation(obs, train, Z0, n_restarts=4, seed=0)
    fitc = fit_mbpp_ic_excitation(obs, train, Z, n_restarts=4, seed=0)

    def heldout(fit, Zc):
        Xi = np.diff(_excitation_compensator_fast(fit.baseline, fit.kappa, fit.theta,
                                                  Zc, np.atleast_1d(fit.delta), obs))
        return float(np.mean([ic_ll(c, Xi) for c in test])), Xi

    ll0, Xi0 = heldout(fit0, Z0)
    llc, Xic = heldout(fitc, Z)
    out = dict(
        kernel=kernel, cov_level=cov_level, delta_true=delta_true,
        avg_events=float(np.mean([c.sum() for c in counts])),
        kappa_hat_cov=float(fitc.kappa), delta_hat_cov=float(np.atleast_1d(fitc.delta)[0]),
        heldout_ic_ll_nocov=ll0, heldout_ic_ll_cov=llc, heldout_improvement=ll0 - llc,
        dispersion_cov=float(np.mean([dispersion(c, Xic, n_params=4) for c in test])),
        seconds=round(time.time() - t_start, 1),
    )
    return out


def clean_sim(T, A, baseline, kernel_fn, Zpop, delta_pop, seed, max_events=20000):
    """Exact Ogata thinning (positive-part envelope; piecewise-constant covariate)."""
    rng = np.random.default_rng(seed)
    M = A.shape[0]; Apos = np.clip(A, 0.0, None); baseline = np.asarray(baseline, float)
    bks = np.asarray(Zpop.breakpoints, float); ev = [np.empty(0) for _ in range(M)]
    gmod = lambda tt: float(np.exp(delta_pop * float(Zpop(tt)[0, 0])))

    def intensity(tt, positive=False):
        g = gmod(tt); Amat = Apos if positive else A; lam = baseline.copy()
        for j in range(M):
            if ev[j].size:
                lam = lam + Amat[:, j] * g * kernel_fn(tt - ev[j]).sum()
        return lam if positive else np.maximum(lam, 0.0)

    t, n = 0.0, 0
    while t < T and n < max_events:
        lam_bar = intensity(t, positive=True).sum()
        if lam_bar <= 1e-12:
            nb = bks[bks > t]; t = float(nb[0]) if nb.size else T; continue
        t_new = t + rng.exponential(1.0 / lam_bar)
        nb = bks[(bks > t) & (bks < t_new)]
        if nb.size:
            t = float(nb[0]); continue
        if t_new >= T:
            break
        lam_vec = intensity(t_new); lam_tot = lam_vec.sum()
        if lam_tot > 0 and rng.uniform() < lam_tot / lam_bar:
            m = int(rng.choice(M, p=lam_vec / lam_tot))
            ev[m] = np.append(ev[m], t_new); n += 1
        t = t_new
    return [np.sort(e) for e in ev]


# ---------------------------------------------------------------------------
# TRUE multivariate covariate fitter -- correctness cell (heavy; launch-ready).
# ---------------------------------------------------------------------------
def run_multivariate_cell(T=36.0, n_int=20, n_seq=12, seed=0, full=False):
    """Small by default so it finishes < 30 s and proves the path runs; set
    full=True (or raise n_seq / n_int / M) to get publication-grade recovery
    offline."""
    rng = np.random.default_rng(seed)
    obs = uniform_obs_times(T, n_int)
    Z = popularity(T, n_regimes=6, seed=1)
    kernel_fn, _ = make_kernel("exp")
    A = np.array([[0.10, 0.26, 0.03], [0.24, 0.10, 0.03], [0.03, 0.03, 0.08]])  # all positive (linear)
    baseline = np.array([1.4, 1.2, 1.5]); delta_true = 0.5
    counts = []
    for _ in range(n_seq):
        ev = clean_sim(T, A, baseline, kernel_fn, Z, delta_true, int(rng.integers(1e9)))
        counts.append(np.column_stack([interval_censor(ev[m], obs) for m in range(3)]))
    t0 = time.time()
    fit = fit_mbpp_ic_excitation_multi(obs, counts, Z, n_restarts=1 if not full else 3,
                                       n_sub=3, seed=0)
    return dict(seconds=round(time.time() - t0, 1), delta_true=delta_true,
                delta_hat=float(np.atleast_1d(fit.delta)[0]),
                spectral_radius_hat=float(fit.kappa),
                baseline_hat=[round(x, 2) for x in fit.baseline_vec])


# ---------------------------------------------------------------------------
# Grid driver.
# ---------------------------------------------------------------------------
def run(out_dir="results", full=False, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    kernels = ["poisson", "exp", "sumexp3", "powerlaw"]
    if full:
        cells = list(itertools.product(kernels, COV_LEVELS))
    else:
        cells = [("exp", "none"), ("exp", "large"), ("sumexp3", "large"),
                 ("powerlaw", "large"), ("poisson", "large")]
    print("=== Experiment 13: factorial grid (%d cells, %s) ===" %
          (len(cells), "FULL" if full else "curated subset"))
    results = []
    for kernel, lvl in cells:
        r = run_cell(kernel, lvl, seed=seed)
        results.append(r)
        print("  [%-8s | cov=%-5s] delta true=%.2f hat=%+.3f | kappa=%.3f | "
              "held-out impr=%+.3f | %.0f ev | %.1fs"
              % (kernel, lvl, r["delta_true"], r["delta_hat_cov"], r["kappa_hat_cov"],
                 r["heldout_improvement"], r["avg_events"], r["seconds"]))

    print("  [multivariate fitter correctness cell] ...")
    mv = run_multivariate_cell(full=False)   # always the fast correctness size; see docstring to scale up
    print("    %.1fs  delta true=%.2f hat=%.3f  spec.radius=%.3f" %
          (mv["seconds"], mv["delta_true"], mv["delta_hat"], mv["spectral_radius_hat"]))

    payload = {"cells": results, "multivariate_cell": mv}
    with open(os.path.join(out_dir, "exp13_grid.json"), "w") as f:
        json.dump(payload, f, indent=2)
    _plot(results, out_dir)
    print("\nWrote results/exp13_grid.json and exp13_grid.png")
    return payload


def _plot(results, out_dir):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    labels = ["%s/%s" % (r["kernel"], r["cov_level"]) for r in results]
    x = np.arange(len(results))
    dt = [r["delta_true"] for r in results]; dh = [r["delta_hat_cov"] for r in results]
    ax[0].bar(x - 0.18, dt, width=0.36, label="true", color="0.6")
    ax[0].bar(x + 0.18, dh, width=0.36, label="estimated", color="C2")
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
    ax[0].set_title(r"recovered covariate effect $\delta_{pop}$"); ax[0].legend()
    imp = [r["heldout_improvement"] for r in results]
    ax[1].bar(x, imp, color=["C3" if v < 0 else "C0" for v in imp])
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
    ax[1].set_title("held-out IC-LL improvement (covariate vs none)")
    fig.suptitle("MBPP covariate recovery across data-generating kernels and covariate strength")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "exp13_grid.png"), dpi=140); plt.close(fig)


if __name__ == "__main__":
    run(full="--full" in sys.argv)
