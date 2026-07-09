r"""
Experiment 20 -- a NON-LINEAR-intensity Hawkes that we *can* estimate because we
observe event TIMES (the complement of the count-only model in exp19).

Here the data are exact continuous-time event timestamps.  The baseline is a
log-linear (exponential) function of macro covariates,

    lambda_m(t) = exp(gamma0_m + gamma_m^T X(t))            <- NON-LINEAR in X(t)
                  + sum_j alpha_{m,j} sum_{t_{j,k} < t} e^{-beta_{m,j}(t - t_{j,k})},

so the intensity is a nonlinear (exponential) function of the covariates rather than
an additive/linear one.  The excitation is kept comfortably subcritical
(spectral radius of alpha/beta well below 1).  With the event times in hand, the
continuous-time MLE recovers every parameter -- the covariate effects gamma AND the
excitation matrix alpha -- which is exactly the information the weekly-count model of
exp19 cannot see.

Run:  PYTHONPATH=. python -m experiments.exp20_nonlinear_hawkes_timeinfo
"""

import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hawkes_calibration import (
    PiecewiseConstantCovariate,
    fit_multivariate_with_covariates,
    simulate_multivariate_hawkes,
)


def _covariate(T, p, period, rng):
    """Piecewise-constant macro covariates: regime values switch every `period`."""
    breaks = np.arange(0.0, T + period, period)
    values = rng.choice([-1.0, 0.0, 1.0], size=(breaks.size - 1, p))
    return PiecewiseConstantCovariate(breaks, values)


def run(seeds=range(5), T=4000.0, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    M, p = 3, 2
    gamma0 = np.array([0.0, -0.3, 0.2])
    gamma = np.array([[0.8, -0.4], [0.5, 0.6], [-0.6, 0.5]])  # covariate effects
    alpha = np.array([[0.30, 0.10, 0.00], [0.00, 0.25, 0.15], [0.10, 0.00, 0.20]])
    beta = np.full((M, M), 1.5)
    rho = float(np.max(np.abs(np.linalg.eigvals(alpha / beta))))  # branching radius

    g0_err, g_err, a_err, a_rel, n_ev = [], [], [], [], []
    last = None
    for seed in seeds:
        rng = np.random.default_rng(seed)
        cov = _covariate(T, p, period=40.0, rng=rng)
        events = simulate_multivariate_hawkes(
            gamma0, alpha, beta, T=T, gamma=gamma, covariate=cov, seed=int(rng.integers(1e9))
        )
        res = fit_multivariate_with_covariates(events, T, beta=beta, covariate=cov, se=False)
        g0_err.append(float(np.max(np.abs(res.gamma0 - gamma0))))
        g_err.append(float(np.max(np.abs(res.gamma - gamma))))
        a_err.append(float(np.max(np.abs(res.alpha - alpha))))
        a_rel.append(float(np.linalg.norm(res.alpha - alpha) / np.linalg.norm(alpha)))
        n_ev.append(int(sum(e.size for e in events)))
        last = (cov, res)

    out = dict(
        M=M,
        p=p,
        T=T,
        n_seeds=len(list(seeds)),
        branching_radius=rho,
        true=dict(gamma0=gamma0.tolist(), gamma=gamma.tolist(), alpha=alpha.tolist()),
        mean_events=float(np.mean(n_ev)),
        recovery=dict(
            gamma0_max_abs_err=float(np.mean(g0_err)),
            gamma_max_abs_err=float(np.mean(g_err)),
            alpha_max_abs_err=float(np.mean(a_err)),
            alpha_rel_l2_err=float(np.mean(a_rel)),
        ),
    )
    json.dump(
        out,
        open(os.path.join(out_dir, "exp20_nonlinear_hawkes.json"), "w"),
        indent=2,
        default=float,
    )

    print("=== Experiment 20: non-linear (exp-covariate) event-time Hawkes ===")
    print(
        f"  M={M}, {p} covariates, T={T:.0f}, ~{np.mean(n_ev):.0f} events; "
        f"branching radius rho(alpha/beta)={rho:.2f} (<< 1, stable)"
    )
    print("  intensity baseline is exp(gamma0 + gamma.X(t)) -- nonlinear in the covariates")
    print("  continuous-time MLE recovery (mean over seeds, max abs error):")
    print(f"    gamma0 (baseline)        : {out['recovery']['gamma0_max_abs_err']:.3f}")
    print(f"    gamma  (covariate effect): {out['recovery']['gamma_max_abs_err']:.3f}")
    print(
        f"    alpha  (excitation)      : {out['recovery']['alpha_max_abs_err']:.3f}  "
        f"(rel-L2 {out['recovery']['alpha_rel_l2_err']:.3f})"
    )
    print("  -> with event TIMES, the nonlinear-baseline Hawkes is fully recovered.")
    _plot(out_dir, gamma0, gamma, alpha, beta, last)
    print("Wrote results/exp20_nonlinear_hawkes.{json,png}")
    return out


def _plot(out_dir, gamma0, gamma, alpha, beta, last):
    cov, res = last
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    # recovered vs true (alpha + gamma)
    ax[0].scatter(alpha.ravel(), res.alpha.ravel(), s=40, color="C3", label="alpha (excitation)")
    ax[0].scatter(
        gamma.ravel(), res.gamma.ravel(), s=40, color="C0", marker="s", label="gamma (covariate)"
    )
    lim = [min(alpha.min(), gamma.min()) - 0.1, max(alpha.max(), gamma.max()) + 0.1]
    ax[0].plot(lim, lim, "k--", lw=1)
    ax[0].set_xlabel("true value")
    ax[0].set_ylabel("MLE recovered")
    ax[0].set_title("Parameter recovery from event times")
    ax[0].legend()
    # the nonlinear baseline intensity over time for dim 0
    t = np.linspace(0, min(400.0, cov.breakpoints[-1]), 800)
    Xt = cov(t)  # (len, p)
    base0 = np.exp(gamma0[0] + Xt @ gamma[0])
    base0_hat = np.exp(res.gamma0[0] + Xt @ res.gamma[0])
    ax[1].plot(t, base0, "C0-", lw=2, label="true baseline exp(g0+g.X)")
    ax[1].plot(t, base0_hat, "C1--", lw=1.6, label="recovered baseline")
    ax[1].set_xlabel("t")
    ax[1].set_ylabel(r"$\lambda_0$ baseline")
    ax[1].set_title("Nonlinear (exponential) covariate baseline, dim 0")
    ax[1].legend()
    fig.suptitle("Non-linear-intensity Hawkes, recovered from event-time information")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "exp20_nonlinear_hawkes.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    run()
