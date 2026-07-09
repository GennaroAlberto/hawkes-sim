"""
Experiment 3 -- Multivariate Hawkes with covariates in the baseline.

We use a 2-component process with a piecewise-constant covariate X(t) that
switches between two regimes. The covariate enters the baseline log-rate
linearly:  mu_m(t) = exp(gamma0_m + gamma_m * X(t)).

We check that MLE jointly recovers (gamma0, gamma, A).
"""

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


def run(seed=3, T=5000.0, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)

    # Build a covariate that alternates between 0 and 1 on intervals of length 200.
    breaks = list(np.arange(0, T + 1.0, 200.0))
    if breaks[-1] < T:
        breaks.append(T)
    values = []
    for k in range(len(breaks) - 1):
        values.append([float(k % 2)])  # 0,1,0,1,...
    cov = PiecewiseConstantCovariate(breakpoints=breaks, values=values)

    # True parameters:
    # baselines:  gamma0 = (log 0.3, log 0.4)
    # covariate effects: gamma = ((+0.8,), (-0.5,))  -- multiplies log-rate by +/-
    # in regime X=1, baselines become 0.3*e^0.8 ≈ 0.668, 0.4*e^-0.5 ≈ 0.243
    gamma0_true = np.log(np.array([0.3, 0.4]))
    gamma_true = np.array([[0.8], [-0.5]])
    A_true = np.array([[0.20, 0.10], [0.10, 0.25]])
    B = np.ones((2, 2))

    print(
        "True baselines (regime X=0):",
        np.exp(gamma0_true).round(4).tolist(),
        " (regime X=1):",
        (np.exp(gamma0_true + gamma_true[:, 0])).round(4).tolist(),
    )

    events = simulate_multivariate_hawkes(
        gamma0=gamma0_true,
        alpha=A_true,
        beta=B,
        T=T,
        gamma=gamma_true,
        covariate=cov,
        seed=seed,
    )
    counts = [len(e) for e in events]
    print(f"Simulated {sum(counts)} events, counts={counts}")

    res = fit_multivariate_with_covariates(events, T=T, beta=B, covariate=cov, se=True)
    print()
    print(res.summary())

    # Summarize side-by-side
    print()
    print("Recovery summary:")
    print(f"  gamma0:  true {gamma0_true.round(3).tolist()}   hat {res.gamma0.round(3).tolist()}")
    print(
        f"  gamma :  true {gamma_true.ravel().round(3).tolist()}   hat {res.gamma.ravel().round(3).tolist()}"
    )
    print(f"  alpha :  true\n{A_true}\n  hat\n{res.alpha.round(3)}")

    # Bar charts of parameters
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    # baseline (gamma0)
    width = 0.35
    x = np.arange(2)
    axes[0].bar(x - width / 2, np.exp(gamma0_true), width, label="true baseline (X=0)")
    axes[0].bar(x + width / 2, np.exp(res.gamma0), width, label="estimated baseline")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["m=0", "m=1"])
    axes[0].set_title("Baseline $\\mu$ (regime X=0)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # covariate effects
    axes[1].bar(x - width / 2, gamma_true.ravel(), width, label="true $\\gamma$")
    axes[1].bar(x + width / 2, res.gamma.ravel(), width, label="estimated $\\gamma$")
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["m=0", "m=1"])
    axes[1].set_title("Covariate effect $\\gamma$")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # alpha
    se_A = res.se_alpha if res.se_alpha is not None else np.full_like(res.alpha, np.nan)
    axes[2].errorbar(A_true.ravel(), res.alpha.ravel(), yerr=se_A.ravel(), fmt="o", capsize=3)
    mx = max(A_true.max(), res.alpha.max()) * 1.2
    axes[2].plot([0, mx], [0, mx], "k--", alpha=0.5)
    axes[2].set_xlabel("true $\\alpha_{m,j}$")
    axes[2].set_ylabel("estimated $\\hat\\alpha_{m,j}$")
    axes[2].set_title("Elicitation recovery")
    axes[2].grid(alpha=0.3)

    fig.suptitle(f"2D Hawkes with regime-switching covariate (T={int(T)}, events={sum(counts)})")
    fig.tight_layout()
    fig_path = os.path.join(out_dir, "exp3_covariates.png")
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)
    print(f"\nWrote {fig_path}")

    np.savez(
        os.path.join(out_dir, "exp3_covariates.npz"),
        gamma0_true=gamma0_true,
        gamma_true=gamma_true,
        A_true=A_true,
        gamma0_hat=res.gamma0,
        gamma_hat=res.gamma,
        A_hat=res.alpha,
        se_A=se_A,
        T=T,
    )
    return res


if __name__ == "__main__":
    run()
