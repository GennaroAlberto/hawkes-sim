r"""
Experiment 5 -- The Mean Behavior Poisson process as the mean of a Hawkes process.

This reproduces Figures 2 and 3 of Rizoiu et al. (2022): the deterministic MBPP
intensity xi(t) tracks the *mean* intensity of a Hawkes process taken over many
realizations.  We verify the closed-form impulse-response solution (Theorem 2 /
Corollary 3) against a Monte-Carlo average of simulated Hawkes intensities, for
several exogenous functions.

Panel (a): a single exogenous impulse (Fig. 2).
Panels (b-d): rectangle, Dassios-Zhao, and sinusoidal exogenous functions (Fig. 3).

Output: results/exp5_mbpp_impulse.png and results/exp5_mbpp_impulse.json.
"""

import os
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hawkes_calibration.mbpp import MBPP, ExponentialKernel
from hawkes_calibration.exogenous import MultiImpulse, Rectangle, Dassios, Sine
from hawkes_calibration.ic_simulate import simulate_separable_hawkes


def _hawkes_mean_intensity(exo, kappa, theta, T, grid, n_real, seed,
                           include_exo_curve=None):
    """Monte-Carlo mean of the Hawkes intensity on `grid` over `n_real` runs."""
    rng = np.random.default_rng(seed)
    acc = np.zeros_like(grid)
    acc_sq = np.zeros_like(grid)
    for _ in range(n_real):
        imm, off = simulate_separable_hawkes(exo, kappa, theta, T, seed=int(rng.integers(1e9)))
        events = np.concatenate([imm, off])
        lam = np.zeros_like(grid)
        if include_exo_curve is not None:
            lam += include_exo_curve(grid)
        for ti in events:
            m = grid > ti
            lam[m] += kappa * theta * np.exp(-theta * (grid[m] - ti))
        acc += lam
        acc_sq += lam ** 2
    mean = acc / n_real
    var = np.maximum(acc_sq / n_real - mean ** 2, 0.0)
    return mean, np.sqrt(var)


def run(seed=0, out_dir="results", n_real=4000):
    os.makedirs(out_dir, exist_ok=True)
    T = 30.0
    grid = np.linspace(0, T, 600)
    summary = {}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # ---- (a) single impulse: Fig. 2 (near-critical kappa) ----------------
    kappa, theta = 0.9, 1.15
    a = 2.0
    exo = MultiImpulse([a])
    mean, sd = _hawkes_mean_intensity(exo, kappa, theta, T, grid, n_real, seed)
    xi = MBPP(ExponentialKernel(kappa, theta), exo, method="closed").intensity(grid)
    ax = axes[0, 0]
    ax.plot(grid, mean, "k-", lw=1.5, label="Hawkes mean (MC)")
    ax.fill_between(grid, np.maximum(mean - 1.96 * sd, 0.0), mean + 1.96 * sd, color="grey", alpha=0.25,
                    label="Hawkes 95% band")
    ax.plot(grid, xi, "b--", lw=2, label="MBPP $\\xi(t)$ (closed form)")
    ax.axvline(a, color="r", lw=1, alpha=0.7, label="exogenous impulse")
    ax.set_title(f"(a) single impulse  $\\kappa$={kappa}, $\\theta$={theta}")
    ax.set_xlabel("time"); ax.set_ylabel("intensity"); ax.legend(fontsize=8)
    summary["impulse"] = dict(kappa=kappa, theta=theta,
                              max_abs_err=float(np.max(np.abs(mean[grid > a + 0.2] - xi[grid > a + 0.2]))))

    # ---- (b-d) rectangle, Dassios, sine: Fig. 3 (subcritical) ------------
    kappa, theta = 0.6, 0.8
    panels = [
        ("(b) rectangle", Rectangle(5.0, 12.0, 2.0), axes[0, 1], lambda g: Rectangle(5.0, 12.0, 2.0).intensity(g)),
        ("(c) Dassios-Zhao", Dassios(u0=4.0, kappa=kappa, theta=theta), axes[1, 0], None),
        ("(d) sine", Sine(alpha=2.0), axes[1, 1], lambda g: Sine(alpha=2.0).intensity(g)),
    ]
    for title, exo, ax, exo_curve in panels:
        m = MBPP(ExponentialKernel(kappa, theta), exo, method="closed")
        # exogenous curve for the Monte-Carlo intensity (Dassios depends on kappa,theta)
        if isinstance(exo, Dassios):
            exo_curve = lambda g, e=exo: e.s_of(g, kappa, theta)
        mean, sd = _hawkes_mean_intensity(exo, kappa, theta, T, grid, n_real,
                                          seed + 1, include_exo_curve=exo_curve)
        xi = m.intensity(grid)
        if exo_curve is not None:
            xi = xi  # closed-form already includes s(t) for rate-type exogenous
        ax.plot(grid, mean, "k-", lw=1.5, label="Hawkes mean (MC)")
        ax.fill_between(grid, np.maximum(mean - 1.96 * sd, 0.0), mean + 1.96 * sd, color="grey", alpha=0.25)
        ax.plot(grid, xi, "b--", lw=2, label="MBPP $\\xi(t)$")
        if exo_curve is not None:
            ax.plot(grid, exo_curve(grid), "r:", lw=1.3, label="exogenous $s(t)$")
        ax.set_title(f"{title}  $\\kappa$={kappa}, $\\theta$={theta}")
        ax.set_xlabel("time"); ax.set_ylabel("intensity"); ax.legend(fontsize=8)
        key = title.split()[1]
        summary[key] = dict(max_abs_err=float(np.max(np.abs(mean - xi))))

    fig.suptitle("MBPP intensity vs. mean Hawkes intensity (closed-form impulse-response solution)")
    fig.tight_layout()
    path = os.path.join(out_dir, "exp5_mbpp_impulse.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)

    with open(os.path.join(out_dir, "exp5_mbpp_impulse.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Closed-form MBPP vs Monte-Carlo Hawkes mean (max abs intensity error):")
    for k, v in summary.items():
        print(f"  {k:10s}: {v['max_abs_err']:.4f}")
    print(f"\nWrote {path}")
    return summary


if __name__ == "__main__":
    run()
