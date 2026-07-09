"""
Experiment 1 -- Univariate Hawkes with exponential kernel.

We simulate a 1D Hawkes process with known (mu, alpha, beta) and study, as a
function of the horizon T, the bias and the standard error of the MLE for
(mu, alpha). beta is held fixed at its true value (standard assumption).
"""

import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hawkes_calibration import fit_univariate, simulate_multivariate_hawkes


def run(seed=1, out_dir="results", n_reps=30):
    os.makedirs(out_dir, exist_ok=True)
    mu_true, alpha_true, beta_true = 0.5, 0.4, 1.0
    Ts = [200, 500, 1000, 2000, 5000]

    rows = []
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 10**9, size=(len(Ts), n_reps))

    for ti, T in enumerate(Ts):
        mu_hats, alpha_hats = [], []
        for r in range(n_reps):
            ev = simulate_multivariate_hawkes(
                gamma0=np.array([np.log(mu_true)]),
                alpha=np.array([[alpha_true]]),
                beta=np.array([[beta_true]]),
                T=T,
                seed=int(seeds[ti, r]),
            )
            res = fit_univariate(ev[0], T=T, beta=beta_true, se=False)
            mu_hats.append(float(np.exp(res.gamma0[0])))
            alpha_hats.append(float(res.alpha[0, 0]))
        mu_hats = np.array(mu_hats)
        alpha_hats = np.array(alpha_hats)
        rows.append(
            dict(
                T=T,
                mu_mean=float(mu_hats.mean()),
                mu_std=float(mu_hats.std(ddof=1)),
                mu_bias=float(mu_hats.mean() - mu_true),
                alpha_mean=float(alpha_hats.mean()),
                alpha_std=float(alpha_hats.std(ddof=1)),
                alpha_bias=float(alpha_hats.mean() - alpha_true),
            )
        )
        print(
            f"T={T:>5d}  mu_hat={rows[-1]['mu_mean']:.4f} ± {rows[-1]['mu_std']:.4f}"
            f"   alpha_hat={rows[-1]['alpha_mean']:.4f} ± {rows[-1]['alpha_std']:.4f}"
        )

    # Plot
    Ts_arr = np.array(Ts)
    mu_mean = np.array([r["mu_mean"] for r in rows])
    mu_std = np.array([r["mu_std"] for r in rows])
    a_mean = np.array([r["alpha_mean"] for r in rows])
    a_std = np.array([r["alpha_std"] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].errorbar(Ts_arr, mu_mean, yerr=mu_std, fmt="o-", capsize=4, label="MLE mean ± sd")
    axes[0].axhline(mu_true, color="r", linestyle="--", label=f"truth {mu_true}")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("horizon T")
    axes[0].set_ylabel(r"$\hat\mu$")
    axes[0].set_title("Baseline rate recovery (1D)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].errorbar(Ts_arr, a_mean, yerr=a_std, fmt="o-", capsize=4, label="MLE mean ± sd")
    axes[1].axhline(alpha_true, color="r", linestyle="--", label=f"truth {alpha_true}")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("horizon T")
    axes[1].set_ylabel(r"$\hat\alpha$")
    axes[1].set_title("Self-excitation recovery (1D)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(
        rf"Univariate Hawkes, $\mu={mu_true}$, $\alpha={alpha_true}$, $\beta={beta_true}$  ({n_reps} replications)"
    )
    fig.tight_layout()
    fig_path = os.path.join(out_dir, "exp1_1d.png")
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)

    with open(os.path.join(out_dir, "exp1_1d.json"), "w") as f:
        json.dump(
            {"truth": dict(mu=mu_true, alpha=alpha_true, beta=beta_true), "rows": rows}, f, indent=2
        )
    print(f"\nWrote {fig_path}")
    return rows


if __name__ == "__main__":
    run()
