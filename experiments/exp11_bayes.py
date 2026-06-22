r"""
Experiment 11 -- Bayesian calibration of the MBPP.

Four panels make the case for going Bayesian on a weakly-identified model:
  (a) the joint (kappa, theta) posterior -- the identifiability ridge, visible as a
      correlated banana rather than a single point + SE;
  (b) the marginals -- kappa tight around truth, theta honestly broad;
  (c) an informative theta-prior legitimately tightens theta;
  (d) hierarchical (partial) pooling sharpens the population branching ratio from
      many short sequences.

Output: results/exp11_bayes.png and .json.
"""

import os
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hawkes_calibration import (
    Constant, simulate_separable_hawkes, interval_censor, uniform_obs_times, MultiImpulse,
    fit_mbpp_bayes, fit_mbpp_bayes_hierarchical, default_prior, GaussianPrior,
)


def run(seed=0, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    kappa, theta, T = 0.5, 1.0, 30.0
    obs = uniform_obs_times(T, 30)
    imm, off = simulate_separable_hawkes(Constant(5.0, T), kappa, theta, T, seed=1)
    counts, Z = interval_censor(off, obs), MultiImpulse(imm)

    res = fit_mbpp_bayes(obs, counts, Z, endogenous=True, method="mcmc",
                         n_samples=6000, burn=2000, n_chains=4, seed=0)
    tight_prior = GaussianPrior(means=[default_prior().means[0], np.log(1.0)], sds=[1.5, 0.12])
    res_t = fit_mbpp_bayes(obs, counts, Z, endogenous=True, method="mcmc",
                           n_samples=6000, burn=2000, n_chains=4, prior=tight_prior, seed=0)
    corr = float(np.corrcoef(res.samples["kappa"], res.samples["theta"])[0, 1])

    # hierarchical: several SHORT sequences (each weak alone), pool the branching ratio
    Tj, mj, J = 12.0, 12, 6
    obsj = uniform_obs_times(Tj, mj)
    cl, el = [], []
    for _ in range(J):
        i2, o2 = simulate_separable_hawkes(Constant(5.0, Tj), kappa, theta, Tj, seed=int(rng.integers(1e9)))
        cl.append(interval_censor(o2, obsj)); el.append(MultiImpulse(i2))
    resh = fit_mbpp_bayes_hierarchical(obsj, cl, el, endogenous=True, theta=theta,
                                       n_samples=2000, burn=800, n_chains=2, seed=0)

    summary = dict(
        kappa_mean=float(res.samples["kappa"].mean()),
        kappa_CrI=[float(x) for x in np.percentile(res.samples["kappa"], [2.5, 97.5])],
        theta_CrI=[float(x) for x in np.percentile(res.samples["theta"], [2.5, 97.5])],
        theta_CrI_tight=[float(x) for x in np.percentile(res_t.samples["theta"], [2.5, 97.5])],
        ridge_corr=corr,
        kappa_pop_CrI=[float(x) for x in np.percentile(resh.samples["kappa_pop"], [2.5, 97.5])],
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    ax.scatter(res.samples["kappa"], res.samples["theta"], s=4, alpha=0.15, color="C0")
    ax.axvline(kappa, color="r", ls="--", lw=1); ax.axhline(theta, color="r", ls="--", lw=1)
    ax.plot(kappa, theta, "r*", ms=14, label="truth")
    ax.set_xlabel(r"$\kappa$"); ax.set_ylabel(r"$\theta$")
    ax.set_title(f"(a) joint posterior: the ridge (corr {corr:+.2f})"); ax.legend()

    ax = axes[0, 1]
    ax.hist(res.samples["kappa"], bins=40, density=True, alpha=0.7, color="C0", label=r"$\kappa$")
    ax.axvline(kappa, color="r", ls="--"); ax.set_xlabel("value"); ax.set_ylabel("posterior density")
    ax.set_title("(b) marginal of $\\kappa$ (tight, covers truth)"); ax.legend()

    ax = axes[1, 0]
    ax.hist(res.samples["theta"], bins=40, density=True, alpha=0.6, color="C1", label="flat prior")
    ax.hist(res_t.samples["theta"], bins=40, density=True, alpha=0.6, color="C2", label="informative prior")
    ax.axvline(theta, color="r", ls="--", label="truth")
    ax.set_xlabel(r"$\theta$"); ax.set_ylabel("posterior density")
    ax.set_title("(c) a $\\theta$-prior legitimately tightens $\\theta$"); ax.legend()

    ax = axes[1, 1]
    ax.hist(resh.samples["kappa_pop"], bins=40, density=True, alpha=0.7, color="C3")
    ax.axvline(kappa, color="r", ls="--", label="truth")
    lo, hi = np.percentile(resh.samples["kappa_pop"], [2.5, 97.5])
    ax.set_title(f"(d) hierarchical pooling: $\\kappa_{{pop}}$\n95% CrI [{lo:.3f}, {hi:.3f}] from {J} short series")
    ax.set_xlabel(r"$\kappa_{pop}$"); ax.set_ylabel("posterior density"); ax.legend()

    fig.suptitle("Bayesian calibration: honest uncertainty, the identifiability ridge, priors, pooling")
    fig.tight_layout()
    path = os.path.join(out_dir, "exp11_bayes.png")
    fig.savefig(path, dpi=140); plt.close(fig)

    print("=== exp11 Bayesian ===")
    print(f"  kappa 95% CrI = {summary['kappa_CrI']} (truth {kappa})")
    print(f"  theta 95% CrI = {summary['theta_CrI']} (flat) -> {summary['theta_CrI_tight']} (informative)")
    print(f"  ridge corr(kappa,theta) = {corr:+.2f}")
    print(f"  kappa_pop 95% CrI (hierarchical) = {summary['kappa_pop_CrI']}")
    with open(os.path.join(out_dir, "exp11_bayes.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {path}")
    return summary


if __name__ == "__main__":
    run()
