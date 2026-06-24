r"""
Experiment 18 -- recovering covariate coefficients AND the excitation matrix of a
covariate-modulated multivariate Hawkes/MBPP, from the (noisy) observed intensity.

Setup (see hawkes_calibration.operators.covariate_inverse).  ``M`` groups; the
excitation of group ``m`` is modulated in time by

    alpha_{m,j}(t) = A_{m,j} * exp( sum_k delta^s_{m,k} Z^s_k(t) + delta^p_m Z^p_m(t) ),

so each group reacts to ``K_shared`` SHARED covariates ``Z^s`` (the overlap -- every
group with its own coefficients) plus ONE PRIVATE covariate ``Z^p_m`` (the different
one).  Given the observed covariates and the intensity path ``xi`` (optionally
corrupted with relative Gaussian observation noise), we recover the excitation
matrix ``A`` and the live covariate coefficients ``delta`` by differentiable
analysis-by-synthesis (gradient inversion through the validated forward solver).

Outputs:  results/exp18_inverse_M{M}.json, results/exp18_inverse_M{M}.png

Run:  PYTHONPATH=. python -m experiments.exp18_covariate_inverse --M 3
      PYTHONPATH=. python -m experiments.exp18_covariate_inverse --M 5
"""

import os
import json
import time
import argparse

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hawkes_calibration.operators.covariate_inverse import (
    CovariateDesign, sample_dataset, recover_params,
)

REGIME = dict(n_switch=7, diag=(0.2, 0.55), off=(0.05, 0.4), rho_max=0.85)


def run(M, K_shared=2, n_test=48, steps=8000, seed=3, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    design = CovariateDesign(M, K_shared)
    t = np.linspace(0.0, 16.0, 100)
    data = sample_dataset(n_test, design, t, seed=seed, **REGIME)
    print(f"=== exp18 inverse: M={M}, {K_shared} shared + {M} private covariates "
          f"(p={design.p}, {design.n_delta} live coeffs); recover A ({M*M}) + delta ({design.n_delta}) ===",
          flush=True)

    # (1) single-observation noise sweep
    single = []
    for sigma in [0.0, 0.01, 0.02, 0.05, 0.1]:
        t0 = time.time()
        r = recover_params(design, data, sigma=sigma, n_obs=1, steps=steps, early_stop=False, seed=1)
        single.append(dict(sigma=sigma, **{k: r[k] for k in
                      ("A_rel_mean", "A_rel_median", "delta_rel_mean", "delta_rel_median")}))
        print(f"  single obs  sigma={sigma:.2f}  A-rel={r['A_rel_mean']:.4f}  "
              f"delta-rel={r['delta_rel_mean']:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    # (2) averaging R repeated noisy observations at a fixed sigma (the robust fix)
    avg = []
    sigma_a = 0.1
    for R in [1, 16, 64, 256]:
        r = recover_params(design, data, sigma=sigma_a, n_obs=R, steps=steps, early_stop=False, seed=1)
        avg.append(dict(R=R, sigma_eff=sigma_a / np.sqrt(R),
                        A_rel_mean=r["A_rel_mean"], delta_rel_mean=r["delta_rel_mean"]))
        print(f"  avg R={R:3d} (sigma={sigma_a}, eff={sigma_a/np.sqrt(R):.4f})  "
              f"A-rel={r['A_rel_mean']:.4f}  delta-rel={r['delta_rel_mean']:.4f}", flush=True)

    # recovered-vs-true scatter at a representative low-effective-noise setting
    r_demo = recover_params(design, data, sigma=0.05, n_obs=64, steps=steps, early_stop=False, seed=2)

    res = dict(M=M, K_shared=K_shared, p=design.p, n_delta=design.n_delta, n_test=n_test,
               steps=steps, regime=REGIME, single_obs=single, averaged=avg,
               clean_A_rel=single[0]["A_rel_mean"], clean_delta_rel=single[0]["delta_rel_mean"])
    json.dump(res, open(os.path.join(out_dir, f"exp18_inverse_M{M}.json"), "w"), indent=2, default=float)
    _plot(out_dir, M, single, avg, sigma_a, data, r_demo, design)
    print(f"  CLEAN: A-rel={single[0]['A_rel_mean']:.4f}  delta-rel={single[0]['delta_rel_mean']:.4f}", flush=True)
    return res


def _plot(out_dir, M, single, avg, sigma_a, data, r_demo, design):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    sg = [s["sigma"] for s in single]
    ax[0].plot(sg, [s["A_rel_mean"] for s in single], "C3-o", label="A (excitation)")
    ax[0].plot(sg, [s["delta_rel_mean"] for s in single], "C0-s", label="delta (covariate coeffs)")
    ax[0].axhline(0.05, color="k", ls=":", label="5% target")
    ax[0].set_xlabel("observation noise sigma (single obs)"); ax[0].set_ylabel("relative recovery error")
    ax[0].set_title(f"M={M}: recovery vs noise"); ax[0].legend(); ax[0].set_ylim(bottom=0)

    eff = [a["sigma_eff"] for a in avg]
    ax[1].plot([a["R"] for a in avg], [a["A_rel_mean"] for a in avg], "C3-o", label="A")
    ax[1].plot([a["R"] for a in avg], [a["delta_rel_mean"] for a in avg], "C0-s", label="delta")
    ax[1].axhline(0.05, color="k", ls=":", label="5% target")
    ax[1].set_xscale("log"); ax[1].set_xlabel(f"# averaged observations R (sigma={sigma_a})")
    ax[1].set_ylabel("relative recovery error")
    ax[1].set_title("averaging repeated obs (sigma_eff = sigma/sqrt R)"); ax[1].legend(); ax[1].set_ylim(bottom=0)

    # recovered vs true scatter (A entries and delta coeffs)
    ax[2].scatter(data["A"].ravel(), r_demo["A"].ravel(), s=8, alpha=0.4, color="C3", label="A entries")
    ax[2].scatter(data["delta_vec"].ravel(), r_demo["delta_vec"].ravel(), s=8, alpha=0.4, color="C0", label="delta coeffs")
    lim = [min(data["A"].min(), data["delta_vec"].min()) - 0.1, max(data["A"].max(), data["delta_vec"].max()) + 0.1]
    ax[2].plot(lim, lim, "k--", lw=1)
    ax[2].set_xlabel("true value"); ax[2].set_ylabel("recovered")
    ax[2].set_title("recovered vs true (sigma=0.05, R=64)"); ax[2].legend()
    fig.suptitle(f"Recovering excitation A + per-group covariate coefficients delta, M={M} "
                 f"({design.n_delta} covariate coeffs: {design.K_shared} shared + {M} private)")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"exp18_inverse_M{M}.png"), dpi=140); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=3)
    ap.add_argument("--steps", type=int, default=8000)
    a = ap.parse_args()
    run(a.M, steps=a.steps)
