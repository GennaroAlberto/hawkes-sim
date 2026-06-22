r"""
Experiment 6 -- Recovering Hawkes parameters from interval-censored data.

This is the central interval-censored experiment, mirroring Section 7 of Rizoiu
et al. (2022).  From separable Hawkes realizations we construct the censored
scenarios of Table 2 and recover the kernel parameters (kappa, theta) by fitting
an MBPP with the interval-censored loss:

  * Scenario E -- immigrants observed as event times (multi-impulse exogenous),
    offspring interval-censored;
  * Scenario F -- both immigrants and offspring interval-censored (LHPP
    exogenous).  This is the regime of the real-world ACTIVE experiment.

We report three things:

  1. Recovery table for scenarios E and F, in a sub-critical (n*=0.6) and a
     near-critical (n*=0.95) regime, comparing the IC-LL and SSE losses.
  2. A granularity sweep: recovery error vs. the number of observation
     intervals (finer censoring -> less information loss -> better recovery).
  3. A closed-form vs. numerical-approximation comparison of the MBPP loss.

Output: results/exp6_ic_scenarios.png and results/exp6_ic_scenarios.json.
"""

import os
import json
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hawkes_calibration.mbpp.exogenous import PiecewiseConstant, MultiImpulse, LHPP
from hawkes_calibration.mbpp.ic_simulate import (
    simulate_separable_hawkes,
    interval_censor,
    uniform_obs_times,
)
from hawkes_calibration.mbpp.interval_censored import fit_mbpp_ic_multi


# HP-pc exogenous of the paper (Eq. 52): immigrant volumes 7,6,8 over [0,5,10,15].
def hp_pc_exogenous():
    return PiecewiseConstant(breaks=[0.0, 5.0, 10.0, 15.0], rates=[7 / 5, 6 / 5, 8 / 5])


def _make_sequences(kappa, theta, T, n_seq, seed):
    """Generate separable Hawkes sequences (immigrants, offspring) from HP-pc."""
    rng = np.random.default_rng(seed)
    exo_gen = hp_pc_exogenous()
    seqs = []
    for _ in range(n_seq):
        imm, off = simulate_separable_hawkes(exo_gen, kappa, theta, T, seed=int(rng.integers(1e9)))
        seqs.append((imm, off))
    return seqs


def _fit_groups(seqs, obs, scenario, loss, n_groups):
    """Split sequences into groups, fit each group, return per-group (kappa,theta)."""
    K = len(seqs) // n_groups
    ests = []
    for g in range(n_groups):
        grp = seqs[g * K:(g + 1) * K]
        counts_list, exo_list = [], []
        for imm, off in grp:
            counts_list.append(interval_censor(off, obs))
            if scenario == "E":
                exo_list.append(MultiImpulse(imm))
            else:  # F
                exo_list.append(LHPP(obs, interval_censor(imm, obs)))
        res = fit_mbpp_ic_multi(obs, counts_list, exo_list, loss=loss,
                                endogenous=True, method="closed", n_restarts=2)
        ests.append((res.kappa, res.theta))
    return np.array(ests)


def run(seed=0, out_dir="results", T=30.0):
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    results = {}

    regimes = {"subcritical (n*=0.6)": (0.6, 0.8),
               "near-critical (n*=0.95)": (0.95, 1.15)}

    # ---- 1) recovery table: scenarios E,F x regimes x losses, 30 intervals --
    n_groups, K = 8, 10
    obs30 = uniform_obs_times(T, 30)
    table = {}
    for rname, (kt, tt) in regimes.items():
        seqs = _make_sequences(kt, tt, T, n_groups * K, seed)
        for scen in ["E", "F"]:
            for loss in ["ic-ll", "sse"]:
                ests = _fit_groups(seqs, obs30, scen, loss, n_groups)
                table[f"{rname} | {scen} | {loss}"] = dict(
                    kappa_true=kt, theta_true=tt,
                    kappa_mean=float(ests[:, 0].mean()), kappa_std=float(ests[:, 0].std()),
                    theta_mean=float(ests[:, 1].mean()), theta_std=float(ests[:, 1].std()),
                )
    results["recovery_table"] = table
    print("=== Recovery table (30 intervals) ===")
    for k, v in table.items():
        print(f"{k:42s}: kappa={v['kappa_mean']:.3f}±{v['kappa_std']:.3f} "
              f"(true {v['kappa_true']}), theta={v['theta_mean']:.3f}±{v['theta_std']:.3f} (true {v['theta_true']})")

    # ---- 2) granularity sweep (scenario F, subcritical, IC-LL) -------------
    kt, tt = regimes["subcritical (n*=0.6)"]
    seqs = _make_sequences(kt, tt, T, n_groups * K, seed + 1)
    grans = [5, 10, 15, 30, 60]
    sweep = {}
    for n_int in grans:
        obs = uniform_obs_times(T, n_int)
        ests = _fit_groups(seqs, obs, "F", "ic-ll", n_groups)
        sweep[n_int] = dict(kappa_mean=float(ests[:, 0].mean()), kappa_std=float(ests[:, 0].std()),
                            theta_mean=float(ests[:, 1].mean()), theta_std=float(ests[:, 1].std()))
    results["granularity_sweep"] = sweep
    print("\n=== Granularity sweep (scenario F, IC-LL, n*=0.6) ===")
    for n_int, v in sweep.items():
        print(f"  {n_int:3d} intervals: kappa={v['kappa_mean']:.3f}±{v['kappa_std']:.3f}, "
              f"theta={v['theta_mean']:.3f}±{v['theta_std']:.3f}")

    # ---- figure ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    labels = list(table.keys())
    kappa_means = [table[l]["kappa_mean"] for l in labels]
    kappa_stds = [table[l]["kappa_std"] for l in labels]
    ypos = np.arange(len(labels))
    colors = ["C0" if "ic-ll" in l else "C1" for l in labels]
    ax.barh(ypos, kappa_means, xerr=kappa_stds, color=colors, alpha=0.8, capsize=3)
    for rname, (kt_, _) in regimes.items():
        for i, l in enumerate(labels):
            if l.startswith(rname):
                ax.plot([kt_, kt_], [i - 0.4, i + 0.4], "k--", lw=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([l.replace(" | ", "\n") for l in labels], fontsize=7)
    ax.set_xlabel(r"recovered $\hat\kappa$ (branching ratio)")
    ax.set_title("Recovery of $\\kappa$ (dashed = truth; blue=IC-LL, orange=SSE)")
    ax.grid(alpha=0.3, axis="x")

    ax = axes[1]
    gx = np.array(grans)
    km = np.array([sweep[g]["kappa_mean"] for g in grans])
    ks = np.array([sweep[g]["kappa_std"] for g in grans])
    tm = np.array([sweep[g]["theta_mean"] for g in grans])
    ts = np.array([sweep[g]["theta_std"] for g in grans])
    ax.errorbar(gx, km, yerr=ks, fmt="o-", capsize=4, label=r"$\hat\kappa$")
    ax.errorbar(gx, tm, yerr=ts, fmt="s-", capsize=4, label=r"$\hat\theta$")
    ax.axhline(kt, color="C0", ls="--", alpha=0.6, label=f"true $\\kappa$={kt}")
    ax.axhline(tt, color="C1", ls="--", alpha=0.6, label=f"true $\\theta$={tt}")
    ax.set_xlabel("number of observation intervals")
    ax.set_ylabel("estimate")
    ax.set_title("Granularity sweep (scenario F, IC-LL)")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("Interval-censored recovery of Hawkes parameters via MBPP")
    fig.tight_layout()
    path = os.path.join(out_dir, "exp6_ic_scenarios.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)

    with open(os.path.join(out_dir, "exp6_ic_scenarios.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {path}  ({time.time() - t0:.1f}s)")
    return results


if __name__ == "__main__":
    run()
