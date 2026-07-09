r"""
Experiment 10 -- the "deep" extensions, end to end.

Four panels:
  (a) Excitation-covariate recovery from *true Hawkes* data (not MBPP-Poisson):
      simulate event streams whose excitation is covariate-modulated, interval-
      censor them, and recover delta with fit_mbpp_ic_excitation.
  (b) The recovered time-varying branching profile kappa(t)=kappa*exp(delta^T Z(t)).
  (c) Sum-of-exponentials (multi-timescale) recovery over a fixed bank of decays.
  (d) Goodness-of-fit: Pearson dispersion separates the correct model (~1) from a
      misspecified kernel (>1).

Output: results/exp10_deep.png and .json.
"""

import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hawkes_calibration import (
    MBPP,
    Constant,
    ExponentialKernel,
    MultiImpulse,
    PiecewiseConstantCovariate,
    dispersion,
    fit_mbpp_ic_excitation,
    fit_mbpp_ic_sumexp,
    interval_censor,
    simulate_hawkes_excitation,
    simulate_separable_hawkes,
    uniform_obs_times,
)
from hawkes_calibration.mbpp.interval_censored import (
    _sumexp_compensator_const,
)


def run(seed=0, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    summary = {}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # ---- (a,b) excitation covariates from TRUE Hawkes data ----
    T, m = 60.0, 60
    obs = np.arange(m + 1, dtype=float)
    Z = PiecewiseConstantCovariate(
        obs, np.array([[((i // 10) % 2) - 0.5] for i in range(m)], float)
    )
    kappa_t, theta_t, mu_t, delta_t = 0.4, 1.0, 2.0, 1.2
    n_groups, K = 8, 8
    deltas = []
    last = None
    for g in range(n_groups):
        counts = []
        for _ in range(K):
            ev = simulate_hawkes_excitation(
                Constant(mu_t, T), kappa_t, theta_t, Z, [delta_t], T, seed=int(rng.integers(1e9))
            )
            counts.append(interval_censor(ev, obs))
        res = fit_mbpp_ic_excitation(obs, counts, Z, n_restarts=3, seed=g)
        deltas.append(res.delta[0])
        last = res
    deltas = np.array(deltas)
    summary["excitation"] = dict(
        delta_true=delta_t,
        delta_mean=float(deltas.mean()),
        delta_std=float(deltas.std()),
        kappa=last.kappa,
        theta=last.theta,
    )

    ax = axes[0, 0]
    ax.axhline(delta_t, color="r", ls="--", label=f"true $\\delta$={delta_t}")
    ax.plot(
        np.arange(n_groups), deltas, "o", color="C0", label="recovered $\\hat\\delta$ (per group)"
    )
    ax.axhline(deltas.mean(), color="C0", alpha=0.5, label=f"mean {deltas.mean():.2f}")
    ax.set_xlabel("group")
    ax.set_ylabel("$\\hat\\delta$ (excitation-covariate effect)")
    ax.set_title("(a) excitation $\\delta$ recovered from true Hawkes counts")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    tg = np.linspace(0, T, 2000)
    Zt = Z(tg)[:, 0]
    ax.plot(
        tg,
        kappa_t * np.exp(delta_t * Zt),
        "k-",
        lw=2,
        label=r"true $\kappa(t)=\kappa e^{\delta Z(t)}$",
    )
    ax.plot(tg, last.kappa * np.exp(last.delta[0] * Zt), "b--", lw=2, label="recovered")
    ax.axhline(1.0, color="grey", lw=0.5, ls=":")
    ax.set_xlim(0, 40)
    ax.set_xlabel("time")
    ax.set_ylabel("time-varying branching ratio")
    ax.set_title("(b) recovered excitation modulation")
    ax.legend(fontsize=8)

    # ---- (c) sum-of-exponentials recovery ----
    mu_s = 1.5
    thetas_true = np.array([2.0, 0.4])
    kappas_true = np.array([0.15, 0.45])
    obs_s = np.arange(81, dtype=float)
    dXi = np.diff(_sumexp_compensator_const(mu_s, kappas_true * thetas_true, thetas_true, obs_s))
    counts_s = [rng.poisson(np.maximum(dXi, 0)).astype(float) for _ in range(25)]
    bank = np.array([0.4, 1.0, 2.0])
    res_s = fit_mbpp_ic_sumexp(obs_s, counts_s, bank, l1=0.02, n_restarts=4)
    summary["sumexp"] = dict(
        bank=bank.tolist(),
        kappas=res_s.kappas.tolist(),
        total_true=0.60,
        total_fit=float(res_s.kappas.sum()),
    )
    ax = axes[1, 0]
    x = np.arange(bank.size)
    true_w = np.array([0.45, 0.0, 0.15])  # weights aligned to bank [0.4,1.0,2.0]
    ax.bar(x - 0.2, true_w, width=0.4, color="k", alpha=0.6, label="true weights")
    ax.bar(x + 0.2, res_s.kappas, width=0.4, color="C2", label="recovered (L1)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"$\\theta$={th}" for th in bank])
    ax.set_ylabel("branching weight $\\kappa_q$")
    ax.set_title(f"(c) sum-of-exp: total $\\kappa$ {res_s.kappas.sum():.2f} vs 0.60")
    ax.legend(fontsize=8)

    # ---- (d) goodness-of-fit: dispersion separates correct vs wrong model ----
    kappa_g, theta_g, Tg = 0.5, 1.0, 30.0
    obs_g = uniform_obs_times(Tg, 30)
    disp_correct, disp_wrong = [], []
    for _ in range(40):
        imm, off = simulate_separable_hawkes(
            Constant(5.0, Tg), kappa_g, theta_g, Tg, seed=int(rng.integers(1e9))
        )
        counts = interval_censor(off, obs_g)
        # correct exogenous (multi-impulse) vs wrong (ignore immigrants -> constant baseline)
        Xi_ok = MBPP(
            ExponentialKernel(kappa_g, theta_g), MultiImpulse(imm), method="closed"
        ).compensator_interval(obs_g, endogenous=True)
        Xi_bad = MBPP(
            ExponentialKernel(0.1, 5.0), MultiImpulse(imm), method="closed"
        ).compensator_interval(obs_g, endogenous=True)
        disp_correct.append(dispersion(counts, np.maximum(Xi_ok, 1e-9), n_params=2))
        disp_wrong.append(dispersion(counts, np.maximum(Xi_bad, 1e-9), n_params=2))
    summary["gof"] = dict(
        disp_correct_mean=float(np.mean(disp_correct)), disp_wrong_mean=float(np.mean(disp_wrong))
    )
    ax = axes[1, 1]
    ax.hist(
        disp_correct,
        bins=15,
        alpha=0.6,
        color="C0",
        label=f"correct kernel (mean {np.mean(disp_correct):.1f})",
    )
    ax.hist(
        disp_wrong,
        bins=15,
        alpha=0.6,
        color="C3",
        label=f"wrong kernel (mean {np.mean(disp_wrong):.1f})",
    )
    ax.axvline(1.0, color="k", ls="--", label="Poisson (dispersion=1)")
    ax.set_xlabel("Pearson dispersion")
    ax.set_ylabel("count")
    ax.set_title("(d) goodness-of-fit separates models")
    ax.legend(fontsize=8)

    fig.suptitle("Deep extensions: excitation covariates, multi-timescale kernels, goodness-of-fit")
    fig.tight_layout()
    path = os.path.join(out_dir, "exp10_deep.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)

    print("=== exp10 deep extensions ===")
    print(f"(a) excitation delta: {deltas.mean():.2f} ± {deltas.std():.2f} (true {delta_t})")
    print(
        f"(c) sum-of-exp total branching: {res_s.kappas.sum():.2f} (true 0.60); weights {np.round(res_s.kappas, 3).tolist()}"
    )
    print(f"(d) dispersion correct {np.mean(disp_correct):.2f} vs wrong {np.mean(disp_wrong):.1f}")
    with open(os.path.join(out_dir, "exp10_deep.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {path}")
    return summary


if __name__ == "__main__":
    run()
