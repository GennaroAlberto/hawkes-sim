r"""
Experiment 9 -- Functional / operator views of the MBPP, on simple scenarios.

The MBPP solution map  s |-> xi  is a linear, translation-invariant operator.
This experiment exercises four interchangeable representations of it and
cross-validates them, so one can pick whichever suits a problem:

  (a) EXACT backends agree.  For an exponential kernel and a chosen forcing, the
      closed form, the state-space ODE reduction, and the exact Fourier operator
      R(w)=1/(1-phi_hat) all produce the same xi.

  (b) LEARN the operator (spectral).  From (forcing, response) pairs we fit the
      transfer function R(w) by per-frequency least squares, generalise to a new
      forcing, and recover the kernel phi_hat = 1 - 1/R.

  (c) DeepONet surrogate.  A branch/trunk network learns s |-> xi without
      assuming linearity -- the route for no-closed-form regimes.

  (d) Amortized inference.  A network learns the inverse map counts |-> (kappa,
      theta); fitting a new series becomes a single forward pass.

Output: results/exp9_functional_operators.png and .json.
"""

import os
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hawkes_calibration.mbpp import MBPP, ExponentialKernel
from hawkes_calibration.mbpp.exogenous import PiecewiseConstant, Sine, Constant
from hawkes_calibration.operators import FunctionalMBPP, SpectralOperator, kernel_exponentials
from hawkes_calibration.operators.nn import DeepONetOperator, AmortizedInference
from hawkes_calibration.mbpp.ic_simulate import simulate_separable_hawkes, interval_censor, uniform_obs_times


def run(seed=0, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    kappa, theta = 0.6, 0.8
    ker = ExponentialKernel(kappa, theta)
    summary = {}

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # ---------------------------------------------------------------
    # (a) exact backends agree: closed vs ODE vs spectral
    # ---------------------------------------------------------------
    T = 40.0
    t = np.linspace(0, T, 2001)
    exo = Sine(2.0)
    xi_closed = MBPP(ker, exo, method="closed").intensity(t)
    xi_ode = FunctionalMBPP(ker, method="ode").solve(exo.intensity, t)
    xi_spec = FunctionalMBPP(ker, method="spectral").solve(exo.intensity, t)
    m = (t > 1) & (t < 35)
    err_ode = float(np.max(np.abs(xi_ode - xi_closed)[m]))
    err_spec = float(np.max(np.abs(xi_spec - xi_closed)[m]))
    summary["exact_backends"] = dict(max_err_ode=err_ode, max_err_spectral=err_spec)
    ax = axes[0, 0]
    ax.plot(t, exo.intensity(t), color="0.6", lw=1, label="forcing $s(t)$")
    ax.plot(t, xi_closed, "k-", lw=3, alpha=0.5, label="closed form")
    ax.plot(t, xi_ode, "b--", lw=1.5, label=f"ODE (err {err_ode:.1e})")
    ax.plot(t, xi_spec, "r:", lw=1.8, label=f"spectral (err {err_spec:.1e})")
    ax.set_xlim(0, T); ax.set_xlabel("time"); ax.set_ylabel(r"$\xi(t)$")
    ax.set_title("(a) exact backends agree: closed = ODE = spectral")
    ax.legend(fontsize=8)

    # ---------------------------------------------------------------
    # (b) learn the operator from data; recover the kernel
    # ---------------------------------------------------------------
    tL = np.linspace(0, T, 2001)

    def rand_pc():
        nb = rng.integers(4, 9)
        br = np.sort(np.concatenate([[0], rng.uniform(0, T, nb - 1), [T]]))
        return PiecewiseConstant(br, rng.uniform(0.2, 3.0, br.size - 1))

    exos = [rand_pc() for _ in range(40)]
    S = np.array([e.intensity(tL) for e in exos])
    XI = np.array([MBPP(ker, e, method="closed").intensity(tL) for e in exos])
    spec = SpectralOperator(tL, pad=4).fit(S, XI)
    exo_new = PiecewiseConstant([0, 7, 18, 30, 40], [1.5, 0.5, 2.5, 1.0])
    xi_pred = spec(exo_new.intensity(tL))
    xi_tru = MBPP(ker, exo_new, method="closed").intensity(tL)
    mm = (tL > 1) & (tL < 35)
    gen_err = float(np.linalg.norm((xi_pred - xi_tru)[mm]) / np.linalg.norm(xi_tru[mm]))
    tt, phi_rec = spec.recover_kernel()
    phi_true = ker(tt)
    br_rec = float(np.trapezoid(np.maximum(phi_rec, 0)[tt < 20], tt[tt < 20]))
    summary["spectral_learned"] = dict(operator_gen_rel_err=gen_err,
                                       recovered_branching=br_rec, true_branching=kappa)
    ax = axes[0, 1]
    ax.plot(tt, phi_true, "k-", lw=2, label=r"true $\phi(t)=\kappa\theta e^{-\theta t}$")
    ax.plot(tt, phi_rec, "m--", lw=1.5, label="recovered from data")
    ax.axhline(0, color="0.8", lw=0.5)
    ax.set_xlim(0, 10); ax.set_ylim(-0.12, max(0.6, float(phi_true.max()) * 1.15))
    ax.set_xlabel("time"); ax.set_ylabel(r"$\phi(t)$")
    ax.set_title(f"(b) learned operator: kernel recovered\n"
                 f"(gen. err {gen_err:.1%}, branching {br_rec:.2f} vs {kappa})")
    ax.legend(fontsize=8)

    # ---------------------------------------------------------------
    # (c) DeepONet surrogate
    # ---------------------------------------------------------------
    Td = 30.0
    td = np.linspace(0, Td, 300)
    sensors = np.linspace(0, Td, 40)
    ker_d = ExponentialKernel(kappa, theta)

    def rand_pc_d():
        nb = rng.integers(3, 7)
        br = np.sort(np.concatenate([[0], rng.uniform(0, Td, nb - 1), [Td]]))
        return PiecewiseConstant(br, rng.uniform(0.2, 3.0, br.size - 1))

    tr = [rand_pc_d() for _ in range(300)]
    Ftr = np.array([e.intensity(sensors) for e in tr])
    Ytr = np.array([MBPP(ker_d, e, method="closed").intensity(td) for e in tr])
    don = DeepONetOperator(n_sensors=40, p=24, hidden=48).fit(Ftr, td, Ytr, epochs=600, lr=2e-3)
    te = [rand_pc_d() for _ in range(40)]
    Fte = np.array([e.intensity(sensors) for e in te])
    Yte = np.array([MBPP(ker_d, e, method="closed").intensity(td) for e in te])
    Pte = don.predict(Fte, td)
    don_rel = float(np.mean(np.linalg.norm(Pte - Yte, axis=1) / np.linalg.norm(Yte, axis=1)))
    summary["deeponet"] = dict(test_rel_L2=don_rel)
    ax = axes[1, 0]
    for k in range(3):
        ax.plot(td, Yte[k], "-", color=f"C{k}", lw=2, alpha=0.7)
        ax.plot(td, Pte[k], "--", color=f"C{k}", lw=1.5)
    ax.plot([], [], "k-", label="true $\\xi$"); ax.plot([], [], "k--", label="DeepONet")
    ax.set_xlabel("time"); ax.set_ylabel(r"$\xi(t)$")
    ax.set_title(f"(c) DeepONet operator surrogate (test rel. L2 {don_rel:.1%})")
    ax.legend(fontsize=8)

    # ---------------------------------------------------------------
    # (d) amortized inference: counts -> (kappa, theta)
    # ---------------------------------------------------------------
    Ta = 40.0
    obs = uniform_obs_times(Ta, 40)
    exo_a = Constant(3.0, Ta)

    def make_amort(K):
        X, Y = [], []
        for _ in range(K):
            ka = rng.uniform(0.2, 0.85); th = rng.uniform(0.4, 2.0)
            imm, off = simulate_separable_hawkes(exo_a, ka, th, Ta, seed=int(rng.integers(1e9)))
            X.append(interval_censor(np.sort(np.concatenate([imm, off])), obs))
            Y.append([ka, th])
        return np.array(X, float), np.array(Y)

    Xtr, Ytr2 = make_amort(1500)
    Xte, Yte2 = make_amort(300)
    amort = AmortizedInference(n_features=40, hidden=64).fit(Xtr, Ytr2, epochs=800, lr=2e-3)
    P = amort.predict(Xte)
    corr_k = float(np.corrcoef(P[:, 0], Yte2[:, 0])[0, 1])
    corr_t = float(np.corrcoef(P[:, 1], Yte2[:, 1])[0, 1])
    mae_k = float(np.mean(np.abs(P[:, 0] - Yte2[:, 0])))
    summary["amortized"] = dict(kappa_corr=corr_k, theta_corr=corr_t, kappa_mae=mae_k)
    ax = axes[1, 1]
    ax.scatter(Yte2[:, 0], P[:, 0], s=10, alpha=0.5, label=f"$\\kappa$ (corr {corr_k:.2f})")
    ax.scatter(Yte2[:, 1], P[:, 1], s=10, alpha=0.5, color="C3", label=f"$\\theta$ (corr {corr_t:.2f})")
    lims = [0, 2.1]
    ax.plot(lims, lims, "k--", alpha=0.5)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("true parameter"); ax.set_ylabel("amortized prediction (1 forward pass)")
    ax.set_title("(d) amortized inference: counts $\\to(\\kappa,\\theta)$")
    ax.legend(fontsize=8)

    fig.suptitle("Functional / operator views of the MBPP (exact ODE & spectral, learned spectral, DeepONet, amortized)")
    fig.tight_layout()
    path = os.path.join(out_dir, "exp9_functional_operators.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)

    print("=== Functional / operator backends ===")
    print(f"(a) exact agreement:  ODE err={err_ode:.2e},  spectral err={err_spec:.2e}")
    print(f"(b) learned spectral: operator gen. err={gen_err:.1%}, branching {br_rec:.3f} (true {kappa})")
    print(f"(c) DeepONet:         test rel. L2 = {don_rel:.1%}")
    print(f"(d) amortized:        kappa corr={corr_k:.2f} (MAE {mae_k:.3f}), theta corr={corr_t:.2f}")
    print("    -> exploit known structure (ODE/spectral) when you have it; the learned")
    print("       operators are for the unknown-kernel / nonlinear / amortized regimes.")
    with open(os.path.join(out_dir, "exp9_functional_operators.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {path}")
    return summary


if __name__ == "__main__":
    run()
