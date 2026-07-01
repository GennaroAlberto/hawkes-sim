r"""
Deliverable driver for the pure-learning experiments (docs/package_guide.md).

Runs the three experiment families on the *stability-constrained* investment
parameter box and writes machine-readable metrics (JSON) plus figures into
``results/learning/`` so they can be dropped straight into the report.

Blocks
------
  P  -- physics-informed JAX solvers: P1 vanilla PINN, P2 PINO held-out covariate
        paths, P3 contamination study (raw vs stability-constrained box).
  F  -- TensorFlow forward neural operators: F1 FNO accuracy vs the exact solver,
        FNO vs StateSpaceMBPP, and a small clean-vs-noisy robustness check.
  A  -- TensorFlow amortised inverse inference: recover the branching matrix from
        interval counts in one forward pass; accuracy and inference speed.

Usage:
  PYTHONPATH=. .venv/bin/python experiments/learning/run_deliverable.py --block P
  PYTHONPATH=. .venv/bin/python experiments/learning/run_deliverable.py --block F
  PYTHONPATH=. .venv/bin/python experiments/learning/run_deliverable.py --block A
  PYTHONPATH=. .venv/bin/python experiments/learning/run_deliverable.py --all
"""

import argparse
import json
import os
import time

import numpy as np

from experiments.learning import configs

RESULTS = configs.results_dir()
ZMAX = 0.5                       # the regime covariate lives in [-0.5, 0.5]


def _regime_Z(t):
    """The fixed regime covariate used for the parametric-PINN experiments."""
    return ((np.floor(t / 10) % 2) - 0.5)[:, None]


def _save(name, payload):
    path = os.path.join(RESULTS, name)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"  wrote {path}", flush=True)


# ===========================================================================
# Block P -- physics-informed JAX solvers
# ===========================================================================
def block_P(n_pinn=30000, n_pino=30000):
    from hawkes_calibration.operators.neural_solver import (
        make_neural_solver, solver_accuracy_report, sample_covariate_paths)

    t = configs.collocation_grid()
    T = float(t[-1])
    Z = _regime_Z(t)
    out = {"grid": {"T": T, "n": int(t.size)}}

    # ---- P3 (contamination study) first: it motivates everything else -----
    # How much of the shipped box is supercritical, and how big do targets get?
    from hawkes_calibration.operators.neural_solver import make_anchor_data
    Praw = configs.sample_params(2000, seed=1)
    keff_raw = configs.effective_branching(Praw, ZMAX)
    Zrep = np.repeat(Z.reshape(1, -1), len(Praw), axis=0)
    XIraw = make_anchor_data(t, Praw, Zrep)
    finite = np.isfinite(XIraw).all(axis=1)
    mx = np.where(finite, np.nanmax(np.abs(np.where(finite[:, None], XIraw, 0.0)), axis=1), np.inf)
    out["P3_contamination"] = {
        "frac_supercritical": float(np.mean(keff_raw >= 1.0)),
        "max_finite_intensity_stable": float(np.nanmax(mx[np.isfinite(mx) & (keff_raw < 0.95)])),
        "max_intensity_supercritical_subset": float(
            np.nanmax(np.where(np.isfinite(mx), mx, np.nan)[keff_raw >= 1.0])
            if np.any(keff_raw >= 1.0) else np.nan),
        "note": "effective branching = kappa*exp(|delta|*max|Z|); >=1 has no bounded solution",
    }
    print("  P3 contamination:", out["P3_contamination"], flush=True)

    # evaluation sets (stability-constrained)
    P_eval = configs.sample_params_stable(96, seed=99, zmax=ZMAX, cap=0.9)
    pred_pinn = None

    # ---- P1: vanilla parametric PINN on the stable box --------------------
    s = make_neural_solver(backend="jax", mode="pinn", coll_grid=t, Z_on_grid=Z, T=T,
                           n_delta=1, width=96, depth=4,
                           kappa_range=(0.1, 0.95), delta_range=(-1.0, 1.5))
    t0 = time.time()
    s.train(n_steps=n_pinn, batch=96)
    train_s = time.time() - t0
    rep = solver_accuracy_report(
        lambda p, z: s.solve(dict(kappa=p[0], theta=p[1], mu=p[2], delta=[p[3]]), t),
        t, P_eval, Z_array=None,
        plot_path=os.path.join(RESULTS, "P1_pinn_accuracy.png"))
    out["P1_pinn"] = {"n_steps": n_pinn, "train_s": train_s,
                      "rel_l2_mean": rep["mean"], "rel_l2_median": float(np.median(rep["rel_l2"])),
                      "rel_l2_max": rep["max"], "by_kappa": rep["by_kappa"]}
    print("  P1 PINN:", {k: out["P1_pinn"][k] for k in ("rel_l2_mean", "rel_l2_median", "rel_l2_max")}, flush=True)
    _save("P_results.json", out)

    # ---- P2: PINO generalising over covariate paths AND params ------------
    op = make_neural_solver(backend="jax", mode="pino", coll_grid=t, T=T,
                            kappa_range=(0.1, 0.7), theta_range=(0.4, 2.5),
                            mu_range=(0.5, 3.0), delta_range=(-0.4, 0.4))
    t0 = time.time()
    op.train(n_steps=n_pino, batch=32)
    train_s = time.time() - t0
    rng = np.random.default_rng(5)
    P_te = np.column_stack([rng.uniform(0.1, 0.7, 64), rng.uniform(0.4, 2.5, 64),
                            rng.uniform(0.5, 3.0, 64), rng.uniform(-0.4, 0.4, 64)])
    Z_te = sample_covariate_paths(t, 64, seed=5)
    # inference-time speed: one forward pass per instance
    t0 = time.time()
    _ = [op.solve(dict(kappa=p[0], theta=p[1], mu=p[2], delta=[p[3]]), t, Z_on_grid=z)
         for p, z in zip(P_te, Z_te)]
    solve_ms = 1000 * (time.time() - t0) / len(P_te)
    rep = solver_accuracy_report(
        lambda p, z: op.solve(dict(kappa=p[0], theta=p[1], mu=p[2], delta=[p[3]]), t, Z_on_grid=z),
        t, P_te, Z_array=Z_te,
        plot_path=os.path.join(RESULTS, "P2_pino_accuracy.png"))
    out["P2_pino"] = {"n_steps": n_pino, "train_s": train_s, "solve_ms_per_instance": solve_ms,
                      "heldout_rel_l2_mean": rep["mean"], "heldout_rel_l2_median": float(np.median(rep["rel_l2"])),
                      "heldout_rel_l2_max": rep["max"], "by_kappa": rep["by_kappa"]}
    print("  P2 PINO:", {k: out["P2_pino"][k] for k in ("heldout_rel_l2_mean", "heldout_rel_l2_median", "solve_ms_per_instance")}, flush=True)
    _save("P_results.json", out)
    return out


# ===========================================================================
# Block F -- TensorFlow forward neural operators
#
# Key methodological point (mirrors the PINN-vs-PINO result): the operator s->xi
# depends on the system's branching matrix G.  An FNO whose ONLY input is the
# forcing s cannot represent a *family* of operators (different G, same s -> the
# map is not even a function); it can only learn the mean operator and plateaus.
# We therefore report two regimes:
#   (i)  FIXED operator  -- one G, vary only the forcing  -> the FNO's true accuracy;
#   (ii) FAMILY (unconditioned) -- the shipped generate_operator_dataset (new G per
#        sample, G not given to the net) -> the ill-posed regime, as a diagnostic.
# ===========================================================================
def _standardize(X, mu=None, sd=None):
    if mu is None:
        mu, sd = X.mean((0, 1)), X.std((0, 1)) + 1e-6
    return (X - mu) / sd, mu, sd


def _sanitize(S, XI, cap=100.0):
    """Drop the rare (~0.1%) samples where the explicit ODE solver diverges."""
    mx = np.array([np.max(np.abs(x)) for x in XI])
    keep = np.isfinite(mx) & (mx < cap)
    return S[keep], XI[keep], int((~keep).sum())


def _fixed_operator_dataset(n, M, seq_len, T=30.0, max_radius=0.7, seed=0):
    from hawkes_calibration.operators.tf import (
        _random_branching_matrix, _random_forcing, solve_mbpp_ode_multivariate)
    rng = np.random.default_rng(seed)
    B = np.ones((M, M))
    A = _random_branching_matrix(M, 0.4, max_radius, rng) * B      # ONE operator
    S = np.empty((n, seq_len, M), np.float32); XI = np.empty((n, seq_len, M), np.float32)
    for i in range(n):
        t, s = _random_forcing(M, seq_len, T, rng)
        forcing = lambda tt, t=t, s=s: s[min(int(np.searchsorted(t, tt, "right") - 1), seq_len - 1)]
        XI[i] = solve_mbpp_ode_multivariate(forcing, A, B, t); S[i] = s
    return S, XI


def block_F(n_samples=4000, M=6, seq_len=96, epochs=80):
    import tensorflow as tf
    from hawkes_calibration.operators.tf import (
        FourierNeuralOperator, StateSpaceMBPP, generate_operator_dataset)

    out = {"config": {"n_samples": n_samples, "M": M, "seq_len": seq_len, "epochs": epochs}}

    def fit_eval(model, Str, XItr, Ste, XIte, ep=epochs):
        Sn, smu, ssd = _standardize(Str); XIn, xmu, xsd = _standardize(XItr)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
        t0 = time.time(); model.fit(Sn, XIn, epochs=ep, batch_size=128, verbose=0); dt = time.time() - t0
        P = model.predict((Ste - smu) / ssd, verbose=0) * xsd + xmu
        num = np.linalg.norm((P - XIte).reshape(len(Ste), -1), axis=1)
        den = np.linalg.norm(XIte.reshape(len(Ste), -1), axis=1) + 1e-12
        return num / den, dt

    # ---- F1: FIXED operator -- the FNO's true accuracy (system held constant) ----
    S, XI = _fixed_operator_dataset(n_samples, M, seq_len, seed=0)
    ntr = int(0.9 * len(S))
    r_fix, dt = fit_eval(FourierNeuralOperator(seq_len=seq_len, n_channels=M, width=64, modes=32),
                         S[:ntr], XI[:ntr], S[ntr:], XI[ntr:])
    out["F1_fno_fixed_operator"] = {"train_s": dt, "rel_l2_mean": float(r_fix.mean()),
                                    "rel_l2_median": float(np.median(r_fix)), "rel_l2_p90": float(np.percentile(r_fix, 90))}
    print("  F1 FNO (fixed operator):", out["F1_fno_fixed_operator"], flush=True); _save("F_results.json", out)

    # ---- F4: noise robustness on the fixed operator (now the baseline is good) ----
    rob = {}; base = None
    for nl in [0.0, 0.1, 0.25]:
        rng = np.random.default_rng(1)
        XItr = XI[:ntr] * (1.0 + nl * rng.standard_normal(XI[:ntr].shape).astype(np.float32))
        r_n, _ = fit_eval(FourierNeuralOperator(seq_len=seq_len, n_channels=M, width=64, modes=32),
                          S[:ntr], XItr, S[ntr:], XI[ntr:], ep=max(40, epochs // 2))
        rob[f"{nl:.2f}"] = float(r_n.mean())
        if nl == 0.0:
            base = float(r_n.mean())
    out["F4_noise_fixed_operator"] = {"clean_test_rel_l2_vs_train_noise": rob,
                                      "breaking_point": next((k for k, v in rob.items() if base and v > 2 * base), None)}
    print("  F4 noise (fixed operator):", out["F4_noise_fixed_operator"], flush=True); _save("F_results.json", out)

    # ---- F-diag: FAMILY, unconditioned -- the shipped (ill-posed) setup ----------
    Sf, XIf = generate_operator_dataset(n_samples=n_samples, M=M, seq_len=seq_len, seed=0)
    Sf, XIf, dropped = _sanitize(Sf, XIf)
    ntr = int(0.9 * len(Sf))
    r_fam, _ = fit_eval(FourierNeuralOperator(seq_len=seq_len, n_channels=M, width=64, modes=32),
                        Sf[:ntr], XIf[:ntr], Sf[ntr:], XIf[ntr:])
    r_ss, _ = fit_eval(StateSpaceMBPP(n_channels=M, state_dim=32), Sf[:ntr], XIf[:ntr], Sf[ntr:], XIf[ntr:])
    out["Fdiag_family_unconditioned"] = {"diverged_samples_dropped": dropped,
                                         "fno_rel_l2_median": float(np.median(r_fam)),
                                         "statespace_rel_l2_median": float(np.median(r_ss)),
                                         "note": "G varies per sample but is not an input -> map ill-posed; learns only the mean operator"}
    print("  F-diag (family, unconditioned):", out["Fdiag_family_unconditioned"], flush=True); _save("F_results.json", out)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].hist(r_fix, bins=30, alpha=0.75, label=f"fixed operator (med {np.median(r_fix):.2f})")
        ax[0].hist(r_fam, bins=30, alpha=0.75, label=f"family, unconditioned (med {np.median(r_fam):.2f})")
        ax[0].set_xlabel("relative L2 error vs exact solver"); ax[0].set_ylabel("count")
        ax[0].set_title(f"FNO: conditioned vs ill-posed (M={M})"); ax[0].legend()
        ks = list(rob.keys()); vs = [rob[k] for k in ks]
        ax[1].plot([float(k) for k in ks], vs, "o-"); ax[1].axhline(2 * base, ls="--", color="r", label="2x noiseless")
        ax[1].set_xlabel("training-target noise level"); ax[1].set_ylabel("clean-test rel-L2")
        ax[1].set_title("FNO noise robustness (fixed operator)"); ax[1].legend()
        fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "F_operator_accuracy.png"), dpi=140); plt.close(fig)
        print("  wrote", os.path.join(RESULTS, "F_operator_accuracy.png"), flush=True)
    except Exception as e:
        print("  (figure skipped:", e, ")", flush=True)
    return out


# ===========================================================================
# Block A -- amortised inverse inference (TensorFlow)
# ===========================================================================
def _robust_inference_dataset(n, M, seq_len, T=30.0, max_radius=0.7, theta=1.0, seed=0):
    """Like tf.generate_inference_dataset but skips systems whose explicit ODE
    solve diverges (the shipped version crashes in np.random.poisson on inf means)."""
    from hawkes_calibration.operators.tf import _random_branching_matrix, solve_mbpp_ode_multivariate
    rng = np.random.default_rng(seed)
    Bm = np.full((M, M), float(theta)); t = np.linspace(0, T, seq_len + 1)
    C = np.empty((n, seq_len, M), np.float32); Gs = np.empty((n, M, M), np.float32)
    i = 0
    while i < n:
        G = _random_branching_matrix(M, 0.4, max_radius, rng); A = G * Bm
        mu = rng.uniform(0.2, 1.5, size=M)
        _, Xi = solve_mbpp_ode_multivariate(lambda tt: mu, A, Bm, t, return_compensator=True)
        counts = np.diff(Xi, axis=0)
        if not np.all(np.isfinite(counts)) or counts.max() > 1e6:
            continue                                   # diverged / unphysical -> skip
        C[i] = rng.poisson(np.maximum(counts, 0)).astype(np.float32); Gs[i] = G; i += 1
    return C, Gs


def block_A(n_samples=8000, M=5, seq_len=96, epochs=40):
    import tensorflow as tf
    from hawkes_calibration.operators.tf import AmortizedKernelInference

    out = {"config": {"n_samples": n_samples, "M": M, "seq_len": seq_len, "epochs": epochs}}
    C, G = _robust_inference_dataset(n_samples, M, seq_len, seed=0)
    ntr = int(0.9 * n_samples)
    net = AmortizedKernelInference(n_channels=M, hidden=128)
    net.compile(optimizer="adam", loss="mse")
    t0 = time.time()
    net.fit(C[:ntr], G[:ntr], epochs=epochs, batch_size=128, verbose=0)
    train_s = time.time() - t0

    t0 = time.time(); Ghat = net.predict(C[ntr:], verbose=0); infer_ms = 1000 * (time.time() - t0) / len(C[ntr:])
    Gte = G[ntr:]
    mae = float(np.mean(np.abs(Ghat - Gte)))
    # correlation over all matrix entries
    corr = float(np.corrcoef(Ghat.ravel(), Gte.ravel())[0, 1])
    # spectral-radius recovery
    sr_hat = np.array([np.max(np.abs(np.linalg.eigvals(g))) for g in Ghat])
    sr_true = np.array([np.max(np.abs(np.linalg.eigvals(g))) for g in Gte])
    sr_corr = float(np.corrcoef(sr_hat, sr_true)[0, 1])
    out["A1_amortized"] = {"train_s": train_s, "infer_ms_per_instance": infer_ms,
                           "entry_mae": mae, "entry_corr": corr, "spectral_radius_corr": sr_corr}
    print("  A1 amortized:", out["A1_amortized"], flush=True)
    _save("A_results.json", out)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].scatter(Gte.ravel(), Ghat.ravel(), s=4, alpha=0.2)
        lim = [0, max(Gte.max(), Ghat.max())]; ax[0].plot(lim, lim, "k--", lw=1)
        ax[0].set_xlabel("true branching entry G[m,j]"); ax[0].set_ylabel("predicted")
        ax[0].set_title(f"Amortised branching-matrix recovery (M={M}, corr={corr:.2f})")
        ax[1].scatter(sr_true, sr_hat, s=10, alpha=0.5); ax[1].plot([0, 1], [0, 1], "k--", lw=1)
        ax[1].set_xlabel("true spectral radius"); ax[1].set_ylabel("predicted")
        ax[1].set_title(f"Stability recovery (corr={sr_corr:.2f})")
        fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "A_amortized_recovery.png"), dpi=140); plt.close(fig)
        print("  wrote", os.path.join(RESULTS, "A_amortized_recovery.png"), flush=True)
    except Exception as e:
        print("  (figure skipped:", e, ")", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", choices=["P", "F", "A"], default=None)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if args.all or args.block == "P":
        print("=== Block P (JAX physics-informed) ===", flush=True); block_P()
    if args.all or args.block == "F":
        print("=== Block F (TF forward operators) ===", flush=True); block_F()
    if args.all or args.block == "A":
        print("=== Block A (TF amortised inverse) ===", flush=True); block_A()


if __name__ == "__main__":
    main()
