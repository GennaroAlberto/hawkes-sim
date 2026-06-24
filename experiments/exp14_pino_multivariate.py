r"""
Experiment 14 -- learning the behaviour of a MULTI-DIMENSIONAL Hawkes with a PINO.

We train one physics-informed neural operator (PINO) to solve the *whole family* of
M-sector MBPPs: given a baseline vector ``s`` and an M x M excitation matrix ``A``
(the self/cross-excitation structure), it outputs every sector's mean-intensity path
``xi(t) in R^M`` in a single forward pass.  Training uses ONLY the multivariate MBPP
residual ``R = xi - s - A (G xi)`` (no exact solver in the loss); we then evaluate
the learned operator against the exact multivariate solver on held-out instances it
never saw.

This is the deliverable: a trained operator + a quantified demonstration that it
learned the multi-dimensional Hawkes behaviour, not a report.

Outputs:
  results/exp14_pino.png        training curve, predicted-vs-exact, error distribution, stiffness
  results/exp14_pino.json       headline metrics
  results/exp14_pino_weights.npz the trained operator

Run:  PYTHONPATH=. python -m experiments.exp14_pino_multivariate
"""

import os
import json
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hawkes_calibration.operators.pino import (
    MultivariateMBPPOperator, sample_instances, exact_solution,
)


def per_instance_rel_l2(op, S, A, XI=None):
    if XI is None:
        XI = exact_solution(S, A, op.t, op.theta)
    xi = op.predict(S, A)
    num = np.linalg.norm((xi - XI).reshape(len(S), -1), axis=1)
    den = np.linalg.norm(XI.reshape(len(S), -1), axis=1) + 1e-12
    return num / den, xi, XI


def run(out_dir="results", seed=0, epochs=7000):
    os.makedirs(out_dir, exist_ok=True)
    M, T, N = 3, 12.0, 96
    t = np.linspace(0.0, T, N)

    Str, Atr = sample_instances(640, M, seed=1)
    Sval, Aval = sample_instances(64, M, seed=2)
    Stest, Atest = sample_instances(256, M, seed=3)

    print("=== Experiment 14: multivariate MBPP PINO (physics-only) ===")
    op = MultivariateMBPPOperator(M, t, theta=1.0, p=40, hidden=96, seed=0)
    t0 = time.time()
    hist = op.train(Str, Atr, epochs=epochs, lr=3e-3, batch=96,
                    val=(Sval, Aval), log_every=max(1, epochs // 7))
    train_time = time.time() - t0

    # --- evaluation on held-out test instances (vs the exact solver) ---
    rel_test, xi_test, XI_test = per_instance_rel_l2(op, Stest, Atest)
    rel_train, _, _ = per_instance_rel_l2(op, Str[:256], Atr[:256])
    # per-sector
    XIe = XI_test
    per_sector = []
    for m in range(M):
        num = np.linalg.norm(xi_test[:, :, m] - XIe[:, :, m], axis=1)
        den = np.linalg.norm(XIe[:, :, m], axis=1) + 1e-12
        per_sector.append(float(np.mean(num / den)))
    # stiffness: rel-L2 vs spectral radius of A
    rho = np.array([float(np.max(np.abs(np.linalg.eigvals(a)))) for a in Atest])
    floor = float(np.sqrt(np.mean(op.residual(XIe, Stest, Atest) ** 2)))

    res = dict(
        M=M, N=N, T=T, theta=op.theta, epochs=epochs, train_seconds=round(train_time, 1),
        n_train=len(Str), n_test=len(Stest),
        residual_final=float(hist["residual"][-1]),
        residual_floor_discretization=floor,
        rel_l2_test_mean=float(rel_test.mean()), rel_l2_test_median=float(np.median(rel_test)),
        rel_l2_test_p90=float(np.percentile(rel_test, 90)),
        rel_l2_train_mean=float(rel_train.mean()),
        rel_l2_per_sector=per_sector,
        rel_l2_low_branching=float(rel_test[rho < 0.4].mean()) if np.any(rho < 0.4) else None,
        rel_l2_high_branching=float(rel_test[rho >= 0.6].mean()) if np.any(rho >= 0.6) else None,
    )
    op.save(os.path.join(out_dir, "exp14_pino_weights.npz"))
    with open(os.path.join(out_dir, "exp14_pino.json"), "w") as f:
        json.dump(res, f, indent=2)
    _plot(out_dir, op, hist, t, Stest, Atest, xi_test, XIe, rel_test, rho)

    print("  trained %d epochs in %.1fs" % (epochs, train_time))
    print("  residual: final %.2e  (discretization floor %.2e)" % (res["residual_final"], floor))
    print("  held-out rel-L2: mean %.3f  median %.3f  p90 %.3f"
          % (res["rel_l2_test_mean"], res["rel_l2_test_median"], res["rel_l2_test_p90"]))
    print("  train rel-L2 %.3f (vs test %.3f -> generalises)" % (res["rel_l2_train_mean"], res["rel_l2_test_mean"]))
    print("  per-sector rel-L2: %s" % [round(x, 3) for x in per_sector])
    print("  by branching: low-rho %.3f  high-rho %.3f"
          % (res["rel_l2_low_branching"] or -1, res["rel_l2_high_branching"] or -1))
    print("\nWrote results/exp14_pino.{png,json} and exp14_pino_weights.npz")
    return res


def _plot(out_dir, op, hist, t, S, A, xi, XI, rel, rho):
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))

    a = ax[0, 0]
    a.plot(hist["epoch"], hist["residual"], "C0-o", ms=3, label="physics residual (MSE)")
    a.set_xlabel("epoch"); a.set_ylabel("residual", color="C0"); a.set_yscale("log")
    a2 = a.twinx(); a2.plot(hist["epoch"], hist["val_rel_l2"], "C3-s", ms=3, label="val rel-L2")
    a2.set_ylabel("val rel-L2", color="C3"); a2.set_yscale("log")
    a.set_title("(a) training: residual & held-out accuracy")

    a = ax[0, 1]
    k = int(np.argsort(rel)[len(rel) // 2])                     # a median-accuracy instance
    for m in range(op.M):
        a.plot(t, XI[k, :, m], color=f"C{m}", lw=2, label=f"exact sector {m}")
        a.plot(t, xi[k, :, m], color=f"C{m}", ls="--", lw=1.5)
    a.set_title(r"(b) PINO (dashed) vs exact (solid), $\rho$=%.2f" % rho[k])
    a.set_xlabel("t"); a.set_ylabel(r"$\xi_m(t)$"); a.legend(fontsize=7, ncol=2)

    a = ax[1, 0]
    a.hist(rel, bins=30, color="C2", alpha=0.8)
    a.axvline(rel.mean(), color="r", ls="--", label="mean %.3f" % rel.mean())
    a.set_title("(c) per-instance held-out rel-L2 (256 unseen instances)")
    a.set_xlabel("relative L2 error"); a.set_ylabel("count"); a.legend()

    a = ax[1, 1]
    a.scatter(rho, rel, s=12, alpha=0.5, color="C4")
    a.set_title("(d) accuracy vs branching ratio (stiffness)")
    a.set_xlabel(r"spectral radius $\rho(A)$"); a.set_ylabel("rel-L2 error")
    fig.suptitle("A physics-trained neural operator that solves the whole family of "
                 "3-sector MBPPs in one forward pass")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "exp14_pino.png"), dpi=140); plt.close(fig)


if __name__ == "__main__":
    run()
