r"""
Experiment 17 -- the multivariate MBPP PINO, trained *hard* (JAX backend).

Same operator as exp14/exp15 (a DeepONet mapping an M-sector instance ``(s, A)`` to
the whole mean-intensity path ``xi(t) in R^M``), but trained with the fast JAX PINO
(:class:`hawkes_calibration.operators.pino_jax.JAXMultivariateMBPPOperator`):

* a hybrid, *scale-invariant* objective -- relative collocation residual (physics)
  + a relative supervised term on a set of exact "anchor" solutions + an IC penalty;
* large instance/anchor sets, a wider net, Fourier time features, cosine LR decay,
  and long training.

The goal is a genuinely accurate operator: ~5 % held-out rel-L2 at M=5 and M=8 and
lower at M=3, evaluated against the exact multivariate solver on instances never seen.

Run one dimension per call (accumulates into results/exp17_pino_jax.json):
  PYTHONPATH=. python -m experiments.exp17_pino_jax --M 3
  PYTHONPATH=. python -m experiments.exp17_pino_jax --M 5
  PYTHONPATH=. python -m experiments.exp17_pino_jax --M 8
  PYTHONPATH=. python -m experiments.exp17_pino_jax --plot
"""

import argparse
import json
import os
import time

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hawkes_calibration.operators.pino import exact_solution, sample_instances
from hawkes_calibration.operators.pino_jax import JAXMultivariateMBPPOperator

JSON = "results/exp17_pino_jax.json"

# Per-dimension configuration (capacity + data + epochs grow with M).
CFG = {
    3: dict(N=120, p=64, hidden=160, depth=3, n_train=1500, n_anchor=1500, epochs=12000),
    5: dict(N=120, p=96, hidden=224, depth=3, n_train=3000, n_anchor=2000, epochs=12000),
    8: dict(N=120, p=96, hidden=256, depth=3, n_train=4000, n_anchor=2500, epochs=14000),
}


def run(M, out_dir="results", seed=0):
    os.makedirs(out_dir, exist_ok=True)
    c = CFG[M]
    T = 12.0
    t = np.linspace(0.0, T, c["N"])

    Str, Atr = sample_instances(c["n_train"], M, seed=1)
    Sval, Aval = sample_instances(200, M, seed=2)
    Stest, Atest = sample_instances(400, M, seed=3)
    Sa, Aa = sample_instances(c["n_anchor"], M, seed=4)
    XIa = exact_solution(Sa, Aa, t, 1.0)

    print(
        f"=== exp17 PINO (JAX), M={M}  net[h={c['hidden']},p={c['p']}]  "
        f"data[train={c['n_train']},anchor={c['n_anchor']}]  epochs={c['epochs']} ===",
        flush=True,
    )
    op = JAXMultivariateMBPPOperator(
        M, t, theta=1.0, p=c["p"], hidden=c["hidden"], depth=c["depth"], n_fourier=16, seed=0
    )
    t0 = time.time()
    hist = op.train(
        Str,
        Atr,
        epochs=c["epochs"],
        lr=2e-3,
        batch=256,
        anchors=(Sa, Aa, XIa),
        w_res=1.0,
        w_anchor=1.0,
        w_ic=1.0,
        val=(Sval, Aval),
        log_every=max(1, c["epochs"] // 6),
    )
    secs = time.time() - t0

    XI = exact_solution(Stest, Atest, t, 1.0)
    xi = op.predict(Stest, Atest)
    num = np.linalg.norm((xi - XI).reshape(len(Stest), -1), axis=1)
    den = np.linalg.norm(XI.reshape(len(Stest), -1), axis=1) + 1e-12
    rel = num / den
    per_sector = [
        float(
            np.mean(
                np.linalg.norm(xi[:, :, m] - XI[:, :, m], axis=1)
                / (np.linalg.norm(XI[:, :, m], axis=1) + 1e-12)
            )
        )
        for m in range(M)
    ]
    rho = np.array([float(np.max(np.abs(np.linalg.eigvals(a)))) for a in Atest])

    row = dict(
        M=M,
        input_dim=int(M + M * M),
        N=c["N"],
        epochs=c["epochs"],
        seconds=round(secs, 1),
        p=c["p"],
        hidden=c["hidden"],
        n_train=c["n_train"],
        n_anchor=c["n_anchor"],
        rel_l2_mean=float(rel.mean()),
        rel_l2_median=float(np.median(rel)),
        rel_l2_p90=float(np.percentile(rel, 90)),
        rel_l2_p99=float(np.percentile(rel, 99)),
        per_sector=per_sector,
        rel_low_rho=float(rel[rho < 0.4].mean()) if np.any(rho < 0.4) else None,
        rel_high_rho=float(rel[rho >= 0.6].mean()) if np.any(rho >= 0.6) else None,
        val_curve=list(zip(hist["epoch"], hist["val_rel_l2"])),
    )
    op.save(os.path.join(out_dir, f"exp17_pino_jax_M{M}.npz"))

    data = json.load(open(JSON)) if os.path.exists(JSON) else []
    data = [r for r in data if r["M"] != M] + [row]
    data.sort(key=lambda r: r["M"])
    json.dump(data, open(JSON, "w"), indent=2)

    _plot_instance(out_dir, M, t, xi, XI, rel, rho)
    print(
        f"  M={M}: held-out rel-L2 mean={rel.mean():.4f} median={np.median(rel):.4f} "
        f"p90={np.percentile(rel, 90):.4f}  ({secs:.0f}s)",
        flush=True,
    )
    print(f"  per-sector={[round(x, 4) for x in per_sector]}", flush=True)
    print(
        f"  by branching: low-rho={row['rel_low_rho']}  high-rho={row['rel_high_rho']}", flush=True
    )
    return row


def _plot_instance(out_dir, M, t, xi, XI, rel, rho):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    k = int(np.argsort(rel)[len(rel) // 2])
    for m in range(M):
        ax[0].plot(t, XI[k, :, m], color=f"C{m}", lw=2)
        ax[0].plot(t, xi[k, :, m], color=f"C{m}", ls="--", lw=1.4)
    ax[0].set_title(f"M={M}: PINO (dashed) vs exact (solid), median instance")
    ax[0].set_xlabel("t")
    ax[0].set_ylabel(r"$\xi_m(t)$")
    ax[1].hist(rel, bins=30, color="C2", alpha=0.85)
    ax[1].axvline(rel.mean(), color="r", ls="--", label=f"mean {rel.mean():.3f}")
    ax[1].axvline(0.05, color="k", ls=":", label="5% target")
    ax[1].set_title("held-out rel-L2 (400 unseen)")
    ax[1].set_xlabel("rel-L2")
    ax[1].legend()
    ax[2].scatter(rho, rel, s=12, alpha=0.5, color="C4")
    ax[2].set_title("accuracy vs branching ratio")
    ax[2].set_xlabel(r"$\rho(A)$")
    ax[2].set_ylabel("rel-L2")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"exp17_pino_jax_M{M}.png"), dpi=140)
    plt.close(fig)


def plot_summary():
    data = json.load(open(JSON))
    Ms = [r["M"] for r in data]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.plot(Ms, [r["rel_l2_mean"] for r in data], "C0-o", label="mean")
    ax.plot(Ms, [r["rel_l2_median"] for r in data], "C2-^", label="median")
    ax.plot(Ms, [r["rel_l2_p90"] for r in data], "C3--s", label="p90")
    ax.axhline(0.05, color="k", ls=":", label="5% target")
    for r in data:
        ax.annotate(
            f"{r['M']}-d", (r["M"], r["rel_l2_mean"]), textcoords="offset points", xytext=(5, 6)
        )
    ax.set_xlabel("sectors M (excitation matrix M×M)")
    ax.set_ylabel("held-out rel-L2")
    ax.set_title("Multivariate MBPP PINO accuracy vs dimension (JAX, hybrid)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/exp17_pino_jax_summary.png", dpi=140)
    plt.close(fig)
    print("summary:", [(r["M"], round(r["rel_l2_mean"], 4)) for r in data])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=None)
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    if a.plot:
        plot_summary()
    elif a.M is not None:
        run(a.M)
    else:
        print("specify --M {3,5,8} or --plot")
