r"""
Experiment 15 -- scaling the MBPP PINO to higher dimensions.

Trains the physics-informed neural operator (exp14) at several sector counts M and
reports held-out accuracy vs M, to see how learning the *whole family* of M-sector
MBPPs degrades as the cross-excitation matrix grows.  One M per invocation (so each
run fits in a single call); results accumulate in results/exp15_highdim.json.

Run (one point at a time, then plot):
  PYTHONPATH=. python -m experiments.exp15_pino_highdim --M 3 --epochs 2000
  PYTHONPATH=. python -m experiments.exp15_pino_highdim --M 5 --epochs 1800
  PYTHONPATH=. python -m experiments.exp15_pino_highdim --M 8 --epochs 1500
  PYTHONPATH=. python -m experiments.exp15_pino_highdim --plot
"""

import os
import sys
import json
import time
import argparse

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hawkes_calibration.operators.pino import (
    MultivariateMBPPOperator, sample_instances, exact_solution,
)

JSON = "results/exp15_highdim.json"


def train_point(M, epochs, N=80, p=40, hidden=96, n_train=512, seed=0):
    os.makedirs("results", exist_ok=True)
    t = np.linspace(0.0, 12.0, N)
    Str, Atr = sample_instances(n_train, M, seed=1)
    Sval, Aval = sample_instances(48, M, seed=2)
    Stest, Atest = sample_instances(200, M, seed=3)
    op = MultivariateMBPPOperator(M, t, theta=1.0, p=p, hidden=hidden, seed=0)
    t0 = time.time()
    hist = op.train(Str, Atr, epochs=epochs, lr=3e-3, batch=96, val=(Sval, Aval),
                    log_every=max(1, epochs // 4))
    secs = time.time() - t0
    XI = exact_solution(Stest, Atest, t, 1.0)
    xi = op.predict(Stest, Atest)
    num = np.linalg.norm((xi - XI).reshape(len(Stest), -1), axis=1)
    den = np.linalg.norm(XI.reshape(len(Stest), -1), axis=1) + 1e-12
    rel = num / den
    row = dict(M=M, n_params=int(M + M * M), epochs=epochs, seconds=round(secs, 1),
               rel_l2_mean=float(rel.mean()), rel_l2_median=float(np.median(rel)),
               rel_l2_p90=float(np.percentile(rel, 90)),
               residual_final=float(hist["residual"][-1]), p=p, hidden=hidden, N=N)
    data = []
    if os.path.exists(JSON):
        data = json.load(open(JSON))
    data = [r for r in data if r["M"] != M] + [row]      # replace same-M
    data.sort(key=lambda r: r["M"])
    json.dump(data, open(JSON, "w"), indent=2)
    print("M=%d  input_dim=%d  rel-L2 mean=%.3f median=%.3f p90=%.3f  (%.0fs, %d ep)"
          % (M, row["n_params"], row["rel_l2_mean"], row["rel_l2_median"],
             row["rel_l2_p90"], secs, epochs))
    return row


def plot():
    data = json.load(open(JSON))
    M = [r["M"] for r in data]
    mean = [r["rel_l2_mean"] for r in data]
    p90 = [r["rel_l2_p90"] for r in data]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.plot(M, mean, "C0-o", label="mean rel-L2")
    ax.plot(M, p90, "C3--s", label="p90 rel-L2")
    for r in data:
        ax.annotate("%d-d (%d params)" % (r["M"], r["n_params"]),
                    (r["M"], r["rel_l2_mean"]), textcoords="offset points",
                    xytext=(5, 6), fontsize=8)
    ax.set_xlabel("number of sectors  M  (excitation matrix is M x M)")
    ax.set_ylabel("held-out relative L2 error")
    ax.set_title("PINO accuracy vs dimensionality of the Hawkes process")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig("results/exp15_highdim.png", dpi=140); plt.close(fig)
    print("wrote results/exp15_highdim.png from", len(data), "points:",
          [(r["M"], round(r["rel_l2_mean"], 3)) for r in data])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=1800)
    ap.add_argument("--p", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    if a.plot:
        plot()
    elif a.M is not None:
        train_point(a.M, a.epochs, p=a.p, hidden=a.hidden)
    else:
        print("specify --M <int> [--epochs] or --plot")
