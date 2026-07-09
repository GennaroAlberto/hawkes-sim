"""
Experiment 2 -- Multivariate Hawkes process: recovery of the elicitation
matrix A=(alpha_{m,j}).

We pick a 3-component model with a structured A matrix:

       j=0   j=1   j=2
m=0  [ 0.20  0.15  0.00 ]
m=1  [ 0.00  0.25  0.10 ]
m=2  [ 0.10  0.00  0.20 ]

so that 0 elicits {0,1}, 1 elicits {0,1}, 2 elicits {0,2}, and the (0->2),
(1->0), (2->1) entries are zero -- the structure is sparse but not block-
diagonal. We check that MLE recovers both magnitudes and zeros.
"""

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hawkes_calibration import fit_multivariate, simulate_multivariate_hawkes


def run(seed=2, T=5000.0, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    mu_true = np.array([0.3, 0.5, 0.2])
    A_true = np.array(
        [
            [0.20, 0.15, 0.00],
            [0.00, 0.25, 0.10],
            [0.10, 0.00, 0.20],
        ]
    )
    B = np.ones((3, 3))  # common decay rate

    # Spectral radius check
    G = A_true / B
    rho = np.max(np.abs(np.linalg.eigvals(G)))
    print(f"Branching matrix spectral radius = {rho:.3f} (must be < 1 for stationarity)")

    events = simulate_multivariate_hawkes(
        gamma0=np.log(mu_true),
        alpha=A_true,
        beta=B,
        T=T,
        seed=seed,
    )
    counts = [len(e) for e in events]
    print(f"Simulated {sum(counts)} events on [0,{T}], counts per component = {counts}")

    res = fit_multivariate(events, T=T, beta=B, se=True)
    print()
    print(res.summary())

    # Save
    A_hat = res.alpha
    se_A = res.se_alpha if res.se_alpha is not None else np.full_like(A_hat, np.nan)
    np.savez(
        os.path.join(out_dir, "exp2_multidim.npz"),
        mu_true=mu_true,
        A_true=A_true,
        mu_hat=np.exp(res.gamma0),
        A_hat=A_hat,
        se_A=se_A,
        counts=np.array(counts),
        T=T,
    )

    # Heatmap of truth vs estimate vs error
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    vmax = max(A_true.max(), A_hat.max())
    for ax, M, title in zip(
        axes,
        [A_true, A_hat, A_hat - A_true],
        ["true $A$", "estimated $\\hat A$", "error $\\hat A - A$"],
    ):
        im = ax.imshow(
            M,
            vmin=-vmax if "error" in title else 0,
            vmax=vmax,
            cmap="RdBu_r" if "error" in title else "viridis",
        )
        ax.set_title(title)
        ax.set_xlabel("from j")
        ax.set_ylabel("to m")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{M[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color="white" if abs(M[i, j]) > vmax * 0.5 else "black",
                    fontsize=10,
                )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"3D Hawkes -- elicitation matrix recovery (T={int(T)}, total events = {sum(counts)})"
    )
    fig.tight_layout()
    fig_path = os.path.join(out_dir, "exp2_multidim_heatmap.png")
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)
    print(f"\nWrote {fig_path}")

    # Also a parameter recovery scatter
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    ax2.errorbar(A_true.ravel(), A_hat.ravel(), yerr=se_A.ravel(), fmt="o", capsize=3, color="C0")
    lim = max(0.05, vmax * 1.1)
    ax2.plot([0, lim], [0, lim], "k--", alpha=0.5)
    ax2.set_xlabel("true elicitation $\\alpha_{m,j}$")
    ax2.set_ylabel("estimated $\\hat\\alpha_{m,j}$")
    ax2.set_title("True vs estimated elicitation entries")
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2_path = os.path.join(out_dir, "exp2_multidim_scatter.png")
    fig2.savefig(fig2_path, dpi=140)
    plt.close(fig2)
    print(f"Wrote {fig2_path}")

    return res


if __name__ == "__main__":
    run()
