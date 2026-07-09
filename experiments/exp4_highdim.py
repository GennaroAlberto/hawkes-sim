"""
Experiment 4 -- High-dimensional Hawkes (M = 15) with a sparse, structured
elicitation network.

We build a 15-component process whose elicitation matrix A is sparse with
about 4 non-zero entries per row, organized as a "small-world" graph:
each component excites itself, its two neighbors on a ring, and one random
long-range partner. The spectral radius of A/beta is kept well below 1.

We then fit the unregularized MLE and ask two questions:

  (i)  Can the analytical-gradient MLE estimate ~225 parameters at all?
  (ii) How well are the true zeros and non-zeros distinguished, both by the
       point estimates and by a hard threshold at, say, 2 standard errors?
"""

import json
import os
import time

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hawkes_calibration import (
    fit_multivariate,
    fit_multivariate_lasso,
    simulate_multivariate_hawkes,
)


def build_A(M, val_self=0.20, val_neighbor=0.08, val_long=0.10, seed=0):
    rng = np.random.default_rng(seed)
    A = np.zeros((M, M))
    for m in range(M):
        A[m, m] = val_self
        A[m, (m - 1) % M] = val_neighbor
        A[m, (m + 1) % M] = val_neighbor
        # random long-range partner that isn't already set
        choices = [j for j in range(M) if A[m, j] == 0]
        if choices:
            j = rng.choice(choices)
            A[m, j] = val_long
    return A


def run(seed=4, M=12, T=2500.0, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    mu_true = np.linspace(0.1, 0.4, M)
    A_true = build_A(M, seed=seed)
    B = np.ones((M, M))

    G = A_true / B
    rho = float(np.max(np.abs(np.linalg.eigvals(G))))
    print(f"M = {M}, spectral radius of branching matrix = {rho:.3f} (stationarity ok: {rho < 1})")
    nnz = int(np.count_nonzero(A_true))
    print(f"True A has {nnz} non-zero entries out of {M * M} ({nnz / (M * M):.1%} density)")

    # Simulate
    t0 = time.time()
    events = simulate_multivariate_hawkes(
        gamma0=np.log(mu_true), alpha=A_true, beta=B, T=T, seed=seed
    )
    t1 = time.time()
    counts = [len(e) for e in events]
    total = sum(counts)
    print(
        f"Simulated {total} events on [0, {T}] in {t1 - t0:.1f}s. min count = {min(counts)}, max = {max(counts)}"
    )

    # Fit (unregularized MLE)
    t0 = time.time()
    res = fit_multivariate(events, T=T, beta=B, se=True)
    t1 = time.time()
    print(
        f"MLE fit ({M + M * M} params) took {t1 - t0:.1f}s, loglik = {res.loglik:.1f}, ok = {res.success}"
    )

    # Fit a lasso-regularized version using BIC to pick lambda from a grid
    lambdas = [10.0, 30.0, 50.0, 80.0, 120.0, 180.0, 250.0]
    lasso_results = []
    for lam in lambdas:
        t0 = time.time()
        r = fit_multivariate_lasso(events, T=T, beta=B, lam=lam, max_iter=300)
        t1 = time.time()
        # BIC = -2*loglik + k * log(N_total)
        k = M + r.nnz  # baseline + non-zero alpha entries (no covariate here)
        bic = -2 * r.loglik + k * np.log(total)
        lasso_results.append((lam, r, bic, t1 - t0))
        print(
            f"  lasso lam = {lam:6.1f}: nnz = {r.nnz:3d}, ll = {r.loglik:.1f}, BIC = {bic:.1f}, time = {t1 - t0:.1f}s"
        )

    best = min(lasso_results, key=lambda x: x[2])
    lam_best, lasso_best, bic_best, _ = best
    print(f"\nLasso BIC selects lambda = {lam_best} (nnz = {lasso_best.nnz})")

    # Recovery metrics for MLE
    A_hat = res.alpha
    mu_hat = np.exp(res.gamma0)
    abs_err_mu = np.abs(mu_hat - mu_true)
    abs_err_A = np.abs(A_hat - A_true)
    nonzero_mask = A_true > 0
    print()
    print(
        f"  [MLE] Baseline rate mean abs error: {abs_err_mu.mean():.4f}  max: {abs_err_mu.max():.4f}"
    )
    print(
        f"  [MLE] Elicitation mean abs error  : {abs_err_A.mean():.4f}  max: {abs_err_A.max():.4f}"
    )
    print(f"           ... on non-zero entries  : {abs_err_A[nonzero_mask].mean():.4f}")
    print(f"           ... on zero entries      : {abs_err_A[~nonzero_mask].mean():.4f}")

    # Edge selection via 2-SE threshold for MLE
    se_A = res.se_alpha if res.se_alpha is not None else np.full_like(A_hat, np.nan)
    selected_mle = A_hat > 2 * se_A
    tp = int((selected_mle & nonzero_mask).sum())
    fp = int((selected_mle & ~nonzero_mask).sum())
    fn = int((~selected_mle & nonzero_mask).sum())
    tn = int((~selected_mle & ~nonzero_mask).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    print(f"  [MLE 2SE]    TP={tp} FP={fp} FN={fn} TN={tn}  prec={precision:.3f}  rec={recall:.3f}")

    # Lasso metrics
    A_lasso = lasso_best.alpha
    selected_lasso = A_lasso > 0
    tp_l = int((selected_lasso & nonzero_mask).sum())
    fp_l = int((selected_lasso & ~nonzero_mask).sum())
    fn_l = int((~selected_lasso & nonzero_mask).sum())
    tn_l = int((~selected_lasso & ~nonzero_mask).sum())
    prec_l = tp_l / max(tp_l + fp_l, 1)
    rec_l = tp_l / max(tp_l + fn_l, 1)
    abs_err_A_lasso = np.abs(A_lasso - A_true)
    print(
        f"  [LASSO] nz_mae = {abs_err_A_lasso[nonzero_mask].mean():.4f}, z_mae = {abs_err_A_lasso[~nonzero_mask].mean():.4f}"
    )
    print(
        f"  [LASSO BIC]  TP={tp_l} FP={fp_l} FN={fn_l} TN={tn_l}  prec={prec_l:.3f}  rec={rec_l:.3f}"
    )

    # ---- Plots: comparison of true, MLE, lasso
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    vmax = max(A_true.max(), A_hat.max(), A_lasso.max())
    panels = [
        (A_true, f"true $A$ ({nnz} non-zeros)"),
        (A_hat, "unpenalized MLE $\\hat A$"),
        (A_lasso, f"lasso (BIC-$\\lambda$={lam_best}, nnz={lasso_best.nnz})"),
    ]
    for ax, (M_, title) in zip(axes, panels):
        im = ax.imshow(M_, vmin=0, vmax=vmax, cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("from j")
        ax.set_ylabel("to m")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"M = {M} Hawkes -- elicitation matrix recovery (T = {int(T)}, {total} events)")
    fig.tight_layout()
    fig_path = os.path.join(out_dir, "exp4_highdim_heatmap.png")
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)
    print(f"\nWrote {fig_path}")

    # Scatter of estimated vs true (color-coded zero vs non-zero) — MLE and Lasso side by side
    fig2, (axA, axB) = plt.subplots(1, 2, figsize=(11, 5))
    for ax, A_est, se, title, sub in [
        (axA, A_hat, se_A, "unpenalized MLE", f"prec={precision:.2f}, rec={recall:.2f} @ 2SE"),
        (
            axB,
            A_lasso,
            None,
            f"lasso ($\\lambda$={lam_best})",
            f"prec={prec_l:.2f}, rec={rec_l:.2f}",
        ),
    ]:
        jit = np.random.uniform(-0.002, 0.002, (~nonzero_mask).sum())
        if se is not None:
            ax.errorbar(
                A_true[nonzero_mask].ravel(),
                A_est[nonzero_mask].ravel(),
                yerr=se[nonzero_mask].ravel(),
                fmt="o",
                capsize=3,
                color="C0",
                label="true non-zero",
            )
            ax.errorbar(
                A_true[~nonzero_mask].ravel() + jit,
                A_est[~nonzero_mask].ravel(),
                yerr=se[~nonzero_mask].ravel(),
                fmt="o",
                capsize=2,
                alpha=0.5,
                color="C3",
                label="true zero",
            )
        else:
            ax.plot(
                A_true[nonzero_mask].ravel(),
                A_est[nonzero_mask].ravel(),
                "o",
                color="C0",
                label="true non-zero",
            )
            ax.plot(
                A_true[~nonzero_mask].ravel() + jit,
                A_est[~nonzero_mask].ravel(),
                "o",
                color="C3",
                alpha=0.5,
                label="true zero",
            )
        lim = vmax * 1.15
        ax.plot([0, lim], [0, lim], "k--", alpha=0.5)
        ax.set_xlabel("true $\\alpha_{m,j}$")
        ax.set_ylabel("estimated $\\hat\\alpha_{m,j}$")
        ax.set_title(f"{title}\n{sub}")
        ax.grid(alpha=0.3)
        ax.legend()
    fig2.suptitle(f"M={M} elicitation recovery: unpenalized MLE vs lasso")
    fig2.tight_layout()
    fig2_path = os.path.join(out_dir, "exp4_highdim_scatter.png")
    fig2.savefig(fig2_path, dpi=140)
    plt.close(fig2)
    print(f"Wrote {fig2_path}")

    # Lambda path plot
    fig3, ax3 = plt.subplots(figsize=(7, 4.5))
    lams = [x[0] for x in lasso_results]
    bics = [x[2] for x in lasso_results]
    nnzs = [x[1].nnz for x in lasso_results]
    ax3.semilogx(lams, bics, "o-", color="C2", label="BIC")
    ax3.set_xlabel("$\\lambda$")
    ax3.set_ylabel("BIC", color="C2")
    ax3.axvline(lam_best, color="k", linestyle="--", alpha=0.5, label=f"BIC choice {lam_best}")
    ax3b = ax3.twinx()
    ax3b.semilogx(lams, nnzs, "s--", color="C1", alpha=0.7, label="nnz")
    ax3b.axhline(nnz, color="C0", linestyle=":", alpha=0.7, label=f"true nnz={nnz}")
    ax3b.set_ylabel("# non-zero in $\\hat A$", color="C1")
    ax3.set_title(f"Lasso path (M={M}, N={total})")
    ax3.grid(alpha=0.3, which="both")
    fig3.tight_layout()
    fig3_path = os.path.join(out_dir, "exp4_highdim_lasso_path.png")
    fig3.savefig(fig3_path, dpi=140)
    plt.close(fig3)
    print(f"Wrote {fig3_path}")

    np.savez(
        os.path.join(out_dir, "exp4_highdim.npz"),
        mu_true=mu_true,
        A_true=A_true,
        mu_hat=mu_hat,
        A_hat=A_hat,
        A_lasso=A_lasso,
        se_A=se_A,
        T=T,
        M=M,
        counts=np.array(counts),
        lam_best=lam_best,
    )
    with open(os.path.join(out_dir, "exp4_highdim.json"), "w") as f:
        json.dump(
            {
                "M": M,
                "T": T,
                "rho": rho,
                "n_events": int(total),
                "mu_mae": float(abs_err_mu.mean()),
                "mle": dict(
                    A_mae_overall=float(abs_err_A.mean()),
                    A_mae_nonzero=float(abs_err_A[nonzero_mask].mean()),
                    A_mae_zero=float(abs_err_A[~nonzero_mask].mean()),
                    precision_2SE=precision,
                    recall_2SE=recall,
                ),
                "lasso": dict(
                    lambda_best=lam_best,
                    nnz=int(lasso_best.nnz),
                    A_mae_nonzero=float(abs_err_A_lasso[nonzero_mask].mean()),
                    A_mae_zero=float(abs_err_A_lasso[~nonzero_mask].mean()),
                    precision=prec_l,
                    recall=rec_l,
                ),
            },
            f,
            indent=2,
        )
    return res


if __name__ == "__main__":
    run()
