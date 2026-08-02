r"""
Experiment 22 -- covariate measurement noise on the benign case (EXPERIMENTS.md v2).

The benign configuration under the v2 standing assumptions: regime A world,
truthful (oracle) risk sets, the funded-pool conditional-logit choice model of
v2 SS0.3, and modest excitation. Question: **how much does covariate measurement
noise move the results**, at relative noise levels up to 15-20%?

Noise model (both stages): additive Gaussian with sd = sigma * (per-column train
sd), sigma in {0, 5, 10, 15, 20}% -- i.e. "15% noise" corrupts each covariate by
15% of its own signal scale. Noise is present at BOTH fit and prediction time
(measurement noise does not disappear out of sample). Three independent noise
replicates per level; the spread comes from the measurement noise itself.

* **Stage 1 (when/where)** -- weekly sector GLM on the macro covariates
  (`fit_sector_glm_fast`). Metrics vs truth: beta correlation; held-out one-step
  Poisson NLL; the **covariate lift** (held-out NLL of a no-covariate fit minus
  the covariate fit -- v2 E1.4: "covariates are informative" as a measured
  number) and how that lift erodes with noise.

* **Stage 2 (which firm)** -- day-resolution funded-pool conditional logit on
  `build_choice_sets` output (`fit_choice_fast`). Firm features (raised,
  employees, age, stage, funding gap) are noised; the synthetic newcomer
  alternative and the `is_newcomer` indicator are structural, not measured, and
  stay clean. Metrics: held-out NLL (vs random over the same pools), top-5
  accuracy, and attenuation of the fitted weights vs the clean fit.

Run:  PYTHONPATH=. python -m experiments.exp22_covariate_noise
Writes results/exp22_covariate_noise.{json,png}.
"""

import json
import os
import time

import numpy as np

from synthetic.fast_fit import fit_choice_fast, fit_sector_glm_fast
from synthetic.loaders import CHOICE_FEATURES, build_choice_sets, load_dataset

DATA = "data/synthetic_A"
GT = os.path.join(DATA, "ground_truth.npz")
TRAIN_END = 416
NOISE_LEVELS = (0.0, 0.05, 0.10, 0.15, 0.20)
N_REPS = 3


# ---------------------------------------------------------------------------
# Stage 1: weekly sector GLM under macro-covariate noise
# ---------------------------------------------------------------------------
def _rates(a, b, exc, cov, counts, n_lags):
    T = counts.shape[0]
    out = np.zeros_like(counts, dtype=float)
    for t in range(n_lags, T):
        eta = a + b @ cov[t]
        for lag in range(1, n_lags + 1):
            eta += exc[:, :, lag - 1] @ counts[t - lag]
        out[t] = np.exp(np.clip(eta, -30, 20))
    return out


def _poisson_nll(counts, rates):
    from math import lgamma

    rates = np.maximum(rates, 1e-12)
    lg = np.vectorize(lambda x: lgamma(x + 1.0))(counts)
    return float(np.sum(rates - counts * np.log(rates) + lg)) / counts.size


def stage1(ds, gt):
    L = int(gt["n_lags"])
    X = (ds.covariates_raw - gt["cov_mean"]) / gt["cov_sd"]  # standardized: sd = 1
    y = ds.sector_counts
    obs = y[TRAIN_END:]

    # reference: no-covariate fit (covariate lift baseline, fitted once)
    a0, _, e0 = fit_sector_glm_fast(y, X[:, :0], n_lags=L, train_end=TRAIN_END, l2=1e-3)
    nll_nocov = _poisson_nll(
        obs, _rates(a0, np.zeros((y.shape[1], 0)), e0, X[:, :0], y, L)[TRAIN_END:]
    )

    rows = []
    for sigma in NOISE_LEVELS:
        for rep in range(N_REPS if sigma > 0 else 1):
            rng = np.random.default_rng(1000 + int(1e4 * sigma) + rep)
            Xn = X + rng.normal(0.0, sigma, size=X.shape)  # sd(X)=1 -> relative noise
            a, b, exc = fit_sector_glm_fast(y, Xn, n_lags=L, train_end=TRAIN_END, l2=1e-3)
            nll = _poisson_nll(obs, _rates(a, b, exc, Xn, y, L)[TRAIN_END:])
            rows.append(
                dict(
                    sigma=sigma,
                    rep=rep,
                    beta_corr=float(np.corrcoef(b.ravel(), gt["sector_beta"].ravel())[0, 1]),
                    beta_scale=float(
                        (b.ravel() @ gt["sector_beta"].ravel())
                        / max(gt["sector_beta"].ravel() @ gt["sector_beta"].ravel(), 1e-12)
                    ),
                    heldout_nll=nll,
                    covariate_lift=nll_nocov - nll,
                )
            )
    return rows, nll_nocov


# ---------------------------------------------------------------------------
# Stage 2: funded-pool conditional logit under firm-feature noise
# ---------------------------------------------------------------------------
def stage2(cs):
    F0 = cs["F"]
    msk = cs["mask"]
    day = cs["day"]
    newcomer_col = CHOICE_FEATURES.index("is_newcomer")
    noised_cols = [j for j in range(F0.shape[2]) if j != newcomer_col]
    # real candidates only: padded slots and the synthetic newcomer stay clean
    real = msk & ~(F0[:, :, newcomer_col] > 0)
    cut = np.quantile(day, 0.8)
    tr_ev = day <= cut
    col_sd = np.array([F0[tr_ev][msk[tr_ev]][:, j].std() for j in range(F0.shape[2])])

    rows = []
    w_clean = None
    for sigma in NOISE_LEVELS:
        for rep in range(N_REPS if sigma > 0 else 1):
            rng = np.random.default_rng(2000 + int(1e4 * sigma) + rep)
            F = np.array(F0, copy=True)
            for j in noised_cols:
                noise = rng.normal(0.0, sigma * col_sd[j], size=F.shape[:2]).astype(F.dtype)
                F[:, :, j] += noise * real
            cs_n = dict(cs)
            cs_n["F"] = F
            w0, u, tr_nll, te_nll, rand_nll, top5, ok = fit_choice_fast(cs_n)
            if sigma == 0.0:
                w_clean = w0.copy()
            rows.append(
                dict(
                    sigma=sigma,
                    rep=rep,
                    test_nll=te_nll,
                    random_nll=rand_nll,
                    lift_nats=rand_nll - te_nll,
                    top5=top5,
                    w0=[round(float(x), 3) for x in w0],
                    weight_attenuation=float(
                        (w0[noised_cols] @ w_clean[noised_cols])
                        / max(w_clean[noised_cols] @ w_clean[noised_cols], 1e-12)
                    ),
                )
            )
    return rows


def _agg(rows, key):
    out = {}
    for s in NOISE_LEVELS:
        v = [r[key] for r in rows if r["sigma"] == s]
        out[s] = (float(np.mean(v)), float(np.std(v)))
    return out


def main(out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    gt = np.load(GT)
    ds = load_dataset(DATA, active="true", ground_truth=GT)
    s1_rows, nll_nocov = stage1(ds, gt)
    print(f"[stage 1 done {time.time() - t0:.0f}s]", flush=True)
    cs = build_choice_sets(DATA, max_candidates=64, seed=0)
    s2_rows = stage2(cs)
    print(f"[stage 2 done {time.time() - t0:.0f}s]", flush=True)

    res = dict(
        config=dict(data=DATA, train_end=TRAIN_END, noise_levels=NOISE_LEVELS, n_reps=N_REPS),
        stage1_no_covariate_nll=nll_nocov,
        stage1=s1_rows,
        stage2=s2_rows,
    )
    with open(os.path.join(out_dir, "exp22_covariate_noise.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=float)

    # ---- report ----
    print(
        "\n=== exp22: covariate measurement noise on the benign case (regime A, oracle pools) ==="
    )
    print("\nSTAGE 1 -- weekly sector GLM (macro covariates)")
    print(f"  no-covariate held-out NLL/cell: {nll_nocov:.3f}")
    print("  sigma   beta corr        beta scale       held-out NLL     covariate lift")
    for s in NOISE_LEVELS:
        bc, bs = _agg(s1_rows, "beta_corr")[s], _agg(s1_rows, "beta_scale")[s]
        nl, lf = _agg(s1_rows, "heldout_nll")[s], _agg(s1_rows, "covariate_lift")[s]
        print(
            f"  {s:4.0%}   {bc[0]:.3f} ± {bc[1]:.3f}   {bs[0]:.3f} ± {bs[1]:.3f}"
            f"   {nl[0]:.3f} ± {nl[1]:.3f}   {lf[0]:.3f} ± {lf[1]:.3f}"
        )
    print("\nSTAGE 2 -- funded-pool conditional logit (firm features)")
    print("  sigma   test NLL         lift over random   top-5 acc        weight attenuation")
    for s in NOISE_LEVELS:
        nl, lf = _agg(s2_rows, "test_nll")[s], _agg(s2_rows, "lift_nats")[s]
        t5, at = _agg(s2_rows, "top5")[s], _agg(s2_rows, "weight_attenuation")[s]
        print(
            f"  {s:4.0%}   {nl[0]:.3f} ± {nl[1]:.3f}   {lf[0]:.3f} ± {lf[1]:.3f}"
            f"     {t5[0]:.3f} ± {t5[1]:.3f}   {at[0]:.3f} ± {at[1]:.3f}"
        )
    _plot(out_dir, s1_rows, s2_rows, nll_nocov)
    print(f"\nWrote results/exp22_covariate_noise.{{json,png}}  ({time.time() - t0:.0f}s)")
    return res


def _plot(out_dir, s1_rows, s2_rows, nll_nocov):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [100 * s for s in NOISE_LEVELS]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    ax = axes[0]
    for key, label, style in (
        ("beta_corr", "beta correlation", "o-"),
        ("beta_scale", "beta scale (attenuation)", "s--"),
    ):
        m = [_agg(s1_rows, key)[s][0] for s in NOISE_LEVELS]
        e = [_agg(s1_rows, key)[s][1] for s in NOISE_LEVELS]
        ax.errorbar(x, m, yerr=e, fmt=style, capsize=3, label=label)
    ax.set_xlabel("covariate noise (% of signal sd)")
    ax.set_title("Stage 1: macro-beta recovery")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    m = [_agg(s1_rows, "heldout_nll")[s][0] for s in NOISE_LEVELS]
    e = [_agg(s1_rows, "heldout_nll")[s][1] for s in NOISE_LEVELS]
    ax.errorbar(x, m, yerr=e, fmt="o-", capsize=3, label="with covariates")
    ax.axhline(nll_nocov, color="gray", ls=":", label="no-covariate fit")
    ax.set_xlabel("covariate noise (% of signal sd)")
    ax.set_ylabel("held-out Poisson NLL / cell")
    ax.set_title("Stage 1: held-out NLL (lift = gap to dotted)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[2]
    m = [_agg(s2_rows, "top5")[s][0] for s in NOISE_LEVELS]
    e = [_agg(s2_rows, "top5")[s][1] for s in NOISE_LEVELS]
    ax.errorbar(x, m, yerr=e, fmt="o-", capsize=3, label="top-5 accuracy")
    m2 = [_agg(s2_rows, "lift_nats")[s][0] for s in NOISE_LEVELS]
    ax2 = ax.twinx()
    ax2.errorbar(
        x,
        m2,
        yerr=[_agg(s2_rows, "lift_nats")[s][1] for s in NOISE_LEVELS],
        fmt="s--",
        color="C1",
        capsize=3,
        label="NLL lift over random (nats)",
    )
    ax.set_xlabel("feature noise (% of signal sd)")
    ax.set_title("Stage 2: choice model")
    lines, labels = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(lines + l2, labels + lb2, loc="lower left")
    ax.grid(alpha=0.3)

    fig.suptitle("Covariate measurement noise, benign case (regime A, truthful pools)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "exp22_covariate_noise.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
