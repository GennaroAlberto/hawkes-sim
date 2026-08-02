r"""
Experiment 24 -- E1b: what day resolution buys for the decay (theta).

Regime B's sector contagion has a **7-day half-life**: true decay
``theta = ln 2 / 7 ~ 0.099 / day``. The v2 briefing asks for the bias/SE of
(kappa, theta) at timestamp resolutions {1d, 7d, 28d} -- where does bucketing
start destroying timing information?

Two parts, one estimator (profile MLE over a decay grid; within-bucket uniform
imputation gives the honest interval-censored treatment of coarse timestamps):

* **Part A -- controlled recovery.** Simulated univariate Hawkes with the
  regime-B kernel (kappa 0.35, theta 0.099/day) and a constant baseline matched
  to sector volumes, 4 replicates. This isolates the *resolution* effect: same
  process, same estimator, only the timestamps coarsen.

* **Part B -- regime B empirical.** The busiest sector's pooled stream, fitted
  WITH the weekly macro-covariate baseline (`fit_multivariate_with_covariates`,
  M=1). A first pass with a constant baseline produced kappa-hat ~ 0.92 and a
  flat profile -- the baseline variation and latent-quality clustering masquerade
  as near-critical excitation and swamp the decay (the latent-confounding
  artifact of E1.3, in its most extreme form). The covariate baseline removes
  the macro part of that confound; the latent-quality part remains and is
  reported honestly.

Run:  PYTHONPATH=. python -m experiments.exp24_theta_identifiability
Writes results/exp24_theta.{json,png}.
"""

import json
import os
import time

import numpy as np

from hawkes_calibration import (
    PiecewiseConstantCovariate,
    fit_multivariate_with_covariates,
    fit_univariate,
    simulate_multivariate_hawkes,
)

DATA_GT = "data/synthetic_B/ground_truth.npz"
T_DAYS = 3640.0
WIDTHS = (1, 7, 28)
THETA_GRID = np.array([0.02, 0.035, 0.05, 0.07, 0.099, 0.14, 0.20, 0.28, 0.40])
KAPPA_TRUE, THETA_TRUE = 0.35, np.log(2) / 7.0
N_SIM_REPS = 4
IMPUTATIONS = {1: 1, 7: 2, 28: 2}


def _impute(days, width, rng):
    """Uniform imputation inside the observed bucket (width=1: within-day)."""
    bucket = np.floor(days / width) * width
    return np.sort(bucket + rng.uniform(0.0, width, size=days.size))


# ---------------------------------------------------------------------------
# Part A -- controlled: simulated Hawkes with the regime-B kernel
# ---------------------------------------------------------------------------
def part_a(volumes=(1, 4)):
    """Volumes: baseline rate multipliers. 1x ~ one sector (~4k events); 4x ~ a
    pooled market (~16k events) -- the resolution effect only emerges once the
    sampling noise of the kappa-theta ridge has shrunk."""
    alpha = KAPPA_TRUE * THETA_TRUE
    rows, curves = [], {}
    for vol in volumes:
        mu = vol * 1.1 * (1 - KAPPA_TRUE)
        for rep in range(N_SIM_REPS):
            ev = simulate_multivariate_hawkes(
                [np.log(mu)], [[alpha]], [[THETA_TRUE]], T=T_DAYS, seed=300 + rep
            )[0]
            days = np.floor(ev)  # the recorder only keeps the DAY, like the data
            for width in WIDTHS:
                for imp in range(IMPUTATIONS[width]):
                    rng = np.random.default_rng(1000 * rep + 10 * width + imp)
                    times = _impute(days, width, rng)
                    lls, kas = [], []
                    for th in THETA_GRID:
                        res = fit_univariate(times, T_DAYS, beta=float(th), se=False)
                        lls.append(res.loglik)
                        kas.append(float(res.alpha[0, 0]) / float(th))
                    k = int(np.argmax(lls))
                    rows.append(
                        dict(
                            vol=vol,
                            rep=rep,
                            width=width,
                            imp=imp,
                            theta_hat=float(THETA_GRID[k]),
                            kappa_hat=float(kas[k]),
                        )
                    )
                    if rep == 0 and imp == 0:
                        curves[f"v{vol}_w{width}"] = (np.asarray(lls) - max(lls)).tolist()
            print(f"  [A] vol {vol}x replicate {rep} done", flush=True)
    return rows, curves


# ---------------------------------------------------------------------------
# Part B -- regime B, busiest sector, macro-covariate baseline
# ---------------------------------------------------------------------------
def part_b(gt):
    ev = np.asarray(gt["events"], int)
    s = int(np.argmax(np.bincount(ev[:, 1], minlength=12)))
    days = np.sort(ev[ev[:, 1] == s, 0]).astype(float)
    Xw = np.asarray(gt["covariates_std"], float)  # (520, 3) standardized weekly macro
    cov = PiecewiseConstantCovariate(np.arange(521) * 7.0, Xw)
    rows = []
    for width in WIDTHS:
        for imp in range(IMPUTATIONS[width]):
            rng = np.random.default_rng(50 * width + imp)
            times = _impute(days, width, rng)
            lls, kas = [], []
            for th in THETA_GRID:
                res = fit_multivariate_with_covariates(
                    [times], T_DAYS, beta=np.array([[th]]), covariate=cov, se=False
                )
                lls.append(res.loglik)
                kas.append(float(res.alpha[0, 0]) / float(th))
            k = int(np.argmax(lls))
            rows.append(
                dict(
                    sector=s,
                    width=width,
                    imp=imp,
                    theta_hat=float(THETA_GRID[k]),
                    kappa_hat=float(kas[k]),
                )
            )
            print(
                f"  [B] sector {s} width {width:>2}d imp {imp}: "
                f"theta_hat={THETA_GRID[k]:.3f} kappa_hat={kas[k]:.3f}",
                flush=True,
            )
    return rows


def _summ(rows, width, key, vol=None):
    v = [r[key] for r in rows if r["width"] == width and (vol is None or r.get("vol") == vol)]
    return float(np.mean(v)), float(np.std(v))


def main(out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    gt = np.load(DATA_GT, allow_pickle=True)
    a_rows, a_curves = part_a()
    b_rows = part_b(gt)

    res = dict(
        theta_true=THETA_TRUE,
        kappa_true=KAPPA_TRUE,
        theta_grid=THETA_GRID.tolist(),
        part_a=a_rows,
        part_a_curves=a_curves,
        part_b=b_rows,
    )
    with open(os.path.join(out_dir, "exp24_theta.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=float)

    print(
        f"\n=== exp24: decay identifiability vs resolution "
        f"(true kappa {KAPPA_TRUE}, theta {THETA_TRUE:.3f}/day = 7d half-life) ==="
    )
    print("\nPART A -- controlled simulation (constant baseline, well-specified):")
    for vol in sorted({r["vol"] for r in a_rows}):
        n_ev = "~4k" if vol == 1 else f"~{4 * vol}k"
        print(f"  volume {vol}x ({n_ev} events):   width   theta_hat            kappa_hat")
        for w in WIDTHS:
            tm, ts = _summ(a_rows, w, "theta_hat", vol=vol)
            km, ks = _summ(a_rows, w, "kappa_hat", vol=vol)
            print(
                f"                            {w:>3}d    {tm:.3f} ± {ts:.3f}      {km:.3f} ± {ks:.3f}"
            )
    print("\nPART B -- regime B busiest sector (macro-covariate baseline; latent")
    print("           quality still clusters events, so kappa keeps an upward bias):")
    print("  width   theta_hat            kappa_hat")
    for w in WIDTHS:
        tm, ts = _summ(b_rows, w, "theta_hat")
        km, ks = _summ(b_rows, w, "kappa_hat")
        print(f"  {w:>3}d    {tm:.3f} ± {ts:.3f}      {km:.3f} ± {ks:.3f}")
    _plot(out_dir, res)
    print(f"\nWrote results/exp24_theta.{{json,png}}  ({time.time() - t0:.0f}s)")
    return res


def _plot(out_dir, res):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.asarray(res["theta_grid"])
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))

    ax = axes[0]
    styles = {1: "o-", 7: "s--", 28: "^:"}
    vol_hi = max(r["vol"] for r in res["part_a"])
    for j, w in enumerate(WIDTHS):
        ax.plot(
            grid,
            res["part_a_curves"][f"v{vol_hi}_w{w}"],
            styles[w],
            color=f"C{j}",
            label=f"{w}d buckets",
        )
    ax.axvline(res["theta_true"], color="k", lw=1, ls="--", label="true theta")
    ax.set_xscale("log")
    ax.set_xlabel("decay theta (1/day)")
    ax.set_ylabel("profile log-lik (rel. max)")
    ax.set_title(f"A: profile likelihood, {vol_hi}x volume")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    for ax, rows, title in (
        (
            axes[1],
            [r for r in res["part_a"] if r["vol"] == vol_hi],
            f"A: controlled recovery ({vol_hi}x volume)",
        ),
        (axes[2], res["part_b"], "B: regime B (covariate baseline)"),
    ):
        for j, w in enumerate(WIDTHS):
            tv = [r["theta_hat"] for r in rows if r["width"] == w]
            kv = [r["kappa_hat"] for r in rows if r["width"] == w]
            ax.errorbar(
                [w],
                [np.mean(tv)],
                yerr=[np.std(tv)],
                fmt="o",
                capsize=4,
                color="C0",
                label="theta_hat" if j == 0 else None,
            )
            ax.errorbar(
                [w * 1.25],
                [np.mean(kv)],
                yerr=[np.std(kv)],
                fmt="s",
                capsize=4,
                color="C3",
                label="kappa_hat" if j == 0 else None,
            )
        ax.axhline(res["theta_true"], color="C0", lw=1, ls="--")
        ax.axhline(res["kappa_true"], color="C3", lw=1, ls=":")
        ax.set_xscale("log")
        ax.set_xticks(list(WIDTHS))
        ax.set_xticklabels([f"{w}d" for w in WIDTHS])
        ax.set_xlabel("bucket width")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("E1b: exact dates identify the decay; buckets ride the kappa-theta ridge")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "exp24_theta.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
