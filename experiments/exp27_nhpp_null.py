r"""
Experiment 27 -- the non-homogeneous Poisson NULL and the improvement
decomposition ("why Hawkes at all", REVIEW.md B2).

Ladder of four predictors of weekly sector deal counts, evaluated on the SAME
held-out one-step Poisson NLL/cell (train weeks [0,416), test [416,520); three
worlds: synthetic_A, synthetic_A_strong, synthetic_B). Covariates are
standardized on TRAIN-window moments (point-in-time).

  (0) seasonal-naive : per-sector historical mean by week-of-year (52 bins
      from the train window; overall train sector mean for empty bins).
  (1) NHPP weekly    : lambda_s(w) = exp(a_s + beta_s' X_w), i.e.
      ``fit_sector_glm_fast(n_lags=0)`` -- a genuine covariates-only fit.
  (2) +excitation    : same GLM with 4 weekly own/cross count lags
      (``n_lags=4``), one-step-ahead on realized past-week counts.
  (3) +event-time    : per-sector UNIVARIATE continuous-time Hawkes MLE on
      day-resolution sector event times (ground-truth event stream; regime A
      dates are week-level -> day = 7*week + U(0,7) jitter; regime B col0 is
      already days -> + U(0,1) within-day jitter), weekly piecewise-constant
      covariates in the log-baseline, fixed exponential decay profiled over
      {0.05, 0.099, 0.2}/day by train log-likelihood.

Arm (3) is scored on the same weekly grid through the REALIZED compensator:
the expected count for held-out week w = [a, b) conditions on the true event
history before each instant,

  Lambda_s(w) = \int_a^b exp(g0_s + g_s' X(t)) dt
              + (alpha_s/beta_s) * sum_{t_i < b} [ e^{-beta_s (a - t_i)_+}
                                                   - e^{-beta_s (b - t_i)} ],

which is exact for the exponential kernel (events inside [a,b) enter through
the (a - t_i)_+ = 0 branch).  Conditioning caveat: this uses realized events
*within* the scored week, whereas arm (2) only sees realized counts of
previous weeks through its lags -- arm (2) is the fair weekly analogue, but
arm (3) strictly sees more (within-week) history. Stated in the output.

Per arm we report: held-out Poisson NLL/cell, MAE, quasi-Poisson dispersion
(Pearson X^2 / n on test cells), randomized PIT for discrete counts
(u ~ U[F(y-1), F(y)] under the predicted Poisson; KS distance from uniform,
averaged over 5 randomization seeds) and empirical coverage of the central
50%/90% Poisson intervals.

Headline: the decomposition table NLL(seasonal) -> NLL(NHPP) -> NLL(+exc) ->
NLL(+event-time) with the marginal gain of each block, and a printed verdict
on REVIEW's expectation ("A: excitation gain real but modest; B: most of the
gain over NHPP comes from short-horizon timing visible only to the event-time
model").

Run:  PYTHONPATH=. OMP_NUM_THREADS=2 python -m experiments.exp27_nhpp_null
Writes results/exp27_nhpp.json and results/exp27_nhpp.png.
"""

import json
import os
import time

import numpy as np
from scipy.special import gammaln
from scipy.stats import poisson

from hawkes_calibration import fit_multivariate_with_covariates
from hawkes_calibration.eventtime.covariates import PiecewiseConstantCovariate
from synthetic.fast_fit import fit_sector_glm_fast
from synthetic.loaders import load_dataset

TRAIN_END = 416
T_WEEKS = 520
DAY_TRAIN_END = TRAIN_END * 7.0  # 2912.0
N_LAGS = 4
DECAY_GRID = (0.05, 0.099, 0.2)  # per-day exponential decays profiled in arm (3)
PIT_SEEDS = 5
JITTER_SEED = 0
DATASETS = ("data/synthetic_A", "data/synthetic_A_strong", "data/synthetic_B")
ARMS = ("seasonal", "nhpp", "excitation", "eventtime")
ARM_LABELS = {
    "seasonal": "(0) seasonal-naive",
    "nhpp": "(1) NHPP (cov only)",
    "excitation": "(2) +excitation",
    "eventtime": "(3) +event-time",
}


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def _metrics(obs, pred):
    """Held-out Poisson NLL/cell, MAE and quasi-Poisson dispersion."""
    lam = np.maximum(np.asarray(pred, float), 1e-12)
    y = np.asarray(obs, float)
    nll = float(np.mean(lam - y * np.log(lam) + gammaln(y + 1.0)))
    mae = float(np.abs(y - lam).mean())
    dispersion = float(np.mean((y - lam) ** 2 / lam))  # Pearson X^2 / n
    return dict(nll=nll, mae=mae, dispersion=dispersion)


def _pit(obs, pred, n_seeds=PIT_SEEDS):
    """Randomized PIT for discrete counts + central-interval coverage."""
    lam = np.maximum(np.asarray(pred, float).ravel(), 1e-12)
    y = np.asarray(obs, float).ravel()
    f_hi = poisson.cdf(y, lam)
    f_lo = poisson.cdf(y - 1.0, lam)  # cdf(-1) = 0
    n = y.size
    grid_hi = np.arange(1, n + 1) / n
    grid_lo = np.arange(0, n) / n
    ks = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(1000 + seed)
        u = np.sort(f_lo + rng.uniform(0.0, 1.0, n) * (f_hi - f_lo))
        ks.append(float(np.max(np.maximum(np.abs(u - grid_hi), np.abs(u - grid_lo)))))
    cover = {}
    for level in (0.5, 0.9):
        lo = poisson.ppf(0.5 - level / 2.0, lam)
        hi = poisson.ppf(0.5 + level / 2.0, lam)
        cover[level] = float(np.mean((y >= lo) & (y <= hi)))
    return dict(
        pit_ks=float(np.mean(ks)),
        pit_ks_sd=float(np.std(ks)),
        coverage50=cover[0.5],
        coverage90=cover[0.9],
    )


def _evaluate(obs, pred):
    out = _metrics(obs, pred)
    out.update(_pit(obs, pred))
    return out


# ---------------------------------------------------------------------------
# arms (0)-(2): weekly predictors
# ---------------------------------------------------------------------------
def seasonal_naive_rates(y):
    """Per-sector train mean by week-of-year (52 bins); test-window rates."""
    T, M = y.shape
    woy = np.arange(T) % 52
    rates = np.zeros((T_WEEKS - TRAIN_END, M))
    overall = np.maximum(y[:TRAIN_END].mean(0), 1e-12)
    for s in range(M):
        by_bin = np.full(52, np.nan)
        for b in range(52):
            sel = woy[:TRAIN_END] == b
            if sel.any():
                by_bin[b] = y[:TRAIN_END, s][sel].mean()
        by_bin = np.where(np.isnan(by_bin), overall[s], by_bin)
        rates[:, s] = np.maximum(by_bin[woy[TRAIN_END:]], 1e-12)
    return rates


def glm_rates(a, b, exc, cov, counts, n_lags):
    """One-step-ahead GLM rates on realized past counts (exp25 convention)."""
    T = counts.shape[0]
    out = np.zeros_like(counts, dtype=float)
    for t in range(n_lags, T):
        eta = a + b @ cov[t]
        for lag in range(1, n_lags + 1):
            eta += exc[:, :, lag - 1] @ counts[t - lag]
        out[t] = np.exp(np.clip(eta, -30, 20))
    return out


# ---------------------------------------------------------------------------
# arm (3): per-sector univariate event-time Hawkes with weekly covariates
# ---------------------------------------------------------------------------
def day_resolution_times(gt, regime, rng):
    """Day-resolution (sector, time) stream from the ground-truth event list."""
    ev = np.asarray(gt["events"])
    if regime == "A":  # col0 is the week -> uniform jitter within the week
        t_days = ev[:, 0] * 7.0 + rng.uniform(0.0, 7.0, len(ev))
    else:  # regime B: col0 is already the day -> within-day jitter
        t_days = ev[:, 0] + rng.uniform(0.0, 1.0, len(ev))
    return ev[:, 1].astype(int), t_days


def fit_eventtime_sector(times_train, covariate):
    """Profile the fixed decay over DECAY_GRID by train log-likelihood."""
    best = None
    for beta in DECAY_GRID:
        with np.errstate(all="ignore"):
            res = fit_multivariate_with_covariates(
                [times_train],
                DAY_TRAIN_END,
                beta=np.array([[beta]]),
                covariate=covariate,
                se=False,
            )
        if best is None or res.loglik > best["loglik"]:
            best = dict(
                beta=beta,
                loglik=float(res.loglik),
                gamma0=float(res.gamma0[0]),
                gamma=res.gamma[0].copy(),
                alpha=float(res.alpha[0, 0]),
                success=bool(res.success),
            )
    return best


def eventtime_weekly_rates(fit, times_all, X):
    """Realized compensator of the fitted intensity over each test week.

    Lambda(w) = 7 * exp(g0 + g'X_w)                     (covariate const. in-week)
              + (alpha/beta) sum_{t_i < b} [e^{-beta (a-t_i)_+} - e^{-beta (b-t_i)}]
    conditioning on the TRUE realized events before each instant.
    """
    beta, ab = fit["beta"], fit["alpha"] / fit["beta"]
    g0, g = fit["gamma0"], fit["gamma"]
    rates = np.zeros(T_WEEKS - TRAIN_END)
    for k, w in enumerate(range(TRAIN_END, T_WEEKS)):
        a, b = w * 7.0, (w + 1) * 7.0
        base = 7.0 * float(np.exp(np.clip(g0 + g @ X[w], -30, 20)))
        past = times_all[times_all < b]
        exc = ab * float(
            np.sum(np.exp(-beta * np.clip(a - past, 0.0, None)) - np.exp(-beta * (b - past)))
        )
        rates[k] = max(base + exc, 1e-12)
    return rates


# ---------------------------------------------------------------------------
def run_world(path):
    name = os.path.basename(path)
    gt_path = os.path.join(path, "ground_truth.npz")
    gt = np.load(gt_path, allow_pickle=True)
    regime = str(np.asarray(gt["regime"]))
    ds = load_dataset(path, active="true", ground_truth=gt_path)
    y = ds.sector_counts.astype(float)
    M = y.shape[1]
    obs = y[TRAIN_END:]

    # point-in-time covariate standardization (train-window moments)
    mu_tr = ds.covariates_raw[:TRAIN_END].mean(0)
    sd_tr = ds.covariates_raw[:TRAIN_END].std(0) + 1e-12
    X = (ds.covariates_raw - mu_tr) / sd_tr

    arms = {}
    # (0) seasonal-naive
    arms["seasonal"] = _evaluate(obs, seasonal_naive_rates(y))

    # (1) NHPP weekly (covariates only) and (2) +excitation
    for arm, lags in (("nhpp", 0), ("excitation", N_LAGS)):
        a, b, exc = fit_sector_glm_fast(y, X, n_lags=lags, train_end=TRAIN_END, l2=1e-3)
        pred = glm_rates(a, b, exc, X, y, lags)[TRAIN_END:]
        arms[arm] = _evaluate(obs, pred)

    # (3) event-time: per-sector univariate continuous-time fit
    rng = np.random.default_rng(JITTER_SEED)
    sec, t_days = day_resolution_times(gt, regime, rng)
    covariate = PiecewiseConstantCovariate(np.arange(T_WEEKS + 1) * 7.0, X)
    pred_et = np.zeros_like(obs)
    detail = []
    t0 = time.time()
    for s in range(M):
        ts = np.sort(t_days[sec == s])
        ts_train = ts[ts < DAY_TRAIN_END]
        fit = fit_eventtime_sector(ts_train, covariate)
        pred_et[:, s] = eventtime_weekly_rates(fit, ts, X)
        detail.append(
            dict(
                sector=s,
                beta=fit["beta"],
                branching=fit["alpha"] / fit["beta"],
                alpha=fit["alpha"],
                gamma0=fit["gamma0"],
                gamma=fit["gamma"].tolist(),
                train_loglik=fit["loglik"],
                n_train_events=int(len(ts_train)),
                success=fit["success"],
            )
        )
    et_time = time.time() - t0
    arms["eventtime"] = _evaluate(obs, pred_et)

    # ground-truth-stream vs observed-counts mismatch (arm-3 target caveat)
    gt_hist = np.zeros((T_WEEKS, M), int)
    wk = np.minimum(np.asarray(gt["events"])[:, 0] // (1 if regime == "A" else 7), T_WEEKS - 1)
    np.add.at(gt_hist, (wk, np.asarray(gt["events"])[:, 1]), 1)
    mismatch = dict(
        mean_abs_weekly_cell_diff=float(np.abs(gt_hist - ds.sector_counts).mean()),
        gt_total=int(len(gt["events"])),
        observed_total=int(ds.sector_counts.sum()),
    )

    d = {a: arms[a]["nll"] for a in ARMS}
    decomposition = dict(
        nll_ladder=[d[a] for a in ARMS],
        gain_covariates=d["seasonal"] - d["nhpp"],
        gain_excitation=d["nhpp"] - d["excitation"],
        gain_eventtime=d["excitation"] - d["eventtime"],
        gain_over_nhpp_total=d["nhpp"] - d["eventtime"],
    )
    return dict(
        regime=regime,
        arms=arms,
        decomposition=decomposition,
        eventtime_detail=detail,
        eventtime_fit_seconds=et_time,
        gt_vs_observed=mismatch,
    ), name


def _verdict(results):
    """Verify/refute REVIEW's expected decomposition pattern, with numbers."""
    lines, flags = [], {}
    for name, r in results.items():
        dec = r["decomposition"]
        g_exc, g_et, g_tot = (
            dec["gain_excitation"],
            dec["gain_eventtime"],
            dec["gain_over_nhpp_total"],
        )
        if r["regime"] == "A":
            modest = 0.0 < g_exc < 0.10
            word = "VERIFIED" if modest else "REFUTED"
            lines.append(
                f"[{name}] excitation gain over NHPP = {g_exc:+.3f} NLL/cell "
                f"(event-time adds {g_et:+.3f} on top) -> 'real but modest' {word}."
            )
            flags[name] = dict(expected="excitation real but modest", verified=bool(modest))
        else:
            timing_dominates = g_tot > 0 and g_et > 0.5 * g_tot
            word = "VERIFIED" if timing_dominates else "REFUTED"
            share = g_et / g_tot if g_tot > 0 else float("nan")
            lines.append(
                f"[{name}] total gain over NHPP = {g_tot:+.3f}; weekly excitation "
                f"captures {g_exc:+.3f}, event-time timing adds {g_et:+.3f} "
                f"({share:.0%} of the total) -> 'most gain is short-horizon timing' {word}."
            )
            flags[name] = dict(
                expected="most NHPP gain from short-horizon timing (event-time only)",
                verified=bool(timing_dominates),
                timing_share=float(share),
            )
    return lines, flags


def _plot(results, out_png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(results), figsize=(4.2 * len(results), 3.6), sharey=False)
    axes = np.atleast_1d(axes)
    colors = ["#9aa0a6", "#4c72b0", "#dd8452", "#55a868"]
    for ax, (name, r) in zip(axes, results.items()):
        nlls = [r["arms"][a]["nll"] for a in ARMS]
        ax.bar(range(len(ARMS)), nlls, color=colors)
        for i, v in enumerate(nlls):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(ARMS)))
        ax.set_xticklabels(
            [ARM_LABELS[a].split(") ")[1] for a in ARMS], rotation=20, ha="right", fontsize=8
        )
        ax.set_title(name.replace("synthetic_", "regime "), fontsize=10)
        ax.set_ylabel("held-out Poisson NLL/cell")
        lo = min(nlls)
        ax.set_ylim(max(0.0, lo - 0.25 * (max(nlls) - lo + 0.05)), None)
    fig.suptitle("Exp27 -- NHPP null and improvement decomposition", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


CAVEATS = [
    "Conditioning: arm (3) scores week w through the realized compensator, which "
    "conditions on true events before each instant INCLUDING events inside week w; "
    "arm (2) only sees realized counts of previous weeks through its 4 lags. Arm (2) "
    "is the fair one-step weekly analogue, but arm (3) strictly sees more "
    "(within-week) history.",
    "Arm (3) trains and conditions on the ground-truth day-resolution event stream "
    "while all arms are scored against OBSERVED weekly counts from deals.csv "
    "(dedup/mislabels/date-jitter noise); the gt stream has ~0.3-1.4% more events "
    "than the observed table (per-world mismatch reported in gt_vs_observed).",
    "Regime A (and A_strong) deal dates are week-level by construction; the "
    "within-week day jitter is synthetic, so arm (3) has no genuine sub-week timing "
    "information there -- exactly the misspecification probe intended.",
    "Single jitter seed (0) for day-time construction; randomized PIT averaged over "
    "5 randomization seeds. Coverage of discrete central intervals overshoots "
    "nominal levels by construction (ppf-based intervals are conservative).",
]


def main(out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    results = {}
    for path in DATASETS:
        if not os.path.isdir(path):
            print(f"  !! missing {path} -- skipping")
            continue
        print(f"[exp27] running {path} ...", flush=True)
        r, name = run_world(path)
        results[name] = r

    verdict_lines, verdict_flags = _verdict(results)

    # ---------------- printed summary ----------------
    for name, r in results.items():
        print(f"\n=== {name} (regime {r['regime']}) ===")
        print(
            f"  {'arm':22s} {'NLL/cell':>9s} {'MAE':>7s} {'disp':>6s} "
            f"{'PIT-KS':>7s} {'cov50':>6s} {'cov90':>6s}"
        )
        for a in ARMS:
            v = r["arms"][a]
            print(
                f"  {ARM_LABELS[a]:22s} {v['nll']:9.3f} {v['mae']:7.3f} "
                f"{v['dispersion']:6.2f} {v['pit_ks']:7.3f} "
                f"{v['coverage50']:6.2f} {v['coverage90']:6.2f}"
            )
        dec = r["decomposition"]
        print(
            f"  decomposition: seasonal {dec['nll_ladder'][0]:.3f} "
            f"-> NHPP {dec['nll_ladder'][1]:.3f} (gain {dec['gain_covariates']:+.3f}) "
            f"-> +exc {dec['nll_ladder'][2]:.3f} (gain {dec['gain_excitation']:+.3f}) "
            f"-> +event-time {dec['nll_ladder'][3]:.3f} (gain {dec['gain_eventtime']:+.3f})"
        )
        betas = [d["beta"] for d in r["eventtime_detail"]]
        brs = [d["branching"] for d in r["eventtime_detail"]]
        print(
            f"  event-time: profiled beta/day counts "
            f"{{{', '.join(f'{b}: {betas.count(b)}' for b in DECAY_GRID)}}}, "
            f"branching alpha/beta median {np.median(brs):.2f} "
            f"(max {max(brs):.2f}), fit {r['eventtime_fit_seconds']:.1f}s"
        )

    print("\n--- verdict on REVIEW B2 expectations ---")
    for line in verdict_lines:
        print("  " + line)
    print("\n--- caveats ---")
    for c in CAVEATS:
        print("  * " + c)

    payload = dict(
        config=dict(
            train_end_week=TRAIN_END,
            weeks=T_WEEKS,
            n_lags=N_LAGS,
            decay_grid=list(DECAY_GRID),
            jitter_seed=JITTER_SEED,
            pit_seeds=PIT_SEEDS,
            datasets=list(DATASETS),
        ),
        worlds=results,
        verdict=verdict_flags,
        verdict_text=verdict_lines,
        caveats=CAVEATS,
        runtime_seconds=time.time() - t0,
    )
    out_json = os.path.join(out_dir, "exp27_nhpp.json")
    with open(out_json, "w") as fh:
        json.dump(payload, fh, indent=2, default=float)
    out_png = os.path.join(out_dir, "exp27_nhpp.png")
    _plot(results, out_png)
    print(f"\nWrote {out_json} and {out_png} ({payload['runtime_seconds']:.0f}s total)")
    return payload


if __name__ == "__main__":
    main()
