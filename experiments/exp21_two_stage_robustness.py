"""
Experiment 21: strength and noise-robustness of the two-stage startup ranker.

Deeper validation of the application model in ``paper/complete_account.tex``
(Sec. "Two-stage startup funding ranker"): the business question is *which
startup is about to raise*, so the focus is the second stage (risk-set ranker /
survival selector) under corruptions modeled after the real PitchBook/FactSet
data we will have:

* **feature measurement noise** -- operating/scale fields (revenue, employees,
  valuations) are noisy; add Gaussian noise to the startup features.
* **stale point-in-time (LOCF) covariates** -- PitchBook firm covariates are
  refreshed at financing events and carried forward; freeze each firm's
  features at its last funding event (entry value before the first event).
* **sector label noise** -- industry taxonomy is imperfect; reassign a fraction
  of firms to a wrong sector, which corrupts every risk set they appear in.
* **partial universe** -- only a tracked watch-list is observable; off-list
  events hit the survival model's outside option.
* **cooldown misspecification** -- the true refractory window is unknown; fit
  with the wrong ``cooldown_weeks``.
* **short history** -- fit on less training data, score on the same test weeks.

Every corrupted fit is compared against (a) a random pick over the same risk
set and (b) the *oracle under the same corruption* (true parameters scored on
the corrupted features), so model error is separated from irreducible loss.

Run:
    PYTHONPATH=. python -m experiments.exp21_two_stage_robustness

Writes ``results/exp21_robustness.json`` and ``results/exp21_robustness.png``.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from hawkes_calibration import backtest_synthetic_pipeline
from hawkes_calibration.sector_ranker import (
    StartupRankerResult,
    evaluate_ranker,
    fit_startup_ranker,
    simulate_synthetic_startup_market,
)
from hawkes_calibration.sector_survival import (
    evaluate_survival,
    fit_startup_survival,
    make_tracked_mask,
)

# Paper configuration (complete_account.tex, "Two-stage startup funding ranker").
T = 180
TRAIN_END = 120
N_SECTORS = 11
STARTUPS_PER_SECTOR = 35
N_LAGS = 4
COOLDOWN_WEEKS = 26

STRENGTH_SEEDS = list(range(10))
ROBUST_SEEDS = list(range(5))
TOPK = (1, 5, 10)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def oracle_ranker(data, cooldown_weeks=COOLDOWN_WEEKS):
    """Ranker built from the true data-generating parameters."""
    M, p = data.true_ranker_weights.shape
    return StartupRankerResult(
        global_weights=np.zeros(p),
        sector_deviations=np.asarray(data.true_ranker_weights, dtype=float),
        cooldown_coef=np.asarray(data.true_ranker_cooldown, dtype=float),
        cooldown_weeks=int(cooldown_weeks),
        loss=float("nan"),
        success=True,
        message="oracle (true parameters)",
    )


def corrupt_features_noise(Z, sigma, rng):
    return Z + rng.normal(0.0, sigma, size=Z.shape)


def corrupt_features_locf(Z, startup_counts, active):
    """Freeze features at the firm's last funding event (LOCF point-in-time).

    Before a firm's first event its features stay at the value they had in the
    first week it was active (its 'onboarding snapshot')."""
    Zc = np.array(Z, dtype=float, copy=True)
    T_, N, _ = Z.shape
    for i in range(N):
        act = np.flatnonzero(active[:, i])
        if act.size == 0:
            continue
        last = int(act[0])  # onboarding snapshot
        for t in range(T_):
            if not active[t, i]:
                continue
            Zc[t, i] = Z[last, i]
            if startup_counts[t, i] > 0:  # refresh AT the event -> visible next week
                last = t
    return Zc


def corrupt_sector_labels(startup_sector, frac, n_sectors, rng):
    lab = np.array(startup_sector, dtype=int, copy=True)
    N = lab.size
    n_bad = int(round(frac * N))
    idx = rng.choice(N, size=n_bad, replace=False)
    for i in idx:
        choices = [s for s in range(n_sectors) if s != lab[i]]
        lab[i] = int(rng.choice(choices))
    return lab


def eval_with_coverage(result, data, Z, sector_labels, *, start_week, end_week):
    """evaluate_ranker + the fraction of test events actually covered by the
    observed risk sets (events whose funded firm is missing from the candidate
    set are skipped by evaluate_ranker; that is exactly the taxonomy-error
    failure mode, so it must be reported)."""
    m = evaluate_ranker(
        result,
        data.events,
        Z,
        sector_labels,
        data.active,
        data.startup_counts,
        start_week=start_week,
        end_week=end_week,
        topk=TOPK,
    )
    total = int(np.sum((data.events[:, 0] >= start_week) & (data.events[:, 0] < end_week)))
    m["coverage"] = m.get("n_events", 0) / max(total, 1)
    m["n_events_total"] = total
    return m


def fit_and_eval(data, Z, sector_labels, *, cooldown_weeks=COOLDOWN_WEEKS, train_end=TRAIN_END):
    fit = fit_startup_ranker(
        data.events,
        Z,
        sector_labels,
        data.active,
        data.startup_counts,
        train_end=train_end,
        cooldown_weeks=cooldown_weeks,
        l2_global=1e-3,
        l2_sector=5e-2,
    )
    fitted = eval_with_coverage(fit, data, Z, sector_labels, start_week=TRAIN_END, end_week=T)
    orc = eval_with_coverage(
        oracle_ranker(data), data, Z, sector_labels, start_week=TRAIN_END, end_week=T
    )
    return {
        "fitted": {
            k: fitted.get(k)
            for k in (
                "nll",
                "mrr",
                "top1",
                "top5",
                "top10",
                "coverage",
                "random_nll",
                "random_mrr",
                "random_top1",
                "random_top5",
                "random_top10",
                "n_events",
            )
        },
        "oracle": {k: orc.get(k) for k in ("mrr", "top1", "top5", "top10")},
    }


def agg(runs, path):
    """Mean/sd of a metric across per-seed run dicts; path like ('fitted','top5')."""
    vals = []
    for r in runs:
        v = r
        for k in path:
            v = v[k]
        if v is not None and np.isfinite(v):
            vals.append(float(v))
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))


def fmt(runs, path, pct=True):
    m, s = agg(runs, path)
    if pct:
        return f"{100 * m:5.1f} ± {100 * s:4.1f}"
    return f"{m:.3f} ± {s:.3f}"


# ---------------------------------------------------------------------------
# Part A: headline strength (paper config, 10 seeds, full pipeline)
# ---------------------------------------------------------------------------
def part_a():
    print("=" * 78)
    print(
        "PART A -- headline strength, paper config "
        f"(T={T}, train_end={TRAIN_END}, {N_SECTORS} sectors x {STARTUPS_PER_SECTOR}), "
        f"{len(STRENGTH_SEEDS)} seeds"
    )
    print("=" * 78)
    runs = []
    for seed in STRENGTH_SEEDS:
        t0 = time.time()
        out = backtest_synthetic_pipeline(
            seed=seed,
            T=T,
            train_end=TRAIN_END,
            n_sectors=N_SECTORS,
            startups_per_sector=STARTUPS_PER_SECTOR,
            n_lags=N_LAGS,
            cooldown_weeks=COOLDOWN_WEEKS,
            n_paths=4,
            max_events_per_week=2,
        )
        m = out["metrics"]
        data = out["data"]
        orc = evaluate_ranker(
            oracle_ranker(data),
            data.events,
            data.startup_features,
            data.startup_sector,
            data.active,
            data.startup_counts,
            start_week=TRAIN_END,
            end_week=T,
            topk=TOPK,
        )
        runs.append(
            {
                "seed": seed,
                "fitted": m["ranker"],
                "oracle": orc,
                "survival": m["survival"],
                "hazard": m["discrete_hazard"],
                "sector_nll_improvement": m["sector_nll_improvement"],
                "sector_spectral_radius": m["sector_spectral_radius"],
                "n_events_test": m["n_events_test"],
            }
        )
        print(
            f"  seed {seed}: ranker top5={m['ranker'].get('top5', float('nan')):.3f} "
            f"oracle top5={orc.get('top5', float('nan')):.3f} "
            f"survival top5={m['survival'].get('top5', float('nan')):.3f} "
            f"outside AUC={m['survival'].get('outside_auc', float('nan')):.3f} "
            f"({time.time() - t0:.1f}s)"
        )

    print("\n  metric            fitted           oracle           random")
    for key, paper_f, paper_o, paper_r in (
        ("top1", 16.7, 21.2, 2.9),
        ("top5", 49.2, 53.7, 14.3),
        ("top10", 69.2, 71.6, 28.6),
    ):
        print(
            f"  {key:<8}  {fmt(runs, ('fitted', key))}   "
            f"{fmt(runs, ('oracle', key))}   {fmt(runs, ('fitted', 'random_' + key))}"
            f"    (paper: {paper_f}/{paper_o}/{paper_r})"
        )
    print(
        f"  {'mrr':<8}  {fmt(runs, ('fitted', 'mrr'), pct=False)}   "
        f"{fmt(runs, ('oracle', 'mrr'), pct=False)}   "
        f"{fmt(runs, ('fitted', 'random_mrr'), pct=False)}"
        f"    (paper: 0.328/0.371/0.118)"
    )
    print(f"  survival top5     {fmt(runs, ('survival', 'top5'))}")
    print(f"  survival AUC      {fmt(runs, ('survival', 'outside_auc'), pct=False)}")
    print(f"  hazard top5       {fmt(runs, ('hazard', 'top5'))}")
    print(
        f"  sector NLL improvement over mean baseline: "
        f"{fmt(runs, ('sector_nll_improvement',), pct=False)}"
    )
    return runs


# ---------------------------------------------------------------------------
# Part B: noise robustness sweeps (ranker-focused, 5 seeds each)
# ---------------------------------------------------------------------------
def part_b(markets):
    print("\n" + "=" * 78)
    print(f"PART B -- noise robustness, {len(ROBUST_SEEDS)} seeds per config")
    print("=" * 78)
    sweeps = {}

    def run_config(name, transform):
        runs = []
        for seed in ROBUST_SEEDS:
            data = markets[seed]
            rng = np.random.default_rng(10_000 + seed)
            Z, labels, kw = transform(data, rng)
            runs.append(fit_and_eval(data, Z, labels, **kw))
        sweeps[name] = runs
        print(
            f"  {name:<28} top5 {fmt(runs, ('fitted', 'top5'))}   "
            f"oracle {fmt(runs, ('oracle', 'top5'))}   "
            f"mrr {fmt(runs, ('fitted', 'mrr'), pct=False)}   "
            f"cover {fmt(runs, ('fitted', 'coverage'))}"
        )

    ident = lambda d, r: (d.startup_features, d.startup_sector, {})
    run_config("baseline (clean)", ident)

    for sigma in (0.25, 0.5, 1.0, 2.0):
        run_config(
            f"feature noise sigma={sigma}",
            lambda d, r, s=sigma: (
                corrupt_features_noise(d.startup_features, s, r),
                d.startup_sector,
                {},
            ),
        )

    run_config(
        "LOCF (stale point-in-time)",
        lambda d, r: (
            corrupt_features_locf(d.startup_features, d.startup_counts, d.active),
            d.startup_sector,
            {},
        ),
    )

    for frac in (0.05, 0.15, 0.30):
        run_config(
            f"sector label noise {int(100 * frac)}%",
            lambda d, r, f=frac: (
                d.startup_features,
                corrupt_sector_labels(d.startup_sector, f, N_SECTORS, r),
                {},
            ),
        )

    for cw in (4, 12, 52):
        run_config(
            f"cooldown misspec K={cw} (true 26)",
            lambda d, r, c=cw: (d.startup_features, d.startup_sector, {"cooldown_weeks": c}),
        )

    for te in (60, 90):
        run_config(
            f"short history train_end={te}",
            lambda d, r, t=te: (d.startup_features, d.startup_sector, {"train_end": t}),
        )
    return sweeps


# ---------------------------------------------------------------------------
# Part C: realistic combined scenario + partial universe (survival stage)
# ---------------------------------------------------------------------------
def part_c(markets):
    print("\n" + "=" * 78)
    print("PART C -- realistic combined corruption + partial tracked universe")
    print("=" * 78)
    combo_runs, tracked_runs = [], {0.9: [], 0.7: [], 0.5: []}

    for seed in ROBUST_SEEDS:
        data = markets[seed]
        rng = np.random.default_rng(20_000 + seed)
        # realistic combo: LOCF + measurement noise 0.5 + 10% sector-label noise
        Z = corrupt_features_locf(data.startup_features, data.startup_counts, data.active)
        Z = corrupt_features_noise(Z, 0.5, rng)
        labels = corrupt_sector_labels(data.startup_sector, 0.10, N_SECTORS, rng)
        combo_runs.append(fit_and_eval(data, Z, labels))

        # partial universe on the combo-corrupted data (survival + outside option)
        for frac in tracked_runs:
            tracked = make_tracked_mask(data.active, fraction=frac, seed=seed + 7)
            surv = fit_startup_survival(
                data.events,
                Z,
                labels,
                data.active,
                data.startup_counts,
                tracked=tracked,
                train_end=TRAIN_END,
                cooldown_weeks=COOLDOWN_WEEKS,
                l2_global=1e-3,
                l2_sector=5e-2,
                l2_outside=5e-2,
            )
            sm = evaluate_survival(
                surv,
                data.events,
                Z,
                labels,
                data.active,
                data.startup_counts,
                tracked=tracked,
                start_week=TRAIN_END,
                end_week=T,
                topk=TOPK,
            )
            tracked_runs[frac].append({"survival": sm})

    print(
        f"  {'realistic combo (ranker)':<28} top5 {fmt(combo_runs, ('fitted', 'top5'))}   "
        f"oracle {fmt(combo_runs, ('oracle', 'top5'))}   "
        f"cover {fmt(combo_runs, ('fitted', 'coverage'))}"
    )
    for frac, runs in tracked_runs.items():
        print(
            f"  tracked={frac:<21} top5 {fmt(runs, ('survival', 'top5'))}   "
            f"outside AUC {fmt(runs, ('survival', 'outside_auc'), pct=False)}   "
            f"outside rate/pred "
            f"{fmt(runs, ('survival', 'outside_rate'), pct=False)} / "
            f"{fmt(runs, ('survival', 'outside_pred_mean'), pct=False)}"
        )
    return combo_runs, {str(k): v for k, v in tracked_runs.items()}


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def make_plot(sweeps, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    base = agg(sweeps["baseline (clean)"], ("fitted", "top5"))[0]
    base_orc = agg(sweeps["baseline (clean)"], ("oracle", "top5"))[0]
    base_rnd = agg(sweeps["baseline (clean)"], ("fitted", "random_top5"))[0]

    # (1) feature noise
    ax = axes[0, 0]
    sig = [0.0, 0.25, 0.5, 1.0, 2.0]
    names = ["baseline (clean)"] + [f"feature noise sigma={s}" for s in sig[1:]]
    for path_, label, style in (
        (("fitted", "top5"), "fitted", "o-"),
        (("oracle", "top5"), "oracle under noise", "s--"),
        (("fitted", "random_top5"), "random", ":"),
    ):
        m = [agg(sweeps[n], path_)[0] for n in names]
        e = [agg(sweeps[n], path_)[1] for n in names]
        ax.errorbar(sig, m, yerr=e, fmt=style, capsize=3, label=label)
    ax.set_xlabel("feature noise sigma")
    ax.set_ylabel("top-5 hit rate")
    ax.set_title("Feature measurement noise")
    ax.legend()
    ax.grid(alpha=0.3)

    # (2) sector label noise: top5 and coverage
    ax = axes[0, 1]
    fr = [0.0, 0.05, 0.15, 0.30]
    names = ["baseline (clean)"] + [f"sector label noise {int(100 * f)}%" for f in fr[1:]]
    for path_, label, style in (
        (("fitted", "top5"), "top-5 (covered events)", "o-"),
        (("fitted", "coverage"), "coverage", "s--"),
    ):
        m = [agg(sweeps[n], path_)[0] for n in names]
        e = [agg(sweeps[n], path_)[1] for n in names]
        ax.errorbar(fr, m, yerr=e, fmt=style, capsize=3, label=label)
    eff = [
        agg(sweeps[n], ("fitted", "top5"))[0] * agg(sweeps[n], ("fitted", "coverage"))[0]
        for n in names
    ]
    ax.plot(fr, eff, "^-.", label="effective top-5 (x coverage)")
    ax.set_xlabel("fraction of mislabeled firms")
    ax.set_ylabel("rate")
    ax.set_title("Sector label (taxonomy) noise")
    ax.legend()
    ax.grid(alpha=0.3)

    # (3) cooldown misspecification + LOCF
    ax = axes[1, 0]
    names = [
        "cooldown misspec K=4 (true 26)",
        "cooldown misspec K=12 (true 26)",
        "baseline (clean)",
        "cooldown misspec K=52 (true 26)",
        "LOCF (stale point-in-time)",
    ]
    ticklabels = ["K=4", "K=12", "K=26\n(true)", "K=52", "LOCF"]
    m = [agg(sweeps[n], ("fitted", "top5"))[0] for n in names]
    e = [agg(sweeps[n], ("fitted", "top5"))[1] for n in names]
    ax.bar(range(len(names)), m, yerr=e, capsize=4, color=["C0"] * 4 + ["C3"])
    ax.axhline(base, color="k", lw=0.8, ls="--", label="clean baseline")
    ax.axhline(base_rnd, color="gray", lw=0.8, ls=":", label="random")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(ticklabels)
    ax.set_ylabel("top-5 hit rate")
    ax.set_title("Cooldown misspecification / stale covariates")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    # (4) training history length
    ax = axes[1, 1]
    te = [60, 90, 120]
    names = ["short history train_end=60", "short history train_end=90", "baseline (clean)"]
    m = [agg(sweeps[n], ("fitted", "top5"))[0] for n in names]
    e = [agg(sweeps[n], ("fitted", "top5"))[1] for n in names]
    ax.errorbar(te, m, yerr=e, fmt="o-", capsize=3, label="fitted")
    ax.axhline(base_orc, color="C1", lw=0.8, ls="--", label="oracle ceiling")
    ax.axhline(base_rnd, color="gray", lw=0.8, ls=":", label="random")
    ax.set_xlabel("training weeks")
    ax.set_ylabel("top-5 hit rate")
    ax.set_title("Training-history length (test weeks fixed)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle(
        "Two-stage startup ranker: robustness to data corruptions "
        "(paper config, 5 seeds, held-out weeks 120-180)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"\nWrote {path}")


def main():
    t0 = time.time()
    strength = part_a()

    markets = {
        s: simulate_synthetic_startup_market(
            T=T,
            n_sectors=N_SECTORS,
            startups_per_sector=STARTUPS_PER_SECTOR,
            n_lags=N_LAGS,
            cooldown_weeks=COOLDOWN_WEEKS,
            seed=s,
        )
        for s in ROBUST_SEEDS
    }

    sweeps = part_b(markets)
    combo, tracked = part_c(markets)

    os.makedirs("results", exist_ok=True)
    payload = {
        "config": {
            "T": T,
            "train_end": TRAIN_END,
            "n_sectors": N_SECTORS,
            "startups_per_sector": STARTUPS_PER_SECTOR,
            "cooldown_weeks": COOLDOWN_WEEKS,
            "strength_seeds": STRENGTH_SEEDS,
            "robust_seeds": ROBUST_SEEDS,
        },
        "part_a_strength": strength,
        "part_b_sweeps": sweeps,
        "part_c_combo": combo,
        "part_c_tracked": tracked,
    }
    with open("results/exp21_robustness.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=float)
    print("\nWrote results/exp21_robustness.json")

    make_plot(sweeps, "results/exp21_robustness.png")
    print(f"Total runtime: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
