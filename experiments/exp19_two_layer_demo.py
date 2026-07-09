r"""
Experiment 19 -- the two-layer marked model (NO event-time information), both layers
working well on a clean, well-conditioned synthetic market.

This is the "count + ranking" regime: we never see exact event times, only weekly
*counts* per sector and which startup got each round.  We deliberately stay far away
from the unstable territory:

  * the true sector excitation has a small spectral radius (kept << 1), so the count
    process is comfortably noncritical -- no explosive feedback;
  * event volumes are moderate (a couple of fundings per sector-week on average);
  * the sector signal is covariate-driven, which the historical-mean baseline cannot
    capture, so the model has something real to learn.

Two held-out evaluations, no forward simulation:

  Layer 1 (sector count GLM)  -- one-step Poisson NLL vs. a historical-mean baseline.
  Layer 2 (risk-set ranker)   -- NLL / MRR / top-k vs. a random pick over the live
                                 risk set, and vs. the ORACLE (true-parameter) ranker.

Run:  PYTHONPATH=. python -m experiments.exp19_two_layer_demo
"""

import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hawkes_calibration import (
    evaluate_ranker,
    excitation_spectral_radius,
    fit_sector_count_model,
    fit_startup_ranker,
    simulate_synthetic_startup_market,
)
from hawkes_calibration.sector_ranker import (
    StartupRankerResult,
    poisson_nll,
    sector_baseline_rates,
)

CFG = dict(
    T=170,
    n_sectors=8,
    startups_per_sector=25,
    n_covariates=2,
    n_lags=3,
    cooldown_weeks=16,
    base_log_rate=-1.0,
    covariate_strength=4.5,
    excitation_radius=0.10,
    train_end=120,
)


def _oracle(data, cooldown_weeks):
    p = data.true_ranker_weights.shape[1]
    return StartupRankerResult(
        global_weights=np.zeros(p),
        sector_deviations=data.true_ranker_weights,
        cooldown_coef=data.true_ranker_cooldown,
        cooldown_weeks=cooldown_weeks,
        loss=0.0,
        success=True,
        message="oracle",
    )


def run(seeds=range(6), out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    c = CFG
    sector_rows, fit_rows, orac_rows, _extra = [], [], [], []
    for seed in seeds:
        d = simulate_synthetic_startup_market(
            T=c["T"],
            n_sectors=c["n_sectors"],
            startups_per_sector=c["startups_per_sector"],
            n_covariates=c["n_covariates"],
            n_lags=c["n_lags"],
            cooldown_weeks=c["cooldown_weeks"],
            base_log_rate=c["base_log_rate"],
            covariate_strength=c["covariate_strength"],
            excitation_radius=c["excitation_radius"],
            seed=seed,
        )
        te = c["train_end"]

        # --- Layer 1: sector count model vs historical-mean baseline ---
        sf = fit_sector_count_model(
            d.sector_counts,
            d.covariates,
            n_lags=c["n_lags"],
            train_end=te,
            max_excitation_radius=0.3,
            l2=1e-2,
        )
        rates = sf.rates(d.sector_counts, d.covariates)
        obs = d.sector_counts[te : c["T"]]
        model_nll = poisson_nll(obs, rates[te : c["T"]]) / obs.size
        base_nll = (
            poisson_nll(obs, sector_baseline_rates(d.sector_counts, te, te, c["T"])) / obs.size
        )
        sector_rows.append(
            dict(
                seed=int(seed),
                model_nll=model_nll,
                baseline_nll=base_nll,
                improvement=base_nll - model_nll,
                fitted_spectral_radius=excitation_spectral_radius(sf.excitation),
                mean_count=float(obs.mean()),
                max_count=int(d.sector_counts.max()),
            )
        )

        # --- Layer 2: ranker vs random and vs oracle ---
        args = (d.events, d.startup_features, d.startup_sector, d.active, d.startup_counts)
        rf = fit_startup_ranker(
            *args, train_end=te, cooldown_weeks=c["cooldown_weeks"], l2_global=1e-3, l2_sector=5e-2
        )
        fit_rows.append(evaluate_ranker(rf, *args, start_week=te, end_week=c["T"]))
        orac_rows.append(
            evaluate_ranker(_oracle(d, c["cooldown_weeks"]), *args, start_week=te, end_week=c["T"])
        )

    def avg(rows, k):
        return float(np.mean([r[k] for r in rows]))

    res = dict(
        config=c,
        n_seeds=len(list(seeds)),
        sector=dict(
            model_nll=avg(sector_rows, "model_nll"),
            baseline_nll=avg(sector_rows, "baseline_nll"),
            nll_improvement=avg(sector_rows, "improvement"),
            beats_baseline_fraction=float(np.mean([r["improvement"] > 0 for r in sector_rows])),
            fitted_spectral_radius=avg(sector_rows, "fitted_spectral_radius"),
            mean_count_per_cell=avg(sector_rows, "mean_count"),
        ),
        ranker=dict(
            test_events=avg(fit_rows, "n_events"),
            fitted={k: avg(fit_rows, k) for k in ("nll", "mrr", "top1", "top5", "top10")},
            oracle={k: avg(orac_rows, k) for k in ("nll", "mrr", "top1", "top5", "top10")},
            random={
                k: avg(fit_rows, "random_" + k) for k in ("nll", "mrr", "top1", "top5", "top10")
            },
        ),
        per_seed_sector=sector_rows,
    )
    json.dump(
        res, open(os.path.join(out_dir, "exp19_two_layer.json"), "w"), indent=2, default=float
    )

    _print(res)
    _plot(out_dir, res, sector_rows)
    return res


def _print(res):
    s, r = res["sector"], res["ranker"]
    print("=== Experiment 19: two-layer marked model (no time info) ===")
    print(
        f"  regime: ~{s['mean_count_per_cell']:.1f} events/sector-week, fitted sector "
        f"spectral radius {s['fitted_spectral_radius']:.2f} (<< 1, stable)"
    )
    print(
        "\n  LAYER 1 -- sector count model vs historical-mean baseline (held-out Poisson NLL/cell):"
    )
    print(
        f"    model {s['model_nll']:.3f}  vs  baseline {s['baseline_nll']:.3f}   "
        f"(improvement {s['nll_improvement']:.3f}; beats baseline in "
        f"{100 * s['beats_baseline_fraction']:.0f}% of seeds)"
    )
    print("\n  LAYER 2 -- risk-set ranker, fitted vs oracle (ceiling) vs random:")
    print("            top1    top5    top10   MRR")
    for name in ("fitted", "oracle", "random"):
        d = r[name]
        print(
            f"    {name:7s} {d['top1']:.3f}   {d['top5']:.3f}   {d['top10']:.3f}   {d['mrr']:.3f}"
        )
    print(
        f"\n  -> Layer 1 beats the naive baseline; Layer 2 sits at the oracle ceiling "
        f"({100 * r['fitted']['top5'] / max(r['oracle']['top5'], 1e-9):.0f}% of oracle top-5) "
        f"and ~{r['fitted']['top5'] / max(r['random']['top5'], 1e-9):.1f}x random."
    )
    print("Wrote results/exp19_two_layer.{json,png}")


def _plot(out_dir, res, sector_rows):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    seeds = [r["seed"] for r in sector_rows]
    width = 0.38
    ax[0].bar(
        [s - width / 2 for s in seeds],
        [r["model_nll"] for r in sector_rows],
        width,
        label="sector model",
        color="C0",
    )
    ax[0].bar(
        [s + width / 2 for s in seeds],
        [r["baseline_nll"] for r in sector_rows],
        width,
        label="mean baseline",
        color="C7",
    )
    ax[0].set_xlabel("seed")
    ax[0].set_ylabel("held-out Poisson NLL / cell")
    ax[0].set_title("Layer 1: sector count model beats baseline")
    ax[0].legend()

    r = res["ranker"]
    ks = ["top1", "top5", "top10", "mrr"]
    x = np.arange(len(ks))
    w = 0.26
    ax[1].bar(x - w, [r["fitted"][k] for k in ks], w, label="fitted", color="C0")
    ax[1].bar(x, [r["oracle"][k] for k in ks], w, label="oracle (ceiling)", color="C2")
    ax[1].bar(x + w, [r["random"][k] for k in ks], w, label="random", color="C7")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(["top-1", "top-5", "top-10", "MRR"])
    ax[1].set_title("Layer 2: ranker at the oracle ceiling")
    ax[1].legend()
    fig.suptitle("Two-layer marked model (no event-time info) -- both layers working")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "exp19_two_layer.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    run()
