r"""
Experiment 7 -- Interval-censored forecasting (ACTIVE-style popularity prediction).

This mirrors the real-world experiment of Rizoiu et al. (2022, Section 8): an
observed "view" process is driven exogenously by a stream of "tweets" and
amplified by self-excitation; both are observed only as interval-censored
counts per day.  We fit the MBPP on the first 90 days and forecast views on days
91-120, exactly as in the paper (forecasting via the Proposition 7 lower-bound
compensator, Eq. 55), using the augmented exogenous function

    s_hat[i] = nu + mu * (#tweets on day i)                         (Eq. 54, simplified)

so that the model can absorb unobserved exogenous influence (mu, nu fitted).

We generate several synthetic "videos" with different tweet schedules, fit each,
and compare the MBPP forecast to an exogenous-only baseline (no self-excitation).

Output: results/exp7_forecast.png and results/exp7_forecast.json.
"""

import os
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hawkes_calibration.exogenous import PiecewiseConstant, LHPP
from hawkes_calibration.ic_simulate import simulate_separable_hawkes, interval_censor, uniform_obs_times
from hawkes_calibration.interval_censored import fit_mbpp_ic, forecast_counts


def _tweet_schedule(kind, days, rng):
    """Daily tweet counts: an initial burst that decays, plus noise."""
    t = np.arange(days)
    if kind == "burst":
        base = 30 * np.exp(-t / 12.0) + 2
    elif kind == "double":
        base = 25 * np.exp(-t / 10.0) + 18 * np.exp(-((t - 45) ** 2) / (2 * 8.0 ** 2)) + 2
    else:  # steady
        base = 8 + 3 * np.sin(t / 7.0)
    return np.maximum(rng.poisson(base), 0)


def _simulate_video(tweets, kappa, theta, mu, nu, seed):
    """Views = Hawkes with daily baseline nu + mu*tweets and exponential kernel."""
    days = tweets.size
    breaks = np.arange(days + 1, dtype=float)
    rates = nu + mu * tweets  # exogenous view-rate per day
    exo = PiecewiseConstant(breaks, rates)
    imm, off = simulate_separable_hawkes(exo, kappa, theta, float(days), seed=seed)
    views = np.sort(np.concatenate([imm, off]))
    return interval_censor(views, breaks)


def _smape(pred, actual):
    pred = np.asarray(pred, float); actual = np.asarray(actual, float)
    return float(np.mean(np.abs(pred - actual) / ((np.abs(pred) + np.abs(actual)) / 2 + 1e-9)))


def run(seed=0, out_dir="results", n_videos=12):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    DAYS, FIT = 120, 90
    kappa_true, theta_true, mu_true, nu_true = 0.45, 0.9, 0.7, 0.5
    obs_all = uniform_obs_times(DAYS, DAYS)

    rows = []
    example = None
    for v in range(n_videos):
        kind = ["burst", "double", "steady"][v % 3]
        tweets = _tweet_schedule(kind, DAYS, rng)
        views = _simulate_video(tweets, kappa_true, theta_true, mu_true, nu_true,
                                seed=int(rng.integers(1e9)))

        # --- fit MBPP on first 90 days (IC-LL, full process, fit scale+background) ---
        obs_fit = obs_all[:FIT + 1]
        exo_fit = LHPP(obs_fit, tweets[:FIT])       # immigrant "tweets" -> exogenous
        res = fit_mbpp_ic(obs_fit, views[:FIT], exo_fit, loss="ic-ll", endogenous=False,
                          method="closed", fit_scale=True, fit_background=True,
                          n_restarts=5, seed=v)

        # --- forecast days 91..120 ---
        rates_full = res.background + res.scale * tweets   # s_hat over full horizon
        exo_full = PiecewiseConstant(np.arange(DAYS + 1, dtype=float), rates_full)
        future = obs_all[FIT:]
        pred = forecast_counts(res.kappa, res.theta, exo_full, obs_fit, views[:FIT], future)
        pred = np.maximum(pred, 0.0)
        actual = views[FIT:]

        # --- baseline: exogenous-only (no self-excitation) ---
        base = np.array([rates_full[FIT + i] for i in range(DAYS - FIT)])

        rows.append(dict(
            video=v, kind=kind, kappa=res.kappa, theta=res.theta, scale=res.scale,
            background=res.background,
            smape_mbpp=_smape(pred, actual), smape_base=_smape(base, actual),
            cum_pred=float(pred.sum()), cum_base=float(base.sum()), cum_actual=float(actual.sum()),
        ))
        if v == 0:
            example = dict(tweets=tweets, views=views, pred=pred, base=base, res=res)

    # ---- aggregate ----
    smape_mbpp = np.mean([r["smape_mbpp"] for r in rows])
    smape_base = np.mean([r["smape_base"] for r in rows])
    cum_err_mbpp = np.mean([abs(r["cum_pred"] - r["cum_actual"]) / max(r["cum_actual"], 1) for r in rows])
    cum_err_base = np.mean([abs(r["cum_base"] - r["cum_actual"]) / max(r["cum_actual"], 1) for r in rows])
    summary = dict(
        truth=dict(kappa=kappa_true, theta=theta_true, mu=mu_true, nu=nu_true),
        smape_mbpp=smape_mbpp, smape_baseline=smape_base,
        cum_err_mbpp=cum_err_mbpp, cum_err_baseline=cum_err_base,
        per_video=rows,
    )

    print("=== Interval-censored forecasting (views | tweets) ===")
    print(f"  MBPP forecast   : daily sMAPE={smape_mbpp:.3f}, cumulative err={cum_err_mbpp:.1%}")
    print(f"  exo-only baseline: daily sMAPE={smape_base:.3f}, cumulative err={cum_err_base:.1%}")

    # ---- figure ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ex = example
    days = np.arange(DAYS)
    ax = axes[0]
    ax.bar(days, ex["tweets"], color="C1", alpha=0.4, label="tweets/day (exogenous)")
    ax.plot(days, ex["views"], "k-", lw=1.5, label="views/day (observed)")
    ax.plot(np.arange(FIT, DAYS), ex["pred"], "b-", lw=2, label="MBPP forecast")
    ax.plot(np.arange(FIT, DAYS), ex["base"], "g--", lw=1.5, label="exo-only baseline")
    ax.axvline(FIT, color="r", ls=":", lw=1)
    ax.set_xlabel("day"); ax.set_ylabel("count")
    ax.set_title(f"Example video (fit days 0-{FIT}, forecast {FIT}-{DAYS})")
    ax.legend(fontsize=8)

    ax = axes[1]
    x = np.arange(len(rows))
    ax.bar(x - 0.2, [r["smape_mbpp"] for r in rows], width=0.4, color="C0", label="MBPP")
    ax.bar(x + 0.2, [r["smape_base"] for r in rows], width=0.4, color="C2", label="exo-only")
    ax.set_xlabel("video"); ax.set_ylabel("forecast sMAPE (lower=better)")
    ax.set_title(f"Per-video forecast error\n(mean MBPP={smape_mbpp:.3f} vs baseline={smape_base:.3f})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Forecasting interval-censored popularity with the MBPP (Eq. 55)")
    fig.tight_layout()
    path = os.path.join(out_dir, "exp7_forecast.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)

    with open(os.path.join(out_dir, "exp7_forecast.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {path}")
    return summary


if __name__ == "__main__":
    run()
