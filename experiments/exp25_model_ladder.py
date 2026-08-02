r"""
Experiment 25 -- the MODEL LADDER: alternative predictors on the same worlds,
same splits, same pools, so the impact of each modeling choice is measurable.

Two prediction tasks on three synthetic worlds (A, A-strong with
``covariate_scale=1.5``, B), oracle risk sets (v2 assumptions), temporal split
train ``[0,416)`` / test ``[416,520)``:

**Task 1 -- weekly sector counts (when/where).** Ladder from dumb to structural:
    mean | EWMA | GLM covariates-only | GLM lags-only | GLM full | ridge-ML |
    oracle (true parameters incl. the latent path; A-worlds only).
Metric: held-out one-step Poisson NLL/cell (+ MAE). Fitted arms standardize
covariates on TRAIN-window moments (point-in-time); the oracle uses the
generator's scalers (it is an oracle).

**Task 2 -- which firm gets the deal (ranking).** Five model philosophies plus
heuristics, ALL scoring the *same* candidate pool (oracle-active firms of the
event's sector; the Hawkes ranker is fitted with ``drop_last_funded=False`` so
its pool matches) and ALL fitted on the same training events (the discrete
hazard receives the same event subsample through ``label_counts``):
    popularity (cum. past deals) | recency momentum | conditional logit |
    discrete hazard | Cox survival | Hawkes ranker (linear link) |
    Hawkes ranker (exp link) | oracle weights (A-worlds).
Metrics: top-1/5/10, MRR for all (mid-rank tie convention, identical for every
model); NLL for the probabilistic models, each computed under its OWN
likelihood on a common log-score scale (softmax of the linear predictor for
logit/survival/oracle; normalized intensities lam/sum(lam) for the Hawkes
arms; normalized hazards for the discrete hazard -- exact only in the
rare-event limit).

Fairness notes (from the adversarial audit): the conditional logit uses
sampled softmax (max_candidates=64), which is a consistent estimator of the
same likelihood the survival/Hawkes fits optimize in full; training events are
pre-filtered to those whose winner is inside the eval pool, so every fitter
sees exactly the same events; the ranking "oracle" applies the true weights to
the OBSERVED (noisy, LOCF) features, so it is a strong reference rather than a
strict ceiling; within-week multi-deal "taken firm" exclusions of the
generator are not replicated, which costs the oracle a small calibration
margin on multi-deal weeks.

Run:  PYTHONPATH=. python -m experiments.exp25_model_ladder
Writes results/exp25_model_ladder.{json,png}.
"""

import json
import os
import time

import numpy as np

from hawkes_calibration.models.hawkes_ranker import _fields_for, _link, fit_hawkes_ranker
from hawkes_calibration.sector_hazard import discrete_hazard_scores, fit_discrete_hazard
from hawkes_calibration.sector_survival import fit_startup_survival
from synthetic.fast_fit import fit_ranker_fast, fit_sector_glm_fast
from synthetic.loaders import load_dataset

TRAIN_END = 416
T_WEEKS = 520
MAX_TEST_EVENTS = 2500
MAX_TRAIN_EVENTS = 8000  # same training subsample for every second-stage model
DATASETS = ("data/synthetic_A", "data/synthetic_A_strong", "data/synthetic_B")


# ===========================================================================
# Task 1 -- weekly sector counts
# ===========================================================================
def _poisson_nll(counts, rates):
    from math import lgamma

    rates = np.maximum(rates, 1e-12)
    lg = np.vectorize(lambda x: lgamma(x + 1.0))(counts)
    return float(np.sum(rates - counts * np.log(rates) + lg)) / counts.size


def _glm_rates(a, b, exc, cov, counts, L, extra_eta=None):
    T = counts.shape[0]
    out = np.zeros_like(counts, dtype=float)
    for t in range(L, T):
        eta = a + b @ cov[t]
        if extra_eta is not None:
            eta = eta + extra_eta[t]
        for lag in range(1, L + 1):
            eta += exc[:, :, lag - 1] @ counts[t - lag]
        out[t] = np.exp(np.clip(eta, -30, 20))
    return out


def counts_ladder(ds, gt, has_oracle):
    L = 4
    y = ds.sector_counts.astype(float)
    # point-in-time standardization for the FITTED arms (train-window moments)
    mu_tr = ds.covariates_raw[:TRAIN_END].mean(0)
    sd_tr = ds.covariates_raw[:TRAIN_END].std(0) + 1e-12
    X = (ds.covariates_raw - mu_tr) / sd_tr
    obs = y[TRAIN_END:]
    out = {}

    mu = np.maximum(y[:TRAIN_END].mean(0), 1e-12)
    out["mean"] = dict(
        nll=_poisson_nll(obs, np.tile(mu, (obs.shape[0], 1))), mae=float(np.abs(obs - mu).mean())
    )

    # EWMA (8-week half-life), strictly one-step-ahead
    lam = np.log(2) / 8.0
    ew = np.zeros_like(y)
    ew[0] = y[0]
    for t in range(1, T_WEEKS):
        ew[t] = (1 - np.exp(-lam)) * y[t - 1] + np.exp(-lam) * ew[t - 1]
    pred = np.maximum(ew[TRAIN_END:], 1e-12)
    out["ewma"] = dict(nll=_poisson_nll(obs, pred), mae=float(np.abs(obs - pred).mean()))

    # GLM variants -- the covariates-only arm is a genuine no-lag fit (n_lags=0)
    for name, cov_in, lags in (
        ("glm_cov_only", X, 0),
        ("glm_lags_only", X[:, :0], L),
        ("glm_full", X, L),
    ):
        a, b, exc = fit_sector_glm_fast(y, cov_in, n_lags=lags, train_end=TRAIN_END, l2=1e-3)
        rates = _glm_rates(a, b, exc, cov_in, y, lags)
        pred = rates[TRAIN_END:]
        out[name] = dict(nll=_poisson_nll(obs, pred), mae=float(np.abs(obs - pred).mean()))

    # ridge "ML" arm: log1p target, Duan smearing back-transform (audit fix)
    M = y.shape[1]
    rows, targets, keys = [], [], []
    for t in range(L, T_WEEKS):
        mkt = y[t - 1].sum()
        for s in range(M):
            onehot = np.zeros(M)
            onehot[s] = 1.0
            rows.append(
                np.concatenate([[1.0], X[t], y[t - L : t, s][::-1], [np.log1p(mkt)], onehot])
            )
            targets.append(np.log1p(y[t, s]))
            keys.append(t)
    F = np.asarray(rows)
    tgt = np.asarray(targets)
    keys = np.asarray(keys)
    tr = keys < TRAIN_END
    W = np.linalg.solve(F[tr].T @ F[tr] + 1.0 * np.eye(F.shape[1]), F[tr].T @ tgt[tr])
    resid = tgt[tr] - F[tr] @ W
    z = F[~tr] @ W
    pred = np.maximum(np.mean(np.expm1(z[:, None] + resid[None, :]), axis=1), 1e-12)
    pred = pred.reshape(-1, M)
    out["ridge_ml"] = dict(nll=_poisson_nll(obs, pred), mae=float(np.abs(obs - pred).mean()))

    if has_oracle:
        Xg = (ds.covariates_raw - gt["cov_mean"]) / gt["cov_sd"]
        latent_eta = float(gt["latent_strength"]) * np.asarray(gt["latent_path"], float)
        rates = _glm_rates(
            gt["sector_intercept"],
            gt["sector_beta"],
            gt["excitation"],
            Xg,
            y,
            int(gt["n_lags"]),
            extra_eta=latent_eta[:, None] * np.ones((1, M)),
        )
        pred = rates[TRAIN_END:]
        out["oracle"] = dict(nll=_poisson_nll(obs, pred), mae=float(np.abs(obs - pred).mean()))
    return out


# ===========================================================================
# Task 2 -- ranking ladder (uniform pools, uniform training events)
# ===========================================================================
def _sub_events(events, lo, hi, cap, seed):
    ev = events[(events[:, 0] >= lo) & (events[:, 0] < hi)]
    if len(ev) > cap:
        rng = np.random.default_rng(seed)
        ev = ev[np.sort(rng.choice(len(ev), cap, replace=False))]
    return ev


def ranking_ladder(ds, gt, has_oracle, cooldown_weeks):
    Z = ds.features
    fs = ds.firm_sector
    act = ds.active
    cnt = ds.startup_counts
    T, N = cnt.shape

    # pre-filter to events whose winner is inside the eval pool (audit #18):
    # every fitter then trains/evaluates on exactly the same population.
    ev_all = ds.events
    in_pool = np.array([act[t, i] and fs[i] == s for t, s, i in ev_all])
    ev_all = ev_all[in_pool]
    ev_tr = _sub_events(ev_all, 0, TRAIN_END, MAX_TRAIN_EVENTS, seed=1)
    ev_te = _sub_events(ev_all, TRAIN_END, T_WEEKS, MAX_TEST_EVENTS, seed=2)
    cum = np.cumsum(cnt, 0)

    # label matrix restricted to the shared training events (audit critical #2)
    cnt_lab = np.zeros_like(cnt)
    np.add.at(cnt_lab, (ev_tr[:, 0], ev_tr[:, 2]), 1)

    # point-in-time recency (weeks since last deal; large if never funded)
    dealt = cnt > 0
    last = np.full(N, -(10**6))
    rec = np.zeros((T, N), float)
    for t in range(T):
        rec[t] = t - last
        last = np.where(dealt[t], t, last)

    # ---- fit the five model families on the SAME training events ----------
    t0 = time.time()
    w0, u, eta, _, _, _ = fit_ranker_fast(
        ev_tr,
        Z,
        fs,
        act,
        cnt,
        train_end=TRAIN_END,
        cooldown_weeks=cooldown_weeks,
        max_candidates=64,
        seed=0,
    )
    print(f"    [fit logit {time.time() - t0:.0f}s]", flush=True)
    hz = fit_discrete_hazard(
        ev_tr,
        Z,
        fs,
        act,
        cnt,
        train_end=TRAIN_END,
        cooldown_weeks=cooldown_weeks,
        negative_sampling_ratio=10,
        seed=0,
        max_iter=300,
        label_counts=cnt_lab,
    )
    print(f"    [fit hazard {time.time() - t0:.0f}s]", flush=True)
    sv = fit_startup_survival(
        ev_tr,
        Z,
        fs,
        act,
        cnt,
        tracked=None,
        train_end=TRAIN_END,
        cooldown_weeks=cooldown_weeks,
        max_iter=300,
    )
    print(f"    [fit survival {time.time() - t0:.0f}s]", flush=True)
    hks = {}
    for link in ("linear", "exp"):
        hks[link] = fit_hawkes_ranker(
            ev_tr,
            Z,
            fs,
            act,
            cnt,
            link=link,
            train_end=TRAIN_END,
            drop_last_funded=False,
            max_iter=300,
        )
        print(f"    [fit hawkes-{link} {time.time() - t0:.0f}s]", flush=True)
    hk_fields = {k: _fields_for(hks[k], cnt, fs, int(fs.max()) + 1) for k in hks}

    # ---- uniform per-event scoring -----------------------------------------
    # returns (rank_score, log_score): rank_score orders candidates; log_score is
    # the model's own log-probability scale (audit criticals #0/#3: Hawkes NLL is
    # lam/sum(lam), i.e. log-intensity, NOT softmax of intensities).
    def scores_for(model, t, s, cand):
        if t > 0:
            lo = max(0, t - cooldown_weeks)
            cd = ((cum[t - 1, cand] - (cum[lo - 1, cand] if lo > 0 else 0)) > 0).astype(float)
        else:
            cd = np.zeros(cand.size)
        if model == "popularity":
            sc = np.log1p(cum[t - 1, cand]) if t > 0 else np.zeros(cand.size)
            return sc, None
        if model == "recency":
            return -rec[t, cand], None
        if model == "logit":
            sc = Z[t, cand] @ (w0 + u[s]) + eta[s] * cd
            return sc, sc
        if model == "hazard":
            logits = discrete_hazard_scores(hz, Z[t, cand], cd, s)
            from scipy.special import expit

            h = expit(np.clip(logits, -40, 40))
            return logits, np.log(np.maximum(h, 1e-300))
        if model == "survival":
            sc = sv.candidate_score(Z[t, cand], cd, s)
            return sc, sc
        if model in ("hawkes_linear", "hawkes_exp"):
            link = "linear" if model == "hawkes_linear" else "exp"
            own, peer = hk_fields[link]
            r = hks[link]
            et = r.eta(Z[t, cand], peer[t, cand], own[t, cand], s)
            if link == "exp":
                return et, et  # log-intensity
            lam, _ = _link(et, "linear")
            return lam, np.log(np.maximum(lam, 1e-300))
        if model == "oracle":
            w_true = np.asarray(gt["ranker_global"]) + np.asarray(gt["ranker_sector_dev"])[s]
            sc = Z[t, cand] @ w_true + float(gt["ranker_cooldown"][s]) * cd
            return sc, sc
        raise ValueError(model)

    models = [
        "popularity",
        "recency",
        "logit",
        "hazard",
        "survival",
        "hawkes_linear",
        "hawkes_exp",
    ] + (["oracle"] if has_oracle else [])
    ranks = {m: [] for m in models}
    nlls = {m: 0.0 for m in models}
    sizes = []
    used = 0
    for t, s, i in ev_te:
        cand = np.flatnonzero(act[t] & (fs == s))
        pos_arr = np.flatnonzero(cand == i)
        if cand.size == 0 or pos_arr.size == 0:
            continue
        used += 1
        pos = int(pos_arr[0])
        sizes.append(cand.size)
        for m in models:
            sc, logsc = scores_for(m, t, s, cand)
            sc = np.asarray(sc, float)
            # mid-rank tie convention (audit majors #4/#6): identical for all models
            strict = int((sc > sc[pos]).sum())
            ties = int((sc == sc[pos]).sum()) - 1
            ranks[m].append(strict + 1 + ties / 2.0)
            if logsc is not None:
                logsc = np.asarray(logsc, float)
                z = logsc - logsc.max()
                p = np.exp(z)
                nlls[m] -= float(np.log(max(p[pos] / p.sum(), 1e-12)))

    sizes = np.asarray(sizes, float)
    out = {
        "n_events": used,
        "mean_risk_set": float(act[TRAIN_END:].sum(1).mean()),
        "random": dict(
            top1=float(np.mean(np.minimum(1 / sizes, 1))),
            top5=float(np.mean(np.minimum(5 / sizes, 1))),
            top10=float(np.mean(np.minimum(10 / sizes, 1))),
            mrr=float(np.mean([np.mean(1 / np.arange(1, int(n) + 1)) for n in sizes])),
            nll=float(np.mean(np.log(sizes))),
        ),
    }
    for m in models:
        r = np.asarray(ranks[m])
        out[m] = dict(
            top1=float(np.mean(r <= 1)),
            top5=float(np.mean(r <= 5)),
            top10=float(np.mean(r <= 10)),
            mrr=float(np.mean(1 / r)),
        )
        if m not in ("popularity", "recency"):
            out[m]["nll"] = nlls[m] / used
    return out


# ===========================================================================
def run_dataset(path):
    name = os.path.basename(path)
    gt_path = os.path.join(path, "ground_truth.npz")
    gt = np.load(gt_path, allow_pickle=True)
    is_a = str(np.asarray(gt["regime"])) == "A" if "regime" in gt.files else "A" in name
    ds = load_dataset(path, active="true", ground_truth=gt_path)
    cw = int(gt["cooldown_weeks"]) if "cooldown_weeks" in gt.files else 26
    print(f"  [{name}] counts ladder...", flush=True)
    c = counts_ladder(ds, gt, has_oracle=is_a)
    print(f"  [{name}] ranking ladder...", flush=True)
    r = ranking_ladder(ds, gt, has_oracle=is_a, cooldown_weeks=cw)
    return {"counts": c, "ranking": r}


def main(out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    res = {}
    for path in DATASETS:
        if not os.path.isdir(path):
            print(f"  !! missing {path} (generate it first) -- skipping")
            continue
        res[os.path.basename(path)] = run_dataset(path)
    with open(os.path.join(out_dir, "exp25_model_ladder.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=float)

    for name, r in res.items():
        print(f"\n=== {name} ===")
        print("  COUNTS (held-out Poisson NLL/cell | MAE):")
        for m, v in r["counts"].items():
            print(f"    {m:14s} {v['nll']:7.3f} | {v['mae']:.3f}")
        print(
            f"  RANKING ({r['ranking']['n_events']} test events, "
            f"~{r['ranking']['mean_risk_set']:.0f} candidates/pool):"
        )
        print("    model           top1    top5    top10   MRR     NLL")
        for m in (
            "random",
            "popularity",
            "recency",
            "logit",
            "hazard",
            "survival",
            "hawkes_linear",
            "hawkes_exp",
            "oracle",
        ):
            if m not in r["ranking"]:
                continue
            v = r["ranking"][m]
            nll = f"{v['nll']:.3f}" if "nll" in v else "  --"
            print(
                f"    {m:14s} {v['top1']:.3f}   {v['top5']:.3f}   {v['top10']:.3f}"
                f"   {v['mrr']:.3f}   {nll}"
            )
    print("\nWrote results/exp25_model_ladder.json")
    return res


if __name__ == "__main__":
    main()
