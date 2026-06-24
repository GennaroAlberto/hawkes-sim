r"""
Experiment 16 -- survival-analysis-with-ranking for "who is funded next".

A runnable counterpart to the Hawkes approach: instead of modelling the event
intensity, model the **time to the next funding event** per firm with a
discrete-time competing-risks survival model, and rank firms by their predicted
probability of funding within a horizon.

Pipeline (all numpy):
  1. simulate a population of firms with covariates; each generates funding events
     (Weibull inter-event gaps -> increasing "maturity" hazard) and may be removed
     by a competing acquisition risk;
  2. build a survival training set by sampling decision-time *origins* and, for each
     at-risk firm, recording (covariates, time-to-next-funding or censoring, cause);
  3. fit a non-parametric **Kaplan-Meier** curve (population baseline) and a
     **discrete-time logistic-hazard** model (DeepHit-style) trained on the survival
     likelihood, optionally + a pairwise **ranking** loss;
  4. evaluate ranking (concordance index, Recall@k), discrimination over a baseline,
     and calibration against Kaplan-Meier.

Outputs: results/exp16_survival.png and results/exp16_survival.json.
Run:  PYTHONPATH=. python -m experiments.exp16_survival
"""

import os
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hawkes_calibration.operators.nn import MLP


# ===========================================================================
# 1. Simulate a firm population with covariate-driven, maturing funding hazard.
# ===========================================================================
def simulate_population(n_firms=500, T=60.0, seed=0):
    rng = np.random.default_rng(seed)
    pop = rng.normal(0, 1, n_firms)                     # popularity score
    size = rng.normal(0, 1, n_firms)                    # firm size
    sector = rng.integers(0, 3, n_firms)                # sector in {0,1,2}
    sec_oh = np.eye(3)[sector]
    # linear predictor for the funding rate (higher -> shorter gaps)
    beta = dict(pop=0.7, size=0.3, sec=np.array([0.0, 0.4, -0.3]))
    lin = beta["pop"] * pop + beta["size"] * size + sec_oh @ beta["sec"]
    k_shape = 1.6                                        # Weibull shape >1 -> hazard rises with time-since-last (maturity)
    scale = np.exp(-(lin) / k_shape) * 6.0              # base scale in time units
    events = []
    acq_time = np.where(rng.uniform(size=n_firms) < 0.25,   # 25% get acquired (competing risk)
                        rng.uniform(5, T, n_firms), np.inf)
    for i in range(n_firms):
        t, evs = 0.0, []
        while True:
            gap = scale[i] * rng.weibull(k_shape)
            t += gap
            if t > T or t > acq_time[i]:
                break
            evs.append(t)
        events.append(np.array(evs))
    X_static = np.column_stack([pop, size, sec_oh])      # (n,5)
    return dict(events=events, X_static=X_static, acq_time=acq_time, T=T,
                pop=pop, sector=sector, lin=lin)


# ===========================================================================
# 2. Build the survival training set by sampling decision-time origins.
# ===========================================================================
def build_dataset(sim, origins_per_firm=4, horizon=12.0, seed=0):
    rng = np.random.default_rng(seed)
    T = sim["T"]; X_static = sim["X_static"]; events = sim["events"]; acq = sim["acq_time"]
    rows_X, rows_t, rows_e = [], [], []
    for i in range(len(events)):
        ev = events[i]
        for _ in range(origins_per_firm):
            tau = rng.uniform(0, T - 1.0)
            past = ev[ev < tau]
            last = past[-1] if past.size else -1.0
            recency = tau - last if last >= 0 else tau   # time since last round (maturity feature)
            future = ev[ev > tau]
            t_next = future[0] - tau if future.size else np.inf
            t_acq = acq[i] - tau if acq[i] > tau else np.inf
            # observed time = min(next funding, acquisition, horizon)
            t_obs = min(t_next, t_acq, horizon)
            event = 1 if (t_next <= horizon and t_next <= t_acq) else 0   # funding observed?
            feat = np.concatenate([X_static[i], [np.log1p(recency)], [np.log1p(past.size)]])
            rows_X.append(feat); rows_t.append(t_obs); rows_e.append(event)
    return np.array(rows_X), np.array(rows_t), np.array(rows_e), horizon


# ===========================================================================
# 3a. Kaplan-Meier product-limit estimator.
# ===========================================================================
def kaplan_meier(t, e, grid):
    """Product-limit estimate S(g) = prod_{t_i <= g} (1 - d_i / n_i) on a time grid."""
    uniq = np.unique(t[e == 1])
    d_arr = np.array([np.sum((t == ti) & (e == 1)) for ti in uniq])
    n_arr = np.array([np.sum(t >= ti) for ti in uniq])
    factors = np.where(n_arr > 0, 1.0 - d_arr / n_arr, 1.0)
    S = [np.prod(factors[uniq <= g]) if np.any(uniq <= g) else 1.0 for g in grid]
    return np.array(S)


# ===========================================================================
# 3b. Discrete-time logistic-hazard model (DeepHit-style), numpy MLP.
# ===========================================================================
class DiscreteHazard:
    def __init__(self, n_features, n_bins, horizon, hidden=48, seed=0):
        self.K = n_bins
        self.edges = np.linspace(0, horizon, n_bins + 1)
        self.net = MLP([n_features, hidden, hidden, n_bins], seed=seed)
        self.x_mean = self.x_std = None

    def _targets(self, t, e):
        """masked binary targets: y_{j}=1 at the event bin (if uncensored); mask j<=obs bin."""
        b = np.clip(np.digitize(t, self.edges) - 1, 0, self.K - 1)
        Y = np.zeros((len(t), self.K)); Mk = np.zeros((len(t), self.K))
        for i in range(len(t)):
            Mk[i, : b[i] + 1] = 1.0
            if e[i] == 1:
                Y[i, b[i]] = 1.0
        return Y, Mk

    def fit(self, X, t, e, epochs=500, lr=3e-3, rank_weight=0.0, sigma=0.3, seed=0):
        X = np.asarray(X, float)
        self.x_mean, self.x_std = X.mean(0), X.std(0) + 1e-8
        Xs = (X - self.x_mean) / self.x_std
        Y, Mk = self._targets(t, e)
        rng = np.random.default_rng(seed)
        ev_idx = np.where(e == 1)[0]
        for ep in range(epochs):
            o = self.net.forward(Xs)
            h = 1.0 / (1.0 + np.exp(-o))                 # hazards (N,K)
            # survival NLL gradient = masked (h - Y)
            g = Mk * (h - Y) / max(1, Mk.sum())
            # optional pairwise ranking loss on risk R = 1 - prod(1-h)
            if rank_weight > 0 and ev_idx.size > 4:
                R = 1.0 - np.prod(1.0 - h, axis=1)       # (N,) funding-within-horizon risk
                ii = rng.choice(ev_idx, size=min(256, ev_idx.size), replace=False)
                jj = rng.integers(0, len(t), size=ii.size)
                comp = t[ii] < t[jj]                     # i had the event earlier than j's obs time
                ii, jj = ii[comp], jj[comp]
                if ii.size:
                    d = R[ii] - R[jj]; w = np.exp(-d / sigma)
                    dR = np.zeros(len(t))
                    np.add.at(dR, ii, -w / sigma * rank_weight / ii.size)
                    np.add.at(dR, jj, +w / sigma * rank_weight / ii.size)
                    # chain to logits: dR/do_k = (1-R) h_k
                    g = g + (dR[:, None] * (1.0 - R)[:, None]) * h
            self.net.backward(g); self.net.adam_step(lr)
        return self

    def hazards(self, X):
        Xs = (np.atleast_2d(X) - self.x_mean) / self.x_std
        return 1.0 / (1.0 + np.exp(-self.net.forward(Xs)))

    def survival_curve(self, X):
        h = self.hazards(X)
        return np.cumprod(1.0 - h, axis=1)               # S at each bin edge (N,K)

    def risk(self, X):
        return 1.0 - np.prod(1.0 - self.hazards(X), axis=1)   # funding-within-horizon probability


# ===========================================================================
# 4. Metrics.
# ===========================================================================
def concordance_index(risk, t, e):
    """Harrell's C with right censoring: over comparable pairs (i earlier event, j later),
    fraction with higher predicted risk for i."""
    n = len(t); conc = tie = comp = 0.0
    ev = np.where(e == 1)[0]
    for i in ev:
        later = t > t[i]
        c = np.sum(later)
        if c == 0:
            continue
        ri = risk[i]; rj = risk[later]
        conc += np.sum(ri > rj); tie += np.sum(ri == rj); comp += c
    return (conc + 0.5 * tie) / max(comp, 1)


def recall_at_k(risk, t, e, horizon, k_frac=0.1):
    """Of the firms that actually fund within the horizon, the fraction captured in
    the top-k-fraction of the risk ranking."""
    funded = (e == 1)
    order = np.argsort(-risk)
    k = max(1, int(k_frac * len(risk)))
    topk = np.zeros(len(risk), bool); topk[order[:k]] = True
    if funded.sum() == 0:
        return float("nan")
    return float(np.sum(topk & funded) / funded.sum())


# ===========================================================================
# Driver.
# ===========================================================================
def run(out_dir="results", seed=0):
    os.makedirs(out_dir, exist_ok=True)
    sim = simulate_population(n_firms=600, T=60.0, seed=seed)
    X, t, e, horizon = build_dataset(sim, origins_per_firm=5, horizon=5.0, seed=seed)
    n = len(t); idx = np.random.default_rng(0).permutation(n)
    tr, te = idx[: int(0.7 * n)], idx[int(0.7 * n):]

    grid = np.linspace(0, horizon, 25)
    km = kaplan_meier(t[tr], e[tr], grid)

    base = DiscreteHazard(X.shape[1], n_bins=12, horizon=horizon, seed=0).fit(
        X[tr], t[tr], e[tr], epochs=600, rank_weight=0.0)
    ranked = DiscreteHazard(X.shape[1], n_bins=12, horizon=horizon, seed=0).fit(
        X[tr], t[tr], e[tr], epochs=600, rank_weight=1.5, sigma=0.25)

    out = {}
    for name, mdl in [("likelihood", base), ("likelihood+ranking", ranked)]:
        r = mdl.risk(X[te])
        out[name] = dict(
            c_index=round(concordance_index(r, t[te], e[te]), 4),
            recall_at_10pct=round(recall_at_k(r, t[te], e[te], horizon, 0.10), 4),
            recall_at_20pct=round(recall_at_k(r, t[te], e[te], horizon, 0.20), 4),
        )
    # popularity baseline ranker (rank by the single 'popularity' covariate)
    pop_risk = X[te][:, 0]
    out["baseline_popularity"] = dict(
        c_index=round(concordance_index(pop_risk, t[te], e[te]), 4),
        recall_at_10pct=round(recall_at_k(pop_risk, t[te], e[te], horizon, 0.10), 4))
    out["event_rate"] = round(float(e.mean()), 3)
    out["n_examples"] = int(n)

    _plot(out_dir, sim, X, t, e, te, grid, km, base, ranked, out)
    with open(os.path.join(out_dir, "exp16_survival.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("=== Experiment 16: survival-with-ranking ===")
    print("  examples=%d  funding-within-horizon rate=%.2f" % (n, out["event_rate"]))
    for k in ("likelihood", "likelihood+ranking", "baseline_popularity"):
        print("  %-22s c-index=%.3f  recall@10%%=%.3f"
              % (k, out[k]["c_index"], out[k]["recall_at_10pct"]))
    print("\nWrote results/exp16_survival.{png,json}")
    return out


def _plot(out_dir, sim, X, t, e, te, grid, km, base, ranked, out):
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))

    a = ax[0, 0]
    a.step(grid, km, where="post", color="k", lw=2, label="Kaplan-Meier (population)")
    # model survival stratified by popularity tertiles
    pop = X[te][:, 0]
    lo, hi = np.percentile(pop, [33, 66])
    for lab, mask, c in [("low pop", pop < lo, "C0"), ("mid", (pop >= lo) & (pop < hi), "C2"),
                         ("high pop", pop >= hi, "C3")]:
        Smean = base.survival_curve(X[te][mask]).mean(0)
        a.step(base.edges[1:], Smean, where="post", color=c, ls="--", label="model: " + lab)
    a.set_title("(a) survival curves: model strata vs Kaplan-Meier")
    a.set_xlabel("time since decision (to next funding)"); a.set_ylabel("S(t) = P(not yet funded)")
    a.legend(fontsize=7)

    a = ax[0, 1]
    names = ["popularity\nbaseline", "survival\nlikelihood", "survival\n+ranking"]
    cidx = [out["baseline_popularity"]["c_index"], out["likelihood"]["c_index"],
            out["likelihood+ranking"]["c_index"]]
    a.bar(names, cidx, color=["0.6", "C0", "C2"])
    a.axhline(0.5, color="r", ls="--", label="random (0.5)")
    a.set_ylim(0.5, max(cidx) + 0.05); a.set_title("(b) concordance index (ranking quality)")
    a.legend()

    a = ax[1, 0]
    r = ranked.risk(X[te])
    a.hist([r[e[te] == 1], r[e[te] == 0]], bins=20, stacked=True,
           color=["C2", "0.8"], label=["funded within horizon", "not / censored"])
    a.set_title("(c) predicted funding risk, by actual outcome")
    a.set_xlabel("predicted P(funded within horizon)"); a.set_ylabel("count"); a.legend(fontsize=8)

    a = ax[1, 1]
    fr = [0.05, 0.1, 0.15, 0.2, 0.3]
    rk_l = [recall_at_k(base.risk(X[te]), t[te], e[te], 12.0, f) for f in fr]
    rk_r = [recall_at_k(ranked.risk(X[te]), t[te], e[te], 12.0, f) for f in fr]
    rk_p = [recall_at_k(X[te][:, 0], t[te], e[te], 12.0, f) for f in fr]
    a.plot(fr, rk_p, "o--", color="0.6", label="popularity baseline")
    a.plot(fr, rk_l, "s-", color="C0", label="survival likelihood")
    a.plot(fr, rk_r, "^-", color="C2", label="survival+ranking")
    a.plot(fr, fr, "k:", label="random")
    a.set_title("(d) Recall@k: caught funders vs watch-list size")
    a.set_xlabel("watch-list fraction k"); a.set_ylabel("recall (funders captured)"); a.legend(fontsize=8)

    fig.suptitle("Survival-with-ranking for next-funding prediction (KM + discrete-hazard DeepHit-style)")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "exp16_survival.png"), dpi=140); plt.close(fig)


if __name__ == "__main__":
    run()
