"""
Within-sector survival model with an *outside option* for "which startup is funded
next".

This replaces the plain conditional-logit ranker (``sector_ranker``) with a
discrete-time competing-risks survival model that matches the data setting of a
positives-only deals feed:

* The candidates are the **tracked** startups that are live in a sector at a given
  week (a watch-list / portfolio), each with a hazard of receiving the next round.
* An explicit **outside alternative** -- "a firm not in our dataset" -- absorbs the
  events that go to companies we do not track (new entrants, off-list firms). Its
  hazard depends on sector-level features only, since the outside firm has no
  company features.

Conditional on a sector funding event (supplied by the sector count model), the
recipient is drawn from ``{tracked candidates} u {outside}`` by a softmax of
log-hazards. The negative log-likelihood is the Cox/McFadden partial likelihood over
the risk set, extended with the outside risk; minimising it is maximum partial
likelihood. The estimator is numpy-only (SciPy used if available).

Public API
----------
``fit_startup_survival`` -- fit the model.
``survival_predict_proba`` -- per-event probabilities over candidates + outside.
``evaluate_survival`` -- held-out NLL, top-k, MRR, and outside calibration / AUC.
``StartupSurvivalResult`` -- fitted parameters.
``make_tracked_mask`` -- helper to define a tracked universe (watch-list).
``sector_week_counts`` -- per-sector weekly funding counts from startup counts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # SciPy is optional.
    from scipy.optimize import minimize
    from scipy.special import logsumexp
except ImportError:  # pragma: no cover
    from ._opt_fallback import logsumexp, minimize

from .sector_ranker import candidate_set, cooldown_vector


@dataclass
class StartupSurvivalResult:
    global_weights: np.ndarray  # (p,)   firm-feature weights w0
    sector_deviations: np.ndarray  # (M, p) sector-specific deviations u_s
    cooldown_coef: np.ndarray  # (M,)   recency / cooldown coefficient eta_s (<= 0)
    outside_intercept: float  # b0
    outside_weights: np.ndarray  # (g,)   sector-feature weights for the outside option
    cooldown_weeks: int
    loss: float
    success: bool
    message: str

    def candidate_score(self, features, cooldown, sector: int):
        w = self.global_weights + self.sector_deviations[int(sector)]
        return np.asarray(features) @ w + self.cooldown_coef[int(sector)] * np.asarray(cooldown)

    def outside_score(self, sector_feat):
        return float(self.outside_intercept + self.outside_weights @ np.asarray(sector_feat))


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------
def sector_week_counts(startup_counts, startup_sector, n_sectors=None):
    """Aggregate startup funding counts into per-sector weekly counts (T, M)."""
    startup_counts = np.asarray(startup_counts, dtype=float)
    startup_sector = np.asarray(startup_sector, dtype=int)
    T = startup_counts.shape[0]
    M = int(n_sectors) if n_sectors is not None else int(startup_sector.max()) + 1
    out = np.zeros((T, M))
    for s in range(M):
        cols = np.flatnonzero(startup_sector == s)
        if cols.size:
            out[:, s] = startup_counts[:, cols].sum(axis=1)
    return out


def make_tracked_mask(active, fraction=0.7, seed=0):
    """A simple watch-list: a random ``fraction`` of startups are tracked (in the
    dataset/universe); the rest are 'outside'. Returns a boolean (N,) mask."""
    N = np.asarray(active).shape[1]
    rng = np.random.default_rng(seed)
    return rng.random(N) < float(fraction)


def _sector_features(secw, week, sector, cand_size, mom_weeks):
    """Outside-option features: [log(1+risk-set size), recent sector funding rate]."""
    t = int(week)
    lo = max(0, t - int(mom_weeks))
    momentum = secw[lo:t, int(sector)].mean() if t > lo else 0.0
    return np.array([np.log1p(cand_size), momentum], dtype=float)


def _prepare(
    events,
    Z,
    startup_sector,
    active,
    startup_counts,
    tracked,
    secw,
    train_end,
    cooldown_weeks,
    momentum_weeks,
):
    """Per training event: (sector, candidate features, cooldown, outside features,
    chosen position) with position 0 reserved for the outside option."""
    p = Z.shape[2]
    prepared = []
    for t, s, chosen in events[events[:, 0] < train_end]:
        cand = candidate_set(startup_sector, active & tracked[None, :], int(s), int(t))
        g = _sector_features(secw, int(t), int(s), cand.size, momentum_weeks)
        if int(chosen) in set(cand.tolist()):
            pos = int(np.where(cand == int(chosen))[0][0]) + 1
        else:
            pos = 0  # outside event
        if cand.size == 0 and pos != 0:
            continue
        cd = (
            cooldown_vector(startup_counts, int(t), cand, cooldown_weeks)
            if cand.size
            else np.zeros(0)
        )
        F = Z[int(t), cand] if cand.size else np.zeros((0, p))
        prepared.append((int(s), F, cd, g, pos))
    return prepared


def _unpack(theta, dims):
    p, M, g_dim = dims
    n_u = M * p
    base = p + n_u + M
    w0 = theta[:p]
    u = theta[p : p + n_u].reshape(M, p) if p else np.zeros((M, 0))
    eta = theta[p + n_u : p + n_u + M]
    b0 = theta[base]
    gamma = theta[base + 1 : base + 1 + g_dim]
    return w0, u, eta, b0, gamma


def _objective(theta, prepared, dims, l2_global, l2_sector, l2_outside):
    """Partial-likelihood loss + analytic gradient (softmax over candidates + outside)."""
    w0, u, eta, b0, gamma = _unpack(theta, dims)
    loss = 0.0
    gw0 = np.zeros_like(w0)
    gu = np.zeros_like(u)
    geta = np.zeros_like(eta)
    gb0 = 0.0
    ggamma = np.zeros_like(gamma)
    for s, F, cd, g, pos in prepared:
        q0 = b0 + gamma @ g
        if F.shape[0]:
            full = np.concatenate([[q0], F @ (w0 + u[s]) + eta[s] * cd])
        else:
            full = np.array([q0])
        lse = logsumexp(full)
        loss += lse - full[pos]
        d = np.exp(full - lse)
        d[pos] -= 1.0  # dNLL/d(full scores)
        gb0 += d[0]
        ggamma += d[0] * g
        if F.shape[0]:
            dc = d[1:]
            gw0 += dc @ F
            gu[s] += dc @ F
            geta[s] += float(dc @ cd)
    loss += 0.5 * (
        l2_global * np.sum(w0**2)
        + l2_sector * (np.sum(u**2) + np.sum(eta**2))
        + l2_outside * np.sum(gamma**2)
    )
    gw0 += l2_global * w0
    gu += l2_sector * u
    geta += l2_sector * eta
    ggamma += l2_outside * gamma
    grad = np.concatenate([gw0, gu.ravel(), geta, [gb0], ggamma])
    return float(loss), grad


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------
def fit_startup_survival(
    events,
    startup_features,
    startup_sector,
    active,
    startup_counts,
    *,
    tracked=None,
    train_end=None,
    cooldown_weeks=26,
    momentum_weeks=8,
    l2_global=1e-3,
    l2_sector=1e-2,
    l2_outside=1e-2,
    constrain_cooldown_negative=True,
    max_iter=500,
):
    r"""Fit the within-sector competing-risks survival model with an outside option.

    For an observed funding event ``(t, s, i*)`` the recipient is the tracked firm
    ``i*`` if it is in the live risk set, otherwise the *outside* alternative. With
    log-hazards ``q_{j,t} = (w0+u_s)^T Z_{j,t} + eta_s c^{(K)}_{j,t}`` for tracked
    candidates and ``q_0 = b0 + gamma^T g_{s,t}`` for the outside option, the
    contribution is the partial-likelihood term
    ``log( exp(q_0) + sum_{j in R} exp(q_{j,t}) ) - q_{chosen}``.
    """
    events = np.asarray(events, dtype=int)
    Z = np.asarray(startup_features, dtype=float)
    startup_sector = np.asarray(startup_sector, dtype=int)
    active = np.asarray(active, dtype=bool)
    startup_counts = np.asarray(startup_counts, dtype=float)
    T, N, p = Z.shape
    M = int(startup_sector.max()) + 1
    if tracked is None:
        tracked = np.ones(N, dtype=bool)
    tracked = np.asarray(tracked, dtype=bool)
    if train_end is None:
        train_end = T
    train_end = int(train_end)
    secw = sector_week_counts(startup_counts, startup_sector, M)
    g_dim = 2

    prepared = _prepare(
        events,
        Z,
        startup_sector,
        active,
        startup_counts,
        tracked,
        secw,
        train_end,
        cooldown_weeks,
        momentum_weeks,
    )
    if not prepared:
        raise ValueError("no usable training events")

    n_u = M * p
    base = p + n_u + M  # end of (w0, u, eta) block
    n = base + 1 + g_dim  # + outside intercept + outside weights
    x0 = np.zeros(n)
    x0[p + n_u : p + n_u + M] = -0.5  # cooldown start (negative)
    x0[base] = -1.0  # outside intercept (rarely chosen a priori)

    bounds = [(None, None)] * n
    if constrain_cooldown_negative:
        for j in range(M):
            bounds[p + n_u + j] = (None, 0.0)

    dims = (p, M, g_dim)
    opt = minimize(
        lambda th: _objective(th, prepared, dims, l2_global, l2_sector, l2_outside),
        x0,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": int(max_iter), "ftol": 1e-9},
    )
    w0, u, eta, b0, gamma = _unpack(opt.x, dims)
    return StartupSurvivalResult(
        global_weights=w0,
        sector_deviations=u,
        cooldown_coef=eta,
        outside_intercept=float(b0),
        outside_weights=gamma,
        cooldown_weeks=int(cooldown_weeks),
        loss=float(opt.fun),
        success=bool(opt.success),
        message=str(opt.message),
    )


# ---------------------------------------------------------------------------
# Predict / evaluate
# ---------------------------------------------------------------------------
def survival_predict_proba(
    result,
    startup_features,
    startup_sector,
    active,
    startup_counts,
    *,
    week,
    sector,
    tracked=None,
    momentum_weeks=8,
    n_sectors=None,
):
    """Return ``(cand_indices, p_candidates, p_outside)`` for one sector event."""
    Z = np.asarray(startup_features, dtype=float)
    startup_sector = np.asarray(startup_sector, dtype=int)
    active = np.asarray(active, dtype=bool)
    N = Z.shape[1]
    if tracked is None:
        tracked = np.ones(N, dtype=bool)
    tracked = np.asarray(tracked, dtype=bool)
    M = int(n_sectors) if n_sectors is not None else int(startup_sector.max()) + 1
    secw = sector_week_counts(startup_counts, startup_sector, M)
    cand = candidate_set(startup_sector, active & tracked[None, :], int(sector), int(week))
    g = _sector_features(secw, int(week), int(sector), cand.size, momentum_weeks)
    q0 = result.outside_score(g)
    if cand.size:
        cd = cooldown_vector(startup_counts, int(week), cand, result.cooldown_weeks)
        full = np.concatenate([[q0], result.candidate_score(Z[int(week), cand], cd, int(sector))])
    else:
        full = np.array([q0])
    ex = np.exp(full - full.max())
    pr = ex / ex.sum()
    return cand, pr[1:], float(pr[0])


def evaluate_survival(
    result,
    events,
    startup_features,
    startup_sector,
    active,
    startup_counts,
    *,
    tracked=None,
    start_week=0,
    end_week=None,
    momentum_weeks=8,
    topk=(1, 5, 10),
):
    """Held-out metrics: NLL, top-k / MRR over tracked candidates (for in-universe
    events), and the outside option's calibration and discrimination (AUC) for the
    'next funder is outside the tracked set' event."""
    events = np.asarray(events, dtype=int)
    Z = np.asarray(startup_features, dtype=float)
    N = Z.shape[1]
    if tracked is None:
        tracked = np.ones(N, dtype=bool)
    tracked = np.asarray(tracked, dtype=bool)
    if end_week is None:
        end_week = Z.shape[0]
    use = events[(events[:, 0] >= int(start_week)) & (events[:, 0] < int(end_week))]
    nll = mrr = 0.0
    hits = {int(k): 0 for k in topk}
    n_in = 0
    p_out_list, y_out_list = [], []
    for t, s, chosen in use:
        cand, pr, p_out = survival_predict_proba(
            result,
            startup_features,
            startup_sector,
            active,
            startup_counts,
            week=int(t),
            sector=int(s),
            tracked=tracked,
            momentum_weeks=momentum_weeks,
        )
        is_outside = int(chosen) not in set(cand.tolist())
        p_out_list.append(p_out)
        y_out_list.append(1.0 if is_outside else 0.0)
        if is_outside:
            nll -= np.log(max(p_out, 1e-12))
            continue
        pos = int(np.where(cand == int(chosen))[0][0])
        nll -= np.log(max(pr[pos], 1e-12))
        rank = int(np.where(np.argsort(-pr) == pos)[0][0]) + 1
        mrr += 1.0 / rank
        for k in topk:
            hits[int(k)] += int(rank <= int(k))
        n_in += 1
    p_out_arr = np.asarray(p_out_list)
    y_out_arr = np.asarray(y_out_list)
    out = {
        "n_events": int(use.shape[0]),
        "n_in_universe": int(n_in),
        "outside_rate": float(y_out_arr.mean()) if y_out_arr.size else float("nan"),
        "outside_pred_mean": float(p_out_arr.mean()) if p_out_arr.size else float("nan"),
        "outside_auc": _auc(y_out_arr, p_out_arr),
        "nll": float(nll / max(use.shape[0], 1)),
        "mrr": float(mrr / max(n_in, 1)),
    }
    for k in topk:
        out[f"top{int(k)}"] = float(hits[int(k)] / max(n_in, 1))
    return out


def _auc(y, score):
    """Area under the ROC curve (Mann-Whitney form); nan if one class is absent."""
    y = np.asarray(y)
    score = np.asarray(score)
    pos = score[y == 1]
    neg = score[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty(score.size)
    ranks[order] = np.arange(1, score.size + 1)
    r_pos = ranks[y == 1].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


__all__ = [
    "StartupSurvivalResult",
    "fit_startup_survival",
    "survival_predict_proba",
    "evaluate_survival",
    "make_tracked_mask",
    "sector_week_counts",
]
