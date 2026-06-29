"""
Survival-style alternatives for the startup-level second stage.

The sector model answers *when/where* funding events happen.  This module offers
survival/risk-set models for the conditional question: given a sector event at
week t, which active startup in that sector is most likely to receive it?

Two no-new-dependency alternatives are implemented:

1. Cox risk-set survival stage.  This is mathematically the Cox partial
   likelihood stratified by sector-week risk sets.  It is equivalent to a
   conditional softmax over the live candidates, but the survival interpretation
   is explicit.

2. Discrete-time logistic hazard stage.  This uses positive and sampled negative
   firm-weeks, so it can learn from weeks in which an active firm did not raise.
   It is a practical baseline for weekly PitchBook-style data.

DeepSurv/DeepHit can be added later behind optional torch/pycox dependencies, but
these two baselines keep the core package numpy/scipy-only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

from .sector_ranker import candidate_set, cooldown_vector


@dataclass
class CoxSurvivalStageResult:
    """Cox partial-likelihood startup selector.

    The score is

    ``q[i,t] = (w0 + u[sector])' Z[i,t] + eta[sector] * cooldown[i,t]``.

    Baseline hazards are not estimated because they cancel in the sector-week
    partial likelihood and are unnecessary for ranking within a known sector
    event.
    """

    global_weights: np.ndarray
    sector_deviations: np.ndarray
    cooldown_coef: np.ndarray
    cooldown_weeks: int
    loss: float
    success: bool
    message: str

    def weights_for_sector(self, sector: int) -> np.ndarray:
        return self.global_weights + self.sector_deviations[int(sector)]


@dataclass
class DiscreteHazardStageResult:
    """Discrete-time logistic hazard startup selector.

    The per-firm weekly hazard is

    ``logit h[i,t] = b0 + b_sector[sector]
                    + (w0 + u[sector])' Z[i,t]
                    + eta[sector] * cooldown[i,t]``.

    For conditional sector-event ranking, hazards are normalized over the live
    risk set of the sector.
    """

    intercept: float
    sector_intercept: np.ndarray
    global_weights: np.ndarray
    sector_deviations: np.ndarray
    cooldown_coef: np.ndarray
    cooldown_weeks: int
    loss: float
    success: bool
    message: str
    negative_sampling_ratio: int | None

    def weights_for_sector(self, sector: int) -> np.ndarray:
        return self.global_weights + self.sector_deviations[int(sector)]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _softmax(scores):
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return np.zeros(0)
    z = scores - np.max(scores)
    p = np.exp(z)
    s = p.sum()
    if not np.isfinite(s) or s <= 0:
        return np.full(scores.size, 1.0 / scores.size)
    return p / s


def _n_sectors_from_startups(startup_sector):
    startup_sector = np.asarray(startup_sector, dtype=int)
    return int(startup_sector.max()) + 1


def _event_subset(events, start_week, end_week):
    events = np.asarray(events, dtype=int)
    if events.size == 0:
        return events.reshape(0, 3)
    mask = np.ones(events.shape[0], dtype=bool)
    if start_week is not None:
        mask &= events[:, 0] >= int(start_week)
    if end_week is not None:
        mask &= events[:, 0] < int(end_week)
    return events[mask]


# ---------------------------------------------------------------------------
# Cox risk-set survival stage
# ---------------------------------------------------------------------------
def fit_cox_survival_stage(
    events,
    startup_features,
    startup_sector,
    active,
    startup_counts,
    *,
    train_end=None,
    start_week=0,
    cooldown_weeks=26,
    l2_global=1e-3,
    l2_sector=1e-2,
    constrain_cooldown_negative=True,
    max_iter=500,
):
    """Fit a sector-stratified Cox partial-likelihood selector.

    This uses only observed positive funding events, but each event contributes a
    full risk set denominator over all active startups in that event's sector.
    It is therefore appropriate when we trust the risk-set construction and want
    a survival interpretation without fabricating fixed negative labels.
    """
    events = np.asarray(events, dtype=int)
    Z = np.asarray(startup_features, dtype=float)
    startup_sector = np.asarray(startup_sector, dtype=int)
    active = np.asarray(active, dtype=bool)
    startup_counts = np.asarray(startup_counts, dtype=float)
    T, _, p = Z.shape
    M = _n_sectors_from_startups(startup_sector)
    if train_end is None:
        train_end = T
    train_events = _event_subset(events, start_week, train_end)
    if train_events.size == 0:
        raise ValueError("no training events in the requested window")

    prepared = []
    for t, s, chosen in train_events:
        cand = candidate_set(startup_sector, active, int(s), int(t))
        if cand.size == 0:
            continue
        pos = np.where(cand == int(chosen))[0]
        if pos.size == 0:
            continue
        cd = cooldown_vector(startup_counts, int(t), cand, cooldown_weeks)
        prepared.append((int(t), int(s), cand, int(pos[0]), cd))
    if not prepared:
        raise ValueError("no training events had a valid risk set containing the funded startup")

    n_w0 = p
    n_u = M * p
    n_eta = M
    n = n_w0 + n_u + n_eta
    x0 = np.zeros(n, dtype=float)
    x0[n_w0 + n_u:] = -0.5

    bounds = [(None, None)] * n
    if constrain_cooldown_negative:
        for j in range(M):
            bounds[n_w0 + n_u + j] = (None, 0.0)

    def unpack(theta):
        w0 = theta[:p]
        u = theta[p:p + n_u].reshape(M, p) if p else np.zeros((M, 0))
        eta = theta[p + n_u:p + n_u + M]
        return w0, u, eta

    def objective(theta):
        w0, u, eta = unpack(theta)
        loss = 0.0
        gw0 = np.zeros_like(w0)
        gu = np.zeros_like(u)
        geta = np.zeros_like(eta)
        for t, s, cand, chosen_pos, cd in prepared:
            F = Z[t, cand]
            scores = F @ (w0 + u[s]) + eta[s] * cd
            lse = logsumexp(scores)
            prob = np.exp(scores - lse)
            loss += float(lse - scores[chosen_pos])
            diff_f = prob @ F - F[chosen_pos]
            diff_cd = float(prob @ cd - cd[chosen_pos])
            gw0 += diff_f
            gu[s] += diff_f
            geta[s] += diff_cd

        loss += 0.5 * l2_global * np.sum(w0 ** 2)
        loss += 0.5 * l2_sector * (np.sum(u ** 2) + np.sum(eta ** 2))
        gw0 += l2_global * w0
        gu += l2_sector * u
        geta += l2_sector * eta
        grad = np.concatenate([gw0, gu.ravel(), geta])
        return loss, grad

    opt = minimize(
        lambda th: objective(th),
        x0,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": int(max_iter), "ftol": 1e-9},
    )
    w0, u, eta = unpack(opt.x)
    return CoxSurvivalStageResult(
        global_weights=w0,
        sector_deviations=u,
        cooldown_coef=eta,
        cooldown_weeks=int(cooldown_weeks),
        loss=float(opt.fun),
        success=bool(opt.success),
        message=str(opt.message),
    )


def cox_survival_predict_proba(
    result: CoxSurvivalStageResult,
    startup_features,
    startup_sector,
    active,
    startup_counts,
    *,
    week: int,
    sector: int,
):
    """Return candidate indices and conditional probabilities for a sector event."""
    Z = np.asarray(startup_features, dtype=float)
    cand = candidate_set(startup_sector, active, sector, week)
    if cand.size == 0:
        return cand, np.zeros(0)
    cd = cooldown_vector(startup_counts, int(week), cand, result.cooldown_weeks)
    scores = Z[int(week), cand] @ result.weights_for_sector(int(sector))
    scores = scores + result.cooldown_coef[int(sector)] * cd
    return cand, _softmax(scores)


# ---------------------------------------------------------------------------
# Discrete-time logistic hazard stage
# ---------------------------------------------------------------------------
def fit_discrete_hazard_stage(
    events,
    startup_features,
    startup_sector,
    active,
    startup_counts,
    *,
    train_end=None,
    start_week=0,
    cooldown_weeks=26,
    negative_sampling_ratio=20,
    seed=0,
    l2_global=1e-3,
    l2_sector=1e-2,
    constrain_cooldown_negative=True,
    max_iter=500,
):
    """Fit a discrete-time logistic hazard model from firm-week labels.

    Positive rows are active firm-weeks with a funding event.  Negative rows are
    active firm-weeks without a funding event.  To keep the design matrix small,
    negatives can be sampled at ``negative_sampling_ratio`` per positive; set it
    to ``None`` to use all active negative firm-weeks.
    """
    events = np.asarray(events, dtype=int)
    Z = np.asarray(startup_features, dtype=float)
    startup_sector = np.asarray(startup_sector, dtype=int)
    active = np.asarray(active, dtype=bool)
    startup_counts = np.asarray(startup_counts, dtype=float)
    T, N, p = Z.shape
    M = _n_sectors_from_startups(startup_sector)
    if train_end is None:
        train_end = T
    train_end = int(train_end)
    start_week = int(start_week)
    rng = np.random.default_rng(seed)

    rows_t = []
    rows_i = []
    rows_y = []

    for t in range(start_week, train_end):
        live = np.flatnonzero(active[t])
        if live.size == 0:
            continue
        labels = (startup_counts[t, live] > 0).astype(int)
        pos_idx = live[labels == 1]
        neg_idx = live[labels == 0]
        if pos_idx.size:
            rows_t.extend([t] * pos_idx.size)
            rows_i.extend(pos_idx.tolist())
            rows_y.extend([1] * pos_idx.size)
        if neg_idx.size:
            if negative_sampling_ratio is None:
                take = neg_idx
            else:
                n_pos_ref = max(1, int(pos_idx.size))
                n_take = min(neg_idx.size, int(negative_sampling_ratio) * n_pos_ref)
                take = rng.choice(neg_idx, size=n_take, replace=False)
            rows_t.extend([t] * len(take))
            rows_i.extend(np.asarray(take, dtype=int).tolist())
            rows_y.extend([0] * len(take))

    if not rows_y or np.sum(rows_y) == 0:
        raise ValueError("no positive training firm-weeks found")

    rows_t = np.asarray(rows_t, dtype=int)
    rows_i = np.asarray(rows_i, dtype=int)
    y = np.asarray(rows_y, dtype=float)
    sectors = startup_sector[rows_i]
    cd = np.zeros_like(y)
    for n, (t, i) in enumerate(zip(rows_t, rows_i)):
        cd[n] = cooldown_vector(startup_counts, int(t), np.array([int(i)]), cooldown_weeks)[0]
    F = Z[rows_t, rows_i]

    n_intercept = 1
    n_sector_intercept = M
    n_w0 = p
    n_u = M * p
    n_eta = M
    n = n_intercept + n_sector_intercept + n_w0 + n_u + n_eta
    eta_start = n_intercept + n_sector_intercept + n_w0 + n_u
    x0 = np.zeros(n, dtype=float)
    base_rate = np.clip(y.mean(), 1e-4, 1 - 1e-4)
    x0[0] = np.log(base_rate / (1.0 - base_rate))
    x0[eta_start:] = -0.5

    bounds = [(None, None)] * n
    if constrain_cooldown_negative:
        for j in range(M):
            bounds[eta_start + j] = (None, 0.0)

    def unpack(theta):
        k = 0
        intercept = float(theta[k]); k += 1
        sector_intercept = theta[k:k + M]; k += M
        w0 = theta[k:k + p]; k += p
        u = theta[k:k + M * p].reshape(M, p) if p else np.zeros((M, 0)); k += M * p
        eta = theta[k:k + M]
        return intercept, sector_intercept, w0, u, eta

    def objective(theta):
        intercept, sector_intercept, w0, u, eta = unpack(theta)
        linear = intercept + sector_intercept[sectors]
        linear += np.sum(F * (w0[None, :] + u[sectors]), axis=1)
        linear += eta[sectors] * cd
        prob = expit(np.clip(linear, -40.0, 40.0))
        loss = -float(np.sum(y * np.log(np.maximum(prob, 1e-12))
                            + (1.0 - y) * np.log(np.maximum(1.0 - prob, 1e-12))))
        err = prob - y

        g_intercept = np.array([err.sum()])
        g_sector_intercept = np.zeros(M)
        np.add.at(g_sector_intercept, sectors, err)
        g_w0 = err @ F
        g_u = np.zeros((M, p))
        for s in range(M):
            mask = sectors == s
            if np.any(mask):
                g_u[s] = err[mask] @ F[mask]
        g_eta = np.zeros(M)
        np.add.at(g_eta, sectors, err * cd)

        loss += 0.5 * l2_global * np.sum(w0 ** 2)
        loss += 0.5 * l2_sector * (np.sum(sector_intercept ** 2) + np.sum(u ** 2) + np.sum(eta ** 2))
        g_w0 += l2_global * w0
        g_sector_intercept += l2_sector * sector_intercept
        g_u += l2_sector * u
        g_eta += l2_sector * eta
        grad = np.concatenate([g_intercept, g_sector_intercept, g_w0, g_u.ravel(), g_eta])
        return loss, grad

    opt = minimize(
        lambda th: objective(th),
        x0,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": int(max_iter), "ftol": 1e-9},
    )
    intercept, sector_intercept, w0, u, eta = unpack(opt.x)
    return DiscreteHazardStageResult(
        intercept=intercept,
        sector_intercept=sector_intercept,
        global_weights=w0,
        sector_deviations=u,
        cooldown_coef=eta,
        cooldown_weeks=int(cooldown_weeks),
        loss=float(opt.fun),
        success=bool(opt.success),
        message=str(opt.message),
        negative_sampling_ratio=negative_sampling_ratio,
    )


def discrete_hazard_scores(
    result: DiscreteHazardStageResult,
    features,
    cooldown,
    sector: int,
):
    """Return logits for candidates under a fitted discrete hazard model."""
    s = int(sector)
    features = np.asarray(features, dtype=float)
    cooldown = np.asarray(cooldown, dtype=float)
    return (
        result.intercept
        + result.sector_intercept[s]
        + features @ result.weights_for_sector(s)
        + result.cooldown_coef[s] * cooldown
    )


def discrete_hazard_predict_proba(
    result: DiscreteHazardStageResult,
    startup_features,
    startup_sector,
    active,
    startup_counts,
    *,
    week: int,
    sector: int,
):
    """Return candidates and conditional probabilities for a sector event.

    The model first estimates weekly hazards and then normalizes those hazards over
    the sector risk set.  This is the conditional mark distribution given that a
    sector event occurred.
    """
    Z = np.asarray(startup_features, dtype=float)
    cand = candidate_set(startup_sector, active, sector, week)
    if cand.size == 0:
        return cand, np.zeros(0)
    cd = cooldown_vector(startup_counts, int(week), cand, result.cooldown_weeks)
    logits = discrete_hazard_scores(result, Z[int(week), cand], cd, int(sector))
    hazards = expit(np.clip(logits, -40.0, 40.0))
    if hazards.sum() <= 0 or not np.isfinite(hazards.sum()):
        return cand, np.full(cand.size, 1.0 / cand.size)
    return cand, hazards / hazards.sum()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_survival_stage(
    result,
    events,
    startup_features,
    startup_sector,
    active,
    startup_counts,
    *,
    start_week=0,
    end_week=None,
    topk=(1, 5, 10),
):
    """Evaluate a Cox or discrete-hazard stage on observed held-out events."""
    events = np.asarray(events, dtype=int)
    Z = np.asarray(startup_features, dtype=float)
    if end_week is None:
        end_week = Z.shape[0]
    use = _event_subset(events, start_week, end_week)
    if use.size == 0:
        return {"n_events": 0}

    if isinstance(result, CoxSurvivalStageResult):
        predict = cox_survival_predict_proba
    elif isinstance(result, DiscreteHazardStageResult):
        predict = discrete_hazard_predict_proba
    else:
        raise TypeError("result must be CoxSurvivalStageResult or DiscreteHazardStageResult")

    nll = 0.0
    mrr = 0.0
    hits = {int(k): 0 for k in topk}
    random_nll = 0.0
    random_mrr = 0.0
    random_hits = {int(k): 0.0 for k in topk}
    used = 0

    for t, s, chosen in use:
        cand, prob = predict(result, startup_features, startup_sector, active,
                             startup_counts, week=int(t), sector=int(s))
        if cand.size == 0:
            continue
        where = np.where(cand == int(chosen))[0]
        if where.size == 0:
            continue
        pos = int(where[0])
        order = np.argsort(-prob)
        rank = int(np.where(order == pos)[0][0]) + 1
        nll -= float(np.log(max(prob[pos], 1e-12)))
        mrr += 1.0 / rank
        for k in topk:
            hits[int(k)] += int(rank <= int(k))
        random_nll += np.log(cand.size)
        random_mrr += np.mean(1.0 / np.arange(1, cand.size + 1))
        for k in topk:
            random_hits[int(k)] += min(int(k), cand.size) / cand.size
        used += 1

    if used == 0:
        return {"n_events": 0}
    out = {
        "n_events": used,
        "nll": nll / used,
        "random_nll": random_nll / used,
        "mrr": mrr / used,
        "random_mrr": random_mrr / used,
    }
    for k in topk:
        out[f"top{int(k)}"] = hits[int(k)] / used
        out[f"random_top{int(k)}"] = random_hits[int(k)] / used
    return out


__all__ = [
    "CoxSurvivalStageResult",
    "DiscreteHazardStageResult",
    "fit_cox_survival_stage",
    "cox_survival_predict_proba",
    "fit_discrete_hazard_stage",
    "discrete_hazard_scores",
    "discrete_hazard_predict_proba",
    "evaluate_survival_stage",
]
