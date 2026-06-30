"""
Discrete-time hazard alternative for the startup second stage.

The sector count model predicts where and when funding events happen.  Conditional
on a sector event, this module estimates which startup is most likely to receive it
from active firm-week labels.

Unlike the Cox / outside-option model in ``sector_survival.py``, this model uses
firm-weeks without funding as explicit negatives.  It is a useful supervised
baseline for weekly PitchBook-style data, with optional negative sampling to control
class imbalance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .sector_ranker import candidate_set, cooldown_vector


@dataclass
class DiscreteHazardResult:
    """Fitted discrete-time logistic hazard model.

    The per-firm weekly hazard is

    ``logit h[i,t] = b0 + b_sector[sector]
                    + (w0 + u[sector])' Z[i,t]
                    + eta[sector] * cooldown[i,t]``.

    For a known sector event, hazards are normalized over the current sector risk
    set to produce a conditional mark distribution.
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


def _n_sectors(startup_sector):
    return int(np.asarray(startup_sector, dtype=int).max()) + 1


def fit_discrete_hazard(
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
    """Fit a discrete-time logistic hazard model from active firm-weeks.

    Parameters
    ----------
    events : array-like
        Included for API symmetry with the Cox/survival stage. Labels are read
        from ``startup_counts``.
    startup_features : (T, N, p) array
        Point-in-time startup features.
    startup_sector : (N,) array
        Sector id for each startup.
    active : (T, N) bool array
        Active/risk-set membership by week.
    startup_counts : (T, N) array
        Funding event count by startup/week.
    negative_sampling_ratio : int or None
        Number of active negative firm-weeks sampled per positive row. If None,
        all active negative firm-weeks are used.
    """
    _ = events  # labels are in startup_counts; keep argument for stable API.
    Z = np.asarray(startup_features, dtype=float)
    startup_sector = np.asarray(startup_sector, dtype=int)
    active = np.asarray(active, dtype=bool)
    startup_counts = np.asarray(startup_counts, dtype=float)
    T, N, p = Z.shape
    M = _n_sectors(startup_sector)
    if train_end is None:
        train_end = T
    train_end = int(train_end)
    start_week = int(start_week)
    rng = np.random.default_rng(seed)

    rows_t, rows_i, rows_y = [], [], []
    for t in range(start_week, train_end):
        live = np.flatnonzero(active[t])
        if live.size == 0:
            continue
        labels = (startup_counts[t, live] > 0).astype(int)
        pos = live[labels == 1]
        neg = live[labels == 0]
        if pos.size:
            rows_t.extend([t] * pos.size)
            rows_i.extend(pos.tolist())
            rows_y.extend([1] * pos.size)
        if neg.size:
            if negative_sampling_ratio is None:
                take = neg
            else:
                n_ref = max(1, int(pos.size))
                n_take = min(neg.size, int(negative_sampling_ratio) * n_ref)
                take = rng.choice(neg, size=n_take, replace=False)
            rows_t.extend([t] * len(take))
            rows_i.extend(np.asarray(take, dtype=int).tolist())
            rows_y.extend([0] * len(take))

    if not rows_y or np.sum(rows_y) == 0:
        raise ValueError("no positive training firm-weeks found")

    rows_t = np.asarray(rows_t, dtype=int)
    rows_i = np.asarray(rows_i, dtype=int)
    y = np.asarray(rows_y, dtype=float)
    sectors = startup_sector[rows_i]
    F = Z[rows_t, rows_i]
    cd = np.zeros_like(y)
    for n, (t, i) in enumerate(zip(rows_t, rows_i)):
        cd[n] = cooldown_vector(startup_counts, int(t), np.array([int(i)]), cooldown_weeks)[0]

    n_intercept = 1
    n_sector_intercept = M
    n_w0 = p
    n_u = M * p
    n_eta = M
    n = n_intercept + n_sector_intercept + n_w0 + n_u + n_eta
    eta_start = n_intercept + n_sector_intercept + n_w0 + n_u

    x0 = np.zeros(n, dtype=float)
    base_rate = np.clip(y.mean(), 1e-4, 1.0 - 1e-4)
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
        loss = -float(np.sum(
            y * np.log(np.maximum(prob, 1e-12))
            + (1.0 - y) * np.log(np.maximum(1.0 - prob, 1e-12))
        ))
        err = prob - y

        g_intercept = np.array([err.sum()])
        g_sector_intercept = np.zeros(M)
        np.add.at(g_sector_intercept, sectors, err)
        g_w0 = err @ F
        g_u = np.zeros((M, p))
        for s in range(M):
            m = sectors == s
            if np.any(m):
                g_u[s] = err[m] @ F[m]
        g_eta = np.zeros(M)
        np.add.at(g_eta, sectors, err * cd)

        loss += 0.5 * l2_global * np.sum(w0 ** 2)
        loss += 0.5 * l2_sector * (
            np.sum(sector_intercept ** 2) + np.sum(u ** 2) + np.sum(eta ** 2)
        )
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
    return DiscreteHazardResult(
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


def discrete_hazard_scores(result, features, cooldown, sector: int):
    """Return logits for candidates under a fitted discrete hazard model."""
    s = int(sector)
    return (
        result.intercept
        + result.sector_intercept[s]
        + np.asarray(features, dtype=float) @ result.weights_for_sector(s)
        + result.cooldown_coef[s] * np.asarray(cooldown, dtype=float)
    )


def discrete_hazard_predict_proba(
    result,
    startup_features,
    startup_sector,
    active,
    startup_counts,
    *,
    week: int,
    sector: int,
):
    """Return ``(candidate_indices, probabilities)`` for one sector event."""
    Z = np.asarray(startup_features, dtype=float)
    cand = candidate_set(startup_sector, active, sector, week)
    if cand.size == 0:
        return cand, np.zeros(0)
    cd = cooldown_vector(startup_counts, int(week), cand, result.cooldown_weeks)
    logits = discrete_hazard_scores(result, Z[int(week), cand], cd, int(sector))
    hazards = expit(np.clip(logits, -40.0, 40.0))
    total = hazards.sum()
    if total <= 0 or not np.isfinite(total):
        return cand, np.full(cand.size, 1.0 / cand.size)
    return cand, hazards / total


def evaluate_discrete_hazard(
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
    """Held-out conditional mark NLL, MRR, and top-k metrics."""
    events = np.asarray(events, dtype=int)
    Z = np.asarray(startup_features, dtype=float)
    if end_week is None:
        end_week = Z.shape[0]
    use = events[(events[:, 0] >= int(start_week)) & (events[:, 0] < int(end_week))]
    if use.size == 0:
        return {"n_events": 0}

    nll = 0.0
    mrr = 0.0
    hits = {int(k): 0 for k in topk}
    random_nll = 0.0
    random_mrr = 0.0
    random_hits = {int(k): 0.0 for k in topk}
    used = 0

    for t, s, chosen in use:
        cand, prob = discrete_hazard_predict_proba(
            result,
            startup_features,
            startup_sector,
            active,
            startup_counts,
            week=int(t),
            sector=int(s),
        )
        if cand.size == 0:
            continue
        pos_arr = np.where(cand == int(chosen))[0]
        if pos_arr.size == 0:
            continue
        pos = int(pos_arr[0])
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
        "nll": float(nll / used),
        "random_nll": float(random_nll / used),
        "mrr": float(mrr / used),
        "random_mrr": float(random_mrr / used),
    }
    for k in topk:
        out[f"top{int(k)}"] = float(hits[int(k)] / used)
        out[f"random_top{int(k)}"] = float(random_hits[int(k)] / used)
    return out


__all__ = [
    "DiscreteHazardResult",
    "fit_discrete_hazard",
    "discrete_hazard_scores",
    "discrete_hazard_predict_proba",
    "evaluate_discrete_hazard",
]
