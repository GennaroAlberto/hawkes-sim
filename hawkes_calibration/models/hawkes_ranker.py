r"""
Within-sector second stage as a per-firm **1-D Hawkes intensity**, in a *linear*
(additive) and a *non-linear* (exp-link) form.

Conditional on a funding event in a sector (supplied by the first stage), we score each
live startup by an intensity built from a shared, sector-level kernel and pick / rank by
that intensity. This is the "which one has the highest rate" view of the second stage,
and it is *exactly* the survival conditional likelihood: with per-firm intensity
``lambda_i(t)`` the probability the event lands on firm ``i`` is the normalized intensity

    P(i | sector event at t) = lambda_i(t) / sum_j lambda_j(t)   =   softmax(log lambda_i)

over the risk set, which is the Cox / conditional-logit partial likelihood. Ranking by
intensity and maximizing that partial likelihood are the same model.

Per-firm intensity
------------------
For firm ``i`` in sector ``s`` at week ``t`` (features ``Z_i``, own recency ``R_i``,
size-normalized peer recency ``E_i``):

    eta_i(t) = b0_s + w . Z_i(t) + a_s * E_i(t) - rho_s * R_i(t),     a_s, rho_s >= 0
    lambda_i(t) = link(eta_i(t)),   link in { softplus (linear/additive), exp (non-linear) }

``R_i`` is the firm's own recent funding (a refractory / cooldown signal, entered with a
non-negative ``rho_s`` so it *lowers* the rate); ``E_i`` is the recent activity of the
*other* firms in the sector (peer excitation, ``a_s >= 0``). The linear link is the
rectified-additive intensity (softplus keeps it positive and differentiable); the exp link
is the log-linear (non-linear) Hawkes intensity, for which the conditional log-likelihood
is concave.

Dropping the last-funded firm
------------------------------
In **both** links the risk set at a sector event excludes the startup funded at the
*previous* event in that sector -- the firm that just raised is not a candidate for the
next round (a hard refractory rule, complementing the soft ``rho_s`` cooldown). If the
observed recipient equals that firm (a rare consecutive re-funding), the removal is skipped
for that event so the likelihood stays defined.

Public API
----------
``HawkesRankerResult``       -- fitted parameters.
``fit_hawkes_ranker``        -- fit the linear or non-linear ranker (partial likelihood).
``hawkes_ranker_intensity``  -- per-candidate intensities for one sector event.
``hawkes_ranker_predict_proba`` -- per-candidate choice probabilities.
``evaluate_hawkes_ranker``   -- held-out NLL, top-k, MRR.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # SciPy is optional.
    from scipy.optimize import minimize
except ImportError:  # pragma: no cover
    from .._opt_fallback import minimize

from ..sector_ranker import candidate_set


# ---------------------------------------------------------------------------
# Links
# ---------------------------------------------------------------------------
def _link(eta, kind):
    """Return (lambda, dlambda/deta) for the chosen positive link."""
    eta = np.clip(eta, -30.0, 30.0)
    if kind == "exp":
        lam = np.exp(eta)
        return lam, lam
    # "linear" == rectified-additive intensity, smoothed by softplus (>0, differentiable)
    lam = np.logaddexp(0.0, eta)  # softplus
    dlam = 1.0 / (1.0 + np.exp(-eta))  # sigmoid
    return lam, dlam


# ---------------------------------------------------------------------------
# Recency fields (weekly geometric decay)
# ---------------------------------------------------------------------------
def _recency_fields(startup_counts, startup_sector, n_sectors, decay_self, decay_cross):
    """Return own-recency ``R[t, i]`` and size-normalized peer-recency ``E[t, i]``.

    Left-continuous (week ``t`` uses only counts strictly before ``t``): no leakage.
    """
    C = np.asarray(startup_counts, dtype=float)
    T, N = C.shape
    sec = np.asarray(startup_sector, int)
    M = int(n_sectors)
    own = np.zeros((T, N))
    cross = np.zeros((T, N))
    ds, dc = float(decay_self), float(decay_cross)
    for t in range(1, T):
        own[t] = ds * (own[t - 1] + C[t - 1])
        cross[t] = dc * (cross[t - 1] + C[t - 1])
    nb = np.maximum(np.bincount(sec, minlength=M), 1)
    sector_sum = np.zeros((T, M))
    for s in range(M):
        idx = np.flatnonzero(sec == s)
        if idx.size:
            sector_sum[:, s] = cross[:, idx].sum(axis=1)
    peer = (sector_sum[:, sec] - cross) / np.maximum(nb[sec] - 1, 1)[None, :]
    return own, peer


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
@dataclass
class HawkesRankerResult:
    link: str  # "linear" | "exp"
    weights: np.ndarray  # (p,)  global covariate weights w
    sector_baseline: np.ndarray  # (M,)  b0_s
    sector_excite: np.ndarray  # (M,)  a_s   (>= 0, peer excitation)
    sector_inhibit: np.ndarray  # (M,)  rho_s (>= 0, own-recency cooldown)
    decay_self: float
    decay_cross: float
    drop_last_funded: bool
    loss: float
    success: bool
    message: str
    n_events: int

    def eta(self, Z, peer, own, sector):
        s = int(sector)
        return (
            self.sector_baseline[s]
            + np.asarray(Z) @ self.weights
            + self.sector_excite[s] * np.asarray(peer)
            - self.sector_inhibit[s] * np.asarray(own)
        )


# ---------------------------------------------------------------------------
# Risk-set assembly (with the last-funded firm dropped)
# ---------------------------------------------------------------------------
def _last_funded_before(events, sector, week):
    """Startup funded at the most recent event in ``sector`` strictly before ``week``."""
    ev = events[(events[:, 1] == int(sector)) & (events[:, 0] < int(week))]
    if ev.shape[0] == 0:
        return None
    return int(ev[ev[:, 0].argmax(), 2])


def _reduced_candidates(startup_sector, active, sector, week, last_funded, chosen=None):
    cand = candidate_set(startup_sector, active, int(sector), int(week))
    if last_funded is not None and last_funded != chosen:
        cand = cand[cand != last_funded]
    return cand


def _prepare(events, Z, startup_sector, active, own, peer, train_end, drop_last_funded):
    ev = events[np.argsort(events[:, 0], kind="stable")]
    prepared = []
    last = {}
    for week, s, chosen in ev:
        week, s, chosen = int(week), int(s), int(chosen)
        lf = last.get(s) if drop_last_funded else None
        if week < train_end:
            cand = _reduced_candidates(startup_sector, active, s, week, lf, chosen)
            if chosen in set(cand.tolist()):
                pos = int(np.where(cand == chosen)[0][0])
                prepared.append((s, Z[week, cand], peer[week, cand], own[week, cand], pos))
        last[s] = chosen
    return prepared


# ---------------------------------------------------------------------------
# Objective (partial likelihood + analytic gradient)
# ---------------------------------------------------------------------------
def _unpack(theta, p, M):
    w = theta[:p]
    b0 = theta[p : p + M]
    a = theta[p + M : p + 2 * M]
    rho = theta[p + 2 * M : p + 3 * M]
    return w, b0, a, rho


def _objective(theta, prepared, p, M, link, l2):
    w, b0, a, rho = _unpack(theta, p, M)
    loss = 0.0
    gw = np.zeros(p)
    gb0 = np.zeros(M)
    ga = np.zeros(M)
    grho = np.zeros(M)
    for s, Z, peer, own, pos in prepared:
        eta = b0[s] + Z @ w + a[s] * peer - rho[s] * own
        lam, dlam = _link(eta, link)
        S = lam.sum()
        loss += float(np.log(S) - np.log(lam[pos]))
        g = dlam / S  # d(-logP)/deta_k
        g[pos] -= dlam[pos] / lam[pos]
        gb0[s] += g.sum()
        gw += g @ Z
        ga[s] += float(g @ peer)
        grho[s] += -float(g @ own)
    loss += 0.5 * l2 * (float(w @ w) + float(a @ a) + float(rho @ rho))
    gw += l2 * w
    ga += l2 * a
    grho += l2 * rho
    return loss, np.concatenate([gw, gb0, ga, grho])


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------
def fit_hawkes_ranker(
    events,
    startup_features,
    startup_sector,
    active,
    startup_counts,
    *,
    link="exp",
    train_end=None,
    decay_self=0.6,
    decay_cross=0.6,
    drop_last_funded=True,
    l2=1e-2,
    max_iter=300,
):
    r"""Fit the within-sector 1-D Hawkes ranker by conditional maximum partial likelihood.

    ``link="linear"`` fits the rectified-additive intensity; ``link="exp"`` fits the
    log-linear (non-linear) intensity (concave objective). In both, the risk set at each
    event drops the previously-funded startup (``drop_last_funded``).
    """
    if link not in ("linear", "exp"):
        raise ValueError("link must be 'linear' or 'exp'")
    events = np.asarray(events, dtype=int)
    Z = np.asarray(startup_features, dtype=float)
    startup_sector = np.asarray(startup_sector, int)
    active = np.asarray(active, dtype=bool)
    T, N, p = Z.shape
    M = int(startup_sector.max()) + 1
    if train_end is None:
        train_end = T
    own, peer = _recency_fields(startup_counts, startup_sector, M, decay_self, decay_cross)
    prepared = _prepare(
        events, Z, startup_sector, active, own, peer, int(train_end), drop_last_funded
    )
    if not prepared:
        raise ValueError("no usable training events")

    n = p + 3 * M
    x0 = np.zeros(n)
    x0[p + M : p + 2 * M] = 0.1  # a start
    x0[p + 2 * M : p + 3 * M] = 0.1  # rho start
    bounds = [(None, None)] * n
    for j in range(M):  # a_s >= 0, rho_s >= 0
        bounds[p + M + j] = (0.0, None)
        bounds[p + 2 * M + j] = (0.0, None)

    opt = minimize(
        lambda th: _objective(th, prepared, p, M, link, l2),
        x0,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": int(max_iter), "ftol": 1e-9},
    )
    w, b0, a, rho = _unpack(opt.x, p, M)
    return HawkesRankerResult(
        link=link,
        weights=w,
        sector_baseline=b0,
        sector_excite=a,
        sector_inhibit=rho,
        decay_self=float(decay_self),
        decay_cross=float(decay_cross),
        drop_last_funded=bool(drop_last_funded),
        loss=float(opt.fun),
        success=bool(opt.success),
        message=str(opt.message),
        n_events=len(prepared),
    )


# ---------------------------------------------------------------------------
# Predict / evaluate
# ---------------------------------------------------------------------------
def _fields_for(result, startup_counts, startup_sector, M):
    return _recency_fields(startup_counts, startup_sector, M, result.decay_self, result.decay_cross)


def hawkes_ranker_intensity(
    result,
    startup_features,
    startup_sector,
    active,
    startup_counts,
    *,
    week,
    sector,
    events=None,
    _fields=None,
):
    """Return ``(candidate_indices, intensities)`` for one sector event."""
    Z = np.asarray(startup_features, dtype=float)
    startup_sector = np.asarray(startup_sector, int)
    M = int(startup_sector.max()) + 1
    own, peer = (
        _fields if _fields is not None else _fields_for(result, startup_counts, startup_sector, M)
    )
    lf = None
    if result.drop_last_funded and events is not None:
        lf = _last_funded_before(np.asarray(events, int), sector, week)
    cand = _reduced_candidates(startup_sector, np.asarray(active, bool), sector, week, lf)
    eta = result.eta(Z[int(week), cand], peer[int(week), cand], own[int(week), cand], sector)
    lam, _ = _link(eta, result.link)
    return cand, lam


def hawkes_ranker_predict_proba(
    result,
    startup_features,
    startup_sector,
    active,
    startup_counts,
    *,
    week,
    sector,
    events=None,
    _fields=None,
):
    """Return ``(candidate_indices, probabilities)`` = normalized intensities."""
    cand, lam = hawkes_ranker_intensity(
        result,
        startup_features,
        startup_sector,
        active,
        startup_counts,
        week=week,
        sector=sector,
        events=events,
        _fields=_fields,
    )
    tot = lam.sum()
    pr = lam / tot if tot > 0 else np.full(lam.shape, 1.0 / max(lam.size, 1))
    return cand, pr


def evaluate_hawkes_ranker(
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
    """Held-out NLL, top-k hit rate, and MRR over the reduced risk set."""
    events = np.asarray(events, dtype=int)
    Z = np.asarray(startup_features, dtype=float)
    startup_sector = np.asarray(startup_sector, int)
    M = int(startup_sector.max()) + 1
    if end_week is None:
        end_week = Z.shape[0]
    fields = _fields_for(result, startup_counts, startup_sector, M)
    use = events[(events[:, 0] >= int(start_week)) & (events[:, 0] < int(end_week))]
    nll = mrr = 0.0
    hits = {int(k): 0 for k in topk}
    n = 0
    for week, s, chosen in use:
        cand, pr = hawkes_ranker_predict_proba(
            result,
            Z,
            startup_sector,
            active,
            startup_counts,
            week=int(week),
            sector=int(s),
            events=events,
            _fields=fields,
        )
        if int(chosen) not in set(cand.tolist()):
            continue
        pos = int(np.where(cand == int(chosen))[0][0])
        nll -= float(np.log(max(pr[pos], 1e-12)))
        rank = int(np.where(np.argsort(-pr) == pos)[0][0]) + 1
        mrr += 1.0 / rank
        for k in topk:
            hits[int(k)] += int(rank <= int(k))
        n += 1
    out = {"n_events": int(n), "nll": float(nll / max(n, 1)), "mrr": float(mrr / max(n, 1))}
    for k in topk:
        out[f"top{int(k)}"] = float(hits[int(k)] / max(n, 1))
    return out


__all__ = [
    "HawkesRankerResult",
    "fit_hawkes_ranker",
    "hawkes_ranker_intensity",
    "hawkes_ranker_predict_proba",
    "evaluate_hawkes_ranker",
]
