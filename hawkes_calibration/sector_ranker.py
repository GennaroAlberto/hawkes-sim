"""
Sector-level Hawkes-style count model plus a dynamic startup risk-set ranker.

This module implements the two-layer architecture discussed for investment data:

1. A weekly sector process predicts *where and when* funding events occur.
   It is a discrete-time Hawkes/GLM approximation with positive lagged sector
   excitation and optional macro covariates.

2. A conditional risk-set ranker predicts *which startup* receives a funding
   event once the sector is known.  It scores the current startups in the sector
   and normalises over the live risk set, so sectors may contain a changing
   number of startups over time.

The code is deliberately numpy/scipy only and is meant as a practical bridge
between the continuous-time Hawkes package and weekly startup-funding data.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import lgamma
from typing import Iterable

import numpy as np

try:  # SciPy is optional: used when available, else a numpy fallback.
    from scipy.optimize import minimize
    from scipy.special import logsumexp
except ImportError:  # pragma: no cover - exercised only without SciPy
    from ._opt_fallback import logsumexp, minimize


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class SyntheticStartupMarket:
    """Synthetic weekly investment-market dataset.

    Attributes
    ----------
    sector_counts : (T, M) array
        Number of funding events per sector and week.
    startup_counts : (T, N) array
        Number of funding events per startup and week.
    events : (E, 3) array
        Rows are ``(week, sector, startup)``.
    covariates : (T, p_x) array
        Exogenous market covariates used by the sector count model.
    startup_features : (T, N, p_z) array
        Startup-level features used by the ranker.
    startup_sector : (N,) array
        Sector id for each startup.
    active : (T, N) bool array
        Whether a startup belongs to the risk set at a week.
    true_sector_intercept, true_sector_beta, true_excitation, true_ranker_weights,
    true_ranker_cooldown : arrays
        Parameters used by the simulator.
    """

    sector_counts: np.ndarray
    startup_counts: np.ndarray
    events: np.ndarray
    covariates: np.ndarray
    startup_features: np.ndarray
    startup_sector: np.ndarray
    active: np.ndarray
    true_sector_intercept: np.ndarray
    true_sector_beta: np.ndarray
    true_excitation: np.ndarray
    true_ranker_weights: np.ndarray
    true_ranker_cooldown: np.ndarray


@dataclass
class SectorCountResult:
    """Fitted weekly sector Hawkes/Poisson-GLM model."""

    intercept: np.ndarray          # (M,)
    beta: np.ndarray               # (M, p_x)
    excitation: np.ndarray         # (M, M, L), receiver, source, lag
    n_lags: int
    train_end: int
    loss: float
    success: bool
    message: str

    def rates(self, counts_history, covariates):
        """Return one-step rates for all valid weeks in ``covariates``.

        ``counts_history`` must contain the observed/simulated counts on the same
        time grid as ``covariates``.  Rates are computed for weeks ``L..T-1``.
        """
        counts_history = np.asarray(counts_history, float)
        covariates = np.asarray(covariates, float)
        T = covariates.shape[0]
        M = self.intercept.size
        out = np.zeros((T, M), dtype=float)
        for t in range(self.n_lags, T):
            out[t] = sector_rate_at(self, counts_history, covariates, t)
        return out


@dataclass
class StartupRankerResult:
    """Fitted conditional risk-set ranker."""

    global_weights: np.ndarray     # (p_z,)
    sector_deviations: np.ndarray  # (M, p_z)
    cooldown_coef: np.ndarray      # (M,), constrained <= 0 by default
    cooldown_weeks: int
    loss: float
    success: bool
    message: str

    def weights_for_sector(self, sector: int) -> np.ndarray:
        return self.global_weights + self.sector_deviations[int(sector)]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _as_2d_covariates(covariates, T):
    if covariates is None:
        return np.zeros((T, 0), dtype=float)
    covariates = np.asarray(covariates, dtype=float)
    if covariates.ndim == 1:
        covariates = covariates[:, None]
    return covariates


def _softmax_from_scores(scores):
    scores = np.asarray(scores, dtype=float)
    z = scores - np.max(scores)
    p = np.exp(z)
    s = p.sum()
    if not np.isfinite(s) or s <= 0:
        return np.full(scores.size, 1.0 / scores.size)
    return p / s


def cooldown_vector(startup_counts, week: int, candidate_idx, cooldown_weeks: int):
    """Return 0/1 indicator: candidate funded in previous ``cooldown_weeks``.

    The current week is excluded, preventing outcome leakage.
    """
    lo = max(0, int(week) - int(cooldown_weeks))
    hi = int(week)
    if hi <= lo:
        return np.zeros(len(candidate_idx), dtype=float)
    past = np.asarray(startup_counts[lo:hi, candidate_idx], dtype=float)
    return (past.sum(axis=0) > 0).astype(float)


def sector_rate_at(result: SectorCountResult, counts_history, covariates, week: int):
    """One-step sector rates at ``week`` from a fitted sector model."""
    counts_history = np.asarray(counts_history, dtype=float)
    covariates = np.asarray(covariates, dtype=float)
    x = covariates[int(week)] if covariates.size else np.zeros(0)
    eta = result.intercept + result.beta @ x
    for lag in range(1, result.n_lags + 1):
        eta += result.excitation[:, :, lag - 1] @ counts_history[int(week) - lag]
    return np.exp(np.clip(eta, -30.0, 20.0))


def poisson_nll(counts, rates):
    """Poisson negative log-likelihood including the log-factorial constant."""
    counts = np.asarray(counts, dtype=float)
    rates = np.maximum(np.asarray(rates, dtype=float), 1e-12)
    lg = np.vectorize(lambda x: lgamma(float(x) + 1.0))(counts)
    return float(np.sum(rates - counts * np.log(rates) + lg))


# ---------------------------------------------------------------------------
# Sector-level weekly Hawkes/GLM
# ---------------------------------------------------------------------------
def fit_sector_count_model(
    sector_counts,
    covariates=None,
    *,
    n_lags=4,
    train_end=None,
    l2=1e-3,
    adjacency=None,
    nonnegative_excitation=True,
    max_iter=500,
):
    r"""Fit a positive-lag sector count model.

    Model:

    .. math::

        Y_{s,t} \sim \mathrm{Poisson}(\Lambda_{s,t}),\qquad
        \log \Lambda_{s,t} = a_s + \beta_s^\top X_t
          + \sum_{r=1}^M\sum_{\ell=1}^L b_{sr\ell}Y_{r,t-\ell}.

    If ``nonnegative_excitation=True``, all lag coefficients ``b`` are constrained
    nonnegative.  ``adjacency`` may be an ``(M, M)`` hard mask; entries with
    ``False`` are forced to zero for every lag.
    """
    y = np.asarray(sector_counts, dtype=float)
    if y.ndim != 2:
        raise ValueError("sector_counts must be a (T, M) array")
    T, M = y.shape
    X = _as_2d_covariates(covariates, T)
    p = X.shape[1]
    L = int(n_lags)
    if train_end is None:
        train_end = T
    train_end = int(train_end)
    if train_end <= L:
        raise ValueError("train_end must be larger than n_lags")

    if adjacency is None:
        mask = np.ones((M, M), dtype=bool)
    else:
        mask = np.asarray(adjacency, dtype=bool)
        if mask.shape != (M, M):
            raise ValueError("adjacency must have shape (M, M)")

    n_intercept = M
    n_beta = M * p
    n_b = M * M * L
    n = n_intercept + n_beta + n_b

    mean_rate = np.maximum(y[L:train_end].mean(axis=0), 1e-3)
    x0 = np.zeros(n, dtype=float)
    x0[:M] = np.log(mean_rate)
    # small positive starting excitation on allowed entries
    b0 = np.zeros((M, M, L), dtype=float)
    b0[mask, :] = 0.01 / max(L, 1)
    x0[n_intercept + n_beta:] = b0.ravel()

    bounds = [(None, None)] * n
    start_b = n_intercept + n_beta
    for idx in range(n_b):
        s = idx // (M * L)
        rem = idx % (M * L)
        r = rem // L
        allowed = bool(mask[s, r])
        if not allowed:
            bounds[start_b + idx] = (0.0, 0.0)
        elif nonnegative_excitation:
            bounds[start_b + idx] = (0.0, None)

    weeks = np.arange(L, train_end)

    def unpack(theta):
        intercept = theta[:M]
        beta = theta[M:M + n_beta].reshape(M, p) if p else np.zeros((M, 0))
        b = theta[start_b:].reshape(M, M, L)
        return intercept, beta, b

    def objective(theta):
        intercept, beta, b = unpack(theta)
        loss = 0.0
        grad_intercept = np.zeros(M)
        grad_beta = np.zeros((M, p))
        grad_b = np.zeros((M, M, L))

        for t in weeks:
            eta = intercept.copy()
            if p:
                eta += beta @ X[t]
            for lag in range(1, L + 1):
                eta += b[:, :, lag - 1] @ y[t - lag]
            eta = np.clip(eta, -30.0, 20.0)
            lam = np.exp(eta)
            err = lam - y[t]
            loss += float(np.sum(lam - y[t] * eta))
            grad_intercept += err
            if p:
                grad_beta += err[:, None] * X[t][None, :]
            for lag in range(1, L + 1):
                grad_b[:, :, lag - 1] += err[:, None] * y[t - lag][None, :]

        # L2 shrinkage, but do not penalize intercepts.
        loss += 0.5 * l2 * (np.sum(beta ** 2) + np.sum(b ** 2))
        grad_beta += l2 * beta
        grad_b += l2 * b
        grad_b[~mask, :] = 0.0

        grad = np.concatenate([grad_intercept, grad_beta.ravel(), grad_b.ravel()])
        return loss, grad

    opt = minimize(lambda th: objective(th), x0, jac=True, method="L-BFGS-B",
                   bounds=bounds, options={"maxiter": int(max_iter), "ftol": 1e-9})
    intercept, beta, b = unpack(opt.x)
    b[~mask, :] = 0.0
    return SectorCountResult(
        intercept=intercept,
        beta=beta,
        excitation=b,
        n_lags=L,
        train_end=train_end,
        loss=float(opt.fun),
        success=bool(opt.success),
        message=str(opt.message),
    )


def sector_baseline_rates(sector_counts, train_end, start, end):
    """Historical-mean baseline rates for held-out sector count scoring."""
    y = np.asarray(sector_counts, dtype=float)
    mu = np.maximum(y[:int(train_end)].mean(axis=0), 1e-12)
    return np.repeat(mu[None, :], int(end) - int(start), axis=0)


# ---------------------------------------------------------------------------
# Dynamic risk-set ranker
# ---------------------------------------------------------------------------
def candidate_set(startup_sector, active, sector: int, week: int):
    """Current live startups in ``sector`` at ``week``."""
    startup_sector = np.asarray(startup_sector)
    active = np.asarray(active, dtype=bool)
    return np.flatnonzero((startup_sector == int(sector)) & active[int(week)])


def ranker_scores(result: StartupRankerResult, features, cooldown, sector: int):
    """Scores for candidate startup feature rows in one sector."""
    features = np.asarray(features, dtype=float)
    cooldown = np.asarray(cooldown, dtype=float)
    s = int(sector)
    return features @ result.weights_for_sector(s) + result.cooldown_coef[s] * cooldown


def ranker_predict_proba(
    result: StartupRankerResult,
    startup_features,
    startup_sector,
    active,
    startup_counts,
    *,
    week: int,
    sector: int,
):
    """Return ``(candidate_indices, probabilities)`` for one sector event."""
    cand = candidate_set(startup_sector, active, sector, week)
    if cand.size == 0:
        return cand, np.zeros(0)
    cd = cooldown_vector(startup_counts, week, cand, result.cooldown_weeks)
    scores = ranker_scores(result, startup_features[int(week), cand], cd, sector)
    return cand, _softmax_from_scores(scores)


def fit_startup_ranker(
    events,
    startup_features,
    startup_sector,
    active,
    startup_counts,
    *,
    train_end=None,
    cooldown_weeks=26,
    l2_global=1e-3,
    l2_sector=1e-2,
    constrain_cooldown_negative=True,
    max_iter=500,
):
    r"""Fit the conditional ranker over dynamic sector risk sets.

    For each observed funding event ``(t, s, i*)`` the objective is

    .. math::

        -\log p(i^*\mid s,t) =
        \log\sum_{j\in R_{s,t}} \exp q_{j,t} - q_{i^*,t},

    where

    .. math::

        q_{j,t} = (w_0 + u_s)^\top Z_{j,t} + \eta_s C^{(K)}_{j,t}.

    The number of candidates may change across weeks and sectors; the softmax is
    always taken over the current risk set only.
    """
    events = np.asarray(events, dtype=int)
    Z = np.asarray(startup_features, dtype=float)
    startup_sector = np.asarray(startup_sector, dtype=int)
    active = np.asarray(active, dtype=bool)
    startup_counts = np.asarray(startup_counts, dtype=float)
    T, N, p = Z.shape
    M = int(startup_sector.max()) + 1
    if train_end is None:
        train_end = T
    train_end = int(train_end)
    train_events = events[events[:, 0] < train_end]
    if train_events.size == 0:
        raise ValueError("no training events before train_end")

    n_w0 = p
    n_u = M * p
    n_eta = M
    n = n_w0 + n_u + n_eta
    x0 = np.zeros(n, dtype=float)
    x0[n_w0 + n_u:] = -0.5

    bounds = [(None, None)] * n
    if constrain_cooldown_negative:
        for j in range(n_eta):
            bounds[n_w0 + n_u + j] = (None, 0.0)

    # Precompute candidate sets and cooldowns for speed.
    prepared = []
    for t, s, chosen in train_events:
        cand = candidate_set(startup_sector, active, int(s), int(t))
        if cand.size == 0 or chosen not in set(cand.tolist()):
            continue
        chosen_pos = int(np.where(cand == chosen)[0][0])
        cd = cooldown_vector(startup_counts, int(t), cand, cooldown_weeks)
        prepared.append((int(t), int(s), cand, chosen_pos, cd))
    if not prepared:
        raise ValueError("no training events had a non-empty risk set containing the funded startup")

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
            w = w0 + u[s]
            scores = F @ w + eta[s] * cd
            lse = logsumexp(scores)
            p_cand = np.exp(scores - lse)
            loss += float(lse - scores[chosen_pos])
            expected_f = p_cand @ F
            chosen_f = F[chosen_pos]
            diff_f = expected_f - chosen_f
            diff_cd = float(p_cand @ cd - cd[chosen_pos])
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

    opt = minimize(lambda th: objective(th), x0, jac=True, method="L-BFGS-B",
                   bounds=bounds, options={"maxiter": int(max_iter), "ftol": 1e-9})
    w0, u, eta = unpack(opt.x)
    return StartupRankerResult(
        global_weights=w0,
        sector_deviations=u,
        cooldown_coef=eta,
        cooldown_weeks=int(cooldown_weeks),
        loss=float(opt.fun),
        success=bool(opt.success),
        message=str(opt.message),
    )


def evaluate_ranker(
    result: StartupRankerResult,
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
    """Evaluate held-out ranker NLL, MRR, and top-k hit rates."""
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
        cand, prob = ranker_predict_proba(result, startup_features, startup_sector, active,
                                          startup_counts, week=int(t), sector=int(s))
        if cand.size == 0 or chosen not in set(cand.tolist()):
            continue
        pos = int(np.where(cand == chosen)[0][0])
        order = np.argsort(-prob)
        rank = int(np.where(order == pos)[0][0]) + 1
        nll -= float(np.log(max(prob[pos], 1e-12)))
        mrr += 1.0 / rank
        for k in topk:
            hits[int(k)] += int(rank <= int(k))
        # Random baseline over the same risk set.
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


# ---------------------------------------------------------------------------
# Synthetic market simulator and end-to-end backtest
# ---------------------------------------------------------------------------
def simulate_synthetic_startup_market(
    *,
    T=180,
    n_sectors=11,
    startups_per_sector=40,
    n_covariates=2,
    n_features=5,
    n_lags=4,
    cooldown_weeks=26,
    seed=0,
):
    """Generate a synthetic weekly sector + startup funding dataset.

    The simulator mirrors the intended production decomposition: sector counts are
    generated by a positive Hawkes-style weekly GLM, and each sector event is
    assigned to a live startup through a sector-specific risk-set softmax with a
    negative recent-funding cooldown effect.
    """
    rng = np.random.default_rng(seed)
    M = int(n_sectors)
    if np.isscalar(startups_per_sector):
        n_by_sector = np.full(M, int(startups_per_sector), dtype=int)
    else:
        n_by_sector = np.asarray(startups_per_sector, dtype=int)
        if n_by_sector.size != M:
            raise ValueError("startups_per_sector must be scalar or length n_sectors")
    N = int(n_by_sector.sum())
    startup_sector = np.repeat(np.arange(M), n_by_sector)

    # AR(1) market covariates.
    X = np.zeros((T, n_covariates), dtype=float)
    noise = rng.normal(size=(T, n_covariates))
    for t in range(1, T):
        X[t] = 0.85 * X[t - 1] + 0.35 * noise[t]
    X -= X.mean(axis=0, keepdims=True)
    X /= X.std(axis=0, keepdims=True) + 1e-8

    # Static startup features plus a slowly improving age/log-traction feature.
    base_features = rng.normal(size=(N, n_features))
    quality = rng.normal(size=N)
    base_features[:, 0] = quality
    if n_features > 1:
        base_features[:, 1] = rng.normal(0, 0.8, size=N)  # founder / investor signal
    Z = np.repeat(base_features[None, :, :], T, axis=0)
    if n_features > 2:
        age = np.linspace(0.0, 1.0, T)[:, None]
        Z[:, :, 2] = base_features[None, :, 2] + age
    if n_features > 3:
        # sector-momentum-like feature visible to the ranker, not the sector GLM.
        Z[:, :, 3] = rng.normal(0.0, 0.2, size=(T, N)) + quality[None, :] * 0.2

    active = np.ones((T, N), dtype=bool)
    # A small fraction of companies enter after week 20, so risk-set sizes change.
    entry_week = rng.integers(0, max(1, T // 3), size=N)
    late = rng.random(N) < 0.25
    entry_week[~late] = 0
    for i in range(N):
        active[:entry_week[i], i] = False

    # True sector model: sparse positive excitation matrix with strong diagonals.
    intercept = rng.normal(-2.5, 0.25, size=M)
    beta = rng.normal(0.0, 0.15, size=(M, n_covariates))
    if n_covariates:
        beta[:, 0] += rng.normal(0.25, 0.05, size=M)  # risk-on covariate
    excitation = np.zeros((M, M, n_lags), dtype=float)
    decay = np.exp(-0.7 * np.arange(n_lags))
    decay /= decay.sum()
    for s in range(M):
        excitation[s, s, :] = rng.uniform(0.06, 0.11) * decay
        # two cross-sector links per sector.
        peers = rng.choice([r for r in range(M) if r != s], size=min(2, M - 1), replace=False)
        for r in peers:
            excitation[s, r, :] = rng.uniform(0.015, 0.045) * decay

    # True ranker: global signal, sector deviations, and negative cooldown.
    w_global = rng.normal(0.0, 0.35, size=n_features)
    w_global[0] += 1.0
    if n_features > 1:
        w_global[1] += 0.5
    w_sector = rng.normal(0.0, 0.12, size=(M, n_features))
    eta = -rng.uniform(1.2, 2.2, size=M)

    sector_counts = np.zeros((T, M), dtype=int)
    startup_counts = np.zeros((T, N), dtype=int)
    events = []

    for t in range(n_lags, T):
        log_rate = intercept + beta @ X[t]
        for lag in range(1, n_lags + 1):
            log_rate += excitation[:, :, lag - 1] @ sector_counts[t - lag]
        lam = np.exp(np.clip(log_rate, -10.0, 3.0))
        y_t = rng.poisson(lam)
        sector_counts[t] = y_t

        chosen_this_week: set[int] = set()
        for s in range(M):
            for _ in range(int(y_t[s])):
                cand = candidate_set(startup_sector, active, s, t)
                if cand.size == 0:
                    continue
                # No duplicate funding events for the same startup in the same week.
                cand = np.array([i for i in cand if i not in chosen_this_week], dtype=int)
                if cand.size == 0:
                    continue
                cd = cooldown_vector(startup_counts, t, cand, cooldown_weeks)
                scores = Z[t, cand] @ (w_global + w_sector[s]) + eta[s] * cd
                prob = _softmax_from_scores(scores)
                chosen = int(rng.choice(cand, p=prob))
                startup_counts[t, chosen] += 1
                chosen_this_week.add(chosen)
                events.append((t, s, chosen))

    return SyntheticStartupMarket(
        sector_counts=sector_counts,
        startup_counts=startup_counts,
        events=np.asarray(events, dtype=int),
        covariates=X,
        startup_features=Z,
        startup_sector=startup_sector,
        active=active,
        true_sector_intercept=intercept,
        true_sector_beta=beta,
        true_excitation=excitation,
        true_ranker_weights=w_global + w_sector,
        true_ranker_cooldown=eta,
    )


def simulate_marked_paths(
    sector_model: SectorCountResult,
    ranker: StartupRankerResult,
    startup_features,
    startup_sector,
    active,
    initial_sector_counts,
    initial_startup_counts,
    covariates,
    *,
    start_week,
    end_week,
    n_paths=100,
    seed=0,
):
    """Simulate sector counts and startup marks from fitted two-layer model."""
    rng = np.random.default_rng(seed)
    Z = np.asarray(startup_features, dtype=float)
    sector_hist0 = np.asarray(initial_sector_counts, dtype=int)
    startup_hist0 = np.asarray(initial_startup_counts, dtype=int)
    T, N, _ = Z.shape
    M = sector_hist0.shape[1]
    start_week = int(start_week)
    end_week = int(end_week)
    H = end_week - start_week
    sector_paths = np.zeros((n_paths, H, M), dtype=int)
    startup_paths = np.zeros((n_paths, H, N), dtype=int)

    for pth in range(int(n_paths)):
        sector_counts = sector_hist0.copy()
        startup_counts = startup_hist0.copy()
        for t in range(start_week, end_week):
            rates = sector_rate_at(sector_model, sector_counts, covariates, t)
            y_t = rng.poisson(rates)
            sector_counts[t] = y_t
            chosen_this_week: set[int] = set()
            for s in range(M):
                for _ in range(int(y_t[s])):
                    cand, prob = ranker_predict_proba(ranker, Z, startup_sector, active,
                                                       startup_counts, week=t, sector=s)
                    if cand.size == 0:
                        continue
                    keep = np.array([i not in chosen_this_week for i in cand], dtype=bool)
                    cand = cand[keep]
                    prob = prob[keep]
                    if cand.size == 0:
                        continue
                    prob = prob / prob.sum()
                    chosen = int(rng.choice(cand, p=prob))
                    startup_counts[t, chosen] += 1
                    chosen_this_week.add(chosen)
            sector_paths[pth, t - start_week] = sector_counts[t]
            startup_paths[pth, t - start_week] = startup_counts[t]
    return {"sector_counts": sector_paths, "startup_counts": startup_paths}


def backtest_synthetic_pipeline(
    *,
    seed=0,
    T=180,
    train_end=120,
    n_sectors=11,
    startups_per_sector=35,
    n_lags=4,
    cooldown_weeks=26,
    n_paths=100,
):
    """Run an end-to-end synthetic backtest for the two-layer architecture."""
    data = simulate_synthetic_startup_market(
        T=T,
        n_sectors=n_sectors,
        startups_per_sector=startups_per_sector,
        n_lags=n_lags,
        cooldown_weeks=cooldown_weeks,
        seed=seed,
    )
    sector_fit = fit_sector_count_model(
        data.sector_counts,
        data.covariates,
        n_lags=n_lags,
        train_end=train_end,
        l2=1e-3,
    )
    ranker_fit = fit_startup_ranker(
        data.events,
        data.startup_features,
        data.startup_sector,
        data.active,
        data.startup_counts,
        train_end=train_end,
        cooldown_weeks=cooldown_weeks,
        l2_global=1e-3,
        l2_sector=5e-2,
    )

    # Sector held-out one-step scores, using actual lag history.
    rates_all = sector_fit.rates(data.sector_counts, data.covariates)
    model_rates = rates_all[train_end:T]
    baseline_rates = sector_baseline_rates(data.sector_counts, train_end, train_end, T)
    observed = data.sector_counts[train_end:T]
    sector_model_nll = poisson_nll(observed, model_rates) / observed.size
    sector_baseline_nll = poisson_nll(observed, baseline_rates) / observed.size

    rank_metrics = evaluate_ranker(
        ranker_fit,
        data.events,
        data.startup_features,
        data.startup_sector,
        data.active,
        data.startup_counts,
        start_week=train_end,
        end_week=T,
        topk=(1, 5, 10),
    )

    paths = simulate_marked_paths(
        sector_fit,
        ranker_fit,
        data.startup_features,
        data.startup_sector,
        data.active,
        data.sector_counts.copy(),
        data.startup_counts.copy(),
        data.covariates,
        start_week=train_end,
        end_week=T,
        n_paths=n_paths,
        seed=seed + 123,
    )
    sim_mean_sector = paths["sector_counts"].mean(axis=0)
    sim_sector_mae = float(np.mean(np.abs(sim_mean_sector - observed)))
    base_sector_mae = float(np.mean(np.abs(baseline_rates - observed)))

    return {
        "data": data,
        "sector_fit": sector_fit,
        "ranker_fit": ranker_fit,
        "metrics": {
            "n_events_total": int(data.events.shape[0]),
            "n_events_train": int(np.sum(data.events[:, 0] < train_end)),
            "n_events_test": int(np.sum(data.events[:, 0] >= train_end)),
            "sector_model_nll_per_cell": float(sector_model_nll),
            "sector_baseline_nll_per_cell": float(sector_baseline_nll),
            "sector_nll_improvement": float(sector_baseline_nll - sector_model_nll),
            "sim_sector_mae": sim_sector_mae,
            "baseline_sector_mae": base_sector_mae,
            "ranker": rank_metrics,
        },
    }


__all__ = [
    "SyntheticStartupMarket",
    "SectorCountResult",
    "StartupRankerResult",
    "fit_sector_count_model",
    "sector_rate_at",
    "sector_baseline_rates",
    "fit_startup_ranker",
    "candidate_set",
    "cooldown_vector",
    "ranker_scores",
    "ranker_predict_proba",
    "evaluate_ranker",
    "simulate_synthetic_startup_market",
    "simulate_marked_paths",
    "backtest_synthetic_pipeline",
]
