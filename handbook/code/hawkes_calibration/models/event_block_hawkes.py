r"""
One-layer, marked, block-structured **log-linear (exp-link) Hawkes** for EXACT
event times in an **open population** (firms enter and leave).

This is the "different road" from the interval-censored two-stage model: here we
assume the event *times* are observed (the only censoring is that we see positive
events -- fundings -- and never an explicit "tried and failed").  With exact times
the full point-process likelihood is available, which buys two things the linear
interval-censored route cannot have:

1. an **exponential link**, ``lambda = exp(eta)``, so the self term can be
   **negative** (a refractory / cooldown self-inhibition) while the intensity stays
   positive -- impossible for a nonnegative linear Hawkes kernel; and
2. a **globally concave** log-likelihood (the parameters enter ``eta`` linearly), so
   the MLE is unique.

Model
-----
Firm ``i`` lives in sector ``s(i)`` and is *at risk* on ``[entry_i, exit_i)`` (the
open-population part).  Its conditional intensity is

    lambda_i(t) = 1{entry_i <= t < exit_i} * exp( eta_i(t) ),

    eta_i(t) = a_{s(i)} + beta_{s(i)} . X_i
               - rho_{s(i)} * R_i(t)                      # own recency -> self-inhibition
               + sum_b A[s(i), b] * E_{i,b}(t),           # block (sector) excitation

    R_i(t)    = sum_{own events u<t}        exp(-w_self  (t-u)),
    E_{i,b}(t)= sum_{events u<t of OTHER firms in sector b} exp(-w_cross (t-u)).

``A`` is the ``M x M`` **sector** excitation matrix (not ``N x N`` over firms): firms
are exchangeable within a sector and differ only through covariates ``X_i``.  A
sparsity ``mask`` zeroes disallowed sector pairs (block / adjacency structure), so the
parameter count is ``O(M^2 + M p)`` regardless of how many firms come and go.

Estimation
----------
On a time grid this is the Berman--Turner device: a Poisson regression with offset
``log(dt)`` whose log-likelihood

    l(Theta) = sum_{i,g} [ dN_i[g] * eta_i[g] - Y_i[g] * exp(eta_i[g]) * dt ]

is concave in ``Theta = (a, beta, rho, A)``.  ``fit_block_hawkes`` minimises the
penalised negative log-likelihood with an analytic gradient (the point-process GLM
"observed minus expected" score), box constraints ``rho>=0`` (genuine inhibition) and
``A>=0`` on the mask (genuine excitation), via L-BFGS-B (SciPy if present, else the
numpy fallback).

Public API
----------
``BlockHawkesData``    -- container for a simulated/real event-time panel.
``BlockHawkesResult``  -- fitted parameters.
``simulate_block_hawkes`` -- synthetic open-population market with known parameters.
``block_hawkes_loglik`` -- (loglik, gradient) at given parameters.
``fit_block_hawkes``   -- concave MLE.
``block_hawkes_intensity`` -- per-firm intensity paths for inspection.
``evaluate_block_hawkes`` -- held-out log-likelihood and time-rescaling GOF.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:  # SciPy is optional.
    from scipy.optimize import minimize
except ImportError:  # pragma: no cover
    from .._opt_fallback import minimize  # numpy projected-Adam fallback


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------
@dataclass
class BlockHawkesData:
    """An event-time panel for an open population of firms."""

    events: np.ndarray  # (n, 2) columns [time, firm_index]
    firm_sector: np.ndarray  # (N,) sector id per firm
    entry: np.ndarray  # (N,) at-risk start time
    exit: np.ndarray  # (N,) at-risk end time
    X: np.ndarray  # (N, p) static firm covariates (may be (N,0))
    T: float  # observation horizon
    n_sectors: int

    @property
    def n_firms(self):
        return int(self.firm_sector.shape[0])

    @property
    def n_events(self):
        return int(self.events.shape[0])


@dataclass
class BlockHawkesResult:
    a: np.ndarray  # (M,) sector log-baselines
    beta: np.ndarray  # (M, p) sector covariate coefficients
    rho: np.ndarray  # (M,) self-inhibition (>=0 => inhibitory via exp link)
    A: np.ndarray  # (M, M) block excitation matrix (masked)
    w_self: float  # fixed self decay
    w_cross: float  # fixed cross decay
    mask: np.ndarray  # (M, M) bool, allowed excitation entries
    loglik: float
    success: bool
    message: str
    n_events: int
    n_grid: int
    l2: float = 0.0
    history: list = field(default_factory=list)

    def summary(self):
        lines = [
            f"  log-likelihood = {self.loglik:.3f}   (events={self.n_events}, grid={self.n_grid})",
            f"  sector baselines a      = {np.round(self.a, 3).tolist()}",
            f"  self-inhibition rho     = {np.round(self.rho, 3).tolist()}   "
            f"(>=0 => recent own funding lowers the rate)",
            f"  block excitation A diag = {np.round(np.diag(self.A), 3).tolist()}",
            f"  decays: w_self={self.w_self:.3f}  w_cross={self.w_cross:.3f}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Grid machinery (Berman--Turner)
# ---------------------------------------------------------------------------
def _bin_events(events, n_firms, n_grid, dt):
    """Counts ``dN[i, g]`` of firm-i events falling in grid cell g."""
    dN = np.zeros((n_firms, n_grid))
    if events.shape[0]:
        cells = np.clip(np.floor(events[:, 0] / dt).astype(int), 0, n_grid - 1)
        firms = events[:, 1].astype(int)
        np.add.at(dN, (firms, cells), 1.0)
    return dN


def _decay_field(dN, dt, w):
    r"""Left-continuous exponential field ``F[i, g] = sum_{c<g} dN[i,c] e^{-w (g-c) dt}``.

    Recursion ``F[:, g] = (F[:, g-1] + dN[:, g-1]) e^{-w dt}`` (excludes the current
    cell, so the intensity at a firing time does not depend on that firing).
    """
    N, G = dN.shape
    F = np.zeros((N, G))
    decay = float(np.exp(-w * dt))
    for g in range(1, G):
        F[:, g] = (F[:, g - 1] + dN[:, g - 1]) * decay
    return F


def _at_risk(entry, exit, n_grid, dt):
    """``Y[i, g] = 1`` while firm i is at risk at the left edge of cell g."""
    g = (np.arange(n_grid) + 0.5) * dt  # cell midpoints
    Y = ((entry[:, None] <= g[None, :]) & (g[None, :] < exit[:, None])).astype(float)
    return Y


def _prepare_grid(data: BlockHawkesData, n_grid, w_self, w_cross):
    """Precompute everything that does not depend on the parameters."""
    N, M = data.n_firms, data.n_sectors
    T = float(data.T)
    dt = T / n_grid
    dN = _bin_events(data.events, N, n_grid, dt)
    Y = _at_risk(np.asarray(data.entry, float), np.asarray(data.exit, float), n_grid, dt)
    F_self = _decay_field(dN, dt, w_self)  # (N, G) -> R_i (own recency)
    F_cross = _decay_field(dN, dt, w_cross)  # (N, G) own field at cross decay
    sec = np.asarray(data.firm_sector, int)
    nb = np.maximum(np.bincount(sec, minlength=M), 1)  # firms per sector
    # sector aggregate, normalised by sector size so excitation responds to the
    # *average* recent activity (N-independent, keeps the process well-scaled):
    #   Sn[b, g] = (1/n_b) sum_{j in b} F_cross[j, g]
    Sn = np.zeros((M, n_grid))
    for b in range(M):
        idx = np.flatnonzero(sec == b)
        if idx.size:
            Sn[b] = F_cross[idx].sum(axis=0) / nb[b]
    Fc_norm = F_cross / nb[sec][:, None]  # own contribution, same 1/n_b scale
    return dict(
        dt=dt,
        dN=dN,
        Y=Y,
        F_self=F_self,
        Fc_norm=Fc_norm,
        Sn=Sn,
        sec=sec,
        nb=nb,
        N=N,
        M=M,
        G=n_grid,
        X=np.asarray(data.X, float),
    )


# ---------------------------------------------------------------------------
# Parameter packing
# ---------------------------------------------------------------------------
def _dims(M, p):
    return M, M * p, M, M * M  # a, beta, rho, A


def _unpack(theta, M, p):
    na, nb, nr, nA = _dims(M, p)
    a = theta[:na]
    beta = theta[na : na + nb].reshape(M, p) if p else np.zeros((M, 0))
    rho = theta[na + nb : na + nb + nr]
    A = theta[na + nb + nr : na + nb + nr + nA].reshape(M, M)
    return a, beta, rho, A


def _eta(prep, a, beta, rho, A):
    """Log-intensity ``eta[i, g]`` for every firm and grid cell."""
    sec, X = prep["sec"], prep["X"]
    base = a[sec].astype(float)
    if X.shape[1]:
        base = base + (X * beta[sec]).sum(axis=1)  # per-firm covariate baseline
    # cross excitation: sum_b A[s(i), b] Sn[b]  minus the firm's own contribution at diag
    cross = A[sec] @ prep["Sn"]  # (N, G)
    self_in_cross = A[sec, sec][:, None] * prep["Fc_norm"]
    eta = base[:, None] - rho[sec][:, None] * prep["F_self"] + cross - self_in_cross
    return eta


# ---------------------------------------------------------------------------
# Likelihood + gradient (concave)
# ---------------------------------------------------------------------------
def _nll_and_grad(theta, prep, M, p, l2, mask_flat):
    a, beta, rho, A = _unpack(theta, M, p)
    dt, dN, Y = prep["dt"], prep["dN"], prep["Y"]
    sec, X = prep["sec"], prep["X"]

    eta = _eta(prep, a, beta, rho, A)
    np.clip(eta, -30.0, 20.0, out=eta)
    lam = np.exp(eta)
    comp = Y * lam * dt  # expected counts per cell
    # log-likelihood: sum dN*eta - sum comp
    ll = float(np.sum(dN * eta) - np.sum(comp))
    r = dN - comp  # (N, G) residual (observed - expected)

    Rsum = r.sum(axis=1)  # (N,)
    g_a = np.bincount(sec, weights=Rsum, minlength=M)
    if p:
        g_beta = np.zeros((M, p))
        for s in range(M):
            idx = np.flatnonzero(sec == s)
            if idx.size:
                g_beta[s] = (X[idx] * Rsum[idx][:, None]).sum(axis=0)
    else:
        g_beta = np.zeros((M, 0))
    rF_self = (r * prep["F_self"]).sum(axis=1)  # (N,)
    g_rho = -np.bincount(sec, weights=rF_self, minlength=M)

    # g_A[s, b] = sum_{i in s} ( r_i . Sn[b] )  - (b==s) sum_{i in s} r_i.Fc_norm_i
    RS = r @ prep["Sn"].T  # (N, M)
    rF_cross = (r * prep["Fc_norm"]).sum(axis=1)  # (N,)
    g_A = np.zeros((M, M))
    for s in range(M):
        idx = np.flatnonzero(sec == s)
        if idx.size:
            g_A[s] = RS[idx].sum(axis=0)
            g_A[s, s] -= rF_cross[idx].sum()

    # ridge on beta and A (not on a, rho)
    ll -= 0.5 * l2 * (float(np.sum(beta**2)) + float(np.sum(A**2)))
    g_beta -= l2 * beta
    g_A -= l2 * A

    grad = np.concatenate([g_a, g_beta.ravel(), g_rho, g_A.ravel()])
    grad *= mask_flat  # zero gradient on disallowed A entries
    return -ll, -grad


def block_hawkes_loglik(data, result_or_theta, *, n_grid=None, w_self=None, w_cross=None, l2=0.0):
    """Return ``(loglik, gradient)`` at a parameter point (a fitted result or a vector)."""
    if isinstance(result_or_theta, BlockHawkesResult):
        res = result_or_theta
        M, p = res.a.shape[0], res.beta.shape[1]
        theta = np.concatenate([res.a, res.beta.ravel(), res.rho, res.A.ravel()])
        w_self, w_cross = res.w_self, res.w_cross
        n_grid = n_grid or res.n_grid
        mask = res.mask
    else:
        raise ValueError("pass a BlockHawkesResult, or use fit_block_hawkes")
    prep = _prepare_grid(data, n_grid, w_self, w_cross)
    mask_flat = _mask_flat(mask, M, p)
    nll, ngrad = _nll_and_grad(theta, prep, M, p, l2, mask_flat)
    return -nll, -ngrad


def _mask_flat(mask, M, p):
    na, nb, nr, nA = _dims(M, p)
    mf = np.ones(na + nb + nr + nA)
    mf[na + nb + nr :] = np.asarray(mask, float).ravel()
    return mf


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------
def fit_block_hawkes(
    data: BlockHawkesData,
    *,
    n_grid=400,
    w_self=1.0,
    w_cross=1.0,
    mask=None,
    l2=1e-2,
    nonneg_excitation=True,
    nonneg_self_inhibition=True,
    max_iter=300,
):
    r"""Concave MLE of the block log-linear Hawkes.

    Parameters
    ----------
    n_grid : Berman--Turner time-grid resolution (compensator quadrature).
    w_self, w_cross : fixed exponential decays for the self and cross fields.
    mask : (M, M) bool of allowed excitation entries (block/adjacency sparsity).
        Defaults to all-ones (full ``M x M``).
    l2 : ridge on ``beta`` and ``A``.
    nonneg_excitation : constrain on-mask ``A >= 0`` (genuine excitation; the self
        term carries inhibition separately).
    nonneg_self_inhibition : constrain ``rho >= 0`` so recent own events *lower* the
        rate (the exp link makes this a multiplicative refractory factor < 1).
    """
    M = int(data.n_sectors)
    p = int(np.asarray(data.X, float).shape[1])
    if mask is None:
        mask = np.ones((M, M), dtype=bool)
    mask = np.asarray(mask, bool)
    prep = _prepare_grid(data, n_grid, w_self, w_cross)
    mask_flat = _mask_flat(mask, M, p)

    na, nb, nr, nA = _dims(M, p)
    n = na + nb + nr + nA
    # warm start: baseline ~ log(mean firm rate), small excitation, mild inhibition
    dt = prep["dt"]
    at_risk_time = prep["Y"].sum(axis=1) * dt
    rate0 = data.n_events / max(at_risk_time.sum(), 1e-6)
    x0 = np.zeros(n)
    x0[:na] = np.log(max(rate0, 1e-3))
    x0[na + nb : na + nb + nr] = 0.2  # rho start (>0)
    A0 = np.zeros((M, M))
    A0[mask] = 0.05
    x0[na + nb + nr :] = A0.ravel()

    bounds = [(None, None)] * n
    if nonneg_self_inhibition:
        for j in range(nr):
            bounds[na + nb + j] = (0.0, None)
    for idx in range(nA):
        s, b = divmod(idx, M)
        if not mask[s, b]:
            bounds[na + nb + nr + idx] = (0.0, 0.0)
        elif nonneg_excitation:
            bounds[na + nb + nr + idx] = (0.0, None)

    hist = []

    def fun(theta):
        nll, ngrad = _nll_and_grad(theta, prep, M, p, l2, mask_flat)
        hist.append(nll)
        return nll, ngrad

    opt = minimize(
        fun,
        x0,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": int(max_iter), "ftol": 1e-9},
    )
    a, beta, rho, A = _unpack(opt.x, M, p)
    A = A * mask
    return BlockHawkesResult(
        a=a,
        beta=beta,
        rho=rho,
        A=A,
        w_self=float(w_self),
        w_cross=float(w_cross),
        mask=mask,
        loglik=float(-opt.fun),
        success=bool(opt.success),
        message=str(opt.message),
        n_events=int(data.n_events),
        n_grid=int(n_grid),
        l2=float(l2),
        history=hist,
    )


# ---------------------------------------------------------------------------
# Intensity inspection + evaluation
# ---------------------------------------------------------------------------
def block_hawkes_intensity(data, result, *, n_grid=None):
    """Return ``(grid_times, lambda[N, G])`` under a fitted result (for plotting)."""
    n_grid = n_grid or result.n_grid
    prep = _prepare_grid(data, n_grid, result.w_self, result.w_cross)
    eta = _eta(prep, result.a, result.beta, result.rho, result.A)
    np.clip(eta, -30.0, 20.0, out=eta)
    lam = prep["Y"] * np.exp(eta)
    tg = (np.arange(n_grid) + 0.5) * prep["dt"]
    return tg, lam


def evaluate_block_hawkes(data, result, *, n_grid=None):
    """Held-out average log-likelihood per event and a time-rescaling GOF.

    Under a correctly specified fit the compensators between consecutive events of a
    firm are i.i.d. Exp(1); we report the mean rescaled inter-event time (target 1.0)
    and the KS distance of its distribution to Exp(1).
    """
    n_grid = n_grid or result.n_grid
    M, p = result.a.shape[0], result.beta.shape[1]
    theta = np.concatenate([result.a, result.beta.ravel(), result.rho, result.A.ravel()])
    prep = _prepare_grid(data, n_grid, result.w_self, result.w_cross)
    mask_flat = _mask_flat(result.mask, M, p)
    nll, _ = _nll_and_grad(theta, prep, M, p, 0.0, mask_flat)
    ll = -nll
    # time-rescaling: cumulative compensator per firm, differenced at its events
    eta = _eta(prep, result.a, result.beta, result.rho, result.A)
    np.clip(eta, -30.0, 20.0, out=eta)
    lam = prep["Y"] * np.exp(eta)
    Lam = np.cumsum(lam * prep["dt"], axis=1)  # (N, G)
    dt = prep["dt"]
    taus = []
    ev = data.events[np.argsort(data.events[:, 0])]
    last = {}
    for t, i in ev:
        i = int(i)
        g = int(min(t / dt, n_grid - 1))
        if i in last:
            taus.append(float(Lam[i, g] - last[i]))
        last[i] = float(Lam[i, g])
    taus = np.asarray(taus) if taus else np.zeros(0)
    ks = _ks_exp1(taus) if taus.size else float("nan")
    return {
        "loglik": float(ll),
        "loglik_per_event": float(ll / max(data.n_events, 1)),
        "n_rescaled": int(taus.size),
        "rescaled_mean": float(taus.mean()) if taus.size else float("nan"),
        "rescaled_ks_exp1": ks,
    }


def _ks_exp1(taus):
    x = np.sort(taus[np.isfinite(taus)])
    if x.size == 0:
        return float("nan")
    emp = np.arange(1, x.size + 1) / x.size
    cdf = 1.0 - np.exp(-x)
    return float(np.max(np.abs(emp - cdf)))


# ---------------------------------------------------------------------------
# Simulation (open population, exp link, at-risk indicators)
# ---------------------------------------------------------------------------
def simulate_block_hawkes(
    *,
    T=60.0,
    n_sectors=4,
    firms_per_sector=20,
    p_covariates=2,
    n_grid=600,
    w_self=1.0,
    w_cross=1.0,
    a=None,
    rho=None,
    A=None,
    beta=None,
    mask=None,
    late_entry_frac=0.3,
    early_exit_frac=0.1,
    seed=0,
):
    r"""Simulate an open-population market from known block-Hawkes parameters.

    The ground truth has **positive** within-sector excitation (``A`` diagonal > 0),
    sparse positive cross-sector excitation on ``mask``, and **positive** ``rho`` so a
    firm's own recent round *suppresses* its rate (self-inhibition through the exp
    link).  About ``late_entry_frac`` of firms enter after t=0 and ``early_exit_frac``
    exit before T, exercising the open population.  Returns ``BlockHawkesData`` plus the
    ground-truth parameter dict.
    """
    rng = np.random.default_rng(seed)
    M = n_sectors
    N = M * firms_per_sector
    sec = np.repeat(np.arange(M), firms_per_sector)
    X = rng.normal(0.0, 1.0, size=(N, p_covariates)) if p_covariates else np.zeros((N, 0))

    if mask is None:
        mask = np.eye(M, dtype=bool)
        for s in range(M):  # a couple of cross links
            mask[s, (s + 1) % M] = True
    mask = np.asarray(mask, bool)
    if a is None:
        a = np.full(M, -3.2) + rng.normal(0, 0.1, M)
    if rho is None:
        rho = rng.uniform(0.6, 1.1, M)  # genuine self-inhibition
    if beta is None:
        beta = rng.normal(0.0, 0.3, size=(M, p_covariates)) if p_covariates else np.zeros((M, 0))
    if A is None:
        A = np.zeros((M, M))
        A[np.arange(M), np.arange(M)] = rng.uniform(0.5, 0.9, M)  # within-sector
        for s in range(M):
            A[s, (s + 1) % M] = rng.uniform(0.1, 0.3)  # cross-sector
    A = A * mask

    dt = T / n_grid
    entry = np.zeros(N)
    exit = np.full(N, T)
    n_late = int(late_entry_frac * N)
    late = rng.choice(N, size=n_late, replace=False)
    entry[late] = rng.uniform(0.0, 0.5 * T, size=n_late)
    n_early = int(early_exit_frac * N)
    early = rng.choice(N, size=n_early, replace=False)
    exit[early] = np.minimum(T, entry[early] + rng.uniform(0.2 * T, 0.6 * T, size=n_early))

    nb = np.maximum(np.bincount(sec, minlength=M), 1)  # firms per sector
    base = a[sec] + ((X * beta[sec]).sum(axis=1) if X.shape[1] else 0.0)  # (N,)
    F_self = np.zeros(N)
    F_cross = np.zeros(N)
    decay_s = np.exp(-w_self * dt)
    decay_c = np.exp(-w_cross * dt)
    events = []
    for g in range(n_grid):
        t = g * dt
        Sraw = np.zeros(M)
        np.add.at(Sraw, sec, F_cross)
        Sn = Sraw / nb  # size-normalised sector field
        cross = A[sec] @ Sn - A[sec, sec] * (F_cross / nb[sec])
        eta = base - rho[sec] * F_self + cross
        at_risk = (entry <= t) & (t < exit)
        lam = np.where(at_risk, np.exp(np.clip(eta, -30, 20)), 0.0)
        counts = rng.poisson(lam * dt)
        # decay fields, then inject this cell's events
        F_self *= decay_s
        F_cross *= decay_c
        fired = np.flatnonzero(counts > 0)
        for i in fired:
            for _ in range(int(counts[i])):
                events.append((t + rng.uniform(0, dt), i))
            F_self[i] += counts[i]
            F_cross[i] += counts[i]
    events = np.array(sorted(events), dtype=float) if events else np.zeros((0, 2))
    data = BlockHawkesData(
        events=events, firm_sector=sec, entry=entry, exit=exit, X=X, T=float(T), n_sectors=M
    )
    truth = dict(a=a, beta=beta, rho=rho, A=A, mask=mask, w_self=w_self, w_cross=w_cross)
    return data, truth


__all__ = [
    "BlockHawkesData",
    "BlockHawkesResult",
    "simulate_block_hawkes",
    "fit_block_hawkes",
    "block_hawkes_loglik",
    "block_hawkes_intensity",
    "evaluate_block_hawkes",
]
