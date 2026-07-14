"""Vectorized reference fitters for the two-stage model (validation utilities).

Mathematically identical to hawkes_calibration.sector_ranker fitters but built on
stacked design matrices instead of per-week / per-event Python loops:

* the sector GLM is separable by receiving sector -> M independent Poisson GLMs
  on a (T-L, 1+p+M*L) design (each fits in <1 s vs minutes for the loop version);
* the ranker objective is evaluated with padded candidate tensors.

These should eventually be upstreamed into the package (see EXPERIMENTS.md).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def fit_sector_glm_fast(
    sector_counts,
    covariates,
    *,
    n_lags=4,
    train_end=None,
    l2=1e-3,
    nonnegative_excitation=True,
    max_iter=500,
):
    """Per-sector vectorized Poisson GLM with lagged own/cross counts.

    Returns (intercept (M,), beta (M,p), excitation (M,M,L)).
    ``n_lags=0`` fits a genuine covariates-only GLM (no lag columns in the design).
    """
    y = np.asarray(sector_counts, float)
    X = np.asarray(covariates, float)
    T, M = y.shape
    p = X.shape[1]
    L = int(n_lags)
    train_end = T if train_end is None else int(train_end)
    w = np.arange(L, train_end)
    if L:
        lagmat = np.concatenate([y[w - l] for l in range(1, L + 1)], axis=1)  # (n, M*L)
    else:
        lagmat = np.zeros((len(w), 0))
    D = np.concatenate([np.ones((len(w), 1)), X[w], lagmat], axis=1)  # (n, 1+p+M*L)

    intercept = np.zeros(M)
    beta = np.zeros((M, p))
    exc = np.zeros((M, M, L))
    n_par = 1 + p + M * L
    bounds = [(None, None)] * (1 + p) + (
        [(0.0, None)] * (M * L) if nonnegative_excitation else [(None, None)] * (M * L)
    )

    for s in range(M):
        ys = y[w, s]
        x0 = np.zeros(n_par)
        x0[0] = np.log(max(ys.mean(), 1e-3))

        def obj(th, ys=ys):  # bind the per-sector target explicitly (B023)
            eta = np.clip(D @ th, -30, 20)
            lam = np.exp(eta)
            loss = float(np.sum(lam - ys * eta)) + 0.5 * l2 * float(np.sum(th[1:] ** 2))
            g = D.T @ (lam - ys)
            g[1:] += l2 * th[1:]
            return loss, g

        r = minimize(
            obj,
            x0,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options=dict(maxiter=max_iter, ftol=1e-10),
        )
        intercept[s] = r.x[0]
        beta[s] = r.x[1 : 1 + p]
        b = r.x[1 + p :].reshape(L, M)  # lag-major from lagmat construction
        exc[s] = b.T  # -> (M source, L)
    return intercept, beta, exc


def fit_choice_fast(
    cs, *, train_frac_days=0.8, l2_global=1e-4, l2_sector=1e-2, sector_deviations=True, max_iter=300
):
    """Conditional logit on padded choice sets from loaders.build_choice_sets.

    Score of candidate j at event e:  q = (w0 + u_{s(e)})' F[e, j].
    The newcomer alternative is just a candidate whose only nonzero feature is
    ``is_newcomer`` — its weight acts as the newcomer intercept (ASC).

    Returns (w0, u, train_nll, test_nll, test_random_nll, acc_top5, success).
    """
    F, msk, lab, sec, day = cs["F"], cs["mask"], cs["label"], cs["sector"], cs["day"]
    E, K, p = F.shape
    M = int(cs["n_sectors"])
    cut = np.quantile(day, train_frac_days)
    tr, te = day <= cut, day > cut
    S1 = np.zeros((E, M))
    S1[np.arange(E), sec] = 1.0

    def nll_and_grad(th, idx, want_grad=True):
        w0 = th[:p]
        u = th[p:].reshape(M, p) if sector_deviations else np.zeros((M, p))
        Wev = w0[None, :] + u[sec[idx]]
        sc = np.einsum("ekp,ep->ek", F[idx], Wev)
        sc = np.where(msk[idx], sc, -1e30)
        mx = sc.max(1, keepdims=True)
        lse = mx[:, 0] + np.log(np.exp(sc - mx).sum(1))
        n = idx.sum() if idx.dtype == bool else len(idx)
        ll = float(np.sum(lse - sc[np.arange(len(sc)), lab[idx]]))
        if not want_grad:
            return ll / max(n, 1)
        P = np.exp(sc - lse[:, None]) * msk[idx]
        dsc = P.copy()
        dsc[np.arange(len(sc)), lab[idx]] -= 1.0
        gW = np.einsum("ek,ekp->ep", dsc, F[idx])
        gw0 = gW.sum(0) + l2_global * w0
        gu = (S1[idx].T @ gW + l2_sector * u) if sector_deviations else np.zeros((M, p))
        ll += 0.5 * l2_global * float(np.sum(w0**2)) + 0.5 * l2_sector * float(np.sum(u**2))
        return ll, np.concatenate([gw0, gu.ravel()])

    npar = p + (M * p if sector_deviations else 0)
    r = minimize(
        lambda th: nll_and_grad(th, tr),
        np.zeros(npar),
        jac=True,
        method="L-BFGS-B",
        options=dict(maxiter=max_iter, ftol=1e-9),
    )
    w0 = r.x[:p]
    u = r.x[p:].reshape(M, p) if sector_deviations else np.zeros((M, p))
    train_nll = float(r.fun) / max(int(tr.sum()), 1)
    test_nll = nll_and_grad(r.x, te, want_grad=False)
    rand_nll = float(np.mean(np.log(msk[te].sum(1))))
    # top-5 accuracy on test
    Wev = w0[None, :] + u[sec[te]]
    sc = np.einsum("ekp,ep->ek", F[te], Wev)
    sc = np.where(msk[te], sc, -1e30)
    top5 = np.argsort(-sc, axis=1)[:, :5]
    acc5 = float(np.mean([lab[te][i] in top5[i] for i in range(len(top5))]))
    return w0, u, train_nll, test_nll, rand_nll, acc5, bool(r.success)


def fit_ranker_fast(
    events,
    Z,
    startup_sector,
    active,
    startup_counts,
    *,
    train_end,
    cooldown_weeks=26,
    l2_global=1e-3,
    l2_sector=1e-2,
    max_iter=300,
    max_candidates=None,
    seed=0,
):
    """Padded-tensor version of fit_startup_ranker (identical likelihood).

    Optional ``max_candidates`` subsamples each risk set (keeping the funded firm)
    for a fast approximate fit — standard sampled-softmax trick.
    """
    rng = np.random.default_rng(seed)
    events = np.asarray(events, int)
    ev = events[events[:, 0] < train_end]
    T, N, p = Z.shape
    M = int(startup_sector.max()) + 1

    cand_list, chosen_pos, sec_list, week_list = [], [], [], []
    for t, s, i in ev:
        cand = np.flatnonzero(active[t] & (startup_sector == s))
        if cand.size == 0 or i not in cand:
            continue
        if max_candidates and cand.size > max_candidates:
            keep = rng.choice(cand[cand != i], size=max_candidates - 1, replace=False)
            cand = np.concatenate([[i], keep])
        cand_list.append(cand)
        chosen_pos.append(int(np.flatnonzero(cand == i)[0]))
        sec_list.append(int(s))
        week_list.append(int(t))
    E = len(cand_list)
    K = max(len(c) for c in cand_list)
    F = np.zeros((E, K, p))
    CD = np.zeros((E, K))
    msk = np.zeros((E, K), bool)
    cum = np.cumsum(startup_counts, axis=0)
    for e, (cand, t) in enumerate(zip(cand_list, week_list)):
        k = len(cand)
        F[e, :k] = Z[t, cand]
        lo = max(0, t - cooldown_weeks)
        CD[e, :k] = ((cum[t - 1, cand] - (cum[lo - 1, cand] if lo > 0 else 0)) > 0) if t > 0 else 0
        msk[e, :k] = True
    sec = np.asarray(sec_list)
    cp = np.asarray(chosen_pos)
    S1 = np.zeros((E, M))
    S1[np.arange(E), sec] = 1.0

    def unpack(th):
        w0 = th[:p]
        u = th[p : p + M * p].reshape(M, p)
        eta = th[p + M * p :]
        return w0, u, eta

    def obj(th):
        w0, u, eta = unpack(th)
        Wev = w0[None, :] + u[sec]  # (E,p)
        sc = np.einsum("ekp,ep->ek", F, Wev) + eta[sec][:, None] * CD
        sc = np.where(msk, sc, -1e30)
        mx = sc.max(1, keepdims=True)
        lse = mx[:, 0] + np.log(np.exp(sc - mx).sum(1))
        loss = float(np.sum(lse - sc[np.arange(E), cp]))
        P = np.exp(sc - lse[:, None]) * msk
        dsc = P.copy()
        dsc[np.arange(E), cp] -= 1.0  # (E,K)
        gW = np.einsum("ek,ekp->ep", dsc, F)  # (E,p)
        gw0 = gW.sum(0) + l2_global * w0
        gu = S1.T @ gW + l2_sector * u
        geta = S1.T @ (dsc * CD).sum(1) + l2_sector * eta
        loss += 0.5 * l2_global * float(np.sum(w0**2))
        loss += 0.5 * l2_sector * (float(np.sum(u**2)) + float(np.sum(eta**2)))
        return loss, np.concatenate([gw0, gu.ravel(), geta])

    x0 = np.zeros(p + M * p + M)
    x0[p + M * p :] = -0.5
    bnd = [(None, None)] * (p + M * p) + [(None, 0.0)] * M
    r = minimize(
        obj, x0, jac=True, method="L-BFGS-B", bounds=bnd, options=dict(maxiter=max_iter, ftol=1e-9)
    )
    w0, u, eta = unpack(r.x)
    return w0, u, eta, float(r.fun / max(E, 1)), bool(r.success), E
