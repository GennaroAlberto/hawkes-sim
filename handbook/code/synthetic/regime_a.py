"""Regime A: well-specified two-stage world.

Weekly sector counts follow the package's discrete-time Hawkes/GLM exactly:

    log lam_{s,t} = a_s + beta_s' X_t + sum_{r,l} b_{s,r,l} Y_{r,t-l} + g * F_t

with X_t the *point-in-time standardized* macro covariates [FFUND, CPI_YOY, RUNEMP]
and F_t a latent risk-appetite factor (small weight ``latent_strength``; it is
independent of X, so it adds over-dispersion without biasing beta).

Given a sector event, the funded firm is a risk-set softmax over ALL live firms of
the sector (tracked or not) with score

    q_i = (w0 + u_s)' z_{i,t} + eta_s * cooldown_{i,t}

where z are the five point-in-time observable features of marks.FEATURE_NAMES and
cooldown is the package's 0/1 funded-in-last-K-weeks indicator.  Latent quality
does NOT enter the score (it only drives deal marks), so the ranker is fully
recoverable from observables up to observation noise.

Stability: raw-count log-link feedback is geometrically explosive (the paper's
"Mechanism B").  The raw excitation matrix is scaled so that
``effective_radius = rho(G) * mean_count`` stays SMALL (default 0.15): at 0.35
the μ=target fixed point is only metastable — a lucky burst tips the process
into a supercritical high-volume state.  Weekly counts are additionally capped
by the live-firm risk set size, and the intercept is calibrated
self-consistently against the realized volume.
"""

from __future__ import annotations

import numpy as np

from .marks import FirmState, features_at


def _cooldown(last_deal_week, n_deals, week, cooldown_weeks):
    gap = week - last_deal_week
    return ((n_deals > 0) & (gap >= 1) & (gap <= cooldown_weeks)).astype(float)


def simulate_regime_a(
    universe,
    quality,
    Xw_raw,
    latent,
    *,
    n_sectors=12,
    n_lags=4,
    cooldown_weeks=26,
    target_mean=8.0,
    effective_radius=0.15,
    latent_strength=0.15,
    covariate_scale=1.0,
    seed=0,
):
    rng = np.random.default_rng(seed + 303)
    T = Xw_raw.shape[0]
    M = n_sectors
    N = len(universe)
    sector_of = universe["sector_idx"].to_numpy()
    entry = universe["entry_week"].to_numpy()
    exitw = universe["exit_week"].to_numpy().copy()

    mu_x, sd_x = Xw_raw.mean(0), Xw_raw.std(0) + 1e-9
    X = (Xw_raw - mu_x) / sd_x

    # --- true sector GLM ---------------------------------------------------
    beta = np.stack(
        [rng.normal(-0.30, 0.08, M), rng.normal(-0.12, 0.06, M), rng.normal(-0.20, 0.07, M)], axis=1
    ) * float(covariate_scale)
    intercept = (
        np.log(target_mean * (1.0 - effective_radius))
        - 0.5 * (beta**2).sum(1).mean()
        + rng.normal(0.0, 0.20, M)
    )
    decay = np.exp(-0.7 * np.arange(n_lags))
    decay /= decay.sum()
    G = np.zeros((M, M))
    for s in range(M):
        G[s, s] = rng.uniform(0.7, 1.0)
        for r in rng.choice([r for r in range(M) if r != s], size=2, replace=False):
            G[s, r] = rng.uniform(0.15, 0.4)
    rho_raw = float(np.max(np.abs(np.linalg.eigvals(G))))
    G *= (effective_radius / target_mean) / rho_raw
    excitation = G[:, :, None] * decay[None, None, :]

    # --- true ranker ---------------------------------------------------------
    p_z = 5
    # moderate weights: keep the funded-firm distribution realistic (no
    # rich-get-richer collapse from the raised/stage feedback loop)
    w_global = np.array([0.30, 0.30, -0.15, 0.15, 0.0])
    w_sector = rng.normal(0.0, 0.15, size=(M, p_z))
    eta = -rng.uniform(2.2, 3.2, size=M)

    # --- calibrate intercept self-consistently (Mechanism B drift) -----------
    # The log-link raw-count feedback amplifies volume above exp(intercept)'s
    # naive prediction (Jensen + near-critical drift).  Run a counts-only
    # simulation and shift the intercept until realized volume hits the target.
    crng = np.random.default_rng(seed + 909)
    for _ in range(4):
        yc = np.zeros((T, M))
        for t in range(T):
            lr = intercept + beta @ X[t] + latent_strength * latent[t]
            for lag in range(1, min(n_lags, t) + 1):
                lr = lr + excitation[:, :, lag - 1] @ yc[t - lag]
            yc[t] = crng.poisson(np.exp(np.clip(lr, -12.0, 4.0)))
        realized = yc.mean()
        intercept += np.log(target_mean / max(realized, 1e-6))
    # rescale excitation so the *effective* radius holds at the realized volume
    excitation *= 1.0  # G is defined per-count; effective radius = rho(G)*mean

    # --- simulate ------------------------------------------------------------
    state = FirmState(universe, quality, rng)
    sector_counts = np.zeros((T, M), dtype=int)
    startup_counts = np.zeros((T, N), dtype=int)
    events = []  # (week, sector, firm_idx)
    marks = []

    for t in range(T):
        log_rate = intercept + beta @ X[t] + latent_strength * latent[t]
        for lag in range(1, min(n_lags, t) + 1):
            log_rate = log_rate + excitation[:, :, lag - 1] @ sector_counts[t - lag]
        lam = np.exp(np.clip(log_rate, -12.0, 4.0))
        y_t = rng.poisson(lam)

        live = (entry <= t) & (exitw > t)
        # start-of-week features: choices within the week all use the same z
        z_all = features_at(state, universe, t)
        cd_all = _cooldown(state.last_deal_week, state.n_deals, t, cooldown_weeks)
        taken = set()
        for s in range(M):
            cand = np.flatnonzero(live & (sector_of == s))
            y_s = min(int(y_t[s]), len(cand))  # risk-set cap
            placed = 0
            for _ in range(y_s):
                c = np.array([i for i in cand if i not in taken], dtype=int)
                if c.size == 0:
                    break
                scores = z_all[c] @ (w_global + w_sector[s]) + eta[s] * cd_all[c]
                zsc = scores - scores.max()
                p = np.exp(zsc)
                p /= p.sum()
                i = int(rng.choice(c, p=p))
                taken.add(i)
                placed += 1
                startup_counts[t, i] += 1
                events.append((t, s, i))
                marks.append(state.draw_deal(i, t, latent[t], Xw_raw[t, 0]))
                # graduation: late-stage firms exit the private risk pool
                st_i = int(state.stage[i])
                if (st_i >= 5 and rng.random() < 0.35) or (st_i == 4 and rng.random() < 0.10):
                    exitw[i] = min(exitw[i], t + int(rng.integers(4, 13)))
            sector_counts[t, s] = placed

    universe["exit_week"] = exitw  # graduations feed the true active mask
    truth = dict(
        regime="A",
        n_lags=n_lags,
        cooldown_weeks=cooldown_weeks,
        sector_intercept=intercept,
        sector_beta=beta,
        excitation=excitation,
        raw_spectral_radius=float(np.max(np.abs(np.linalg.eigvals(excitation.sum(2))))),
        effective_radius=effective_radius,
        target_mean=target_mean,
        ranker_global=w_global,
        ranker_sector_dev=w_sector,
        ranker_cooldown=eta,
        latent_strength=latent_strength,
        latent_path=latent,
        cov_mean=mu_x,
        cov_sd=sd_x,
        cov_names=["FFUND", "CPI_YOY", "RUNEMP"],
    )
    return dict(
        events=np.asarray(events, int),
        marks=marks,
        sector_counts=sector_counts,
        startup_counts=startup_counts,
        covariates_std=X,
        truth=truth,
    )
