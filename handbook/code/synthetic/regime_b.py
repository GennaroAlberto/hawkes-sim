"""Regime B: misspecified world — daily firm-level block Hawkes with inhibition.

Ground truth is a continuous-time-style (daily grid) firm intensity

    log lam_{i}(d) = a_{s(i)} + beta_{s(i)}' X_w(d) + 0.8*latent_w(d)*chi_s
                     - rho_{s(i)} R_i(d) + A_{s(i),b} E_b(d) + g_q * quality_i

* R_i(d) = sum over own past events of exp(-om_self * dt)      (cooldown, ~30d half-life)
* E_b(d) = size-normalized sector field: mean over firms in b of decayed event
  sums with ~7d half-life                                       (peer contagion)
* quality enters the intensity directly (firm heterogeneity the weekly two-stage
  model cannot see), and the covariate response has sector-specific cyclicality.

The weekly two-stage model is *not* this DGP: intensity is firm-level, inhibition
is continuous, contagion decays within the week, latent quality drives selection.
Fitting the two-stage model here answers the robustness question.
"""

from __future__ import annotations

import numpy as np

from .marks import FirmState


def simulate_regime_b(
    universe,
    quality,
    Xw_raw,
    latent,
    *,
    n_sectors=12,
    target_total_events=55_000,
    quality_strength=0.45,
    self_inhibition=(1.2, 2.2),
    peer_excitation=0.35,
    contagion_exponent=0.12,
    contagion_cap=0.6,
    latent_coef=0.5,
    covariate_scale=1.0,
    seed=0,
):
    rng = np.random.default_rng(seed + 505)
    Tw = Xw_raw.shape[0]
    Td = Tw * 7
    M, N = n_sectors, len(universe)
    sector_of = universe["sector_idx"].to_numpy()
    entry_d = universe["entry_week"].to_numpy() * 7
    exit_d = np.minimum(universe["exit_week"].to_numpy() * 7, Td)

    mu_x, sd_x = Xw_raw.mean(0), Xw_raw.std(0) + 1e-9
    X = (Xw_raw - mu_x) / sd_x

    beta = np.stack(
        [rng.normal(-0.28, 0.10, M), rng.normal(-0.10, 0.07, M), rng.normal(-0.18, 0.08, M)], axis=1
    ) * float(covariate_scale)
    chi = rng.uniform(0.4, 1.2, M)  # sector cyclicality
    rho = rng.uniform(*self_inhibition, size=M)  # self-inhibition strength
    om_self = np.log(2) / 30.0  # 30-day half-life
    om_cross = np.log(2) / 7.0  # 7-day half-life
    A = np.zeros((M, M))
    for s in range(M):
        A[s, s] = peer_excitation
        for r in rng.choice([r for r in range(M) if r != s], size=2, replace=False):
            A[s, r] = peer_excitation * rng.uniform(0.2, 0.5)
    # E is a *mean field per firm* (size-normalized), so its typical magnitude is
    # daily_sector_rate / om_cross / n_firms_in_sector — tiny.  Rescale A so the
    # typical exponent contribution is ``contagion_exponent`` and hard-cap the
    # total contribution (saturation) — hot firms make the exponentiated field
    # locally supercritical otherwise.
    n_by_sector = np.bincount(sector_of, minlength=M).astype(float)
    typical_field = (target_total_events / Td / M) / om_cross / n_by_sector.mean()
    A_eff = A * (contagion_exponent / peer_excitation) / max(typical_field, 1e-12)

    # Baseline intercept to hit the target volume (mean-field calibration).
    lam_bar = target_total_events / Td / N  # per live firm per day (approx)
    a = np.log(lam_bar) + rng.normal(0.0, 0.25, M)
    a -= 0.5 * quality_strength**2  # E[exp(g q)] correction

    q_term = quality_strength * quality
    R = np.zeros(N)
    E = np.zeros(M)
    ds, df = np.exp(-om_self), np.exp(-om_cross)
    events = []  # (day, sector, firm)

    for d in range(Td):
        w = min(d // 7, Tw - 1)
        live = (entry_d <= d) & (exit_d > d)
        contag = np.minimum((A_eff @ E)[sector_of], contagion_cap)
        base = (
            a[sector_of]
            + (beta[sector_of] * X[w]).sum(1)
            + latent_coef * latent[w] * chi[sector_of]
            + q_term
            - rho[sector_of] * R
            + contag
        )
        lam = np.exp(np.clip(base, -18.0, -1.0)) * live
        fired = np.flatnonzero(rng.random(N) < -np.expm1(-lam))
        R *= ds
        E *= df
        for i in fired:
            s = int(sector_of[i])
            events.append((d, s, i))
            R[i] += 1.0
            E[s] += 1.0 / max(n_by_sector[s], 1.0)

    # Draw marks chronologically (intensity does not depend on marks).
    state = FirmState(universe, quality, np.random.default_rng(seed + 606))
    marks = []
    for d, _s, i in events:
        w = min(d // 7, Tw - 1)
        marks.append(state.draw_deal(i, d // 7, latent[w], Xw_raw[w, 0]))

    ev = np.asarray(events, int)
    sector_counts = np.zeros((Tw, M), int)
    startup_counts = np.zeros((Tw, N), int)
    for d, s, i in events:
        w = min(d // 7, Tw - 1)
        sector_counts[w, s] += 1
        startup_counts[w, i] += 1

    truth = dict(
        regime="B",
        day_events=ev,
        om_self=om_self,
        om_cross=om_cross,
        sector_intercept=a,
        sector_beta=beta,
        sector_cyclicality=chi,
        self_inhibition=rho,
        peer_excitation_matrix=A,
        quality_strength=quality_strength,
        quality=quality,
        latent_path=latent,
        cov_mean=mu_x,
        cov_sd=sd_x,
        cov_names=["FFUND", "CPI_YOY", "RUNEMP"],
    )
    return dict(
        events=ev,
        marks=marks,
        sector_counts=sector_counts,
        startup_counts=startup_counts,
        covariates_std=X,
        truth=truth,
    )
