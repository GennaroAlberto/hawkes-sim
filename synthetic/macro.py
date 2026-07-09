"""FactSet Standardized Economics style macro feed (synthetic).

Series (codes follow the FactSet subfile convention used in production):

* ``FFUNDT``  -- Fed Funds target rate, daily step path (policy cycles).
* ``FFUND``   -- effective Fed Funds, daily, small noise around target.
* ``CPI_RAW`` -- CPI index level, monthly, published with ~14 day lag.
* ``CPI_YOY`` -- CPI year-over-year %, monthly, published with ~14 day lag.
* ``RUNEMP``  -- unemployment rate %, monthly, published with ~7 day lag.

The feed is *long format* ``(series_code, ref_date, publish_date, value)`` so the
loader must do an as-of join — exactly the point-in-time discipline the real
pipeline needs.  A latent weekly ``risk_appetite`` factor (NOT in the feed) is
returned separately; regimes use it as unobserved dispersion / confounding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MACRO_SERIES = ["FFUNDT", "FFUND", "CPI_RAW", "CPI_YOY", "RUNEMP"]

CPI_PUB_LAG_DAYS = 14
UNEMP_PUB_LAG_DAYS = 7


def generate_macro(start="2015-01-05", weeks=520 + 60, seed=0):
    """Return (macro_long_df, latent_weekly, week_index).

    ``weeks`` should cover a burn-in before the deal window so lagged features and
    YoY transforms are well defined from week 0 of the deal window.
    """
    rng = np.random.default_rng(seed + 101)
    week_index = pd.date_range(start=start, periods=weeks, freq="W-MON")
    days = pd.date_range(week_index[0], week_index[-1] + pd.Timedelta(days=6), freq="D")
    nd = len(days)

    # --- Fed funds target: regime-switching policy cycles (hold/hike/cut). ---
    state = 0  # -1 cutting, 0 holding, +1 hiking
    target = 0.50
    tpath = np.zeros(nd)
    P = {  # weekly-ish transition probabilities checked every 42 days
        -1: [0.55, 0.43, 0.02],
        0: [0.08, 0.84, 0.08],
        1: [0.02, 0.43, 0.55],
    }
    for i, _d in enumerate(days):
        if i and i % 42 == 0:  # FOMC-ish cadence
            state = rng.choice([-1, 0, 1], p=P[state])
            if state != 0:
                target = float(np.clip(target + 0.25 * state, 0.0, 6.5))
        tpath[i] = target
    ffund = np.clip(
        tpath + rng.normal(0.0, 0.02, nd).cumsum() * 0.0 + rng.normal(0.0, 0.03, nd), 0.0, None
    )

    # --- Monthly CPI YoY: AR(1) with occasional inflation-spike regime. ---
    months = pd.date_range(days[0], days[-1], freq="MS")
    nm = len(months)
    cpi_yoy = np.zeros(nm)
    level, spike = 2.0, 0.0
    for m in range(nm):
        if rng.random() < 0.015:
            spike = rng.uniform(2.0, 5.0)  # inflation shock
        spike *= 0.94
        level = 2.0 + 0.85 * (level - 2.0) + rng.normal(0.0, 0.18)
        cpi_yoy[m] = max(-1.0, level + spike)
    cpi_raw = 100.0 * np.cumprod((1.0 + cpi_yoy / 100.0) ** (1.0 / 12.0))

    # --- Monthly unemployment: slow AR pushed up when policy is tight. ---
    unemp = np.zeros(nm)
    u = 5.0
    tight = pd.Series(tpath, index=days).resample("MS").mean().reindex(months).ffill().to_numpy()
    for m in range(nm):
        u = 5.0 + 0.94 * (u - 5.0) + 0.020 * (tight[m] - 2.0) + rng.normal(0.0, 0.10)
        unemp[m] = float(np.clip(u, 2.5, 12.0))

    rows = []
    for d, tv, ev in zip(days, tpath, ffund):
        rows.append(("FFUNDT", d, d, round(float(tv), 4)))
        rows.append(("FFUND", d, d, round(float(ev), 4)))
    for m, yy, rawv, uv in zip(months, cpi_yoy, cpi_raw, unemp):
        pub_c = m + pd.offsets.MonthEnd(1) + pd.Timedelta(days=CPI_PUB_LAG_DAYS)
        pub_u = m + pd.offsets.MonthEnd(1) + pd.Timedelta(days=UNEMP_PUB_LAG_DAYS)
        rows.append(("CPI_RAW", m, pub_c, round(float(rawv), 3)))
        rows.append(("CPI_YOY", m, pub_c, round(float(yy), 3)))
        rows.append(("RUNEMP", m, pub_u, round(float(uv), 2)))
    macro = pd.DataFrame(rows, columns=["series_code", "ref_date", "publish_date", "value"])

    # --- Latent weekly risk-appetite factor (unobserved). ---
    latent = np.zeros(weeks)
    for t in range(1, weeks):
        latent[t] = 0.90 * latent[t - 1] + 0.44 * rng.normal()
    latent = (latent - latent.mean()) / (latent.std() + 1e-9)

    return macro, latent, week_index


def weekly_macro_matrix(macro: pd.DataFrame, week_index) -> np.ndarray:
    """Point-in-time weekly covariate matrix (T, 3): [FFUND, CPI_YOY, RUNEMP].

    For week starting at ``w`` we use the latest value *published strictly before*
    ``w``.  This is what a real-time system would have seen, and it is also what
    the generator feeds the true model — so loader and generator agree exactly.
    """
    out = np.zeros((len(week_index), 3))
    for j, code in enumerate(["FFUND", "CPI_YOY", "RUNEMP"]):
        s = macro[macro.series_code == code].sort_values("publish_date")
        pub = s.publish_date.to_numpy()
        val = s.value.to_numpy(float)
        idx = np.searchsorted(pub, np.asarray(week_index, dtype="datetime64[ns]"), side="left") - 1
        idx = np.clip(idx, 0, len(val) - 1)
        out[:, j] = val[idx]
    return out
