"""Rebuild fitter-ready arrays from the noisy PitchBook/FactSet tables.

This is the mirror of the observation layer: everything here uses ONLY what a
production pipeline would see (deals.csv, companies.csv, macro.csv), enforcing
point-in-time discipline:

* macro covariates are as-of joined on publish_date (no look-ahead);
* firm features are last-observation-carried-forward from *past deals only*;
* the active mask is inferred from observables (founding year, first deal),
  optionally replaced by the true mask from ground_truth.npz for oracle runs.

Output shapes match hawkes_calibration.sector_ranker fitters:
  sector_counts (T, M) int, covariates (T, p) float, features (T, N, 5) float32,
  startup_counts (T, N) int, active (T, N) bool, events (E, 3) int.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .marks import FEATURE_NAMES


@dataclass
class LoadedDataset:
    week_index: pd.DatetimeIndex
    sector_names: list
    sector_counts: np.ndarray
    covariates_raw: np.ndarray
    cov_names: list
    events: np.ndarray
    features: np.ndarray
    feature_names: list
    startup_counts: np.ndarray
    active: np.ndarray
    firm_ids: np.ndarray
    firm_sector: np.ndarray
    off_universe_share: float


def _weekly_asof(macro, codes, week_index):
    out = np.zeros((len(week_index), len(codes)))
    wk = np.asarray(week_index, dtype="datetime64[ns]")
    for j, code in enumerate(codes):
        s = macro[macro.series_code == code].sort_values("publish_date")
        pub = s.publish_date.to_numpy(dtype="datetime64[ns]")
        val = s.value.to_numpy(float)
        idx = np.clip(np.searchsorted(pub, wk, side="left") - 1, 0, len(val) - 1)
        out[:, j] = val[idx]
    return out


def deduplicate_deals(deals: pd.DataFrame, window_days=3) -> pd.DataFrame:
    """Drop likely duplicate rows: same company within ``window_days``."""
    d = deals.sort_values(["company_id", "deal_date"]).copy()
    dt = pd.to_datetime(d.deal_date)
    gap = dt.groupby(d.company_id).diff().dt.days
    return d[(gap.isna()) | (gap > window_days)].sort_values("deal_date")


CHOICE_FEATURES = [
    "log1p_total_raised",
    "log_employees",
    "log1p_age_years",
    "stage_num",
    "log1p_months_since_last_funding",
    "is_newcomer",
]


def build_choice_sets(data_dir, *, max_candidates=64, exclusion="sector", newcomer=True, seed=0):
    """Event-level (day-resolution) choice sets under the truthful-risk-set rule.

    For each deal, chronologically:

    * pool = tracked firms of the event's sector that were funded at least once
      strictly before the event and are active (true mask from ground_truth.npz),
      EXCLUDING the most recently funded firm of that sector ("exclude the last
      firm that got funding"); ties within a day break by deal order;
    * one synthetic "newcomer" alternative represents a first-time/untracked
      entrant (features all zero except is_newcomer=1);
    * label = funded firm's position, or the newcomer slot if the funded firm has
      no prior funding history or no companies.csv row (out-of-universe);
    * events whose winner is exactly the excluded last-funded firm are dropped
      and counted (`n_dropped_repeat`).

    Features per candidate (CHOICE_FEATURES) are LOCF from deals strictly before
    the event day; time-to-last-funding enters as log1p(days/30).
    Risk sets larger than ``max_candidates`` are subsampled keeping the winner
    (sampled softmax); set ``max_candidates=None`` for full sets.

    Returns dict with padded tensors: F (E,K,6) float32, mask (E,K) bool,
    label (E,), sector (E,), day (E,), plus bookkeeping counts.
    """
    import json

    if not max_candidates:
        raise ValueError("max_candidates must be a positive int (padded output)")
    rng = np.random.default_rng(seed)
    deals = pd.read_csv(f"{data_dir}/deals.csv", low_memory=False)
    companies = pd.read_csv(f"{data_dir}/companies.csv")
    with open(f"{data_dir}/meta.json") as fh:
        meta = json.load(fh)
    gt = np.load(f"{data_dir}/ground_truth.npz", allow_pickle=True)
    week0 = pd.Timestamp(meta["week_index"][0])
    sector_names = meta["sector_names"]
    sec_idx = {s: k for k, s in enumerate(sector_names)}
    M = len(sector_names)

    deals = deduplicate_deals(deals)
    deals = deals.sort_values(["deal_date", "deal_id"]).reset_index(drop=True)
    day = (pd.to_datetime(deals.deal_date) - week0).dt.days.to_numpy()

    firm_ids = companies.company_id.to_numpy()
    N = len(firm_ids)
    fidx = {c: i for i, c in enumerate(firm_ids)}
    firm_sector = companies.primary_industry_sector.map(sec_idx).fillna(0).astype(int).to_numpy()
    tp = gt["tracked_positions"]
    entry_d = gt["entry_week"][tp] * 7
    exit_d = gt["exit_week"][tp] * 7

    # per-firm LOCF state
    raised = np.zeros(N)
    emp = np.full(N, 9.0)
    ndeal = np.zeros(N)
    last_day = np.full(N, -(10**9))
    pool_by_sector = [[] for _ in range(M)]  # funded-at-least-once members
    last_funded_by_sector = np.full(M, -1)  # firm to exclude
    e_sector = deals.primary_industry_sector.map(sec_idx).fillna(0).astype(int).to_numpy()
    rtd = pd.to_numeric(deals.total_raised_to_date, errors="coerce").to_numpy(float)
    nemp = pd.to_numeric(deals.number_of_employees, errors="coerce").to_numpy(float)
    cid = deals.company_id.map(lambda c: fidx.get(c, -1)).to_numpy()

    K = (max_candidates or 10**9) + 1  # +1 newcomer slot
    F_rows, m_rows, labels, secs, days_out = [], [], [], [], []
    n_drop_repeat = n_newcomer = 0

    for e in range(len(deals)):
        d, s, f = int(day[e]), int(e_sector[e]), int(cid[e])
        pool = pool_by_sector[s]
        cand = np.array(
            [i for i in pool if i != last_funded_by_sector[s] and entry_d[i] <= d < exit_d[i]],
            dtype=int,
        )
        winner_in_pool = f >= 0 and f in set(cand.tolist())
        excluded_repeat = f >= 0 and f == last_funded_by_sector[s]

        if excluded_repeat:
            n_drop_repeat += 1
        else:
            if not winner_in_pool:
                n_newcomer += 1
            if max_candidates and cand.size > max_candidates - (0 if newcomer else 1):
                keep = max_candidates - (1 if winner_in_pool else 0)
                others = cand[cand != f] if winner_in_pool else cand
                cand = np.concatenate(
                    (
                        [f] if winner_in_pool else [],
                        rng.choice(others, size=min(keep, others.size), replace=False),
                    )
                ).astype(int)
            k = cand.size
            Fe = np.zeros((K, 6), np.float32)
            me = np.zeros(K, bool)
            if k:
                gap = np.maximum(d - last_day[cand], 0.0)
                Fe[:k, 0] = np.log1p(raised[cand])
                Fe[:k, 1] = np.log(np.maximum(emp[cand], 2.0))
                Fe[:k, 2] = np.log1p(np.maximum((d - entry_d[cand]) / 365.0, 0.0))
                Fe[:k, 3] = np.minimum(ndeal[cand], 6)
                Fe[:k, 4] = np.log1p(gap / 30.0)
                me[:k] = True
            lab = -1
            if newcomer:
                Fe[k, 5] = 1.0
                me[k] = True
                lab = int(np.flatnonzero(cand == f)[0]) if winner_in_pool else k
            elif winner_in_pool:
                lab = int(np.flatnonzero(cand == f)[0])
            if lab >= 0:
                F_rows.append(Fe)
                m_rows.append(me)
                labels.append(lab)
                secs.append(s)
                days_out.append(d)

        # ---- state update (after constructing the event's risk set) ----
        if f >= 0:
            if ndeal[f] == 0:
                pool_by_sector[firm_sector[f]].append(f)
            if np.isfinite(rtd[e]):
                raised[f] = max(raised[f], float(rtd[e]))
            if np.isfinite(nemp[e]) and nemp[e] > 0:
                emp[f] = float(nemp[e])
            ndeal[f] += 1
            last_day[f] = d
            if exclusion == "sector":
                last_funded_by_sector[firm_sector[f]] = f
        if exclusion == "global" and f >= 0:
            last_funded_by_sector[:] = f

    Kmax = min(K, max(2, max(int(m.sum()) for m in m_rows)))
    F = np.stack([r[:Kmax] for r in F_rows])
    msk = np.stack([r[:Kmax] for r in m_rows])
    return dict(
        F=F,
        mask=msk,
        label=np.array(labels),
        sector=np.array(secs),
        day=np.array(days_out),
        feature_names=list(CHOICE_FEATURES),
        n_events=len(labels),
        n_dropped_repeat=n_drop_repeat,
        n_newcomer_wins=n_newcomer,
        n_sectors=M,
    )


def load_dataset(data_dir, *, dedup=True, active="observed", ground_truth=None):
    data_dir = str(data_dir)
    deals = pd.read_csv(f"{data_dir}/deals.csv", low_memory=False)
    companies = pd.read_csv(f"{data_dir}/companies.csv")
    macro = pd.read_csv(f"{data_dir}/macro.csv", parse_dates=["ref_date", "publish_date"])
    import json

    with open(f"{data_dir}/meta.json") as fh:
        meta = json.load(fh)
    week_index = pd.DatetimeIndex(pd.to_datetime(meta["week_index"]))
    T = len(week_index)
    sector_names = meta["sector_names"]
    M = len(sector_names)
    sec_idx = {s: k for k, s in enumerate(sector_names)}

    if dedup:
        deals = deduplicate_deals(deals)
    deals = deals.copy()
    dt = pd.to_datetime(deals.deal_date).to_numpy(dtype="datetime64[ns]")
    wk = np.asarray(week_index, dtype="datetime64[ns]")
    deals["week"] = np.clip(np.searchsorted(wk, dt, side="right") - 1, 0, T - 1)
    deals["sector_k"] = deals.primary_industry_sector.map(sec_idx).fillna(0).astype(int)

    # --- sector counts: ALL deals (tracked or not) --------------------------
    sector_counts = np.zeros((T, M), dtype=int)
    np.add.at(sector_counts, (deals.week.to_numpy(), deals.sector_k.to_numpy()), 1)

    # --- covariates ----------------------------------------------------------
    cov_names = ["FFUND", "CPI_YOY", "RUNEMP"]
    covariates = _weekly_asof(macro, cov_names, week_index)

    # --- firm panel (tracked universe only) ----------------------------------
    firm_ids = companies.company_id.to_numpy()
    N = len(firm_ids)
    fidx = {c: i for i, c in enumerate(firm_ids)}
    firm_sector = companies.primary_industry_sector.map(sec_idx).fillna(0).astype(int).to_numpy()

    d_tr = deals[deals.company_id.isin(fidx)].sort_values(["week"]).copy()
    d_tr["fi"] = d_tr.company_id.map(fidx)
    off_share = 1.0 - len(d_tr) / max(len(deals), 1)

    startup_counts = np.zeros((T, N), dtype=int)
    np.add.at(startup_counts, (d_tr.week.to_numpy(), d_tr.fi.to_numpy()), 1)

    year0 = int(week_index[0].year)
    entry_week = np.clip((companies.year_founded.to_numpy() - year0) * 52, 0, T - 1)
    first_deal_week = np.full(N, T, dtype=int)
    fw = d_tr.groupby("fi").week.min()
    first_deal_week[fw.index.to_numpy()] = fw.to_numpy()
    entry_week = np.minimum(entry_week, first_deal_week)

    if active == "true" and ground_truth is not None:
        gt = np.load(ground_truth, allow_pickle=True)
        ew, xw = gt["entry_week"], gt["exit_week"]
        tracked_pos = gt["tracked_positions"]  # positions of tracked firms in world order
        act = np.zeros((T, N), dtype=bool)
        for i in range(N):
            act[ew[tracked_pos[i]] : xw[tracked_pos[i]], i] = True
    else:
        act = np.zeros((T, N), dtype=bool)
        weeks_col = np.arange(T)[:, None]
        act[:] = weeks_col >= entry_week[None, :]
        dead = companies.business_status.eq("Out of Business").to_numpy()
        last_deal_week = np.zeros(N, dtype=int)
        lw = d_tr.groupby("fi").week.max()
        last_deal_week[lw.index.to_numpy()] = lw.to_numpy()
        # observed proxy: dead firms leave the risk set 52w after their last deal
        exit_guess = np.where(dead, np.minimum(last_deal_week + 52, T), T)
        act &= weeks_col < exit_guess[None, :]

    # --- point-in-time LOCF features -----------------------------------------
    features = np.zeros((T, N, 5), dtype=np.float32)
    raised = np.zeros(N)
    emp = np.full(N, 9.0)
    ndeal = np.zeros(N)
    last_w = np.full(N, -10_000)
    ptr = 0
    rows = np.column_stack(
        [
            d_tr.week.to_numpy(float),
            d_tr.fi.to_numpy(float),
            pd.to_numeric(d_tr.total_raised_to_date, errors="coerce").to_numpy(float),
            pd.to_numeric(d_tr.number_of_employees, errors="coerce").to_numpy(float),
        ]
    )
    (0 - entry_week) / 52.0
    for t in range(T):
        # apply all deals from weeks < t (features at t use info strictly before t)
        while ptr < len(rows) and rows[ptr, 0] < t:
            _, fi, rtd, ne = rows[ptr]
            fi = int(fi)
            if np.isfinite(rtd):
                raised[fi] = max(raised[fi], float(rtd))
            if np.isfinite(ne) and ne > 0:
                emp[fi] = float(ne)
            ndeal[fi] += 1
            last_w[fi] = int(rows[ptr, 0])
            ptr += 1
        gap = t - last_w
        rec = np.where(ndeal > 0, np.exp(-np.maximum(gap, 0) / 52.0), 0.0)
        features[t, :, 0] = np.log1p(raised)
        features[t, :, 1] = np.log(np.maximum(emp, 2.0))
        features[t, :, 2] = np.log1p(np.maximum((t - entry_week) / 52.0, 0.0))
        features[t, :, 3] = np.minimum(ndeal, 6)
        features[t, :, 4] = rec

    events = d_tr[["week", "sector_k", "fi"]].to_numpy(int)

    return LoadedDataset(
        week_index=week_index,
        sector_names=sector_names,
        sector_counts=sector_counts,
        covariates_raw=covariates,
        cov_names=cov_names,
        events=events,
        features=features,
        feature_names=list(FEATURE_NAMES),
        startup_counts=startup_counts,
        active=act,
        firm_ids=firm_ids,
        firm_sector=firm_sector,
        off_universe_share=float(off_share),
    )
