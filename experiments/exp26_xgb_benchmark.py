r"""
Experiment 26 -- XGBoost ranking benchmark (REVIEW.md B1): a model-free
discriminative reference for the funded-firm choice layer.

Question (REVIEW R8): does gradient boosting beat the conditional logit on the
identical choice sets?  If XGB >> logit we are missing features or
nonlinearities; if XGB ~ logit the parametric choice model is defensible.

Setup
-----
* Data: ``build_choice_sets`` (seed 0, max_candidates=64) on synthetic_A and
  synthetic_B.  One row per (event, incumbent candidate) from the padded
  tensors.  Split: identical to ``fit_choice_fast`` -- train = day <= q80(day),
  test = day > q80.  Early-stopping validation = last 15% of TRAIN days
  (day > q85 of train days), never touching test.
* The builder loop is REPLICATED here (verified bitwise against
  ``synthetic.loaders.build_choice_sets``) because the library version does not
  return candidate firm ids, which the kitchen-sink features require.
* Objectives: (a) ``binary:logistic`` scored per event by softmax over the raw
  margins; (b) ``rank:pairwise`` with group = event (time-ordered).
* Feature sets:
  (i)  PARITY -- exactly the 6 choice-model features (isolates functional form);
  (ii) KITCHEN SINK -- parity + strictly point-in-time extras re-derived from
       deals.csv / macro.csv / companies.csv static columns: raw n prior deals,
       log last deal size + undisclosed flag, disclosed-size share, raw
       days-since-last-funding, age from year_founded, hq_global_region
       one-hots, as-of macro (FFUND, CPI_YOY, RUNEMP; publish_date < event
       day, mirroring the loaders), log pool size, sector id (int).
       FORBIDDEN (end-of-sample snapshots): companies.csv last_*,
       total_money_raised, employees.
* Newcomer handling, BOTH reported:
  (a) ``test_incumbent_only``  -- events whose winner is an incumbent; both XGB
      and the logit renormalized over incumbent slots only;
  (b) ``test_all_events_constASC`` -- all test events; the newcomer slot gets a
      constant score = logit of the train newcomer share (XGB), while the logit
      reference uses its native fitted ASC.
* Model selection: grid depth {3,5} x eta {0.05,0.1}, <=600 trees early-stopped
  (30 rounds) on the validation tail; pick by validation NLL (renormalized
  softmax, incumbent-won events); refit the winner on the full train window
  with the chosen tree count; report test only for the winner.
* Metrics: top-1/5/10 and MRR under the mid-rank tie convention, plus NLL of
  the renormalized scores.  The conditional logit is re-fit in this script on
  the identical split so numbers are exactly comparable (sanity anchors, seed
  0: test NLL 3.346 / top-5 0.446 on A; 3.796 / 0.316 on B).

Run:  OMP_NUM_THREADS=2 PYTHONPATH=. python -m experiments.exp26_xgb_benchmark
Writes results/exp26_xgb.json.
"""

import json
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from synthetic.fast_fit import fit_choice_fast
from synthetic.loaders import build_choice_sets, deduplicate_deals

DATASETS = ("data/synthetic_A", "data/synthetic_B")
MAX_CANDIDATES = 64
SEED = 0
TRAIN_FRAC_DAYS = 0.8
VAL_FRAC_OF_TRAIN = 0.85  # last 15% of TRAIN days = validation tail
GRID = [(d, e) for d in (3, 5) for e in (0.05, 0.1)]
MAX_TREES = 600
EARLY_STOP = 30
MACRO_CODES = ("FFUND", "CPI_YOY", "RUNEMP")
TOP5_EPS = 0.02  # decision-rule thresholds (absolute)
NLL_EPS = 0.05

SINK_EXTRA_NAMES = [
    "n_prior_deals",
    "log1p_last_deal_size",
    "last_size_undisclosed",
    "disclosed_share",
    "gap_days_raw",
    "age_from_founding_years",
]  # + region one-hots, macro as-ofs, log_pool_size, sector_id appended at build


# ===========================================================================
# Choice-set replica with candidate ids + point-in-time extras
# ===========================================================================
def _asof_macro(data_dir, week0):
    """Per-series (publish-day offsets, values), sorted -- for strict as-of joins."""
    macro = pd.read_csv(f"{data_dir}/macro.csv", parse_dates=["ref_date", "publish_date"])
    out = {}
    for code in MACRO_CODES:
        s = macro[macro.series_code == code].sort_values("publish_date")
        pub_days = (s.publish_date - week0).dt.days.to_numpy()
        out[code] = (pub_days, s.value.to_numpy(float))
    return out


def build_choice_sets_with_extras(data_dir, *, max_candidates=64, seed=0):
    """Replicates loaders.build_choice_sets (identical RNG stream) and additionally
    returns per-candidate kitchen-sink features (strictly point-in-time).
    """
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
    dt = pd.to_datetime(deals.deal_date)
    day = (dt - week0).dt.days.to_numpy()
    ev_year = dt.dt.year.to_numpy()

    firm_ids = companies.company_id.to_numpy()
    N = len(firm_ids)
    fidx = {c: i for i, c in enumerate(firm_ids)}
    firm_sector = companies.primary_industry_sector.map(sec_idx).fillna(0).astype(int).to_numpy()
    tp = gt["tracked_positions"]
    entry_d = gt["entry_week"][tp] * 7
    exit_d = gt["exit_week"][tp] * 7

    # static company columns (allowed: identity/region/founding only)
    year_founded = companies.year_founded.to_numpy(float)
    reg_cat = pd.Categorical(companies.hq_global_region)
    regions = list(reg_cat.categories)
    reg_code = reg_cat.codes.astype(int)
    n_reg = len(regions)

    macro_asof = _asof_macro(data_dir, week0)

    # per-firm LOCF state (loaders' state + extras)
    raised = np.zeros(N)
    emp = np.full(N, 9.0)
    ndeal = np.zeros(N)
    last_day = np.full(N, -(10**9))
    last_size = np.full(N, np.nan)  # size of most recent prior deal (NaN if undisclosed)
    n_disc = np.zeros(N)  # prior deals with disclosed size
    pool_by_sector = [[] for _ in range(M)]
    last_funded_by_sector = np.full(M, -1)
    e_sector = deals.primary_industry_sector.map(sec_idx).fillna(0).astype(int).to_numpy()
    rtd = pd.to_numeric(deals.total_raised_to_date, errors="coerce").to_numpy(float)
    nemp = pd.to_numeric(deals.number_of_employees, errors="coerce").to_numpy(float)
    dsize = pd.to_numeric(deals.deal_size, errors="coerce").to_numpy(float)
    cid = deals.company_id.map(lambda c: fidx.get(c, -1)).to_numpy()

    n_extra = len(SINK_EXTRA_NAMES) + n_reg + len(MACRO_CODES) + 2  # + log_pool + sector_id
    K = max_candidates + 1
    F_rows, X_rows, id_rows, m_rows, labels, secs, days_out = [], [], [], [], [], [], []
    n_drop_repeat = 0

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
            if max_candidates and cand.size > max_candidates:  # newcomer=True threshold
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
            Xe = np.zeros((K, n_extra), np.float32)
            ids = np.full(K, -1, np.int32)
            me = np.zeros(K, bool)
            if k:
                gap = np.maximum(d - last_day[cand], 0.0)
                Fe[:k, 0] = np.log1p(raised[cand])
                Fe[:k, 1] = np.log(np.maximum(emp[cand], 2.0))
                Fe[:k, 2] = np.log1p(np.maximum((d - entry_d[cand]) / 365.0, 0.0))
                Fe[:k, 3] = np.minimum(ndeal[cand], 6)
                Fe[:k, 4] = np.log1p(gap / 30.0)
                me[:k] = True
                ids[:k] = cand
                ls = last_size[cand]
                disclosed = np.isfinite(ls)
                Xe[:k, 0] = ndeal[cand]
                Xe[:k, 1] = np.log1p(np.where(disclosed, ls, 0.0))
                Xe[:k, 2] = (~disclosed).astype(np.float32)
                Xe[:k, 3] = n_disc[cand] / np.maximum(ndeal[cand], 1.0)
                Xe[:k, 4] = gap
                Xe[:k, 5] = np.maximum(ev_year[e] - year_founded[cand], 0.0)
                Xe[np.arange(k), 6 + reg_code[cand]] = 1.0
                for j, code in enumerate(MACRO_CODES):
                    pub, val = macro_asof[code]
                    ix = np.clip(np.searchsorted(pub, d, side="left") - 1, 0, len(val) - 1)
                    Xe[:k, 6 + n_reg + j] = val[ix]
                Xe[:k, 6 + n_reg + 3] = np.log(max(k, 1))
                Xe[:k, 6 + n_reg + 4] = s
            Fe[k, 5] = 1.0
            me[k] = True
            lab = int(np.flatnonzero(cand == f)[0]) if winner_in_pool else k
            F_rows.append(Fe)
            X_rows.append(Xe)
            id_rows.append(ids)
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
            last_size[f] = dsize[e] if np.isfinite(dsize[e]) else np.nan
            n_disc[f] += float(np.isfinite(dsize[e]))
            last_funded_by_sector[firm_sector[f]] = f

    Kmax = min(K, max(2, max(int(m.sum()) for m in m_rows)))
    extra_names = (
        SINK_EXTRA_NAMES
        + [f"region_{r}" for r in regions]
        + [f"{c}_asof" for c in MACRO_CODES]
        + ["log_pool_size", "sector_id"]
    )
    return dict(
        F=np.stack([r[:Kmax] for r in F_rows]),
        X_extra=np.stack([r[:Kmax] for r in X_rows]),
        cand_id=np.stack([r[:Kmax] for r in id_rows]),
        mask=np.stack([r[:Kmax] for r in m_rows]),
        label=np.array(labels),
        sector=np.array(secs),
        day=np.array(days_out),
        extra_names=extra_names,
        n_dropped_repeat=n_drop_repeat,
        n_sectors=M,
    )


# ===========================================================================
# Metrics (mid-rank tie convention) and row extraction
# ===========================================================================
def rank_metrics(scores, valid, lab):
    """top-1/5/10, MRR (mid-rank ties) and softmax NLL over the valid slots."""
    s = np.where(valid, scores, -np.inf)
    n = len(lab)
    sw = s[np.arange(n), lab]
    greater = ((s > sw[:, None]) & valid).sum(1)
    equal = ((s == sw[:, None]) & valid).sum(1)  # includes the winner itself
    rank = greater + 0.5 * (equal - 1) + 1.0
    mx = s.max(1)
    lse = mx + np.log(np.sum(np.where(valid, np.exp(s - mx[:, None]), 0.0), 1))
    return dict(
        top1=float(np.mean(rank <= 1)),
        top5=float(np.mean(rank <= 5)),
        top10=float(np.mean(rank <= 10)),
        mrr=float(np.mean(1.0 / rank)),
        nll=float(np.mean(lse - sw)),
        random_nll=float(np.mean(np.log(valid.sum(1)))),
        n_events=int(n),
    )


def extract_rows(feat, inc, lab, ev_ids):
    """Flatten incumbent slots of the selected events into (X, y, group, locs)."""
    sub = inc[ev_ids]
    e_loc, k_loc = np.nonzero(sub)
    X = feat[ev_ids][e_loc, k_loc]
    y = (k_loc == lab[ev_ids][e_loc]).astype(np.int8)
    grp = sub.sum(1).astype(int)
    return X, y, grp, e_loc, k_loc


def score_matrix(pred, e_loc, k_loc, n_ev, K):
    S = np.full((n_ev, K), -np.inf)
    S[e_loc, k_loc] = pred
    return S


# ===========================================================================
# XGB training
# ===========================================================================
def train_config(objective, dtr, dval, depth, eta):
    params = dict(
        objective=objective,
        max_depth=depth,
        eta=eta,
        tree_method="hist",
        nthread=2,
        seed=SEED,
        eval_metric="logloss" if objective == "binary:logistic" else "ndcg@5",
    )
    t0 = time.perf_counter()
    bst = xgb.train(
        params,
        dtr,
        num_boost_round=MAX_TREES,
        evals=[(dval, "val")],
        early_stopping_rounds=EARLY_STOP,
        verbose_eval=False,
    )
    sec = time.perf_counter() - t0
    best_it = int(getattr(bst, "best_iteration", MAX_TREES - 1))
    return bst, best_it, sec


def predict_margin(bst, X, objective, best_it):
    dm = xgb.DMatrix(X)
    kw = dict(iteration_range=(0, best_it + 1))
    if objective == "binary:logistic":
        kw["output_margin"] = True
    return bst.predict(dm, **kw)


def run_combo(objective, feat, cs, splits, const_asc):
    """Grid-search one (objective, feature set), refit winner, return test metrics."""
    inc, lab, msk, nc_slot = splits["inc"], cs["label"], cs["mask"], splits["nc_slot"]
    K = feat.shape[1]
    Xc, yc, gc, _, _ = extract_rows(feat, inc, lab, splits["core_ids"])
    Xv, yv, gv, ev_e, ev_k = extract_rows(feat, inc, lab, splits["val_ids"])
    dtr, dval = xgb.DMatrix(Xc, label=yc), xgb.DMatrix(Xv, label=yv)
    if objective == "rank:pairwise":
        dtr.set_group(gc)
        dval.set_group(gv)

    grid, best = [], None
    for depth, eta in GRID:
        bst, best_it, sec = train_config(objective, dtr, dval, depth, eta)
        Sv = score_matrix(
            predict_margin(bst, Xv, objective, best_it), ev_e, ev_k, len(splits["val_ids"]), K
        )
        val_nll = rank_metrics(Sv, inc[splits["val_ids"]], lab[splits["val_ids"]])["nll"]
        rec = dict(max_depth=depth, eta=eta, best_iter=best_it, val_nll=val_nll, train_s=sec)
        grid.append(rec)
        print(
            f"    {objective:16s} depth={depth} eta={eta:4.2f}  "
            f"trees={best_it + 1:3d}  val_nll={val_nll:.4f}  ({sec:5.1f}s)"
        )
        if best is None or val_nll < best["val_nll"]:
            best = rec

    # refit winner on core+val (full train window) with the selected tree count
    Xt, yt, gt_, _, _ = extract_rows(feat, inc, lab, splits["train_ids"])
    dfull = xgb.DMatrix(Xt, label=yt)
    if objective == "rank:pairwise":
        dfull.set_group(gt_)
    params = dict(
        objective=objective,
        max_depth=best["max_depth"],
        eta=best["eta"],
        tree_method="hist",
        nthread=2,
        seed=SEED,
    )
    t0 = time.perf_counter()
    bst = xgb.train(params, dfull, num_boost_round=best["best_iter"] + 1)
    refit_s = time.perf_counter() - t0

    # (a) incumbent-won test events, renormalized over incumbents only
    Xa, _, _, a_e, a_k = extract_rows(feat, inc, lab, splits["test_inc_ids"])
    Sa = score_matrix(
        predict_margin(bst, Xa, objective, best["best_iter"]),
        a_e,
        a_k,
        len(splits["test_inc_ids"]),
        K,
    )
    m_inc = rank_metrics(Sa, inc[splits["test_inc_ids"]], lab[splits["test_inc_ids"]])

    # (b) all test events; newcomer slot = const score (logit of train newcomer share)
    te_ids = splits["test_ids"]
    Xb, _, _, b_e, b_k = extract_rows(feat, inc, lab, te_ids)
    Sb = score_matrix(
        predict_margin(bst, Xb, objective, best["best_iter"]), b_e, b_k, len(te_ids), K
    )
    Sb[np.arange(len(te_ids)), nc_slot[te_ids]] = const_asc
    m_all = rank_metrics(Sb, msk[te_ids], lab[te_ids])

    return dict(
        grid=grid,
        winner=dict(**best, refit_s=refit_s, n_train_rows=int(len(yt))),
        test_incumbent_only=m_inc,
        test_all_events_constASC=m_all,
    )


# ===========================================================================
# Per-dataset driver
# ===========================================================================
def make_splits(cs):
    day, lab, msk, F = cs["day"], cs["label"], cs["mask"], cs["F"]
    E = len(lab)
    nc_col = (F[:, :, 5] == 1) & msk
    nc_slot = np.argmax(nc_col, axis=1)
    is_new_win = nc_col[np.arange(E), lab]
    cut = np.quantile(day, TRAIN_FRAC_DAYS)
    tr, te = day <= cut, day > cut
    val_cut = np.quantile(day[tr], VAL_FRAC_OF_TRAIN)
    core, val = tr & (day <= val_cut), tr & (day > val_cut)
    inc = msk & ~nc_col
    return dict(
        inc=inc,
        nc_slot=nc_slot,
        is_new_win=is_new_win,
        cut=float(cut),
        val_cut=float(val_cut),
        core_ids=np.flatnonzero(core & ~is_new_win),
        val_ids=np.flatnonzero(val & ~is_new_win),
        train_ids=np.flatnonzero(tr & ~is_new_win),
        test_inc_ids=np.flatnonzero(te & ~is_new_win),
        test_ids=np.flatnonzero(te),
        newcomer_share_train=float(is_new_win[tr].mean()),
    )


def logit_reference(cs, splits):
    """Re-fit the conditional logit on the identical split; score both modes."""
    w0, u, tr_nll, te_nll, rand_nll, top5, ok = fit_choice_fast(cs, train_frac_days=TRAIN_FRAC_DAYS)
    F, msk, lab, sec = cs["F"], cs["mask"], cs["label"], cs["sector"]
    Wev = w0[None, :] + u[sec]
    sc = np.einsum("ekp,ep->ek", F, Wev)
    te_ids, ti_ids = splits["test_ids"], splits["test_inc_ids"]
    m_all = rank_metrics(sc[te_ids], msk[te_ids], lab[te_ids])
    m_inc = rank_metrics(sc[ti_ids], splits["inc"][ti_ids], lab[ti_ids])
    assert abs(m_all["nll"] - te_nll) < 1e-6, (m_all["nll"], te_nll)  # same split sanity
    return dict(
        success=ok,
        train_nll=tr_nll,
        fitfn_test_nll=te_nll,
        fitfn_top5=top5,
        fitfn_random_nll=rand_nll,
        newcomer_asc=float(w0[5]),
        test_all_events=m_all,
        test_incumbent_only=m_inc,
    )


def decide(logit, combos):
    """REVIEW R8 decision rule on the incumbent-only test slice."""
    key = "test_incumbent_only"
    best_name = min(combos, key=lambda n: combos[n][key]["nll"])
    xg, lg = combos[best_name][key], logit[key]
    d_nll, d_top5 = lg["nll"] - xg["nll"], xg["top5"] - lg["top5"]
    parity = [n for n in combos if n.endswith("parity")]
    parity_beats = any(
        lg["nll"] - combos[n][key]["nll"] >= NLL_EPS
        or combos[n][key]["top5"] - lg["top5"] >= TOP5_EPS
        for n in parity
    )
    if d_nll >= NLL_EPS or d_top5 >= TOP5_EPS:
        cause = (
            "nonlinearity in the 6 choice features (parity XGB already beats the logit)"
            if parity_beats
            else "missing features (only the kitchen sink beats the logit)"
        )
        verdict = f"XGB >> logit -> {cause}"
    else:
        verdict = "XGB ~ logit -> parametric choice model defensible"
    return dict(
        best_combo=best_name,
        delta_nll_vs_logit=d_nll,
        delta_top5_vs_logit=d_top5,
        parity_beats_logit=bool(parity_beats),
        verdict=verdict,
    )


def run_dataset(data_dir):
    print(f"\n=== {data_dir} ===")
    t0 = time.perf_counter()
    cs = build_choice_sets(data_dir, max_candidates=MAX_CANDIDATES, seed=SEED)
    ext = build_choice_sets_with_extras(data_dir, max_candidates=MAX_CANDIDATES, seed=SEED)
    for k in ("F", "mask", "label", "sector", "day"):
        assert np.array_equal(cs[k], ext[k]), f"replica mismatch on {k}"
    print(f"  choice sets built + replica verified ({time.perf_counter() - t0:.1f}s)")

    splits = make_splits(cs)
    p_new = splits["newcomer_share_train"]
    const_asc = float(np.log(p_new / (1.0 - p_new)))
    print(
        f"  events={len(cs['label'])}  train newcomer share={p_new:.3f} "
        f"(const ASC={const_asc:+.3f})  cut day={splits['cut']:.0f}  "
        f"val cut day={splits['val_cut']:.0f}"
    )

    logit = logit_reference(cs, splits)
    print(
        f"  logit: test NLL(all)={logit['test_all_events']['nll']:.3f}  "
        f"top5(all)={logit['test_all_events']['top5']:.3f}  "
        f"NLL(inc)={logit['test_incumbent_only']['nll']:.3f}  "
        f"top5(inc)={logit['test_incumbent_only']['top5']:.3f}"
    )

    parity_feat = cs["F"].astype(np.float32)
    sink_feat = np.concatenate([cs["F"], ext["X_extra"]], axis=2).astype(np.float32)
    feature_sets = dict(parity=parity_feat, kitchen_sink=sink_feat)
    combos = {}
    for obj_tag, objective in (("binary", "binary:logistic"), ("pairwise", "rank:pairwise")):
        for fs_tag, feat in feature_sets.items():
            name = f"{obj_tag}_{fs_tag}"
            print(f"  -- {name}")
            combos[name] = run_combo(objective, feat, cs, splits, const_asc)

    decision = decide(logit, combos)
    return dict(
        n_events=int(len(cs["label"])),
        n_dropped_repeat=int(cs["n_dropped_repeat"]),
        newcomer_share_train=p_new,
        const_newcomer_score=const_asc,
        cut_day=splits["cut"],
        val_cut_day=splits["val_cut"],
        parity_features=list(cs["feature_names"]),
        kitchen_sink_extra_features=list(ext["extra_names"]),
        logit=logit,
        xgb=combos,
        decision=decision,
    )


def summarize(tag, res):
    print(f"\n{'=' * 78}\n{tag}: TEST metrics (winner configs)\n{'=' * 78}")
    hdr = f"  {'model':22s} {'NLLinc':>7s} {'top1':>6s} {'top5':>6s} {'top10':>6s} {'MRR':>6s} | {'NLLall':>7s} {'top5':>6s}"
    print(hdr)
    rows = [("logit (reference)", res["logit"])] + [(name, res["xgb"][name]) for name in res["xgb"]]
    for name, r in rows:
        a, b = (
            r["test_incumbent_only"],
            r["test_all_events_constASC" if name != "logit (reference)" else "test_all_events"],
        )
        print(
            f"  {name:22s} {a['nll']:7.3f} {a['top1']:6.3f} {a['top5']:6.3f} "
            f"{a['top10']:6.3f} {a['mrr']:6.3f} | {b['nll']:7.3f} {b['top5']:6.3f}"
        )
    print(
        f"  random NLL: inc {res['logit']['test_incumbent_only']['random_nll']:.3f} / all {res['logit']['test_all_events']['random_nll']:.3f}"
    )
    print(f"  DECISION [{res['decision']['best_combo']}]: {res['decision']['verdict']}")
    print(
        f"    dNLL(inc) vs logit = {res['decision']['delta_nll_vs_logit']:+.4f}  "
        f"dtop5(inc) = {res['decision']['delta_top5_vs_logit']:+.4f}  "
        f"(thresholds {NLL_EPS} / {TOP5_EPS})"
    )


def main():
    out = {}
    for data_dir in DATASETS:
        tag = os.path.basename(data_dir)
        out[tag] = run_dataset(data_dir)
    os.makedirs("results", exist_ok=True)
    with open("results/exp26_xgb.json", "w") as fh:
        json.dump(out, fh, indent=1)
    for tag, res in out.items():
        summarize(tag, res)
    print("\nwrote results/exp26_xgb.json")


if __name__ == "__main__":
    main()
