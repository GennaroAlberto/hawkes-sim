"""Observation layer: world -> noisy PitchBook-style tables.

Noise model (all knobs in NOISE, recorded in meta.json):

* undisclosed deal sizes (`deal_size` NaN, deal_size_status='Undisclosed');
  'Estimated' sizes carry multiplicative error.
* missing valuations (pre/post NaN together, status blank).
* sector mislabels: primary_industry_sector wrong on a small fraction of deals.
* duplicate deal rows (same deal, new deal_id, jittered size) — dedup hygiene test.
* date jitter: announced vs. closed dates a few days apart.
* employees reported with noise and rounding.
* untracked firms: deals exist (company_universe='Out of Universe') but the firm
  has NO row in companies.csv — motivates the ranker's outside option.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .firms import GROUPS, SECTORS

NOISE = dict(
    p_undisclosed_size=0.28,
    p_estimated_size=0.15,
    estimated_size_sd=0.18,
    p_missing_valuation=0.55,
    p_sector_mislabel=0.03,
    p_duplicate=0.012,
    p_date_jitter=0.30,
    max_date_jitter_days=6,
    employees_noise_sd=0.10,
)


def build_pitchbook_tables(
    world, universe, week_index, *, day_resolution=False, seed=0, noise=NOISE
):
    rng = np.random.default_rng(seed + 707)
    events = world["events"]
    marks = world["marks"]
    n_ev = len(events)
    sec_names = np.array(SECTORS[: world["sector_counts"].shape[1]])
    week0 = pd.Timestamp(week_index[0])

    rows = []
    for k in range(n_ev):
        tt, s, i = events[k]
        m = dict(marks[k])
        if day_resolution:
            date = week0 + pd.Timedelta(days=int(tt))
        else:
            date = pd.Timestamp(week_index[tt]) + pd.Timedelta(days=int(rng.integers(0, 5)))
        firm = universe.iloc[i]

        true_sector = sec_names[s]
        obs_sector = true_sector
        if rng.random() < noise["p_sector_mislabel"]:
            obs_sector = str(rng.choice(sec_names))

        size, status = m["deal_size"], "Actual"
        u = rng.random()
        if u < noise["p_undisclosed_size"]:
            size, status = np.nan, "Undisclosed"
        elif u < noise["p_undisclosed_size"] + noise["p_estimated_size"]:
            size = round(size * float(np.exp(rng.normal(0, noise["estimated_size_sd"]))), 2)
            status = "Estimated"
        pre, post, vstat = m["pre_money_valuation"], m["post_valuation"], "Actual"
        if rng.random() < noise["p_missing_valuation"]:
            pre, post, vstat = np.nan, np.nan, ""
        emp = int(
            round(
                m["number_of_employees"] * float(np.exp(rng.normal(0, noise["employees_noise_sd"])))
            )
        )
        if rng.random() < noise["p_date_jitter"]:
            date = date + pd.Timedelta(days=int(rng.integers(1, noise["max_date_jitter_days"] + 1)))

        rows.append(
            dict(
                deal_id=f"D{500000 + k}",
                company_id=firm["company_id"],
                deal_date=date.date().isoformat(),
                primary_industry_sector=obs_sector,
                primary_industry_group=GROUPS.get(str(obs_sector), ""),
                primary_industry_code=f"PIC{list(sec_names).index(obs_sector):03d}",
                current_financing_status="Venture Capital-Backed",
                company_universe=("Venture Capital" if firm["tracked"] else "Out of Universe"),
                deal_size=size,
                deal_size_status=status,
                pre_money_valuation=pre,
                post_valuation=post,
                post_valuation_status=vstat,
                number_of_employees=emp,
                **{
                    k2: m[k2]
                    for k2 in (
                        "pct_acquired",
                        "raised_to_date",
                        "vc_round",
                        "vc_round_up_down_flat",
                        "total_raised_to_date",
                        "total_invested_capital",
                        "total_investor_ownership",
                        "deal_type",
                        "deal_type_2",
                        "deal_type_3",
                        "deal_class",
                        "total_invested_equity",
                        "debt",
                        "total_new_debt",
                        "type_of_stock",
                        "deal_status",
                        "sellers",
                        "revenue",
                        "ebitda",
                        "valuation_cash_flow",
                        "valuation_ebitda",
                        "valuation_revenue",
                        "deal_size_cash_flow",
                        "deal_size_ebitda",
                        "deal_size_revenue",
                        "gross_profit",
                        "net_income",
                        "revenue_growth_since_last_deal",
                        "total_debt",
                    )
                },
                year_founded=int(firm["year_founded"]),
                hq_location=firm["hq_location"],
                company_country=firm["hq_country"],
                hq_global_region=firm["hq_global_region"],
                hq_global_sub_region=firm["hq_global_sub_region"],
                _true_sector_idx=int(s),
                _true_week=int(tt if not day_resolution else tt // 7),
                _firm_idx=int(i),
            )
        )
    deals = pd.DataFrame(rows)

    # duplicates
    dup = deals.sample(frac=noise["p_duplicate"], random_state=int(rng.integers(1e9)))
    dup = dup.assign(
        deal_id=[f"D9{k:06d}" for k in range(len(dup))],
        deal_size=dup.deal_size * np.exp(rng.normal(0, 0.05, len(dup))),
        deal_size_status="Estimated",
    )
    deals = (
        pd.concat([deals, dup], ignore_index=True).sort_values("deal_date").reset_index(drop=True)
    )

    # companies table: tracked firms only, last-* fields from their latest deal.
    tracked = universe[universe.tracked].copy()
    last = (
        deals[deals.company_id.isin(tracked.company_id)]
        .sort_values("deal_date")
        .groupby("company_id")
        .last()
    )
    companies = pd.DataFrame(
        {
            "company_id": tracked.company_id,
            "company_name": tracked.company_name,
            "description": "Synthetic company; sector " + tracked.primary_industry_sector,
            "other_industries": "",
            "financing_status": "Venture Capital-Backed",
            "business_status": np.where(
                tracked.exit_week < len(week_index), "Out of Business", "Generating Revenue"
            ),
            "ownership_status": "Privately Held (backing)",
            "company_universe": "Venture Capital",
            "website": "www."
            + tracked.company_name.str.replace(" ", "").str.lower()
            + ".example.com",
            "year_founded": tracked.year_founded,
            "hq_location": tracked.hq_location,
            "hq_country": tracked.hq_country,
            "hq_global_region": tracked.hq_global_region,
            "hq_global_sub_region": tracked.hq_global_sub_region,
            "primary_industry_sector": tracked.primary_industry_sector,
        }
    ).set_index("company_id")
    for col_src, col_dst in [
        ("total_raised_to_date", "total_money_raised"),
        ("post_valuation", "last_financing_valuation"),
        ("deal_date", "last_financing_date"),
        ("deal_size", "last_financing_size"),
        ("deal_type", "last_financing_deal_type"),
        ("deal_date", "last_known_valuation_date"),
        ("deal_type", "last_known_valuation_deal_type"),
    ]:
        companies[col_dst] = last[col_src] if col_src in last else np.nan
    companies["employees"] = last["number_of_employees"]
    companies = companies.reset_index()

    # investors table
    n_inv = 2500
    inv_ids = np.array([f"I{30000 + j}" for j in range(n_inv)])
    inv_pop = rng.zipf(1.6, size=n_inv).astype(float)
    inv_p = inv_pop / inv_pop.sum()
    irows = []
    for _, r in deals.iterrows():
        for j in rng.choice(n_inv, size=int(rng.integers(1, 5)), replace=False, p=inv_p):
            irows.append((r.company_id, inv_ids[j], f"Investor {j:04d}", "Active", r.deal_date))
    investors = pd.DataFrame(
        irows,
        columns=["company_id", "investor_id", "investorname", "investorstatus", "investorsince"],
    )
    investors = (
        investors.sort_values("investorsince")
        .groupby(["company_id", "investor_id"], as_index=False)
        .first()
    )

    return deals, companies, investors
