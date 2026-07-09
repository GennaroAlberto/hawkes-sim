"""Deal-mark machinery: sizes, valuations, rounds, employees, financials.

Shared by both regimes.  Maintains per-firm state (stage, raised, employees,
revenue, last valuation) and produces the mark fields of a PitchBook deal row.
Marks depend on latent quality and on market conditions (risk appetite, rates),
so they carry signal but are NOT part of the ranker's true score in regime A.
"""

from __future__ import annotations

import numpy as np

from .firms import STAGES


class FirmState:
    def __init__(self, universe, quality, rng):
        n = len(universe)
        self.rng = rng
        self.quality = quality
        self.stage = np.zeros(n, dtype=int)  # index into STAGES
        self.raised = np.zeros(n)  # $M, true cumulative
        self.employees = universe["employees_0"].to_numpy().astype(float)
        self.revenue = np.zeros(n)  # $M/yr, 0 pre-revenue
        self.last_val = np.full(n, np.nan)  # post-money $M
        self.last_deal_week = np.full(n, -10_000, dtype=int)
        self.n_deals = np.zeros(n, dtype=int)

    def draw_deal(self, i, week, risk_appetite, ffund):
        """Advance firm ``i`` through a funding round at ``week``; return marks."""
        rng = self.rng
        q = self.quality[i]
        st = int(self.stage[i])
        hot = 0.18 * risk_appetite - 0.05 * (ffund - 2.0)

        # Deal size: grows with stage, quality, and market heat.
        base = [1.5, 6.0, 15.0, 35.0, 70.0, 120.0][st]
        size = base * float(np.exp(0.35 * q + hot + rng.normal(0.0, 0.45)))
        size = max(0.1, round(size, 2))

        # Valuation: multiple of size, pro-cyclical.
        multiple = float(np.exp(rng.normal(1.35, 0.30) + 0.5 * hot))
        post = round(size * (1.0 + multiple), 2)
        pre = round(max(post - size, 0.05), 2)
        prev_val = self.last_val[i]
        if np.isfinite(prev_val):
            updf = (
                "Up" if post > 1.05 * prev_val else ("Down" if post < 0.95 * prev_val else "Flat")
            )
        else:
            updf = ""

        # Growth between rounds; employees/revenue update ONLY at deals (LOCF world).
        gap_w = week - self.last_deal_week[i] if self.n_deals[i] else np.nan
        emp_growth = float(np.exp(0.12 + 0.08 * q + rng.normal(0.0, 0.25)))
        # size-damped growth with a hard cap: firms plateau, no runaway feature loop
        self.employees[i] = float(
            np.clip(
                self.employees[i] * emp_growth * (1.0 - 0.1 * np.log1p(self.employees[i] / 200.0)),
                2.0,
                2000.0,
            )
        )
        rev_prev = self.revenue[i]
        if st >= 1 or rng.random() < 0.3:
            self.revenue[i] = max(
                0.05,
                (self.revenue[i] if self.revenue[i] > 0 else 0.3)
                * float(np.exp(0.35 + 0.15 * q + rng.normal(0.0, 0.4))),
            )
        rev = self.revenue[i]
        rev_growth = (rev / rev_prev - 1.0) if rev_prev > 0 else np.nan
        ebitda = (
            round(-0.4 * rev + 0.15 * rev * q + rng.normal(0.0, 0.2 * max(rev, 1.0)), 2)
            if rev > 0
            else np.nan
        )
        gross = round(0.6 * rev, 2) if rev > 0 else np.nan
        net = round(ebitda - 0.05 * rev, 2) if rev > 0 and np.isfinite(ebitda) else np.nan

        self.raised[i] += size
        marks = dict(
            deal_size=size,
            pre_money_valuation=pre,
            post_valuation=post,
            vc_round=STAGES[st],
            vc_round_up_down_flat=updf,
            raised_to_date=round(self.raised[i], 2),
            total_raised_to_date=round(self.raised[i], 2),
            total_invested_capital=round(self.raised[i] * float(rng.uniform(0.95, 1.05)), 2),
            number_of_employees=int(round(self.employees[i])),
            pct_acquired=round(100.0 * size / post, 2),
            total_investor_ownership=round(min(95.0, 15.0 + 12.0 * st + rng.normal(0, 5)), 2),
            deal_type="VC Round" if st < 5 else "Later Stage VC",
            deal_type_2=STAGES[st],
            deal_type_3="",
            deal_class="Venture Capital",
            total_invested_equity=size,
            debt=0.0,
            total_new_debt=0.0,
            total_debt=0.0,
            type_of_stock="Preferred",
            deal_status="Completed",
            sellers="",
            revenue=round(rev, 2) if rev > 0 else np.nan,
            ebitda=ebitda,
            gross_profit=gross,
            net_income=net,
            revenue_growth_since_last_deal=round(rev_growth, 3)
            if np.isfinite(rev_growth)
            else np.nan,
            valuation_revenue=round(post / rev, 2) if rev > 0 else np.nan,
            valuation_ebitda=round(post / abs(ebitda), 2)
            if np.isfinite(ebitda) and ebitda != 0
            else np.nan,
            valuation_cash_flow=np.nan,
            deal_size_revenue=round(size / rev, 2) if rev > 0 else np.nan,
            deal_size_ebitda=np.nan,
            deal_size_cash_flow=np.nan,
            weeks_since_prev_deal=gap_w,
        )
        self.last_val[i] = post
        self.last_deal_week[i] = week
        self.n_deals[i] += 1
        if self.stage[i] < len(STAGES) - 1 and self.rng.random() < 0.85:
            self.stage[i] += 1
        return marks


# --- Point-in-time ranker features (the loader rebuilds exactly these). --------
FEATURE_NAMES = [
    "log1p_total_raised",  # LOCF from deals (0 before first deal)
    "log_employees",  # LOCF from deals (founding size before first deal)
    "log1p_age_years",  # deterministic from year_founded
    "stage_num",  # LOCF: number of completed rounds, capped at 6
    "recency",  # exp(-(weeks since last deal)/52); 0 if never funded
]


def features_at(state: FirmState, universe, week, week_per_year=52.0):
    """(N, 5) point-in-time observable features implied by the current state."""
    len(state.raised)
    age_w = np.maximum(week - universe["entry_week"].to_numpy(), 0)
    gap = week - state.last_deal_week
    rec = np.where(state.n_deals > 0, np.exp(-np.maximum(gap, 0) / week_per_year), 0.0)
    z = np.stack(
        [
            np.log1p(state.raised),
            np.log(np.maximum(state.employees, 2.0)),
            np.log1p(age_w / week_per_year),
            np.minimum(state.n_deals, 6).astype(float),
            rec,
        ],
        axis=1,
    )
    return z
