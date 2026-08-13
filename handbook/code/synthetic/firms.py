"""Company universe generator (PitchBook-style)."""

from __future__ import annotations

import numpy as np
import pandas as pd

SECTORS = [
    "Information Technology",
    "Healthcare",
    "B2B",
    "B2C",
    "Financial Services",
    "Energy",
    "Materials and Resources",
    "Consumer Products",
    "Media",
    "Transportation",
    "AgTech",
    "Software",
]
GROUPS = {s: f"{s} Group" for s in SECTORS}
REGIONS = [
    ("United States", "San Francisco, CA", "Americas", "North America", 0.34),
    ("United States", "New York, NY", "Americas", "North America", 0.18),
    ("United States", "Boston, MA", "Americas", "North America", 0.08),
    ("United Kingdom", "London", "Europe", "Western Europe", 0.10),
    ("Germany", "Berlin", "Europe", "Western Europe", 0.06),
    ("France", "Paris", "Europe", "Western Europe", 0.05),
    ("India", "Bangalore", "Asia", "South Asia", 0.07),
    ("China", "Beijing", "Asia", "East Asia", 0.06),
    ("Singapore", "Singapore", "Asia", "Southeast Asia", 0.03),
    ("Israel", "Tel Aviv", "Middle East", "Middle East", 0.03),
]
STAGES = ["Seed Round", "Series A", "Series B", "Series C", "Series D", "Later Stage VC"]


def generate_universe(n_firms=8000, n_sectors=12, week_index=None, seed=0, p_untracked=0.12):
    """Return a firm-state DataFrame plus arrays used by the simulators.

    * ``quality``      -- latent (never observed) firm quality, drives everything.
    * ``entry_week``   -- first week the firm is in the risk set (founded/first tracked).
    * ``exit_week``    -- week the firm leaves (acquisition/IPO/death), T if never.
    * ``tracked``      -- False for firms outside the data vendor's universe: their
                          deals still happen (sector counts see them) but the firm
                          has no row in the companies table (outside option).
    """
    rng = np.random.default_rng(seed + 202)
    T = len(week_index)
    sectors = np.array(SECTORS[:n_sectors])
    firm_sector = rng.integers(0, n_sectors, size=n_firms)

    quality = rng.normal(size=n_firms)
    year0 = int(pd.Timestamp(week_index[0]).year)
    year_founded = rng.integers(year0 - 11, year0 + 9, size=n_firms)
    founded_week = np.clip((year_founded - year0) * 52 + rng.integers(0, 52, n_firms), 0, T - 1)
    founded_week[year_founded < year0] = 0
    # lifetime: most firms survive the window; some die/exit.
    exit_week = np.full(n_firms, T, dtype=int)
    dies = rng.random(n_firms) < 0.18
    exit_week[dies] = np.clip(founded_week[dies] + rng.integers(60, 400, dies.sum()), 1, T)
    tracked = rng.random(n_firms) >= p_untracked

    ridx = rng.choice(len(REGIONS), p=[r[4] for r in REGIONS], size=n_firms)
    emp0 = np.maximum(2, np.round(np.exp(rng.normal(2.2 + 0.25 * quality, 0.7)))).astype(int)

    df = pd.DataFrame(
        {
            "company_id": [f"C{100000 + i}" for i in range(n_firms)],
            "company_name": [f"Startup {i:05d}" for i in range(n_firms)],
            "sector_idx": firm_sector,
            "primary_industry_sector": sectors[firm_sector],
            "year_founded": year_founded,
            "entry_week": founded_week,
            "exit_week": exit_week,
            "tracked": tracked,
            "hq_country": [REGIONS[k][0] for k in ridx],
            "hq_location": [REGIONS[k][1] for k in ridx],
            "hq_global_region": [REGIONS[k][2] for k in ridx],
            "hq_global_sub_region": [REGIONS[k][3] for k in ridx],
            "employees_0": emp0,
        }
    )
    return df, quality
