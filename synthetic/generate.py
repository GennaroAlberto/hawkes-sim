"""Generate a full synthetic FactSet/PitchBook dataset.

Usage (from the repo root):

    PYTHONPATH=. python -m synthetic.generate --regime A --out data/synthetic_A --seed 0
    PYTHONPATH=. python -m synthetic.generate --regime B --out data/synthetic_B --seed 0

Outputs in --out:
    deals.csv, companies.csv, investors.csv, macro.csv    (the observed world)
    ground_truth.npz, private_deal_map.csv, meta.json     (oracle-only files)
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .firms import SECTORS, generate_universe
from .macro import generate_macro, weekly_macro_matrix
from .observe import NOISE, build_pitchbook_tables
from .regime_a import simulate_regime_a
from .regime_b import simulate_regime_b

BURNIN_WEEKS = 60


def simulate_world(
    regime="A",
    seed=0,
    weeks=520,
    n_firms=8000,
    n_sectors=12,
    target_mean=8.0,
    target_total_events_b=55_000,
):
    macro, latent_full, week_full = generate_macro(weeks=weeks + BURNIN_WEEKS, seed=seed)
    week_index = week_full[BURNIN_WEEKS:]
    latent = latent_full[BURNIN_WEEKS:]
    Xw_raw = weekly_macro_matrix(macro, week_index)
    universe, quality = generate_universe(
        n_firms=n_firms, n_sectors=n_sectors, week_index=week_index, seed=seed
    )
    if regime.upper() == "A":
        world = simulate_regime_a(
            universe,
            quality,
            Xw_raw,
            latent,
            n_sectors=n_sectors,
            target_mean=target_mean,
            seed=seed,
        )
    else:
        world = simulate_regime_b(
            universe,
            quality,
            Xw_raw,
            latent,
            n_sectors=n_sectors,
            target_total_events=target_total_events_b,
            seed=seed,
        )
    return dict(
        world=world,
        universe=universe,
        quality=quality,
        macro=macro,
        week_index=week_index,
        regime=regime.upper(),
        seed=seed,
        weeks=weeks,
        n_firms=n_firms,
        n_sectors=n_sectors,
    )


def emit_dataset(sim, out):
    os.makedirs(out, exist_ok=True)
    world, universe, quality = sim["world"], sim["universe"], sim["quality"]
    macro, week_index = sim["macro"], sim["week_index"]
    regime, seed = sim["regime"], sim["seed"]
    day_res = regime == "B"

    deals, companies, investors = build_pitchbook_tables(
        world, universe, week_index, day_resolution=day_res, seed=seed
    )

    private_cols = [c for c in deals.columns if c.startswith("_")]
    deal_map = deals[["deal_id"] + private_cols].copy()
    deals_obs = deals.drop(columns=private_cols)

    deals_obs.to_csv(f"{out}/deals.csv", index=False)
    companies.to_csv(f"{out}/companies.csv", index=False)
    investors.to_csv(f"{out}/investors.csv", index=False)
    macro_obs = macro.copy()
    macro_obs["ref_date"] = macro_obs["ref_date"].dt.date
    macro_obs["publish_date"] = macro_obs["publish_date"].dt.date
    macro_obs.to_csv(f"{out}/macro.csv", index=False)
    deal_map.to_csv(f"{out}/private_deal_map.csv", index=False)

    truth = dict(world["truth"])
    truth.pop("day_events", None)
    reserved = {
        "regime",
        "events",
        "sector_counts_true",
        "startup_counts_true",
        "covariates_std",
        "entry_week",
        "exit_week",
        "tracked",
        "tracked_positions",
        "firm_sector_true",
        "quality",
    }
    np.savez_compressed(
        f"{out}/ground_truth.npz",
        **{
            k: np.asarray(v)
            for k, v in truth.items()
            if not isinstance(v, str) and k not in reserved
        },
        regime=np.array(truth.get("regime", regime)),
        events=world["events"],
        sector_counts_true=world["sector_counts"],
        startup_counts_true=world["startup_counts"],
        covariates_std=world["covariates_std"],
        entry_week=universe.entry_week.to_numpy(),
        exit_week=universe.exit_week.to_numpy(),
        tracked=universe.tracked.to_numpy(),
        tracked_positions=np.flatnonzero(universe.tracked.to_numpy()),
        firm_sector_true=universe.sector_idx.to_numpy(),
        quality=quality,
    )

    meta = dict(
        regime=regime,
        seed=seed,
        weeks=sim["weeks"],
        n_firms=sim["n_firms"],
        n_sectors=sim["n_sectors"],
        sector_names=SECTORS[: sim["n_sectors"]],
        week_index=[d.date().isoformat() for d in week_index],
        n_deals=int(len(deals_obs)),
        n_companies=int(len(companies)),
        n_investor_links=int(len(investors)),
        noise=NOISE,
        feature_names=[
            "log1p_total_raised",
            "log_employees",
            "log1p_age_years",
            "stage_num",
            "recency",
        ],
        notes="Firm features update ONLY at deal events (LOCF). Macro is long-format "
        "with publish_date; always as-of join. company_universe='Out of "
        "Universe' deals have no companies.csv row (outside option).",
    )
    with open(f"{out}/meta.json", "w") as fh:
        json.dump(meta, fh, indent=1)

    print(
        f"[{regime}] {len(deals_obs)} deals, {len(companies)} companies, "
        f"{len(investors)} investor links -> {out}"
    )
    return out


def generate(
    regime="A",
    out="data/synthetic_A",
    seed=0,
    weeks=520,
    n_firms=8000,
    n_sectors=12,
    target_mean=8.0,
    target_total_events_b=55_000,
):
    sim = simulate_world(
        regime=regime,
        seed=seed,
        weeks=weeks,
        n_firms=n_firms,
        n_sectors=n_sectors,
        target_mean=target_mean,
        target_total_events_b=target_total_events_b,
    )
    return emit_dataset(sim, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="A", choices=["A", "B", "a", "b"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weeks", type=int, default=520)
    ap.add_argument("--n-firms", type=int, default=8000)
    ap.add_argument("--n-sectors", type=int, default=12)
    a = ap.parse_args()
    out = a.out or f"data/synthetic_{a.regime.upper()}"
    generate(
        regime=a.regime,
        out=out,
        seed=a.seed,
        weeks=a.weeks,
        n_firms=a.n_firms,
        n_sectors=a.n_sectors,
    )


if __name__ == "__main__":
    main()
