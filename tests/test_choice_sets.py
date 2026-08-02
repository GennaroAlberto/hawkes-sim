"""Unit tests for synthetic.loaders.build_choice_sets (REVIEW.md section N).

A hand-built 4-firm / 1-sector world pins down the risk-set rules exactly:

* funded-pool rule: only previously-funded (strictly earlier) firms enter;
* exclude-last-funded rule, with the same-day tie broken by deal_id order
  (state updates sequentially within a day);
* active window: a firm past its ground-truth exit week leaves the pool;
* newcomer label for first-timers and for out-of-universe winners;
* repeat-by-last-funded events are dropped and counted;
* N1 context columns (newcomer_context=True) land ONLY on the newcomer row and
  match hand-computed values; default output is unchanged by the flag.

A smoke test on data/synthetic_A checks the enlarged feature width with an
unchanged event count.
"""

import json
import os

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from synthetic.loaders import (  # noqa: E402
    CHOICE_CONTEXT_FEATURES,
    CHOICE_FEATURES,
    build_choice_sets,
)

WEEK0 = pd.Timestamp("2020-01-06")
SYNTH_A = "data/synthetic_A"


def _date(day):
    return str((WEEK0 + pd.Timedelta(days=day)).date())


@pytest.fixture()
def tiny_world(tmp_path):
    """4 tracked firms (C1, C2, C3, C4) + out-of-universe C9, one sector.

    Deal script (day, deal_id, company):
      0  D01 C1  first ever          -> newcomer win, empty pool
      1  D02 C3  pool={C1} minus last(C1) -> newcomer win, empty risk set
      2  D03 C2  cand=[C1]           -> newcomer win (C2 never funded: pool rule)
      8  D04 C1  cand=[C1] (C2=last excluded, C3 inactive from day 7) -> label 0
      12 D05 C2  cand=[C2] (C1=last excluded)                         -> label 0
      12 D06 C1  cand=[C1] (C2=last after D05: same-day deal_id order) -> label 0
      16 D07 C1  winner == last-funded C1 -> dropped repeat
      20 D08 C9  out-of-universe winner   -> newcomer win, cand=[C2]
    """
    meta = dict(
        week_index=[_date(7 * i) for i in range(6)],
        sector_names=["S"],
    )
    (tmp_path / "meta.json").write_text(json.dumps(meta))

    pd.DataFrame(
        dict(
            company_id=["C1", "C2", "C3", "C4"],
            primary_industry_sector=["S", "S", "S", "S"],
        )
    ).to_csv(tmp_path / "companies.csv", index=False)

    # world order == companies order; C3 exits at week 1 (day 7)
    np.savez(
        tmp_path / "ground_truth.npz",
        entry_week=np.array([0, 0, 0, 0]),
        exit_week=np.array([100, 100, 1, 100]),
        tracked_positions=np.array([0, 1, 2, 3]),
    )

    deals = pd.DataFrame(
        [
            ("D01", "C1", _date(0), "S", 1.0, 10),
            ("D02", "C3", _date(1), "S", 2.0, 20),
            ("D03", "C2", _date(2), "S", 3.0, 30),
            ("D04", "C1", _date(8), "S", 4.0, 40),
            ("D05", "C2", _date(12), "S", 5.0, 50),
            ("D06", "C1", _date(12), "S", 6.0, 60),
            ("D07", "C1", _date(16), "S", 7.0, 70),
            ("D08", "C9", _date(20), "S", 8.0, 80),
        ],
        columns=[
            "deal_id",
            "company_id",
            "deal_date",
            "primary_industry_sector",
            "total_raised_to_date",
            "number_of_employees",
        ],
    )
    deals.to_csv(tmp_path / "deals.csv", index=False)

    # macro: base values published before day 0; FFUND re-published at day 10
    rows = [
        ("FFUND", "2020-01-01", "2020-01-01", 1.0),
        ("CPI_YOY", "2020-01-01", "2020-01-01", 2.0),
        ("RUNEMP", "2020-01-01", "2020-01-01", 3.0),
        ("FFUND", _date(10), _date(10), 9.0),
    ]
    pd.DataFrame(rows, columns=["series_code", "ref_date", "publish_date", "value"]).to_csv(
        tmp_path / "macro.csv", index=False
    )
    return tmp_path


def test_pool_exclusion_and_labels(tiny_world):
    cs = build_choice_sets(str(tiny_world), max_candidates=8)
    # D07 dropped (winner == last-funded); everything else kept in day order
    assert cs["n_events"] == 7
    assert cs["n_dropped_repeat"] == 1
    assert cs["n_newcomer_wins"] == 4
    n_inc = cs["mask"].sum(1) - 1  # incumbents per event (newcomer slot always on)
    npos = (cs["F"][:, :, 5] > 0.5).argmax(1)

    # D01, D02: empty risk sets, newcomer wins
    assert list(n_inc[:2]) == [0, 0]
    assert cs["label"][0] == npos[0] and cs["label"][1] == npos[1]
    # D03: funded-pool rule -- C2 (tracked, active, never funded) NOT in the pool;
    # cand = [C1] only, winner is a first-timer -> newcomer label
    assert n_inc[2] == 1
    assert cs["label"][2] == npos[2]
    # the single incumbent row is C1's LOCF state: raised=1.0, emp=10
    assert np.isclose(cs["F"][2, 0, 0], np.log1p(1.0))
    assert np.isclose(cs["F"][2, 0, 1], np.log(10.0))
    # D04 (day 8): exclude-last-funded drops C2, active window drops C3 -> [C1]
    assert n_inc[3] == 1 and cs["label"][3] == 0
    # D05/D06 same day: state updates sequentially in deal_id order, each winner
    # is the single non-excluded incumbent
    assert n_inc[4] == 1 and cs["label"][4] == 0
    assert n_inc[5] == 1 and cs["label"][5] == 0
    # D08: out-of-universe winner -> newcomer label; cand = [C2]
    assert n_inc[6] == 1 and cs["label"][6] == npos[6]


def test_newcomer_context_columns(tiny_world):
    cs0 = build_choice_sets(str(tiny_world), max_candidates=8)
    cs = build_choice_sets(str(tiny_world), max_candidates=8, newcomer_context=True)

    # default tensors unchanged by the flag; only extra columns appended
    assert cs["F"].shape[2] == 12
    assert cs["feature_names"] == CHOICE_FEATURES + CHOICE_CONTEXT_FEATURES
    np.testing.assert_array_equal(cs0["F"], cs["F"][:, :, :6])
    np.testing.assert_array_equal(cs0["mask"], cs["mask"])
    np.testing.assert_array_equal(cs0["label"], cs["label"])
    np.testing.assert_array_equal(cs0["day"], cs["day"])
    assert list(cs["deal_id"]) == ["D01", "D02", "D03", "D04", "D05", "D06", "D08"]

    npos = (cs["F"][:, :, 5] > 0.5).argmax(1)
    ar = np.arange(cs["n_events"])
    # context columns land ONLY on the newcomer row
    ctx = cs["F"][:, :, 6:].copy()
    ctx[ar, npos] = 0.0
    assert np.all(ctx == 0.0)

    g8 = cs["F"][6, npos[6], 6:]  # D08, day 20
    # 7 sector deals strictly before day 20 (D01..D07; D07 dropped as an event
    # but still a deal); 3 of them are first financings (D01, D02, D03)
    assert np.isclose(g8[0], np.log1p(7))
    assert np.isclose(g8[1], (3 + 1) / (7 + 2))
    # macro as-of day 20: FFUND re-published at day 10 -> 9.0
    assert np.allclose(g8[2:5], [9.0, 2.0, 3.0])
    assert np.isclose(g8[5], np.log1p(1))  # eligible pool = [C2]

    g1 = cs["F"][0, npos[0], 6:]  # D01, day 0
    assert np.isclose(g1[0], 0.0)  # no deals yet
    assert np.isclose(g1[1], 0.5)  # Laplace prior (0+1)/(0+2)
    assert np.allclose(g1[2:5], [1.0, 2.0, 3.0])
    assert np.isclose(g1[5], 0.0)  # empty pool

    g4 = cs["F"][3, npos[3], 6:]  # D04, day 8 < publish day 10
    assert np.isclose(g4[2], 1.0)


def test_context_requires_newcomer(tiny_world):
    with pytest.raises(ValueError):
        build_choice_sets(str(tiny_world), newcomer=False, newcomer_context=True)


@pytest.mark.skipif(not os.path.isdir(SYNTH_A), reason="synthetic_A dataset not on disk")
def test_synthetic_A_smoke():
    cs0 = build_choice_sets(SYNTH_A)
    csx = build_choice_sets(SYNTH_A, newcomer_context=True)
    assert cs0["F"].shape[2] == 6
    assert csx["F"].shape[2] == 12
    assert csx["n_events"] == cs0["n_events"]
    assert csx["F"].shape[1] == cs0["F"].shape[1]
    assert len(csx["deal_id"]) == csx["n_events"]
    # context only on the newcomer row, everywhere
    npos = (csx["F"][:, :, 5] > 0.5).argmax(1)
    ctx = csx["F"][:, :, 6:].copy()
    ctx[np.arange(csx["n_events"]), npos] = 0.0
    assert np.all(ctx == 0.0)
