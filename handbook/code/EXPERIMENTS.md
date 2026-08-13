# EXPERIMENTS.md (v2) — Synthetic calibration campaign, event-time assumptions

Briefing for the coding agent. Read fully before running. Theory references:
`paper/complete_account.tex` (two-stage model, block Hawkes, stability) and
`paper/book/` ch. 3 (event-time MLE), ch. 7 (identifiability), ch. 8 (covariates),
ch. 12 (PINO). This version supersedes v1 after a change of assumptions.

## 0. Standing assumptions (changed — read carefully)

1. **Deal dates are observed at day resolution → no interval censoring.** The time
   unit is the day. Event-time likelihoods apply directly; weekly bucketing is a
   *choice* (for the count layer), not a data constraint.
2. **Risk sets are given truthfully.** Use the oracle active mask
   (`ground_truth.npz`; loaders handle it). Do not spend effort estimating exits.
3. **Choice-set construction** (implemented in
   `synthetic/loaders.py::build_choice_sets`): for each deal, candidates are the
   active firms of that sector **funded at least once before**, **excluding the
   most recently funded firm** of the sector, **plus one synthetic "newcomer"
   alternative** (all-zero features, `is_newcomer=1`) that wins whenever the
   funded firm is a first-timer or out-of-universe. Every candidate carries a
   **time-since-last-funding** feature (`log1p(days/30)`).
4. **Covariates are informative** — the macro baseline signal is real and should
   be measured, not assumed away (E5 ablation).
5. **Self-excitation is modest** (event volumes are not extreme). Work in the
   low-branching region; do not chase high-excitation regimes.
   Grid for sensitivity: effective radius / contagion ∈ {0.05, 0.15, 0.25} only.

## 1. Data

Two worlds in `data/` (regenerate: `PYTHONPATH=. python -m synthetic.generate
--regime A|B --seed k`; regime A ~50 s — stage via `simulate_world()`/pickle/
`emit_dataset()` if your shell kills long commands):

| | `synthetic_A` | `synthetic_B` |
|---|---|---|
| True DGP | weekly two-stage (sector GLM + risk-set softmax, 26-week cooldown indicator) | **daily firm-level block Hawkes**: self-inhibition 30d half-life, sector mean-field contagion 7d half-life, latent quality, macro baseline |
| Dates | week + uniform weekday (day-level fits are mildly misspecified by construction) | true day resolution |
| Role under v2 assumptions | well-specified for the *choice* layer; misspecification probe for day-level intensity fits | **well-specified for the event-time track** (this is now the primary world) |

~50k deals each, 12 sectors, 8000 firms (7031 tracked), 520 weeks. Observed
tables: `deals.csv` (44 PitchBook fields), `companies.csv`, `investors.csv`,
`macro.csv` (long format, FFUNDT/FFUND/CPI_RAW/CPI_YOY/RUNEMP, publish-date
as-of joins). Oracle-only: `ground_truth.npz`, `private_deal_map.csv`,
`meta.json`. Noise: 28% undisclosed sizes, 15% estimated (±18%), 55% missing
valuations, 3% sector mislabels, 1.2% duplicates, ≤6d date jitter, noisy
employees, publication lags, ~12% out-of-universe firms.

Generator knobs relevant here: `covariate_scale` (both regimes; make a
"covariate-strong" variant at 1.5), `latent_strength` (A), `effective_radius`
(A), `contagion_exponent` (B), `quality_strength` (B).

Loaders: `load_dataset(...)` → weekly arrays (counts, covariates, LOCF features,
events); `build_choice_sets(dir, max_candidates=64)` → padded day-resolution
choice tensors per §0.3 (6.4 s, ~49k events). Fitters:
`synthetic/fast_fit.py::fit_sector_glm_fast`, `fit_choice_fast` (vectorized;
package loop fitters are minutes-slow — verify equivalence once, then use fast).

## 2. Ground rules

* Temporal splits only. Weekly track: train `[0,416)`, test `[416,520)`. Choice
  track: `fit_choice_fast` splits at the 80% day quantile by default.
* Standardize covariates on the train window in real runs (stored scalers in
  `ground_truth.npz` are full-sample, for exact-recovery checks only).
* Oracle files never touch a fitting path — evaluation only.
* Any fitted excitation: report effective spectral radius; refuse forward
  simulation above 0.95 (Mechanism B, §6).
* Report quasi-Poisson dispersion with any Poisson NLL; 5 seeds for error bars.

## 3. Experiment tracks

### E1 — "When/where" layer: weekly GLM vs event-time Hawkes

Two calibrators for sector-level deal flow:

* **E1a (weekly GLM)**: `fit_sector_glm_fast` as in v1. Baselines (seed 0,
  regime A): beta corr 0.675, intercept bias −0.09, excitation entry corr 0.27,
  effective radius fit 0.355 vs true 0.172.
* **E1b (event-time, now primary)**: multivariate sector Hawkes MLE on the
  day-resolution event stream (`hawkes_calibration/eventtime`), exponential
  kernel, log-linear baseline in the as-of macro covariates. On regime B this is
  close to well-specified (mean-field contagion, 7d half-life).

Deliverables:

1. Recovery tables for both, 5 seeds, both regimes.
2. **θ-identifiability payoff**: the κ–θ ridge (ch. 7) afflicts weekly buckets;
   with exact dates θ (decay) should become identifiable. Show
   bias/SE of (κ, θ) for E1b vs E1a-style bucketing at {1d, 7d, 28d} — this
   quantifies what day resolution buys and where bucketing starts destroying
   timing information.
3. **Latent-confounding study** (kept from v1, it survives the new assumptions):
   fitted excitation absorbs the serially-correlated latent factor. Variants
   `latent_strength ∈ {0, 0.15, 0.3}` (A): plot fitted effective radius vs
   latent strength at fixed true radius. Mitigations: lagged aggregate count as
   covariate; l2 on off-diagonals; diagonal adjacency.
4. Covariate lift: held-out NLL with vs without macro covariates (ties to E5).

Success: E1b beats E1a on held-out NLL on regime B; θ recovered within 2 SE on
B at day resolution; a stated recommendation for bucket width on real data.

### E2 — "Which firm" layer: funded-pool conditional logit (primary choice model)

Model: score `q = (w0 + u_s)' z` over the §0.3 choice set; newcomer ASC is the
`is_newcomer` weight. Fit with `fit_choice_fast` on `build_choice_sets` output.

Measured baselines (seed 0, `max_candidates=64`, sector deviations on):

| | events | newcomer share | dropped repeats | test NLL | random NLL | top-5 acc |
|---|---|---|---|---|---|---|
| A | 48,723 | 0.276 | 35 | 3.346 | 4.011 | 0.446 |
| B | 49,032 | 0.273 | 2 | 3.796 | 4.172 | 0.316 |

Fitted w0 (A): [raised 0.34, emp 0.25, age 0.05, stage 0.13, gap **+1.15**,
newcomer 6.86] vs truth [0.30, 0.30, −0.15, 0.15] — raised/emp/stage recover;
the positive gap weight correctly expresses the true cooldown (recently funded →
suppressed). In B the gap weight (+0.44) matches the 30d-half-life inhibition.

Sub-experiments:

1. **Gap functional form**: log1p(days/30) vs binned dummies ({<1m, 1–3m, 3–6m,
   6–12m, >12m}) vs exponential decay exp(−days/τ), τ ∈ {30, 90, 180}d. Pick by
   held-out NLL; report the implied hazard-vs-gap curve against the true one
   (26-week indicator in A, exponential in B) — the binned version should reveal
   the true shape.
2. **Newcomer model**: enrich the ASC to `q_new = b0 + γ' g_{s,t}` with sector
   heat features (sector deals last 90d, macro as-of values). Does newcomer
   share become time-varying and calibrated? (Reliability diagram of predicted
   vs realized newcomer share by quarter.)
3. **Sampled softmax**: max_candidates ∈ {32, 64, 128, full-on-10%-subsample};
   confirm weight stability (document the standard result that sampling the
   denominator leaves the MLE consistent).
4. **Exclusion rule**: `exclusion="sector"` vs `"global"` vs none; count dropped
   repeats; effect on weights (expected: negligible here — 2–35 events — but pin
   it, since the rule is part of the production spec).
5. Sector deviations on/off; l2 grid {1e-4, 1e-3, 1e-2} for u.
6. Cross-regime: fit on A, evaluate on B (and vice versa) — how portable are
   choice weights across DGPs?

Success: A-world signs/magnitudes of [raised, emp, stage] within 2 SE across
seeds; ≥0.6 nats over random held-out; top-5 ≥ 0.40 (A) / 0.30 (B); calibrated
newcomer share (|pred − real| ≤ 2pp per quarter with the enriched ASC).

### E3 — Single-layer event-time block Hawkes vs the two-stage factorization

Fit the full firm-level intensity of `complete_account.tex` §blockhawkes on
day-resolution timestamps: `log λ_i(t) = a_s + β_s'X(t) − ρ_s R_i(t) +
A_{s,b} E_b(t)` with size-normalized sector field (concave MLE per the paper).

* On **B** (well-specified): recovery targets — self-inhibition ρ ordering
  rank-corr ≥ 0.9 (paper achieves 0.93), β corr, contagion magnitude order.
* On **A** (misspecified at day level): robustness read-out.
* **Head-to-head vs E1b+E2**: evaluate both on identical held-out events for
  (i) next-deal sector prediction, (ii) funded-firm ranking conditional on the
  event, (iii) joint NLL. Question to answer in writing: *given exact dates and
  truthful risk sets, what does the two-stage factorization still buy (modularity,
  newcomer handling, calibration) and what does it cost (joint-intensity
  coherence)?* This is the central design decision for production.

### E4 — PINO/neural operators: demoted to accelerator, benchmarked honestly

With no censoring, the analytical event-time MLE is exact and concave — PINO is
**not** needed for calibration. Keep two uses:

1. **Forward acceleration**: train `operators/pino.py::MultivariateMBPPOperator`
   at M=12 on subcritical instances (branching ≤ 0.5 per §0.5) to produce mean
   intensity paths for E5's walk-forward simulation and scenario stress (many
   repeated solves under alternative covariate paths). Protocol of ch. 12: full
   training + anchors, held-out rel-L2 target ≤ 2% (paper: 0.55–2% for M=3–8;
   tiny 600-epoch smoke here gave 8.2% — machinery confirmed). Report
   accuracy-vs-M and solve-time speedup vs the exact ODE solver.
2. **Inverse benchmark**: differentiable-solver inversion
   (`operators/covariate_inverse.py`) vs event-time MLE vs weekly GLM on the
   same data: table "method × (κ/A, θ, β) × (bias, SE, wall-clock)". Expected
   outcome (state it if confirmed): MLE wins at this scale; PINO inversion is
   only worth it for non-exponential kernels or very high M — document the
   crossover.

### E5 — Walk-forward backtest with covariate ablation

Freeze the best of E1b/E3 + E2. Weekly walk-forward over the last 20% of days:
predict sector intensities, rank candidates per event, emit top-k lists +
newcomer probability. Report: top-1/5/10 hit rates and MRR vs oracle
(true-parameter) and random; count calibration (PIT/coverage with quasi-Poisson
correction); lead-time (weeks a funded firm spends in the top decile before its
deal). **Covariate ablation** (assumption §0.4 check): rerun with macro
covariates removed and with `covariate_scale=1.5` data; report the metric deltas
so "covariates are fairly helpful" is a measured statement with a number.

## 4. Operational notes

* `PYTHONPATH=.` from repo root; scipy required. Shell may cap commands (~45 s)
  and kill background jobs between commands — stage long runs via pickles.
* `build_choice_sets`: 6.4 s; `fit_choice_fast`: ~12 s (padded 49k×65×6).
* Package loop fitters (`fit_sector_count_model`, `fit_startup_ranker`) take
  minutes; use fast_fit equivalents after a one-off equivalence check.
* `load_dataset` dedups with a 3-day same-company window; `dedup=False` to study
  duplicate sensitivity. Deal-date jitter moves ~4% of deals across week
  boundaries; 3% sector mislabels put some winners outside the labeled sector's
  pool — they land on the newcomer option by construction (count them).
* Feature panel (520, 7031, 5) float32 ≈ 73 MB; keep float32.

## 5. Baselines to reproduce (seed 0)

| Quantity | Value |
|---|---|
| A weekly stage-1: beta corr / G corr / rho_eff fit vs true | 0.675 / 0.27 / 0.355 vs 0.172 |
| A choice model: test NLL / random / top-5 | 3.346 / 4.011 / 0.446 |
| B choice model: test NLL / random / top-5 | 3.796 / 4.172 / 0.316 |
| Newcomer share (both) | ~0.27 |
| B weekly stage-1: beta corr / rho_eff / dispersion | 0.37 / 0.57 / 2.72 |
| PINO M=3 tiny smoke (600 ep) | rel-L2 0.082 |

## 6. Cautionary results already measured (keep in mind, they generalize)

1. **Risk-set truthfulness matters enormously** — v1 measured what happens when
   it fails: with a naive observed mask, dead/graduated firms contaminate risk
   sets and the ranker *inverts* (recency weight +1.94 vs true 0; NLL 3.886 vs
   3.558 oracle). We now assume truthful risk sets (§0.2), but if production risk
   sets are ever inferred rather than given, this failure mode is the first thing
   to re-check.
2. **Mechanism B**: raw-count log-link feedback tipped the generator itself
   supercritical at effective radius 0.35 (169k events). Radius guard + risk-set
   caps before any forward simulation.
3. **Rich-get-richer**: cumulative features (raised/employees) + positive weights
   concentrate deals absurdly unless winners exit (stage graduation). Watch for
   it in any simulation loop.
4. **Latent factor → spurious excitation**: persistent unobserved risk appetite
   inflates fitted contagion (0.355 vs 0.172 true). Covariates reduce, don't
   eliminate, this.

## 7. Upstreaming (after experiments)

* Vectorize the package fitters using `fast_fit.py` as reference (sector GLM is
  separable per receiving sector; ranker/choice model as padded tensors with
  optional sampled softmax). Keep APIs; add optimum-equivalence tests.
* Promote `build_choice_sets` (funded-pool + newcomer + exclusion rule) into the
  package as the production choice-set constructor, with the gap-feature form
  chosen in E2.1.
* Effective-spectral-radius guard + warning on `SectorCountResult`.
* Day-resolution event-stream loader for the eventtime module (currently the
  synthetic loaders produce it; the package should own it).
