# REVIEW.md — Deep-review briefing for the coding agent

Companion to `EXPERIMENTS.md` (v2), which owns the calibration campaign. This
file owns four additional workstreams: (N) the newcomer/cold-start design,
(B) two independent benchmarks the report must include, (M) a mathematical
review of the project's claims, and (R) the report expansion ("explain in
detail why our choices"). Work top to bottom; N and B feed R.

Environment: `PYTHONPATH=.` from repo root; `scipy` installed; for B1 run
`pip install xgboost scikit-learn --break-system-packages` (neither is present).
Shell may cap commands (~45 s) and kill background jobs — stage via pickles.

---

## N. The newcomer (cold-start) problem

Setting: covariates exist only for firms that already raised (features are
LOCF from past deals). The newcomer alternative in `build_choice_sets` therefore
has no firm covariates — currently a bare alternative-specific constant (ASC),
fitted ≈ 6.9 (A) / 4.7 (B) against ~27% newcomer share. Four designs to
implement and compare (same splits/metrics as E2 in EXPERIMENTS.md):

* **N1 — Context-covariate ASC** (default; matches the outside option
  `q_{0,s,t} = b0 + γ' g_{s,t}` already in `complete_account.tex`). Newcomer
  utility uses *market* covariates: sector deals last 90d, sector historical
  first-financing share, as-of macro (FFUND, CPI_YOY, RUNEMP), log pool size.
  Extend `build_choice_sets` to fill the newcomer row's feature slots with
  g_{s,t} (add columns; incumbents get zeros there, mirroring `is_newcomer`).
  Success: quarterly reliability plot of predicted vs realized newcomer share
  within ±2pp; incumbent weights unchanged vs bare-ASC fit.

* **N2 — Representative entrant**. Entrants ARE observed — at their first deal.
  From the train window only, estimate the at-first-funding feature
  distribution (μ_s, Σ_s) per sector (age at first raise, employees, etc. as
  measured *at that deal*). Two variants: (a) plug-in: newcomer feature row =
  μ_s (shares incumbent weights w — no separate ASC needed except a pool-size
  term); (b) integrated: newcomer utility = w'μ_s + ½ w'Σ_s w + c_s, the exact
  log-MGF under a Gaussian entrant prior — derive and verify the formula in M7.
  On synthetic data the true entrant pool is known (`ground_truth.npz` +
  `private_deal_map.csv`): decompose the fitted ASC into quality vs pool-size
  contributions and report how close (b) gets.

* **N3 — Nested two-step**. Step 1: binary GLM P(winner is newcomer | s, t) on
  g_{s,t}. Step 2: conditional logit among incumbents only (drop newcomer rows).
  Report whether incumbent weights shift vs N1 (they shouldn't, materially) and
  compare joint NLL: `NLL = NLL_binary + (1 - is_new) * NLL_incumbent`.

* **N4 — Immigrant/offspring split (Hawkes-native, preferred narrative)**.
  First financings are the *immigrant* stream of the sector process
  (covariate-driven background); repeats are the history-dependent stream. Split
  each sector's event stream into first-time vs repeat; fit the sector layer
  with two intensities (or one intensity plus thinning probability
  p_new(s,t) = logistic(γ'g_{s,t})); run the funded-pool ranker on the repeat
  channel only. This aligns the choice layer with Hawkes branching semantics —
  write it up as the structural justification of the outside option (feeds R).

Caveat to quantify (synthetic world can, real world can't): "newcomer" conflates
true entrants with untracked incumbents (~12% out-of-universe). Using the
private map, report the fraction of newcomer wins of each type and refit N1
with the two separated — the delta is the cost of the conflation.

## B. Benchmarks (both go in the report)

### B1 — XGBoost ranking benchmark (discriminative ceiling for stage 2)

Purpose: a strong model-free reference for the funded-firm ranking. If XGB
beats the conditional logit by a lot, we are missing features or
nonlinearities; if not, the parametric choice model is defensible — either way
the report gets a number.

Spec:
* Data: the same choice sets as E2 (`build_choice_sets`, same splits, same
  candidate sampling). One row per (event, candidate); label 1 for the winner.
* Two objectives, report both: `binary:logistic` with per-event softmax
  renormalization at eval, and `rank:pairwise` with `group` = event.
* Feature sets: (i) *parity* — exactly the 6 choice-model features (isolates
  functional form); (ii) *kitchen sink* — add raw PitchBook fields as-of the
  event (last deal size/valuation status, region one-hots, vc_round, employees,
  sector, macro as-ofs, pool size, days-since-last-funding raw). STRICT
  point-in-time: only fields from deals strictly before the event day —
  re-derive from `deals.csv`, do NOT use `companies.csv` last_* columns
  (they are end-of-sample snapshots = leakage).
* Newcomer handling: XGB scores incumbents; combine with the N1/N3 newcomer
  probability for a fair joint comparison (or evaluate incumbent-only events
  separately — state which).
* Metrics: top-1/5/10, MRR, and NLL of the renormalized scores; 5 seeds.
  Modest hyperparameter search only (depth {3,5,7}, eta {0.05,0.1}, 300–800
  trees, early stopping on a time-ordered validation tail). Log train time.
* Targets to beat (seed 0 baselines from E2): top-5 0.446 (A) / 0.316 (B).

### B2 — Non-homogeneous Poisson benchmark (null model for stage 1)

Purpose: does self-excitation buy anything beyond covariates? The assumption
"self-excitation is modest" must be *shown* against this null.

Spec:
* Sector level: λ_s(t) = exp(a_s + β_s' X(t)) with the as-of weekly covariates —
  i.e., E1a with excitation forced to zero (`fit_sector_glm_fast` with
  `n_lags=0` or excitation bounds (0,0)). Also the day-resolution version
  (piecewise-constant weekly covariates) for the event-time track.
* Compare vs: full GLM with excitation (E1a), event-time Hawkes (E1b), and a
  seasonal-naive baseline (per-sector historical mean by week-of-year).
* Report per regime: held-out Poisson NLL, dispersion, PIT calibration, and a
  likelihood-ratio-style improvement decomposition: covariates-only → +excitation
  → +event-time timing. The marginal gain of each block, with error bars over 5
  seeds, is the headline table for "why Hawkes at all".
* Expected (verify): in regime A the excitation gain is real but modest
  (true effective radius 0.15); in regime B most apparent gain over NHPP comes
  from short-horizon timing (7d contagion) visible only to the event-time model.

## M. Math review (deliverable: `MATH_NOTES.md` with derivations)

Verify, correct, or sharpen — against `paper/book/` and `complete_account.tex`:

1. **Factorization theorem (cornerstone of R)**: any marked point process
   factorizes as ground intensity × conditional mark distribution
   (Daley–Vere-Jones). Write the exact statement for our case: firm-level
   intensity λ_i(t) = Λ_{s(i)}(t) · P(i | s, t, H_t) ⇔ the two-stage model, and
   state precisely what is restricted by our parametrizations of each factor
   (this turns "two-stage" from heuristic into a lossless decomposition with
   named approximations).
2. **Concavity of the event-time block-Hawkes MLE** (paper claims a concave
   log-likelihood): check the proof covers the log-link with negative
   inhibition term and the size-normalized field; identify any conditions
   (boundedness of R_i, E_b) and edge cases at the constraint boundary.
3. **κ–θ ridge, quantitatively**: reproduce the Fisher-information argument
   (ch. 7) and extend it to day-resolution data — show the ridge collapse rate
   as bucket width → 0 (supports E1.2).
4. **Sampled-softmax consistency**: show the subsampled denominator (uniform
   sampling, winner kept) gives consistent estimates for conditional logit
   (McFadden's result on subsampled alternatives); note it does NOT hold for
   arbitrary non-uniform sampling without correction terms.
5. **MBPP correspondence under the log-link**: the package's weekly GLM uses
   exp(Σ b Y); the MBPP theory is linear/additive. Document the gap (Jensen
   term, Mechanism B metastability we measured: radius 0.35 tipped a 50k-event
   world to 169k) and derive a stability condition for the log-link recursion
   (beyond ρ(G)<1 — relate to Fokianos log-AR conditions).
6. **Cooldown forms**: reconcile the 26-week indicator (regime A truth), the
   exponential inhibition (regime B truth), and the fitted log-gap feature —
   derive the implied hazard-vs-gap curve for each and show what E2.1's binned
   fit should recover in both worlds.
7. **Newcomer log-MGF formula** (N2b): utility of an integrated-out entrant
   under Gaussian features = w'μ + ½w'Σw + log N_pool; derive, state when the
   pool-size term is identified (it isn't, separately from b0, unless N_pool
   varies over time — say so).
8. **Positive-only observation**: justify why the conditional-choice partial
   likelihood needs no "failed attempts" data and where it silently assumes
   independence between the event-timing process and the mark process —
   connect to the factorization in M1.

## R. Report expansion ("why our choices", requested by stakeholders)

Extend `complete_account.tex` with a designed-decisions chapter. Every claim
cites an experiment (EXPERIMENTS.md tracks) or a derivation (M). Skeleton:

1. *Data reality* — event-time-only updates, LOCF covariates, positive-only
   labels, partial universe. (Schema audit: this repo's synthetic tables.)
2. *Losslessness of the two stages* — M1 factorization theorem; the stages are
   the ground/mark decomposition, not an approximation; approximations enter
   only inside each stage (list them explicitly).
3. *Why sector Hawkes for the ground process* — positive contagion is sector
   level; firm-level effects are suppressive and cannot live in a nonnegative
   linear kernel; B2's improvement decomposition shows what excitation adds.
4. *Why a conditional risk-set model for the mark* — positive-only data (M8);
   truthful risk sets by assumption; funded-pool + exclude-last rule; gap
   feature as the inhibition carrier (M6, E2.1).
5. *The newcomer option* — N1–N4 with the immigrant/offspring narrative (N4)
   and the measured ASC decomposition (N2); conflation caveat.
6. *Identifiability-driven parametrization* — κ primary / θ needs timing (M3,
   E1.2); fixed decay banks; effective-radius reporting.
7. *Stability* — Mechanism A vs B (M5), the measured metastability episode,
   guards used in simulation.
8. *Benchmarks and falsifiability* — B1/B2 as the standing challenges: state
   the decision rules ("if XGB ≫ logit → nonlinearity/features missing; if
   NHPP ≈ Hawkes → drop excitation at this scale").
9. *Why analytical MLE first, PINO as accelerator* — E4 results, wall-clock
   tables, when the operator pays off.
10. *Limitations & production checklist* — risk-set truthfulness assumption
    (v1's measured inversion when it fails), point-in-time discipline, universe
    coverage, over-dispersion corrections.

## Repo deep-review checklist (do alongside)

* Leakage audit of every feature path (loaders, choice sets, B1 features):
  nothing may read data at or after the event timestamp; `companies.csv`
  last_* columns are end-of-sample — forbidden as features.
* Equivalence tests: `fast_fit` optima vs package fitters on subsamples (same
  losses to 1e-6); document any divergence.
* `synthetic/` code: check the exclude-last rule tie-breaking (same-day events
  break by deal_id order — is that stated and stable?), dedup window (3d)
  sensitivity, and that `build_choice_sets` state updates happen strictly
  after risk-set construction (no within-event leakage).
* Tests: add unit tests for `build_choice_sets` (a hand-built 3-firm example
  with known pools/labels), the newcomer variants, and the B2 null model.
* Reproducibility: every table in the report regenerable by one script under
  `experiments/` with a seed argument; runtimes logged.

## Done criteria

1. `MATH_NOTES.md` with M1–M8 (derivations, counterexamples where claims fail).
2. Benchmarks B1/B2 implemented, 5 seeds, results tables checked in.
3. Newcomer designs N1–N4 implemented and compared; one recommended with data.
4. Report chapter drafted in `paper/` (tex), every claim cross-referenced to an
   experiment or derivation.
5. Repo checklist items closed or filed as issues with reproduction snippets.
