# STRATEGY.md — The model escalation ladder for funding events

The deployment strategy in one sentence: **start from the status quo (XGBoost),
climb one rung of structure at a time, and let each rung buy its complexity by
beating the previous one on held-out metrics — stop climbing the moment a rung
stops paying.** Every rung below states its model, its math, why it is the next
step (not a later one), what it can and cannot capture, the gate it must pass,
and where it already exists in this repository.

Notation (matches `paper/complete_account.tex` and `MATH_NOTES.md`): firms $i$,
sectors $s$, macro/context covariates $X(t)$, firm features $Z_{i,t}$ (LOCF),
weekly counts $Y_{s,t}$, history $\mathcal H_t$. The marked-point-process
factorization $\lambda_i(t) = \Lambda_{s}(t)\,P(i \mid s,t,\mathcal H_t)$ (M1)
holds at every rung; the rungs differ only in how much structure each factor
gets.

---

## Rung 0 — Status quo: XGBoost on the data

**What it is.** Gradient boosting on event/candidate rows (and, if used for
timing, on firm-week rows): the pragmatic incumbent. In our benchmark protocol
(`exp26`) it is trained on exactly the choice sets the structural models see,
with strictly point-in-time features.

**What it is genuinely good at.** Nonlinearity and interactions with zero
modeling effort. This is measured, not conceded: at *feature parity* XGB beats
the conditional logit by +0.22 nats / +9pp incumbent top-5 on world A (+0.10
nats on B) — there is real nonlinearity in even our six clean features.

**Why it cannot be the destination.**
- No intensity: it scores *who* conditional on an event but has no calibrated
  *when/how many* — no counts distribution, no PIT, no simulation.
- No coherent joint model: probabilities do not compose into
  $p(\text{time},\text{sector},\text{firm})$, so scenario analysis and
  multi-horizon forecasting are out of reach.
- Structurally silent: no stage logic, no inhibition semantics, no
  immigrant/entrant channel; leakage discipline lives entirely in the feature
  pipeline, invisibly.

**Role going forward.** Permanent challenger, never deleted: every rung below
reports against the XGB reference on identical splits (`exp26` protocol), and
the parity gap is the standing measure of how much nonlinearity the structural
rung leaves on the table.

---

## Rung 1 — Calibrated Poisson intensity on covariates (NHPP)

**Model.** The first *calibrated* model: intensity as a function of covariates
only,

$$\Lambda_s(t) = \exp\big(a_s + \beta_s^\top X(t)\big),$$

fit at sector level (weekly: `fit_sector_glm_fast(n_lags=0)`; day-resolution:
piecewise-constant covariates in the event-time MLE). Poisson likelihood,
concave, closed diagnostics (PIT, quasi-Poisson dispersion).

**Why this is step one.** It converts the problem from scoring to *rates*: the
first object that predicts "how many deals, where, next week" with calibrated
uncertainty. And it is empirically the workhorse: in the decomposition
benchmark (`exp27`) covariates carry almost the entire held-out gain over a
seasonal baseline (+3.5 / +5.7 / +1.5 nats across the three synthetic worlds),
while everything added later is decimal-point territory. Interpretable
$\beta$'s connect deal flow to the macro cycle.

**What it cannot capture.** Any history dependence: stage progression,
cooldown, clustering beyond what covariates explain. Overdispersion shows up
immediately where history matters (dispersion 4.8 on the misspecified world).

**Gate to pass (vs rung 0).** Calibrated counts XGB cannot produce at all, plus
sane residual diagnostics; report dispersion and PIT next to every NLL.
**Gate to escalate:** residual autocorrelation / overdispersion in the PIT
diagnostics — evidence that history is missing.

**Repo status.** Done: `fit_sector_glm_fast(n_lags=0)`, `exp27` (NHPP arm),
day-resolution variant via `fit_multivariate_with_covariates`.

---

## Rung 2 — Multi-state model with stage transitions

**Model.** Funding is not an undifferentiated event stream: each round moves a
firm along the stage ladder (Seed → A → B → C → D → E → …, with exits to
IPO/M&A/failure as absorbing states). Model each *transition* with its own
intensity on a constrained state graph:

$$\lambda_{k\to j, i}(t) = \mathbf 1\{j \succeq k\}\cdot
\exp\big(\alpha_{k\to j} + \beta_{k\to j}^\top X(t) + \delta_{k\to j}^\top Z_{i,t}\big),$$

with the admissibility rule $j \succeq k$: from state $k$ the next round may be
$k$ again (extension/bridge — a second C is allowed) or any forward stage
including skips (C → E without D), but **never backward** (from C, no A or B).
The state space is a DAG plus self-loops; each firm's history is a monotone
path on it.

**Why this rung, and why before any excitation.**
- The constraints are free information: backward transitions carry exactly
  zero probability mass by construction. XGB and flat Poisson both waste
  capacity (and can leak absurd predictions) learning what the DAG encodes
  for free.
- Stage is measurably the strongest single firm covariate we have seen: it was
  a top predictor in the exit-hazard analysis and drives the entrant-profile
  distributions; PitchBook carries it natively (`last_financing_deal_type`).
- It fixes two structural problems at once: **rich-get-richer** (late-stage
  firms exit the ladder to absorbing states instead of accumulating
  probability forever — the failure mode we measured when winners don't exit)
  and **risk-set precision** (a Series-C round chooses among C-eligible firms,
  not the whole sector — smaller, more honest pools).
- Estimation decomposes exactly: with transition-specific parameters, the
  likelihood factorizes into independent Poisson/Cox fits per transition, each
  concave, each with its own at-risk indicator. No new estimation theory
  needed.

**What it cannot capture.** Cross-firm interaction: sector heat, contagion,
crowding-out. Each firm still evolves independently given covariates.

**Gate.** Held-out joint NLL vs rung 1 (it should win on the *which-firm* and
*which-stage* components); transition-specific calibration (predicted vs
realized stage mix by quarter); the C→A cell of the empirical transition
matrix must be exactly zero in-model and near-zero in-data (data-quality
check for stage mislabels — we ship with 3% sector mislabels, expect stage
noise too).

**Repo status.** *The one genuinely new build on this ladder.* Ingredients
exist (stage_num feature, per-stage entrant profiles in `exp28`, hazard
machinery in `sector_hazard.py`); the transition-indexed fitter and the
stage-aware risk-set builder are new modules. Everything downstream (rungs
3–5) composes with it by adding covariates or streams, not by rewriting it.

---

## Rung 3 — History as covariates: "Hawkes without Hawkes"

**The distinction this rung makes explicit: we do not need Hawkes per se.**
The two interaction effects that motivate Hawkes — sector momentum and
self-inhibition — can enter the rung-2 intensities as *covariates*:

$$\lambda_{k\to j,i}(t) = \mathbf 1\{j\succeq k\}\exp\Big(
\alpha_{k\to j} + \beta^\top X(t)
\;+\; \phi\,\underbrace{H_{s(i)}(t)}_{\text{same-sector activity}}
\;-\; \psi\,\underbrace{g(t - T_i^{\text{last}})}_{\text{self-inhibition}}\Big),$$

where $H_s(t)$ is a fixed-decay summary of recent same-sector deals (e.g.
log-count over the last 90 days, or an EWMA with a chosen half-life) and
$g(\cdot)$ is a monotone gap transform (our $\log(1+\text{days}/30)$, or
binned dummies to let the data draw the curve — M6).

**Why this is the honest default rather than a compromise.**
- Mathematically, this *is* a Hawkes-type model with the kernel timescale
  **fixed** and only its coefficient estimated. The difference from full
  Hawkes is exactly the part that is fragile: jointly estimating the decay.
- Our identifiability results say fixing the timescale is what the data can
  support at realistic volumes: the κ–θ ridge makes the decay unidentifiable
  from weekly buckets and noisy even at day resolution without pooled volume
  (M3, `exp24`); latent common factors inflate any freely-estimated
  excitation (fitted radius 2.4× truth, `campaign_regimeA`). A coefficient on
  a fixed-decay covariate absorbs the same predictive signal without
  pretending to identify the kernel.
- Sign freedom for free: a covariate coefficient can be negative
  (self-inhibition) inside a concave GLM — the thing a nonnegative Hawkes
  kernel structurally cannot do (the original reason our architecture split
  the two stages).
- Measured: in the model ladder (`exp25`) the feature-based scorers with a
  gap covariate matched the oracle, while the structural Hawkes-recency
  rankers gave up ~30% of the lift; in the decomposition (`exp27`) weekly
  structural excitation added +0.04 nats on A and was *harmful* on A-strong.

**Practical guidance.** Fit $H_s$ at 2–3 fixed half-lives (e.g. 30/90/180d)
and let the coefficients choose; use binned gap dummies once (E2.1 protocol)
to check the inhibition shape, then freeze the transform.

**Gate to escalate to rung 4.** Only if, *after* these history covariates,
residual clustering persists (PIT autocorrelation, dispersion > ~1.5, or
systematic underprediction in hot weeks) — that residual is the only thing a
structural kernel can add.

**Repo status.** Done in pieces: gap/cooldown covariates throughout
(`sector_survival`, `sector_hazard`, choice sets), sector-heat context
features (`build_choice_sets(newcomer_context=True)`, N1), fixed-decay recency
fields (`models/hawkes_ranker._fields_for`). Composing them into the rung-2
transition intensities is configuration, not new code.

---

## Rung 4 — Full Hawkes

**Model.** Structural self/cross-excitation with jointly estimated branching
and decay: weekly stability-constrained sector GLM
(`fit_sector_count_model`, row-sum bound ⇒ spectral radius < 1), event-time
exponential-kernel MLE at day resolution (`eventtime/`), block Hawkes with
self-inhibition for the single-layer firm-level alternative
(`models/event_block_hawkes`), MBPP for counts-only calibration (`mbpp/`).

**When it earns its place — four concrete triggers.**
1. **Rung-3 gate fired**: residual clustering that fixed-decay covariates
   cannot absorb.
2. **Simulation with endogenous feedback**: scenario stress where a hot week
   must *propagate* (covariate models cannot amplify; a branching process
   can). This is where the stability guards and the PINO forward surrogate
   pay off (3ms/solve for many repeated scenario solves).
3. **Timing as a deliverable**: if the contagion half-life itself is a
   quantity stakeholders want, only day-resolution event-time Hawkes with
   sufficient volume identifies it (bucket ≤ one half-life; ≥4 half-lives
   destroys it — `exp24`).
4. **Ordering questions at the firm level**: which firms are most suppressed,
   which sectors most self-exciting — orderings survive even under
   misspecification (self-inhibition rank corr 0.937, `exp23`) where
   individual kernel entries do not.

**Standing caveats (pre-registered).** The fitted radius is an upper bound
(latent factors, S1); cross-sector kernel entries are not structurally
interpretable; never simulate a log-link count model without the caps
(Mechanism B — the measured 169k-event metastability episode).

**Repo status.** Done: all fitters, stability machinery, MBPP theory, PINO
acceleration, and the honest expected outcome from rehearsal — modest
incremental NLL over rung 3, with the real value in simulation coherence and
timing.

---

## Rung 5 — Firms outside the dataset: the two-step, the whole nine yards

The universe is incomplete in two distinct ways: **true entrants** (first
raise — no covariates exist before it, by construction) and **untracked
incumbents** (~12% of deals in our synthetic calibration of the problem).
Both land outside the risk set. The architecture:

**Step 1 — new firm or risk-set firm?** A binary model driven by *global*
covariates only (nothing firm-level exists for the "new" branch):

$$P(\text{new} \mid s, t) = \sigma\big(b_0 + \gamma^\top g_{s,t}\big),
\qquad g_{s,t} = \big[\text{sector deals last 90d},\;
\text{hist. first-financing share},\; \text{macro as-ofs},\;
\log \text{pool size}\big].$$

**Step 2 — conditional on the branch.**
- *Risk-set firm*: the rung 2–4 conditional-choice model over the stage-aware
  pool (softmax over eligible incumbents).
- *New firm*: optionally a profile model of the entrant (which stage/sector
  mix; the representative-entrant distribution $(\mu_s, \Sigma_s)$ of features
  at first funding, estimable from the train window — the exact integrated
  utility is the Gaussian log-MGF $w^\top\mu_s + \tfrac12 w^\top\Sigma_s w$,
  M7).

The likelihood factorizes exactly — no approximation:

$$\mathrm{NLL} \;=\; \mathrm{NLL}_{\text{binary}}
\;+\; (1-\mathbf 1_{\text{new}})\cdot \mathrm{NLL}_{\text{choice}}
\;\big(+\; \mathbf 1_{\text{new}}\cdot \mathrm{NLL}_{\text{profile}}\big).$$

**Why this design, with measurements.**
- The two-step (N3) won the newcomer comparison on joint likelihood on both
  worlds (`exp28`), and its incumbent weights provably match the one-shot
  fit — the split costs nothing on the incumbent side.
- Global-covariate step 1 (the N1 context set) is what makes the newcomer
  *share* calibratable over time (mean quarterly error ~3–4pp vs 6–9pp for a
  bare constant) — the operationally important property, since a drifting
  newcomer share with constant context is the diagnostic that *coverage*
  changed, not the market (S4).
- Entrant quality explains only 13–15% of the newcomer utility; most of it is
  coverage and pool size. So invest step-1 effort in context covariates, not
  in richer entrant profiles.
- The entrant-vs-untracked conflation is nearly free in likelihood (+0.003
  nats when separated with oracle labels) but permanently blocks
  compositional claims — say "newcomer probability," never "entrant
  probability," on real data.
- Structural reading (N4): the "new firm" stream is the **immigrant channel**
  of the branching process — covariate-driven background intensity — and the
  risk-set stream is the offspring channel. The two-step is not a hack; it is
  the immigrant/offspring split seen from the mark factor. (With the measured
  irony that entry waves absorb the latent factor: the immigrant stream
  carried the larger fitted "excitation" radius, 0.30 vs 0.08–0.11.)
- Identifiability honesty (M7): the pool-size term in the newcomer utility is
  not separately identified from the intercept unless pool size varies over
  time — don't over-interpret $b_0$.

**Repo status.** Done: `build_choice_sets(newcomer_context=True)`, N1–N4 in
`exp28`, unit tests pinning pool rules and context placement
(`tests/test_choice_sets.py`).

---

## The ladder at a glance

| rung | model | key object | must beat | main risk | repo |
|---|---|---|---|---|---|
| 0 | XGBoost | scores | — (incumbent) | no intensity, no joint model | `exp26` |
| 1 | Poisson on covariates | calibrated rates | rung 0 on calibration (new capability) | history-blind | `exp27`, `fit_sector_glm_fast(0)` |
| 2 | multi-state transitions | stage-DAG intensities | rung 1 joint NLL + stage calibration | stage-label noise | **new build** |
| 3 | + history covariates | sector heat $H_s$, gap $g$ | rung 2 NLL; PIT clean-up | fixed decay misses true timescale | compose existing |
| 4 | full Hawkes | branching + decay, simulation | rung 3 residual clustering | ridge, latent-factor inflation, Mechanism B | `sector_stability`, `eventtime`, `mbpp` |
| 5 | two-step outside-universe | $P(\text{new}\mid g_{s,t})$ + choice | rung 2–4 without it (joint NLL, share calibration) | conflation, coverage drift | `exp28`, loaders |

Rung 5 is orthogonal — it attaches to whichever of rungs 2–4 is deployed.

**Standing evaluation discipline at every rung** (unchanged from the
campaign): temporal splits only; strict point-in-time features
(`companies.csv` `last_*` columns forbidden); XGB challenger re-run on
identical data; NLL with dispersion + PIT for anything that claims to be an
intensity; top-k/MRR with mid-rank ties for anything that ranks; structural
claims reported as bounds with the identifiability caveat attached (S1–S7 in
`paper/design_decisions.tex`).

**Where we expect the ladder to stop, honestly.** On the synthetic rehearsals
the marginal value concentrated in rungs 1–3: covariates dominate, stage
structure and history covariates capture most history dependence, and full
Hawkes adds decimal points of NLL — its real justification is simulation and
timing, not one-step prediction. If real data reproduces that pattern, the
production configuration is rungs 1–3 + 5, with rung 4 reserved for scenario
analysis and the XGB challenger keeping everyone honest about nonlinearity.
