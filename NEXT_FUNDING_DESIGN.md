# Predicting the Next Funded Firm — Design & Build Plan

**Status:** source-of-truth design document. Engineering agents should build from this
file and iterate. It is intended to be read end-to-end once, then used section by
section (the *Code structure* and *Milestones* sections are the build backlog).

**One-line problem.** From a stream of investment-announcement events (with
covariates), rank the firms most likely to be funded *next* (or within a short
horizon), so we can build relationships early and negotiate better terms.

**Headline recommendation.** Model the announcement stream as a **marked,
covariate-modulated multivariate Hawkes process** and read off a **ranking** of
firms from the per-firm intensity / horizon probability. Validate it against a
**survival-with-ranking** alternative (DeepHit-style) and **neural-TPP** baselines.
Reuse the existing `hawkes_calibration` package (MBPP, covariate excitation,
multivariate LTV solver, goodness-of-fit, Bayesian uncertainty) as the spine.

---

## 1. Problem understanding

### 1.1 Business framing
We want a *leading indicator*: a ranked watch-list of firms about to raise. The value
is in acting before the round closes (warm intros, better terms, allocation). So we
care about **ranking quality at the top of the list** and **lead time**, not about
point-estimating valuations. Concretely the product is: *"given everything known at
time `t`, output the top-`k` firms most likely to announce funding in the next
`h` weeks, with a calibrated probability and an expected time."*

### 1.2 Why this is a temporal, self-exciting problem
Funding is not i.i.d. across firms or time:
- **Self/mutual excitation.** A marquee round advertises a sector and pulls in
  capital; deals beget deals in *similar* firms (positive cross-excitation). Investor
  attention has momentum — the strongest known predictor of "next round within 12
  months" in the empirical ML literature is *financing recency / maturity*
  (see §2.6). That is exactly a self-exciting kernel.
- **Refractoriness.** A firm that *just* closed a round is briefly unlikely to raise
  again (self-inhibition / a cooldown).
- **Covariate modulation.** Popularity scores, sector, stage, macro regime, prior
  investor quality all shift the propensity to transact.
- **Censoring/aggregation.** Sometimes we only have aggregate counts (deals per
  sector per week) rather than exact, attributable event times.

These are precisely the phenomena Hawkes processes encode, and precisely the reasons
a static classifier ("will this firm raise in 12 months?") leaves information on the
table.

### 1.3 Formal statement
Observed history up to decision time `t`:
```
H(t) = { (t_k, m_k, x_k) : t_k < t },
```
where `t_k` is the announcement time, `m_k ∈ {1,…,N}` the **mark** (the firm, or a
(firm, round-type) pair), and `x_k` event/firm covariates. Firm-level, possibly
time-varying covariates `X_i(t)` (sector, stage, popularity, headcount, web traffic,
macro) are available for all firms `i`, including ones with no past events
(cold-start).

**Targets** (any/all of):
- **Next-firm identity:** the distribution over which firm `m` is the *next* to be
  funded; deliver a ranked list.
- **Horizon hit:** for each firm `i`, `P(firm i is funded in (t, t+h] | H(t))`.
- **Time-to-event:** expected/quantile time of firm `i`'s next round.

**Primary metric:** top-`k` ranking quality of the horizon-hit list (Recall@k,
MRR, NDCG) plus calibration; secondary: time-to-event concordance and likelihood.

### 1.4 Scope and non-goals
In scope: ranking firms by imminent-funding propensity, with covariates, at
event-time *or* interval-censored resolution. Out of scope (for v1): valuation
prediction, success/exit prediction (different label), causal effect of an
intervention.

---

## 2. Literature review

The problem sits at the intersection of four literatures. We summarize each and what
we take from it. (Full citations with links in §8.)

### 2.1 Classical Hawkes / self-exciting processes
The self-exciting process (Hawkes 1971) and its cluster/branching representation
(Hawkes & Oakes 1974) are the canonical models for event streams where occurrences
raise the short-term rate of further occurrences. Multivariate Hawkes processes model
*mutual* excitation through a kernel matrix — the natural object for "deals in firm
`j` excite deals in firm `i`." We take: the intensity form, the branching-ratio
stability condition, and the time-rescaling goodness-of-fit theorem.

### 2.2 Interval-censored Hawkes / the MBPP (our base)
Rizoiu et al. (2022) solve the problem of fitting Hawkes processes when only
**aggregate interval counts** are observed, via the **Mean Behavior Poisson Process**:
the deterministic mean intensity `ξ = E[λ]` solves a Volterra equation and, by
Watanabe's theorem, defines a genuine Poisson process whose interval likelihood is
tractable (a Kullback–Leibler / Bregman objective). Our package already implements
this, including **covariate-modulated excitation** (the multivariate LTV solver and
`fit_mbpp_ic_excitation_multi`) and the proof that the covariate augmentation
preserves the construction (see the companion `paper/investment_case_study.pdf`).
We take: the entire interval-censored pathway and the covariate machinery.

### 2.3 Neural temporal point processes (TPPs)
When the intensity's functional form is unknown, neural TPPs learn it from data:
- **RMTPP** (Du et al., KDD 2016) — an RNN embeds event history and jointly models the
  next time and mark.
- **Neural Hawkes Process** (Mei & Eisner, NeurIPS 2017) — a continuous-time LSTM
  gives a neurally self-modulating multivariate intensity (allows inhibition).
- **Transformer / Self-Attentive Hawkes** (Zuo et al., ICML 2020; Zhang et al. 2020) —
  self-attention for long-range dependencies, efficient and accurate.
- **Intensity-free** (Shchur et al., ICLR 2020) — model the inter-event *distribution*
  directly (normalizing flows / mixtures), sidestepping intractable intensity
  integrals.
- **Benchmarking** — EasyTPP (Xue et al., ICLR 2024) provides standardized datasets,
  baselines and metrics; recent variants (Transformer/Mamba Hawkes, 2024) push
  long-range modeling.
We take: strong, flexible **baselines** (RMTPP, Neural Hawkes, THP) and the EasyTPP
harness for fair comparison. The trade-off: accuracy vs interpretability — neural TPPs
do not give us an interpretable branching matrix or covariate coefficients.

### 2.4 Relational / graph TPPs — "who interacts with whom next"
This is the closest analogue to "which firm is funded next":
- **Know-Evolve** (Trivedi et al., ICML 2017) — models a multi-relational dynamic
  graph as a multivariate TPP whose intensity is modulated by a score over learned
  **entity embeddings**.
- **DyRep** (Trivedi et al., ICLR 2019) — a two-time-scale TPP coupling network
  *topology* and *activity*, with temporally-attentive node representations.
We take: the **embedding-structured intensity** idea — with thousands of sparse
firms, parametrize the `N×N` excitation through low-rank firm embeddings (similar
firms excite each other) rather than a dense matrix. This is the key to scaling
multivariate Hawkes to many firms.

### 2.5 Survival analysis & competing risks (the alternative the user named)
"Time until firm `i` is next funded" is a time-to-event problem with **right
censoring** (firms that have not raised yet) and **competing risks** (acquired/dead
firms leave the risk set):
- **Cox proportional hazards** (Cox 1972) and deep variants (**DeepSurv**, Katzman et
  al. 2018) — covariate-driven hazards.
- **DeepHit** (Lee et al., AAAI 2018) — discrete-time, distribution-free, with a
  **ranking loss** that directly optimizes the concordance (ordering) we care about,
  natively handling competing risks; **Dynamic-DeepHit** (Lee et al. 2019) adds
  longitudinal covariates.
- Evaluation: the **time-dependent concordance index** (Antolini et al. 2005),
  integrated Brier score / IPA.
Relationship to Hawkes (made precise in §4.6): a per-firm **cause-specific hazard**
*is* a per-firm conditional intensity. The difference is that standard survival models
assume hazards depend on covariates but **not on the event history of other firms**;
Hawkes adds exactly that mutual excitation. So survival-with-ranking is a strong,
metric-aligned **discriminative baseline**, while Hawkes is the more faithful
**generative** model that also yields self/mutual-excitation structure.

### 2.6 Domain: predicting startup / VC funding with ML
Crunchbase-based studies predict funding, valuation, survival and exits with ML:
CapitalVX (Ross et al. 2021) predicts exits/follow-on funding at 80–89% accuracy;
startup-success and portfolio-simulation work on Crunchbase (2023); interpretable
models for funding/patenting/exits (2025). The consistent empirical finding —
**maturity and financing recency dominate "next round within 12 months"**, reflecting
*momentum in investor attention* — is direct evidence for a self-exciting model and
tells us which covariates matter (time-since-last-round, age, cumulative financing).
We take: feature inspiration and a sanity benchmark (a gradient-boosted "raise in
`h`?" classifier is a baseline we must beat on ranking + lead time).

### 2.7 Synthesis
No existing line does all of: history-dependent **mutual excitation**, **covariates**,
**interval-censoring tolerance**, **scalability to many firms**, and a **ranking**
read-out with calibration. The marked covariate Hawkes with embedding-structured
excitation (below) combines the Hawkes generative fidelity (§2.1–2.2), the
embedding scalability of relational TPPs (§2.4), and a ranking read-out evaluated with
survival metrics (§2.5), with neural TPPs and survival models as the baselines it must
beat.

---

## 3. Methodology — critical discussion and decision

### 3.1 Candidate approaches
| approach | models mutual excitation? | covariates | interpretable | handles counts-only | directly optimizes ranking | scales to N firms |
|---|---|---|---|---|---|---|
| Popularity/recency heuristic | implicitly | no | yes | yes | no | yes |
| Static classifier ("raise in h?") | no | yes | medium | n/a | no (per-firm prob only) | yes |
| Cox / DeepSurv | no | yes | medium | no | partly | yes |
| **DeepHit / Dynamic-DeepHit** | no (covariate-only) | yes | low | no | **yes (ranking loss)** | medium |
| Neural TPP (RMTPP/NHP/THP) | yes (latent) | partial | low | no | via likelihood | medium |
| **Marked covariate Hawkes (ours)** | **yes (explicit)** | **yes** | **yes** | **yes (via MBPP)** | via intensity + ranking read-out | yes (low-rank embeddings) |

### 3.2 Why Hawkes first (the decision)
1. **It matches the data-generating story.** Excitation, refractoriness, covariate
   modulation, sector contagion — all are first-class in a Hawkes intensity, and the
   dominant empirical predictor (financing recency) *is* a self-exciting kernel.
2. **It degrades gracefully to the data we actually have.** With exact attributable
   event times → event-time MLE. With only aggregate counts → the MBPP interval
   likelihood (already implemented and proven sound under covariates).
3. **It is interpretable and auditable.** We get a branching matrix (who excites
   whom), covariate coefficients with signs, and a stability diagnostic — essential
   for a tool whose recommendations humans must trust and act on.
4. **It reuses a validated codebase.** The covariate-excitation MBPP, multivariate
   LTV solver, goodness-of-fit (time-rescaling) and Bayesian uncertainty already
   exist and are tested.
5. **Ranking is a thin read-out on top**, not a separate model: rank by the horizon
   probability `1 − exp(−∫ λ_i)`.

We nonetheless **build the survival-with-ranking model and neural-TPP models as
baselines**, because (a) DeepHit directly optimizes the ranking metric and may win on
pure top-`k`, and (b) neural TPPs bound how much interpretable structure costs us in
accuracy. The decision to ship Hawkes is contingent on it being competitive on
ranking while winning on interpretability, lead time, and counts-only robustness.

### 3.3 Honest risks of the Hawkes choice (carried from the companion paper)
- **Weak identifiability of the decay `θ`** and of the full `N×N` excitation structure
  from coarse data — mitigated by low-rank embeddings, priors, and reporting the
  branching ratio (well identified) rather than every entry.
- **Linearity requirement.** True endogenous self-inhibition is *nonlinear* and breaks
  the MBPP mean-field identity (proven in the paper). We encode refractoriness the
  *lawful* way — as an exogenous "cooldown" covariate with a negative coefficient —
  and treat genuine nonlinear inhibition as a known approximation, or escalate to a
  neural TPP if it dominates.
- **Cold-start firms** have no event history; the baseline covariate term `μ_i(t)`
  and firm embeddings-from-features carry them (see §4.5).

---

## 4. Proposed methodology — the math

### 4.1 Marked multivariate Hawkes intensity
For firm `i ∈ {1,…,N}`, the conditional intensity given history `H(t)`:
```
λ_i(t) = μ_i(t)  +  Σ_{k : t_k < t}  A_{i, m_k}(t) · g(t − t_k),
```
- **Baseline (immigration), covariate-modulated, log-linear:**
  ```
  μ_i(t) = exp( β_0 + β^T X_i(t) ),
  ```
  keeps `μ_i ≥ 0` and carries cold-start firms (depends only on features `X_i`).
- **Excitation strength, structured + covariate-modulated:**
  ```
  A_{i,j}(t) = a_{ij} · exp( η^T Z_{ij}(t) ),     a_{ij} ≥ 0,
  ```
  where `Z_{ij}(t)` are covariates of the *receiving* time / triggering event
  (popularity, deal size, same-sector indicator) and `a_{ij}` is the base
  firm-to-firm excitation.
- **Kernel `g`:** exponential `θ e^{−θτ}` (fast, closed-form, the workhorse),
  optionally sum-of-exponentials (multi-timescale) or power-law (heavy tail) — all
  supported by the package.

This is exactly the covariate-augmented model proven sound in the companion paper
(baseline covariates preserve the construction exactly; excitation covariates replace
the convolution by an exactly solvable linear time-varying ODE).

### 4.2 Scaling: embedding-structured excitation (the key design choice)
A dense `N×N` matrix `a_{ij}` is infeasible and unidentifiable for large `N`. Following
the relational-TPP idea (Know-Evolve / DyRep), factor it through **firm embeddings**:
```
a_{ij} = softplus( u_i^T W u_j + b_sector·1[sec(i)=sec(j)] ),
```
with `u_i ∈ R^d` either learned or initialized from firm features (so a cold-start
firm gets an embedding from its covariates). This encodes "similar firms excite each
other," reduces parameters from `O(N^2)` to `O(Nd + d^2)`, and is differentiable.

### 4.3 Likelihood and estimation
**Event-time data** (exact, attributable timestamps). Log-likelihood on `[0,T]`:
```
ℓ(Θ) = Σ_k log λ_{m_k}(t_k)  −  Σ_{i=1}^N ∫_0^T λ_i(s) ds,
```
with parameters `Θ = (β, η, embeddings u, W, kernel params)`. The compensator integral
has the usual closed form for the exponential kernel; the per-pair `O(N_events)`
recursion (already in the package's event-time module) keeps it linear in events.
Optimize by L-BFGS (few params) or Adam/SGD (embeddings); regularize with L2 on
embeddings and optional L1 on `a_{ij}` for sparsity. Standard errors from the observed
information; full posterior via the package's Bayesian module for the low-dimensional
parameters.

**Interval-censored data** (aggregate counts per window). Use the MBPP: the expected
per-firm intensity `ξ_i = E[λ_i]` solves the multivariate covariate Volterra equation,
integrated by `solve_mbpp_ltv`; fit by the interval-censored log-likelihood
```
L_IC(Θ) = Σ_i Σ_w [ Ξ_{i,w}(Θ) − C_{i,w} log Ξ_{i,w}(Θ) ],     Ξ_{i,w} = ∫_{window w} ξ_i,
```
which is `fit_mbpp_ic_excitation_multi` generalized with the embedding structure.

### 4.4 The ranking read-out (how "who's next" comes out)
Two equivalent facts about marked TPPs give the read-out:
- **Next-mark probability.** Conditional on the next event occurring at time `τ`, the
  probability it is firm `i` is `λ_i(τ) / Σ_j λ_j(τ)`.
- **Horizon hit probability.** The probability that firm `i` has ≥1 event in `(t, t+h]`
  is
  ```
  p_i(t, h) = 1 − exp( −∫_t^{t+h} λ_i(u) du ) = 1 − exp( −(Ξ_i(t+h) − Ξ_i(t)) ).
  ```
**Primary read-out:** rank firms by `p_i(t,h)` for the business horizon `h` (e.g. 4–12
weeks). This is monotone in the compensator increment, which the package computes
directly. We also expose the expected time to firm `i`'s next event and a calibrated
probability (Bayesian posterior predictive for uncertainty bands).

### 4.5 Cold-start and time-varying covariates
- **Cold-start firm** (no events): `λ_i ≈ μ_i(t) = exp(β_0 + β^T X_i(t))` plus
  cross-excitation from similar firms via `a_{ij}` built from its feature embedding —
  so a brand-new firm in a hot sector still ranks.
- **Time-varying covariates** enter piecewise-constantly (regime/popularity updates),
  which keeps the MBPP compensator closed-form per segment (proven in the paper).

### 4.6 The survival / competing-risks alternative (precise relationship)
Define, for firm `i`, the cause-specific hazard of "next funding" `h_i(t | X)`. Then:
- A survival model assumes `h_i(t|X) = h_0(t) exp(γ^T X_i(t))` (Cox) — covariates only,
  **no dependence on other firms' events**.
- The Hawkes intensity is `λ_i(t) = μ_i(t) + (history-excitation)` — the *same* object
  plus the mutual-excitation term. So **Hawkes ⊃ Cox-type hazards**; setting the
  excitation to zero recovers an inhomogeneous-Poisson / proportional-hazards model.
- **DeepHit** discretizes time and learns the joint distribution of (event time, cause)
  with a **ranking loss** on the concordance — it optimizes our top-`k` metric
  directly but discards the generative excitation structure and the counts-only path.

We therefore implement the survival models as **baselines and as a ranking-loss
fine-tuning target**: optionally fine-tune the Hawkes horizon scores with a listwise
ranking loss (NDCG surrogate) to combine generative fidelity with metric alignment.

---

## 5. Testing & validation plan

### 5.1 Synthetic recovery (truth known) — reuse and extend `experiments/exp13_grid.py`
Generate a marked covariate Hawkes with known baseline `β`, excitation embeddings,
covariate coefficients `η`, kernel. Verify:
- parameter recovery (β, η, branching ratio / spectral radius, embedding similarity
  structure) vs truth;
- **ranking recovery**: Recall@k / MRR / NDCG of the predicted next-firm list vs the
  simulator's realized next firm, and vs the *oracle* intensity ranking;
- **calibration**: time-rescaling KS test against Exp(1); horizon-probability
  reliability (PIT) diagram; Pearson dispersion.

### 5.2 Ablations (the factorial grid, extended to marks)
Sweep, reusing the existing grid machinery:
- **covariate importance** ∈ {none, small, large} — does modelling covariates improve
  ranking + held-out likelihood?
- **kernel mis-specification** ∈ {exp, sum-of-exp (3 timescales), mild power-law} —
  graceful vs sharp degradation (already characterized in the paper).
- **resolution** ∈ {event-time, interval-censored} — quantify the cost of aggregation.
- **embedding rank** `d` and **sparsity** penalty — recovery vs N.
- **self-inhibition**: exogenous cooldown (lawful) vs endogenous nonlinear (mis-spec).

### 5.3 Baselines (must implement and beat / contextualize)
1. **Recency/popularity** rank (time-since-last-round, sector momentum) — the empirical
   strong baseline from §2.6.
2. **Per-firm inhomogeneous Poisson** (no excitation) — isolates the value of
   excitation.
3. **Cox / DeepSurv** (covariate hazards, no excitation).
4. **DeepHit / Dynamic-DeepHit** (ranking-loss survival; competing risks).
5. **Neural TPPs** via **EasyTPP**: RMTPP, Neural Hawkes, Transformer Hawkes.
6. **Ours**: marked covariate Hawkes (event-time) and MBPP (interval-censored).

### 5.4 Backtesting protocol (real-data ready)
- **Temporal split**, never random: train on `[0, T1]`, validate `[T1, T2]`, test
  `[T2, T3]`; then **rolling-origin** evaluation (re-fit at each origin, predict the
  next horizon) to get a distribution of metrics with bootstrap CIs.
- **Cold-start holdout**: a set of firms with features but no training events, scored
  by covariates + similarity excitation.
- **Leakage guards**: covariates must be as-of-`t` (point-in-time), no future
  information; mark/firm metadata frozen at decision time.

### 5.5 Metrics
- **Ranking (primary):** Recall@k, Precision@k, MRR, NDCG@k of the horizon-hit list;
  lift over the recency baseline.
- **Time-to-event:** time-dependent concordance index (Antolini), MAE/RMSE of next
  time, integrated Brier score / IPA.
- **Probabilistic / calibration:** held-out (IC-)log-likelihood, time-rescaling KS,
  reliability diagrams, dispersion.
- **Business:** precision@k of flagged firms that raised within `h`, and realized
  **lead time** (days between flag and announcement).
- **Significance:** paired bootstrap over rolling-origin windows; report CIs, not point
  numbers.

### 5.6 Acceptance criteria (v1)
Ship Hawkes if, on the backtest, it (a) beats recency/Poisson on Recall@k and lead
time with non-overlapping bootstrap CIs, (b) is within noise of DeepHit/neural-TPP on
Recall@k while providing interpretable structure and calibrated probabilities, and (c)
passes the time-rescaling GOF test on held-out data.

---

## 6. Code structure (build on `hawkes_calibration`)

Add a new subpackage; **reuse** the existing spine (do not re-implement solvers).

```
hawkes_calibration/                      # existing, reused
    mbpp/        core, exogenous, interval_censored (fit_mbpp_ic_excitation_multi), gof, bayes
    operators/   solve_mbpp_ltv (multivariate covariate solver), linear solvers
    eventtime/   event-time MLE, likelihood + analytic gradient, lasso

nextfund/                                # NEW subpackage (this project)
    data/
        schema.py        Event(t, firm, round_type, amount, ...), Firm(id, sector, features)
        synthetic.py     marked covariate Hawkes generator (generalises exp13 clean_sim to marks+embeddings)
        adapters.py      Crunchbase-like loader -> canonical event/covariate frames (point-in-time)
    model/
        marked_hawkes.py     marked multivariate intensity; embedding-structured A_{ij}; log-lik + grad (event-time)
        mbpp_marked.py       interval-censored path: wraps fit_mbpp_ic_excitation_multi + embeddings
        embeddings.py        firm embeddings from features / learned; softplus(u_i^T W u_j + sector)
        kernels.py           exp / sum-exp / power-law (reuse package kernels)
    rank/
        readout.py       p_i(t,h) = 1 - exp(-(Xi_i(t+h)-Xi_i(t))); next-mark prob; expected time
        ranking_loss.py  optional listwise NDCG-surrogate fine-tuning
    baselines/
        recency.py       time-since-last-round / sector-momentum ranker
        poisson.py       per-firm inhomogeneous Poisson
        survival.py      Cox / DeepSurv / DeepHit / Dynamic-DeepHit wrappers (lifelines / pycox)
        neural_tpp.py    EasyTPP adapter (RMTPP, NeuralHawkes, THP)
    eval/
        metrics.py       recall@k, mrr, ndcg, c-index(Antolini), Brier/IPA, time-rescaling KS, dispersion
        backtest.py      temporal split + rolling-origin harness + bootstrap CIs + leakage guards
    experiments/
        configs.py       grid configs (covariate level x kernel x resolution x embedding rank)
        run_synthetic.py end-to-end synthetic recovery + ablations (reuses exp13 grid)
        run_backtest.py  real-data backtest driver
    README.md            quickstart + how the pieces fit
```

**Interfaces (stable contracts agents code against):**
```python
model.fit(events, firm_covariates, resolution="event"|"interval")           # -> fitted model
model.rank_next(t, horizon_h, candidate_firms) -> [(firm_id, p_hat, exp_time, ci)]  # ranked desc
model.intensity(firm_id, t) ; model.compensator(firm_id, t0, t1)            # diagnostics
eval.backtest(model, data, origins, horizon_h) -> MetricsTable              # CIs over windows
```

---

## 7. Milestones (iteration plan for agents)

- **M0 — Data & synthetic (foundation).** `data/schema.py`, `data/synthetic.py`
  (marked covariate Hawkes generator with embeddings), `eval/metrics.py`. *Exit:* can
  simulate, censor, and score a ranked list against truth.
- **M1 — Baselines + eval harness.** `baselines/recency.py`, `baselines/poisson.py`,
  `eval/backtest.py` (temporal split, rolling origin, bootstrap). *Exit:* recency &
  Poisson scored on synthetic with CIs.
- **M2 — Marked Hawkes (event-time).** `model/marked_hawkes.py` (dense `a_{ij}` for
  small N) + `rank/readout.py`. *Exit:* beats Poisson on synthetic ranking; recovers
  covariate signs; passes time-rescaling GOF.
- **M3 — Embedding-structured excitation (scale).** `model/embeddings.py`; low-rank
  `a_{ij}`; SGD training. *Exit:* scales to N≈10³–10⁴ firms; cold-start firms rank
  sensibly.
- **M4 — Interval-censored path.** `model/mbpp_marked.py` over
  `fit_mbpp_ic_excitation_multi`. *Exit:* counts-only fit recovers ranking within a
  quantified gap of event-time.
- **M5 — Strong baselines.** `baselines/survival.py` (DeepHit/Dynamic-DeepHit via
  pycox), `baselines/neural_tpp.py` (EasyTPP). *Exit:* head-to-head table on synthetic
  + ablation grid.
- **M6 — Backtest + report.** `experiments/run_backtest.py` on real (Crunchbase-like)
  data; lead-time and precision@k; decision memo against §5.6 acceptance criteria.

Dependencies: M0→M1→M2→{M3,M4}→M5→M6. M3 and M4 are parallelizable after M2.

---

## 8. Open questions for stakeholders (resolve before/with M0)
1. **Mark granularity:** individual firms (sparse, large N → embeddings essential) vs
   firm-clusters/sectors (denser, more identifiable)? Affects M2/M3 sizing.
2. **Data resolution:** do we get attributable event times, or only aggregate counts?
   (Determines event-time vs MBPP as the primary path; we build both.)
3. **Horizon `h`** and **`k`** of the watch-list (sets the operating point and the
   primary metric).
4. **Covariate availability & point-in-time correctness** (popularity scores, sector,
   stage, macro) — and update cadence (for piecewise-constant covariates).
5. **Competing risks** scope: do acquired/dead firms leave the risk set, and do we
   model re-entry (a firm raising again)?

---

## 9. References

- A. G. Hawkes (1971). Spectra of some self-exciting and mutually exciting point processes. *Biometrika* 58(1).
- A. G. Hawkes, D. Oakes (1974). A cluster process representation of a self-exciting process. *J. Appl. Probab.* 11(3).
- M.-A. Rizoiu et al. (2022). [Interval-censored Hawkes processes](https://www.jmlr.org/papers/v23/21-0917.html). *JMLR* 23(1):1–84.
- N. Du et al. (2016). [Recurrent Marked Temporal Point Processes: Embedding Event History to Vector](https://www.kdd.org/kdd2016/papers/files/rpp1081-duA.pdf). *KDD*.
- H. Mei, J. Eisner (2017). [The Neural Hawkes Process: A Neurally Self-Modulating Multivariate Point Process](https://arxiv.org/abs/1612.09328). *NeurIPS*.
- S. Zuo et al. (2020). [Transformer Hawkes Process](http://proceedings.mlr.press/v119/zuo20a/zuo20a.pdf). *ICML*.
- Q. Zhang et al. (2020). Self-Attentive Hawkes Process. *ICML*.
- O. Shchur et al. (2020). [Intensity-Free Learning of Temporal Point Processes](https://arxiv.org/abs/1909.12127). *ICLR*.
- S. Xue et al. (2024). [EasyTPP: Towards Open Benchmarking Temporal Point Processes](https://arxiv.org/html/2307.08097v3). *ICLR*.
- R. Trivedi et al. (2017). Know-Evolve: Deep Temporal Reasoning for Dynamic Knowledge Graphs. *ICML*.
- R. Trivedi et al. (2019). [DyRep: Learning Representations over Dynamic Graphs](https://openreview.net/forum?id=HyePrhR5KX). *ICLR*.
- D. R. Cox (1972). Regression Models and Life-Tables. *J. R. Stat. Soc. B* 34(2).
- J. Katzman et al. (2018). DeepSurv: personalized treatment recommender using a Cox proportional hazards deep neural network. *BMC Med. Res. Methodol.*
- C. Lee et al. (2018). [DeepHit: A Deep Learning Approach to Survival Analysis with Competing Risks](https://cdn.aaai.org/ojs/11842/11842-13-15370-1-2-20201228.pdf). *AAAI*.
- C. Lee et al. (2019). Dynamic-DeepHit: A Deep Learning Approach for Dynamic Survival Analysis with Competing Risks Based on Longitudinal Data. *IEEE TBME*.
- L. Antolini et al. (2005). A time-dependent discrimination index for survival data. *Statistics in Medicine* 24.
- G. Ross et al. (2021). [CapitalVX: A machine learning model for startup selection and exit prediction](https://www.sciencedirect.com/science/article/pii/S2405918821000040). *J. Finance and Data Science*.
- Startup success prediction and VC portfolio simulation using Crunchbase data (2023). [arXiv:2309.15552](https://arxiv.org/abs/2309.15552).
- Interpretable Machine Learning for Predicting Startup Funding, Patenting, and Exits (2025). [arXiv:2510.09465](https://arxiv.org/html/2510.09465v1).
- Companion document: `paper/investment_case_study.pdf` (this repo) — the covariate-augmented MBPP, proofs, and experiments this plan builds on.
