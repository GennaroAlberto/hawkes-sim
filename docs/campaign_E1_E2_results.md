# Campaign results — E1 (stage-1 recovery) and E2 (ranker & the risk-set problem)

Regime A (`data/synthetic_A`, well-specified two-stage world), seed 0, temporal split
train weeks `[0,416)` / test `[416,520)`. Reproduce with
`PYTHONPATH=. python -m experiments.campaign_regimeA` → `results/campaign_regimeA.json`.
Fitters are the vectorized references in `synthetic/fast_fit.py`.

## E1 — Stage-1 sector count GLM

| quantity | measured | briefing baseline |
|---|---|---|
| beta correlation (fit vs true) | **0.669** | 0.675 |
| excitation entry correlation | **0.295** | 0.27 |
| effective spectral radius: fit vs true | **0.355 vs 0.150** | 0.355 vs 0.172 |
| held-out Poisson NLL/cell: model vs mean-baseline | **2.13 vs 5.40** | — |
| held-out dispersion | 1.46 | — |

**Reads.** The GLM recovers the covariate effects (β corr ≈ 0.67, ~25 % attenuation
from observation noise) and beats the historical-mean baseline decisively on held-out
NLL. **But the fitted effective excitation radius is ~2.4× the truth (0.355 vs 0.15).**
This is the *latent-factor → spurious-excitation* artifact: a persistent unobserved
common factor (risk appetite) masquerades as cross-sector contagion in the fitted
excitation matrix. Practical implication for real data: **treat a fitted "contagion"
radius as an upper bound**; much of it may be a common factor. Mitigations to pursue
(E1 sub-experiment): add a lagged aggregate-market-count covariate, stronger ℓ2 on
off-diagonals, or a diagonal-only adjacency mask.

## E2 — Stage-2 risk-set ranker (the headline)

Fair held-out evaluation: an event whose funded firm is absent from the risk set
counts as a **miss** (rank = risk-set size), so pruning cannot cheat by dropping hard
events. True weights `w=[0.30, 0.30, −0.15, 0.15, 0.00]`, `η≈−2.67`.

| risk set | fitted w | η | held-out NLL | top-5 | MRR |
|---|---|---|---|---|---|
| **observed** (naive pool) | `[−0.06, 0.17, 0.25, 0.08, +1.94]` | −1.88 | 4.68 | 0.144 | 0.103 |
| **oracle** (true entry/exit) | `[0.28, 0.25, −0.12, 0.18, 0.08]` | −2.48 | 4.20 | **0.288** | **0.196** |
| exit prune (recency ≤ 78) | `[−0.05, 0.17, 0.30, 0.07, +3.97]` | −2.50 | 4.20 | 0.157 | 0.114 |
| exit prune (raised & silent > 78) | `[−0.07, 0.17, 0.22, 0.08, +2.00]` | −1.90 | 4.53 | 0.147 | 0.103 |

**The core result.** The **observed** risk set *inverts the economics*: the weight on
cumulative capital raised jumps to **+1.94** (truth 0.0) and the covariate-3 sign
flips, because graduated/dead "zombie" firms stay in the pool and make *raised* look
spuriously predictive. The **oracle** mask (true entry/exit) recovers the weight signs
and order and **doubles** the ranking metrics (top-5 0.29 vs 0.14). The entire gap is
risk-set contamination — the event-time observation problem, since the data never
labels when a firm leaves the private market.

**What the exit models buy (and don't).** Four exit strategies were tested, from
crude to principled:

1. *recency prune* (drop firms silent > cap): closes the **NLL** gap (4.20 = oracle)
   only by shrinking the pool — ranking barely moves (top-5 0.157), weights stay
   inverted, and it over-prunes first-time funders.
2. *raised-and-silent prune* (exit only firms that raised then went quiet): keeps all
   events, small NLL gain, no ranking recovery, no weight fix.
3. *learned exit hazard, hard prune*: a logistic exit model on
   `[recency, stage, raised, employees, age, weeks-since-entry]`, trained on an
   **observable-only** label ("any deal in the next 26 weeks"). Feasibility is real —
   this model predicts the *oracle* active-status with **AUC 0.877**, and the top
   predictors are exactly the right ones (recency −, stage −, raised − → late-stage /
   high-raised firms graduate out). Yet pruning by it *lowers* top-5 to ~0.11
   (imperfect exit modeling drops true funded firms → misses) and does not de-bias the
   weights.
4. *learned exit hazard, soft weight* (log-odds-active as a fixed score offset, no
   dropping): NLL 4.79, top-5 0.124 — no better, weights still off.

**Conclusion (stronger than the briefing anticipated).** The observed→oracle gap is
**not closable by any observable risk-set reweighting/pruning** tested. The reason is
structural: the observation process (which firms populate the pool, and their
recency/stage) is **confounded with the funding outcome**. Given the contaminated
pool, the "inverted" observed weight on recency (+1.94) is in fact the *Bayes-optimal*
predictive weight — recency is a rational **proxy for the unobserved exit
information**. Recovering the true structural weights requires the true entry/exit
(oracle); an exit model good enough to predict exit (AUC 0.877) is still not good
enough to reconstruct it without dropping real events. This is precisely the
event-time observation problem the campaign is about.

**Recommendation for real data.** The ranking value lives in **exit/graduation
labeling**, not in the ranker model. Invest the pipeline effort in the best possible
firm-lifecycle signal (business-status transitions, IPO / M&A events, going-quiet
hazards by stage) and **report ranking metrics under both the observed pool and the
best available exit mask as a bound** — the spread (top-5 0.14 → 0.29 here) is the
cost of the observation gap and should be quoted for real deployments.

## E-noise — covariate measurement noise on the benign case (v2, `exp22`)

Relative Gaussian noise (sd = σ × signal sd, σ up to 20 %) on the macro covariates
(stage 1) and firm features (stage 2), present at fit **and** prediction, anchored to
the v2 seed-0 baselines at σ=0. **Prediction is nearly untouched**: the stage-1
covariate lift stays ≈1.0 nats (1.007→0.985), stage-2 top-5 drops only 0.7 pp
(0.446→0.439). **Structure erodes first**: β corr 0.67→0.53, choice weights attenuate
×0.88 — textbook errors-in-variables. 15–20 % covariate noise is essentially harmless
for ranking/prediction.

## E3 — misspecification stress on regime B (`exp23`)

Daily firm-level block-Hawkes truth; every weekly model is misspecified by design.
Briefing baselines reproduced exactly (β corr 0.376 vs ~0.37; radius 0.609 vs ~0.57;
choice NLL/random/top-5 = 3.796/4.172/0.316).

| layer | what survives model error | what does not |
|---|---|---|
| weekly sector GLM | covariate *signs/ordering* (β corr 0.38); held-out NLL still beats the mean baseline (2.60 vs 4.30) | β magnitudes (~2× attenuated); the fitted "excitation" (radius 0.61) is aggregation artifact + quality clustering — not structural |
| funded-pool choice model | ranking degrades gracefully: top-5 0.316 (vs 0.446 on regime A); lift 0.38 nats | the gap weight now reflects the 30d inhibition, not a 26w cooldown — functional form is DGP-specific |
| **event-time block Hawkes (day resolution)** | **self-inhibition ordering: rank corr 0.937** (target ≥0.9) — the structural read weekly models cannot deliver | individual contagion entries (diag-A corr ≈ 0): only the *ordering* of structural effects is identified, not entries |

## E1b — decay identifiability vs timestamp resolution (`exp24`)

True contagion decay θ = ln2/7 ≈ 0.099/day (7-day half-life). One estimator (profile
MLE, within-bucket uniform imputation), resolutions {1d, 7d, 28d}:

* **Controlled (well-specified), sector volume (~4k events):** the κ–θ ridge
  dominates — θ̂ spread ±0.05 even at day resolution; 7d buckets lose almost nothing.
  Day resolution alone does not buy a sharp decay at sector volumes.
* **Controlled, 4× volume (~16k events):** 1d/7d unbiased (θ̂ 0.112 ± 0.030); **28d
  systematically biased low (0.081 ± 0.014)**. Rule of thumb: bucket width ≤ one
  kernel half-life is safe; ≥ 4 half-lives destroys the timing information.
  κ is robust at every width (0.28–0.38 around 0.35) — the usual κ-strong/θ-weak split.
* **Regime B empirical (busiest sector, macro-covariate baseline):** θ̂ ≈ 0.20 (2×
  truth) and κ̂ ≈ 0.79 at *every* resolution — a first pass without the covariate
  baseline gave κ̂ ≈ 0.92. Latent-quality clustering, not resolution, is the binding
  constraint: at pooled-sector level the structural (κ, θ) are confounded regardless
  of how exact the dates are. Day resolution pays off for *firm-level* structure
  (E3's ρ ordering), not for pooled-sector kernels.

## Status vs EXPERIMENTS.md (v2)

- E1a recovery + latent-artifact diagnosis: **done** (baselines reproduced). The
  `latent_strength ∈ {0,0.15,0.3}` regeneration sweep is still to run.
- E1b (event-time track): **θ-resolution study done** (`exp24`); the full multivariate
  sector-Hawkes recovery table across 5 seeds remains.
- E2 (v1 risk-set study): **done** — observed/oracle reproduced; exit-model regime (3)
  closed with a structural negative result (four exit models, incl. an AUC-0.877
  learned hazard, cannot close the gap). Under v2 the risk set is assumed truthful.
- E2 (v2 choice model): baselines reproduced on both regimes (`exp22`, `exp23`);
  sub-experiments 1–6 (gap functional form, newcomer ASC, sampled softmax, exclusion
  rule, l2 grid, cross-regime) remain.
- E3: **done** (`exp23`) including the day-resolution block Hawkes (ρ rank corr 0.937).
- E-noise (covariate noise ≤20 %): **done** (`exp22`).
- E4 (PINO vs analytical), E5 (walk-forward backtest + covariate ablation): not started.
