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

**What the exit models buy (and don't).** Two learned exit heuristics were tested:
- *recency prune* (drop firms silent > cap): closes the **NLL** gap (4.20 = oracle)
  but only because the pool shrinks — **ranking barely moves** (top-5 0.157) and the
  weights stay inverted; it also over-prunes first-time funders.
- *raised-and-silent prune* (exit only firms that raised then went quiet; keep
  never-yet-funded firms): keeps all events, small NLL gain, but again **no ranking
  recovery and no weight fix**.

**Conclusion.** Simple recency/silence rules are **not sufficient** to reconstruct the
at-risk set. Graduation/death does not reduce to "has been silent" (a firm can exit
right after a large raise), so the contaminating firms are not cleanly separable by
recency alone. Recovering the oracle-level ranking needs a **genuine survival / exit
model** with richer signal (stage, size, sector hazard) — the `sector_survival.py` /
`fit_discrete_hazard` machinery — used to define a probabilistic risk set that keeps
first-time funders while removing true exits. That is the open piece of E2 track (3).

## Status vs EXPERIMENTS.md

- E1 recovery table + latent-artifact diagnosis: **done** (baselines reproduced). The
  `latent_strength ∈ {0,0.15,0.3}` regeneration sweep is still to run.
- E2 regimes (1) observed and (2) oracle: **done, reproduced exactly**. Regime (3), a
  survival-based risk set that closes the ranking gap, is **open** — recency heuristics
  shown insufficient here.
- E3 (regime B stress), E4 (PINO vs analytical), E5 (forecast backtest): not started.
