# Recovering covariate coefficients + excitation of a covariate-modulated Hawkes

**Question.** If each group's excitation is modulated by covariates — some **shared**
across groups (overlap), some **private** to a group (different) — can we recover both
the **covariate coefficients** and the **excitation matrix** from the observed
intensity? And how does **observation noise** change the picture?

**Answer.** On clean data, **both are recovered essentially exactly** (excitation
≤2.5%, covariate coefficients ≤0.1% relative error — well under the 5% target). Under
noise the **covariate coefficients stay robust** (<5% at realistic noise), while the
**excitation matrix is ~10× more noise-sensitive** but recovers cleanly once repeated
observations are averaged. The recovery error scales with the *effective* noise
`σ/√R` (R = number of averaged observations).

## The model

`M` groups. The excitation of group `m` is modulated in time by a log-linear
covariate response:

```
α_{m,j}(t) = A_{m,j} · exp( Σ_k δˢ_{m,k} Zˢ_k(t)  +  δᵖ_m Zᵖ_m(t) )
                              └ shared covariates ┘   └ private one ┘
```

- **`K_shared` shared covariates** `Zˢ` enter *every* group — each with its **own**
  coefficients `δˢ_{m,k}` (the *overlap*: same drivers, similar-or-different responses).
- **One private covariate** `Zᵖ_m` enters *only* group `m` (the *different* one).

The covariates are observed; the mean intensity `ξ(t)` solves the covariate-modulated
MBPP (validated solver `solve_mbpp_ltv`). We recover the excitation matrix `A` (M²
numbers) and the live covariate coefficients `δ` (M·K_shared + M numbers).

## Method

**Differentiable analysis-by-synthesis.** We built a fast, vectorised JAX forward
solver (matches the numpy `solve_mbpp_ltv` oracle to **1e-6**) and recover the
parameters by gradient-descending `‖forward(A,δ) − ξ_obs‖² / ‖ξ_obs‖²` through it.
Noise is handled physically: **average `R` repeated noisy observations** (σ_eff = σ/√R)
of the same system before inverting.

> We also tried a one-shot **amortised** net (CNN: `(ξ, Z) → (A, δ)`). Even with 20k
> training instances it plateaued at ~38% on `A` / ~49% on `δ`: a direct regression
> can't invert this physics well. The differentiable inversion is far superior — so it
> is the method of record. (Same lesson as the forward operators: use the physics.)

## Results (held-out instances, vs. the true parameters)

**Clean observations — essentially exact:**

| M | recover | A rel-err | δ rel-err |
|---|---|---|---|
| 3 | 9 A + 9 δ | **0.03 %** | **0.00 %** |
| 5 | 25 A + 15 δ | **2.55 %** | **0.07 %** |

**With observation noise (M=3), single observation:**

| σ | A rel-err | δ rel-err |
|---|---|---|
| 0.00 | 0.03 % | 0.00 % |
| 0.01 | 10.1 % | **2.1 %** |
| 0.02 | 20.4 % | **4.2 %** |
| 0.05 | 53 % | 10.5 % |

→ the **covariate coefficients δ are robust** (<5% through σ=0.02), while **A**
(the excitation, dominated by small weakly-conditioned cross-terms) degrades faster.

**Averaging repeated observations drives both down (M=3, σ=0.1):**

| R (averaged obs) | σ_eff = σ/√R | A rel-err | δ rel-err |
|---|---|---|---|
| 1 | 0.100 | 129 % | 22 % |
| 16 | 0.025 | 22 % | 5.3 % |
| 64 | 0.0125 | 11 % | 2.4 % |
| 256 | 0.0063 | **5.6 %** | **1.2 %** |

The error scales cleanly with `σ/√R`: with enough repeated observations both the
excitation **and** the covariate coefficients return to the few-percent regime.

See `results/exp18_inverse_M{3,5}.{png,json}` — each figure shows (left) recovery vs
noise, (middle) the averaging law, (right) recovered-vs-true scatter for every `A`
entry and `δ` coefficient.

## Takeaways

- **Yes — we can learn the covariate coefficients and the excitation jointly**, to ~0
  error on clean data.
- **The covariate effects `δ` are the well-identified part** — recovered to <5% even
  from a single noisy intensity path. (These are "the parameters of the covariates".)
- **The excitation matrix `A` is the harder, noise-sensitive part** (weakly-conditioned
  cross-excitation); it is recovered to <5% on clean data and, under noise, by
  averaging repeated observations — error ∝ effective noise `σ/√R`.

## Reproduce

```
PYTHONPATH=. python -m experiments.exp18_covariate_inverse --M 3
PYTHONPATH=. python -m experiments.exp18_covariate_inverse --M 5
```

Code: `hawkes_calibration/operators/covariate_inverse.py` (model, validated JAX
forward, differentiable recovery, amortised net). Tests:
`tests/test_covariate_inverse.py` (mask design; JAX forward == oracle; clean recovery
<5%).
