# Stability & the "explosion" problem — an agent-oriented report

**Audience.** A future coding/thinking agent working in this repo who needs to
explain to the user *why simulations blow up*, *why a "spectral radius < 1" check is
not always enough*, and *what is already in place to prevent it*. Read this before
touching the sector model, `simulate_marked_paths`, the MBPP/PINO samplers, or any
synthetic data generator.

**TL;DR.** There are **two different excitation models** in this repo and **two
different explosion mechanisms**. One is the familiar continuous-time Hawkes
supercriticality (`ρ(branching) ≥ 1`). The other — the one that actually hung CI and
the backtests — is a **discrete-time log-linear Poisson auto-regression that feeds
raw counts through an `exp` link**, which is explosive *even when the coefficient
matrix has spectral radius well below 1*. Keep the sector excitation tiny, drive the
signal with covariates, and rely on the guards below.

---

## 1. The two model families (do not conflate them)

### A. Continuous-time additive Hawkes / MBPP (event-time models, e.g. `exp20`)

$$
\lambda_m(t) = \mu_m(t) + \sum_j \alpha_{mj}\sum_{t_{j,k}<t} e^{-\beta_{mj}(t-t_{j,k})}.
$$

Excitation is **additive**. The branching matrix is `G_{mj} = α_{mj}/β_{mj}`
(the integral of the kernel = expected direct offspring). The mean intensity solves
`ξ = μ + G ξ`, i.e. `ξ* = (I − G)^{-1} μ`.

* **Stability condition: `ρ(G) < 1`** (spectral radius). This is exact and
  well-behaved. As `ρ(G) → 1`, `(I − G)^{-1}` blows up and the process becomes
  supercritical (infinitely many events). This is the *classic* Hawkes explosion.
* In `exp20` we keep `ρ(α/β) ≈ 0.25` and everything is clean. No surprises here.
* The **MBPP/PINO** work hit a variant: the excitation is covariate-modulated,
  `α_{mj}(t) = α_{mj}\,e^{\delta^\top Z(t)}`, so the *effective* branching is
  `α·e^{δ·Z}/β`. Sampling parameters without enforcing
  `κ·e^{δ·\max|Z|} < 1` let ~18–26 % of draws go supercritical, and the exact target
  intensity blew up to ~`10^{84}`. Fix: constrain the sampler (already done; see
  `operators/covariate_inverse.py` and the PINO config notes).

### B. Discrete-time **log-linear** Poisson count model (the sector layer, `exp19`)

This is **`hawkes_calibration/sector_ranker.py: fit_sector_count_model` /
`simulate_synthetic_startup_market`**:

$$
Y_{s,t}\sim\mathrm{Poisson}(\Lambda_{s,t}),\qquad
\log\Lambda_{s,t} = a_s + \beta_s^\top X_t + \sum_{r,\ell} b_{sr\ell}\,Y_{r,t-\ell}.
$$

Excitation is **multiplicative / in the log domain**, and it feeds **raw counts**
`Y` back into the **log**-rate. **This is the dangerous one.**

---

## 2. The explosion problem, precisely

### Mechanism 1 — additive supercriticality (model A)

Standard: `ρ(G) ≥ 1 ⇒ unbounded`. Diagnose with the spectral radius of `α/β` (or of
the summed kernel). Prevent by keeping `ρ(G)` bounded below 1 (with a margin). This
is the condition everyone expects, and for model A the spectral radius is the right
quantity.

### Mechanism 2 — log-linear positive feedback (model B). **This is the subtle one.**

For the count model, the conditional mean given the recent counts is

$$
\mathbb{E}[Y_{s,t}\mid \text{past}] = \Lambda_{s,t}
 = \exp\!\Big(a_s + \beta_s^\top X_t + \sum_{r,\ell} b_{sr\ell} Y_{r,t-\ell}\Big).
$$

Because the link is `exp(·)` and the inputs are **raw counts**, the rate is a
**convex, exponential** function of past counts. A single unusually busy week
(`Y` large) is exponentiated into the next rate:

> `Y` jumps → `Σ b·Y` jumps → `Λ = exp(…)` jumps **multiplicatively** → next `Y` is
> drawn even larger → runaway.

**Key point that trips people up:** the condition `ρ(b) < 1` (correct stability for
the *additive / identity-link* INGARCH `Λ_t = a + Σ b·Y_{t-ℓ}`) **does NOT make the
log-linear model stable.** With the `exp` link, even `ρ(b) = 0.4` is explosive once
counts get moderately large, because the feedback compounds geometrically rather than
additively. The standard *stable* log-linear Poisson auto-regression (Fokianos &
Tjøstheim) feeds back `log(1 + Y_{t-1})` and/or lagged **log-intensities**, with
`|coeffs| < 1` — **not raw `Y`**. Our model uses raw `Y`, which is the explosive
specification.

**Concrete failure we observed.** The fitted sector model came out with summed-lag
spectral radius `ρ ≈ 3.7` (no stability constraint at the time). In
`simulate_marked_paths` the feedback drove `log Λ` into the `exp(20) ≈ 4.85×10^8`
rate clip; `Poisson(4.85×10^8)` returns on the order of **hundreds of millions** of
events; and the per-event inner loop

```python
for _ in range(int(y_t[s])):   # y_t[s] ~ 4.85e8
    ...
```

iterates that many times → the simulation "hangs" for minutes per path. With no
per-test timeout, GitHub Actions ran until killed → **CI showed 0/3** (all three
Python-matrix jobs). It was never a logic bug; it was an ~`5×10^8`-iteration loop.

---

## 3. Why "I constrained the spectral radius" is not the whole story

We *do* now constrain the fitted `b` (see `sector_stability.py`): a row-sum bound
`Σ_ℓ b_{sr ℓ} ≤ ρ_max` implies `ρ(G) ≤ ρ_max < 1` for non-negative `G`. That bounds
the **self-excited feedback loop** and stops the runaway in *simulation*.

But two things remain true and worth telling the user:

1. **One-step rates can still spike on real inputs.** The stability bound limits the
   feedback, but for held-out *one-step* prediction the model is fed the *observed*
   lagged counts. If a real `Y` is large, `exp(b·Y)` can still hit the clip for a
   single cell, producing an astronomically large *predicted* rate and a Poisson NLL
   in the millions. So a stability-constrained fit can still have a catastrophic
   one-step NLL on an unlucky cell. (Seen at seed 4.)
2. **Stability ≠ predictive value.** Even when stable, the sector log-linear model
   does **not** beat a per-sector historical-mean baseline on this synthetic DGP,
   because the excitation has to be kept tiny to stay stable and the residual signal
   is weak. The way to make the sector layer genuinely good is to drive it with the
   **covariates** (which the mean baseline cannot see), not with count feedback.

---

## 4. Defense in depth — what is already in the code

| Layer | Guard | File | Effect |
|---|---|---|---|
| Sector fit | row-sum stability constraint `Σ_ℓ b ≤ ρ_max (=0.95)` + projection | `sector_stability.py` | fitted `ρ(G) < 1`; bounds feedback |
| Simulation | **risk-set cap**: `Y_{s,t} ≤ #live startups in sector s` (you cannot fund more *distinct* firms than exist) | `sector_ranker.py: simulate_marked_paths` | inner loop bounded **by construction** — un-hangable regardless of the fit |
| Simulation | **first-K cap** `Y_{s,t}=min(Poisson, K)` (`max_events_per_week`) | `simulate_marked_paths`, `sector_backtest.py` | matches the business question ("next K firms") and hard-bounds events |
| Generator | `excitation_radius` knob (set the *true* summed-lag `ρ` far below 1) + drive signal via `covariate_strength` | `simulate_synthetic_startup_market` | DGP itself is noncritical and learnable |
| MBPP/PINO | sampler stability `κ·e^{δ·\max|Z|} < 1` | `operators/covariate_inverse.py`, PINO configs | no supercritical training instances |

**Rule of thumb for the agent:** never simulate a discrete-time log-linear count
model with non-trivial raw-count feedback. Keep `excitation_radius ≲ 0.1`, keep base
rates moderate, and let covariates carry the signal. If you must simulate, the
risk-set cap and/or `max_events_per_week` make it safe.

---

## 5. Current state (what works)

* **CI is green.** In a CI-equivalent env (numpy+scipy, no jax/tf) the whole suite
  passes; the previously-hanging end-to-end test now uses a realistic config + the
  caps and asserts *true* things.
* **`exp19` (two-layer, no time info):** Layer 1 sector GLM beats the mean baseline
  100 % of seeds (held-out Poisson NLL ≈ 1.07 vs 2.71) with fitted `ρ ≈ 0.30` and
  ~1.3 events/cell; Layer 2 ranker sits at the **oracle ceiling** (top-5 0.46 vs
  oracle 0.46, ~2.3× random).
* **`exp20` (non-linear event-time Hawkes):** baseline `exp(γ₀+γ·X(t))` (nonlinear in
  covariates) + subcritical excitation (`ρ(α/β)=0.25`), recovered from ~20k event
  *times* by continuous-time MLE to ~0.05 max abs error on `γ` and `α`.

The contrast is the headline: **with event times** (model A) the nonlinear-intensity
Hawkes is fully identified and well-behaved; **with only weekly counts** (model B) the
log-linear feedback is fragile and must be kept near-zero, so the value comes from the
covariates and the ranker, not from count self-excitation.

---

## 6. If the user asks "why did it explode / why is it slow?"

Say this:

> The weekly *sector* model is a log-linear Poisson auto-regression: it puts the
> lagged event **counts** inside an `exp`. That makes the rate grow *exponentially*
> in recent activity, so a busy week feeds an even busier week — a runaway. Unlike a
> normal (additive) Hawkes, a "spectral radius below 1" on the coefficients is **not**
> enough to keep it stable. When the fit was left unconstrained it became
> supercritical and the forward simulation tried to generate ~half a billion events
> in a single week, which is the multi-minute "hang" (and the red CI). It's fixed
> three ways now: the fit is stability-constrained, the simulation can't draw more
> fundings than there are live startups, and you can cap events per week to the first
> K. For real predictive value, keep the count feedback tiny and let the covariates
> and the startup ranker do the work.
