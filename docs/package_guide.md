# Package guide — how each part works

A part-by-part tour of `hawkes_calibration`: what each piece does, a runnable snippet,
and what we found. The theory and proofs are in `paper/complete_account.pdf`; this guide
is the *code* companion. Everything imports with **numpy only**; SciPy/JAX/TensorFlow are
optional and only needed where noted.

```bash
pip install -e .                 # numpy-only core
pip install -e ".[dev]"          # + scipy, pytest, matplotlib
pip install -e ".[jax]"          # optional: PINN/PINO + differentiable inverse
pip install -e ".[tf]"           # optional: FNO / DeepONet / amortized inference
PYTHONPATH=. pytest              # the executable ground truth for every claim below
```

The package is organized by **data regime**, then by **application model**:

| area | module | regime |
|---|---|---|
| event-time calibration | `eventtime/` | exact timestamps |
| interval-censored calibration | `mbpp/` | weekly counts (MBPP) |
| learned / exact solvers | `operators/` | forward + inverse maps |
| application models | `models/`, `sector_*` | funding-event forecasting |

---

## 1. Event-time calibration — `eventtime/`

Exact timestamps ⇒ the full Hawkes log-likelihood is available; fit by MLE (with
asymptotic SEs from the observed information), optionally L1-penalised for sparse
high-dimensional excitation.

```python
import numpy as np
from hawkes_calibration import simulate_hawkes, fit_hawkes_mle   # eventtime API

# multivariate exponential-kernel Hawkes; recover (mu, alpha, beta) from event times
events = simulate_hawkes(mu=[0.3, 0.3], alpha=[[0.2, 0.1], [0.0, 0.25]], beta=1.0, T=2000, seed=0)
fit = fit_hawkes_mle(events, n_dims=2)
print(fit.summary())
```

**What we found.** On a 12-dimensional process (~14k events) an L1 (BIC-lasso) fit recovers
the excitation support at precision **0.74 / recall 0.94** (vs 0.61 / 0.79 unpenalised), and
drives the true-zero MAE from **0.016 → 0.002**. Event times identify the full structure;
counts alone do not (see §2).

---

## 2. Interval-censored calibration — `mbpp/`

Only weekly counts are observed. The Hawkes likelihood is unavailable, so we fit through
the **Mean Behavior Poisson Process**: the expected intensity solves a Volterra equation
`ξ = s + φ * ξ`, its compensator makes interval counts Poisson (Watanabe), and the
interval-censored loss `Σ Ξ − C·log Ξ` is a KL/Bregman divergence. The exponential-kernel
compensator is **closed-form** (a small linear ODE); no learned solver is used here.

```python
import numpy as np
from hawkes_calibration import fit_mbpp_ic

obs = np.arange(0, 200)                       # weekly bin edges
counts = ...                                   # (T,) events per week
fit = fit_mbpp_ic(obs, counts, exogenous, loss="ic-ll")   # method="closed" by default
print(fit.summary())                           # kappa, theta, SEs (quasi-Poisson inflated)
```

Covariates enter the baseline (`fit_mbpp_ic_covariates`) or the excitation
(`fit_mbpp_ic_excitation[_multi]`, exact per-segment ODE).

**What we found (synthetic investment market).** Correctly specified: branching κ and
baseline recovered to **~0.03**, covariate coefficients (incl. a *negative* cooldown) to
**~0.10**, decay θ left loose (the κ–θ ridge). Triply mis-specified (power-law + nonlinear
self-inhibition + cross-coupling): the covariate's **sign and significance survive** and
held-out likelihood improves, though magnitude inflates ~2×. Multivariate: levels + overall
branching (spectral radius **0.31 vs 0.30**) recover; the full M×M structure needs event
times or more data. Full table in `paper/complete_account.pdf` §"MBPP and covariate-augmented Hawkes".

---

## 3. Learned & exact solvers — `operators/`

Exact solvers (`solve_mbpp_ode*`, `solve_mbpp_ltv`, spectral) are the oracle. On top of
them sit **learned** operators — a PINO/DeepONet forward map and a differentiable inverse —
for when the exact solve becomes the bottleneck (many repeated solves, larger M, or kernels
with no closed form). They are optional accelerators, **not** part of the default fit.

```python
import numpy as np
from hawkes_calibration.operators.pino import MultivariateMBPPOperator, sample_instances

t = np.linspace(0, 12, 96)
S, A = sample_instances(640, 3, seed=1)                     # baselines + excitation matrices
op = MultivariateMBPPOperator(M=3, t_grid=t, theta=1.0, p=40, hidden=96)
op.train(S, A, epochs=4000, lr=3e-3, batch=96)              # physics-only (MBPP residual)
xi = op.predict(np.array([1.2, 0.8, 1.5]),
                np.array([[.3, .1, 0], [.1, .3, 0], [0, 0, .25]]))[0]   # (96, 3), one pass
```

**What we found.**
- **Forward (PINO, held-out rel-L2):** 0.55 % (M=3), 1.52 % (M=5), 2.03 % (M=8), ~**3 ms/solve**.
  Numpy physics-only reference: ~2.5 % at M=3 in 37 s.
- **Conditioning is required (go/no-go):** an *unconditioned* map fails — vanilla PINN median
  **0.345**, FNO with the system hidden **0.492**; conditioning on `(s, A)` is what makes it work.
- **Inverse (differentiable):** clean A **0.03 %** / δ **0 %** (M=3); under noise δ stays <5 %
  to σ=0.02 while A is ~10× more sensitive; averaging R obs ⇒ error ∝ σ/√R (R=256 → A 5.6 %).
  A one-shot amortized regressor plateaus at ~38 %/49 % → direct regression can't invert; use
  differentiable inversion (amortized net only as a warm start).

**How to run the full learning suite.** JAX/TF backends live in `operators/{neural_solver,tf}.py`;
the driver + self-check is `experiments/learning/`:

```bash
PYTHONPATH=. python experiments/learning/run_learning.py --selfcheck   # numpy, runs anywhere
```

The self-check validates the enabling fact — the MBPP residual is **linear in ξ**, so its
global minimum is the exact solution (`family_residual(exact_xi) ≈ 1.8e-15`). The suite is
organised as **F** (forward operators), **P** (physics-informed PINN/PINO), **A** (amortized
inverse); parameter prior box and configs are in `experiments/learning/configs.py`.

---

## 4. Application models — `models/`, `sector_*`

The funding-event forecaster. Two roads (both covered in `complete_account.pdf`):

**(a) Counts / two-stage** — a sector Poisson-Hawkes count model (`fit_sector_count_model`)
for *when/where*, then a within-sector selector for *which firm*: either the survival model
with an **outside option** (`fit_startup_survival`) or the 1-D Hawkes rankers
(`fit_hawkes_ranker`, linear or exp link, dropping the last-funded firm from the risk set).

```python
from hawkes_calibration import (simulate_synthetic_startup_market,
                                fit_startup_survival, evaluate_survival,
                                fit_hawkes_ranker, evaluate_hawkes_ranker, make_tracked_mask)

d = simulate_synthetic_startup_market(T=120, n_sectors=6, startups_per_sector=20, seed=3)

# survival selector with an outside ("not in our universe") option
tracked = make_tracked_mask(d.active, fraction=0.7, seed=1)
surv = fit_startup_survival(d.events, d.startup_features, d.startup_sector, d.active,
                            d.startup_counts, tracked=tracked, train_end=80)

# or the 1-D Hawkes intensity ranker (rank firms by rate == the survival conditional prob.)
rk = fit_hawkes_ranker(d.events, d.startup_features, d.startup_sector, d.active,
                       d.startup_counts, link="exp", train_end=80, drop_last_funded=True)
print(evaluate_hawkes_ranker(rk, d.events, d.startup_features, d.startup_sector,
                             d.active, d.startup_counts, start_week=80, end_week=120))
```

**(b) Event-time / one layer** — `models/event_block_hawkes.py`: a marked, block-structured
**log-linear (exp-link) Hawkes** for exact times in an open population — self-inhibition via
the exp link, an M×M sector (not N×N firm) excitation matrix, at-risk indicators for
entry/exit, concave MLE.

**What we found.** Two-stage ranker: top-5 **0.49** vs oracle **0.54**, ~3–6× random. Survival
with outside option: off-list AUC ≈ **0.68**, top-5 ≈ **0.70**. 1-D Hawkes rankers: top-5 ≈
**0.65** for both links (the ordering is set by covariates + cooldown, not the link; exp gives
better-calibrated probabilities). Event-time block Hawkes: gradient exact, concave, time-
rescaling KS ≈ **0.04**, baselines/covariates and the self-inhibition *ordering* recovered;
absolute excitation vs inhibition weakly identified from one market path.

---

## 5. Stability — read before you simulate

Discrete-time log-linear count models feed **raw counts through an `exp` link**, which can
explode *even when the coefficient spectral radius is < 1*. This is the subtle failure that
once tried to generate ~5×10⁸ events in a single week. The guards (row-sum stability bound,
risk-set cap, `max_events_per_week`, subcritical samplers, size-normalized fields) and the
full analysis are in **`docs/stability_and_explosion_report.md`** and `complete_account.pdf`
§"Stability and the explosion problem". Rule of thumb: keep count-feedback tiny, let
covariates carry the signal, and cap events per week if you must simulate.

---

*Architecture of the built two-stage system: `docs/investment_market_architecture.md`.
Full mathematics: `paper/complete_account.pdf` and `paper/textbook.pdf`.*
