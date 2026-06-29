# hawkes_calibration

[![CI](https://github.com/GennaroAlberto/hawkes-sim/actions/workflows/ci.yml/badge.svg)](https://github.com/GennaroAlberto/hawkes-sim/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Calibrate Hawkes processes in **two data regimes**, with a single consistent API:

1. **Event-time data** — exact timestamps observed. Maximum-likelihood estimation
   of multivariate Hawkes processes with exponential kernels and optional
   time-varying covariates in the baseline, including an L1-regularised fitter for
   sparse high-dimensional excitation networks.

2. **Interval-censored data** — only aggregate counts per interval observed (e.g.
   admissions per day, views per day). The Hawkes likelihood needs event times, so
   we fit through the **Mean Behavior Poisson Process (MBPP)** — a Poisson process
   whose deterministic intensity `ξ(t)` equals the *expected* Hawkes intensity —
   following Rizoiu et al. (2022), *Interval-censored Hawkes processes*, JMLR 23(1).

The full mathematical treatment — derivations, proofs, assumptions, and the link
from each result to the code — is the textbook **[`paper/textbook.pdf`](paper/textbook.pdf)**
(built from `paper/textbook.tex`). This README is the practical package reference.

## Install

```bash
pip install -e .                  # core (numpy only)
pip install -e ".[dev]"           # + pytest, ruff, matplotlib, scipy (development)
pip install -e ".[jax]"           # optional: JAX neural operator / inverse
pip install -e ".[tf]"            # optional: TensorFlow operators
pip install -e ".[all]"           # everything
```

The core imports with **numpy only**; `import hawkes_calibration` has no hard
dependency on SciPy, JAX, TensorFlow or matplotlib. SciPy is used when present for the
`sector_ranker` optimiser (a numpy fallback runs otherwise); JAX/TF are needed only for
the learned-operator and differentiable-inverse modules, which are imported lazily.

## Testing and development

```bash
pytest                            # full suite; JAX/TF tests skip if not installed
ruff check .                      # lint  (config in pyproject.toml, line length 100)
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the development workflow.

## Documentation

| Document | What it is |
|---|---|
| [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md) | technical report: model, learned operator, application |
| [`paper/investment_case_study.pdf`](paper/investment_case_study.pdf) | the rigorous paper (MBPP + covariate augmentation, proofs) |
| [`paper/textbook.pdf`](paper/textbook.pdf) | the full mathematical treatment (pedagogical) |
| [`paper/two_step_ranker.pdf`](paper/two_step_ranker.pdf) | sector Hawkes + within-sector startup ranker |
| [`SURVIVAL_RANKING.md`](SURVIVAL_RANKING.md) | the survival-with-ranking alternative |
| [`PINO_RESULTS.md`](PINO_RESULTS.md), [`LEARNING_REPORT.pdf`](LEARNING_REPORT.pdf) | learned-solver results and feasibility study |
| [`NEXT_FUNDING_DESIGN.md`](NEXT_FUNDING_DESIGN.md), [`PURE_LEARNING.md`](PURE_LEARNING.md) | application design and learning playbook |

## Interval-censored calibration in 30 seconds

```python
from hawkes_calibration import (
    Constant, LHPP, simulate_separable_hawkes, interval_censor,
    uniform_obs_times, fit_mbpp_ic,
)

# simulate a Hawkes cascade (immigrants + offspring kept separate)
imm, off = simulate_separable_hawkes(Constant(3.0, 30.0), kappa=0.5, theta=1.0, T=30.0, seed=0)

# observe ONLY interval counts (e.g. per-day volumes), not event times
obs = uniform_obs_times(30.0, 30)
off_counts = interval_censor(off, obs)
imm_counts = interval_censor(imm, obs)

# recover the branching ratio kappa and decay theta from counts alone
res = fit_mbpp_ic(obs, off_counts, LHPP(obs, imm_counts), loss="ic-ll", endogenous=True)
print(res.summary())            # kappa ~ 0.5, theta ~ 1.0
```

## The model

For an `M`-dimensional point process, component `m` has stochastic intensity

```
lambda_m(t) = s_m(t) + sum_j  alpha_mj(t) * sum_{u < t, events of j}  g_mj(t - u)
```

with each ingredient optionally covariate-driven:

| Ingredient | Forms available | Covariate-driven? |
|---|---|---|
| Baseline `s_m(t)` | constant, piecewise-constant, rectangle, sine, Dassios, multi-impulse, LHPP | **yes** — log-linear `exp(γ₀ + γᵀX(t))` |
| Kernel shape `g_mj` | exponential (closed form), sum-of-exponentials, power-law (numeric) | — |
| Excitation `alpha_mj(t)` | constant `κθ` | **yes** — `α₀·exp(δᵀZ(t))` (time-varying) |

The exponential kernel is parametrised by `(κ, θ)` with `α = κθ`, `β = θ`, so
`κ = α/β` is the branching ratio and `θ` the decay rate
(`kappa_theta_to_alpha_beta` / `alpha_beta_to_kappa_theta` convert).

## What we can fit (observation regime × model)

| | Event times observed | Interval-censored (counts only) |
|---|---|---|
| **Univariate** | `fit_univariate` — `(μ, α, β)` | `fit_mbpp_ic` — `(κ, θ)` |
| **Multivariate** | `fit_multivariate` (elicitation matrix `A`, analytic gradient, SEs); sparse via `fit_multivariate_lasso` | `fit_mbpp_ic_multi` |
| **+ baseline covariates** | `fit_multivariate_with_covariates` | `fit_mbpp_ic_covariates` (recovers `γ`) |
| **+ excitation covariates** | (forward model only) | `fit_mbpp_ic_excitation` (recovers `δ`; fast exact per-regime solve) |
| **+ multi-timescale kernel** | (sum-of-exp via lasso) | `fit_mbpp_ic_sumexp` (fixed θ-bank, fit κ weights) |
| **Forecasting** | — | `forecast_counts` |
| **Standard errors** | observed information | quasi-Poisson observed-information SEs |
| **Goodness of fit** | time-rescaling KS (`ks_test_exp1`) | Pearson `dispersion`, residuals (`gof.py`) |
| **Bayesian posterior** | — | `fit_mbpp_bayes`, `fit_mbpp_bayes_hierarchical` |

## How we solve the forward model

| Situation | Method | Function | Status |
|---|---|---|---|
| Exponential kernel, any baseline | closed-form impulse response | `MBPP.intensity` / `.compensator` | Exact |
| General kernel (power-law) | finite convolution sum | `MBPP(method="numeric")` | Approx. (O(dt)) |
| Sum-of-exponentials, any forcing | state-space ODE | `solve_mbpp_ode` | Exact |
| Multivariate, exponential | M²-state ODE | `solve_mbpp_ode_multivariate` | Exact (=stationary law to 1e-15) |
| Convolution operator (frequency) | transfer function `1/(1-φ̂)` | `SpectralOperator`, `FunctionalMBPP` | Exact |
| Excitation covariates (exp. shape) | linear time-varying ODE (exact per-regime) | `solve_mbpp_ltv` | Exact (=LTI to 1e-16 when δ=0) |
| Excitation covariates (general shape) | Volterra–Nyström quadrature | `solve_mbpp_volterra` | Approx. (O(dt)) |
| Multi-timescale, constant baseline | closed-form matrix exponential | `_sumexp_compensator_const` | Exact (=ODE to 1e-5) |
| Any non-closed model, fast + differentiable | physics-informed neural solver (PINN/PINO) | `neural_solver.make_neural_solver` | Reference impl. (residual validated in numpy; needs JAX/TF to train) |

## Package layout

The package is organised into three subpackages (by data regime / role) plus a
shared optimiser. The full public API is re-exported at the top level, so
`from hawkes_calibration import X` works for every public `X`.

```
hawkes_calibration/
    __init__.py            re-exports the whole public API
    optim.py               shared minimal numpy BFGS (accepts an analytical gradient)

    eventtime/             # --- event-time calibration (exact timestamps) ---
        simulate.py        exact simulation via Ogata thinning
        likelihood.py      log-likelihood + analytical gradient, vectorised per-pair
        estimate.py        MLE wrappers + asymptotic SEs from observed information
        lasso.py           L1-penalised MLE via FISTA (sparse high-M elicitation)
        covariates.py      piecewise-constant covariates, closed-form baseline integral

    mbpp/                  # --- interval-censored calibration via the MBPP ---
        core.py            MBPP intensity & compensator; closed-form + numeric solvers; kernels
        exogenous.py       baseline functions (constant/rectangle/sine/Dassios/
                           multi-impulse/LHPP) + log-linear CovariateExogenous
        ic_simulate.py     separable + covariate-modulated-excitation simulators; censoring
        interval_censored.py  IC-LL & SSE losses; all MBPP fitters; SEs; forecasting
        gof.py             goodness-of-fit: time-rescaling KS, Pearson dispersion, residuals
        bayes.py           Bayesian posterior over (κ,θ): adaptive Metropolis, Laplace,
                           hierarchical pooling, priors, R-hat

    operators/             # --- functional / operator views + learned solvers ---
        linear.py          exact solvers: ODE/state-space (incl. multivariate), Volterra, spectral
        nn.py              numpy DeepONet surrogate + amortized inference net
        tf.py              TensorFlow (optional): high-dim FNO / DeepONet / state-space /
                           amortized kernel inference
        neural_solver.py   residual + family_residual (numpy, tested) + dispatcher + reference
        neural_solver_jax.py   JAX PINN (JAXNeuralMBPP) + PINO (JAXDeepONetPINO)   [needs jax]
        neural_solver_tf.py    TF  PINN (TFNeuralMBPP)  + PINO (TFDeepONetPINO)    [needs tensorflow]
experiments/   exp1–exp11 + tf_lab (synthetic-data noise-stress harness)
results/       figures and saved estimates from the experiments
tests/         the executable ground truth for every numerical claim
paper/         textbook.tex / textbook.pdf — the single mathematical reference
```

Only `eventtime`, `mbpp`, and `operators` (its `linear` + `nn` parts) load on
`import hawkes_calibration`; the TensorFlow/JAX backends (`operators/tf.py`,
`operators/neural_solver*.py`) are imported explicitly so the core stays
dependency-light.

## Usage recipes

### Baseline covariates (interval-censored)

A log-linear baseline `s(t) = exp(γ₀ + γᵀX(t))` recovers the covariate coefficients
jointly with the kernel from counts alone. A piecewise-constant covariate keeps
`s(t)` piecewise-constant, so the closed-form MBPP solver applies unchanged.

```python
from hawkes_calibration import PiecewiseConstantCovariate, fit_mbpp_ic_covariates
cov = PiecewiseConstantCovariate(breakpoints, values)        # values: (K, p)
res = fit_mbpp_ic_covariates(obs_times, counts, cov, loss="ic-ll")
print(res.gamma0, res.gamma, res.kappa, res.theta)
```

### Excitation covariates, multi-timescale, forecasting

```python
from hawkes_calibration import fit_mbpp_ic_excitation, fit_mbpp_ic_sumexp, forecast_counts
res_e = fit_mbpp_ic_excitation(obs, counts, Z, endogenous=True)   # recovers δ (non-convex)
res_s = fit_mbpp_ic_sumexp(obs, counts, exogenous, thetas=[0.3, 1.0, 3.0])  # fit κ-bank
future = forecast_counts(res, future_obs_times)
```

### Bayesian posterior over (κ, θ)

The branching ratio `κ` is well identified from counts but the decay `θ` is weak.
A posterior reports that honestly (a correlated κ–θ ridge), lets an informative
prior tighten `θ`, and — hierarchically — borrows strength across short series.

```python
from hawkes_calibration import fit_mbpp_bayes, fit_mbpp_bayes_hierarchical, GaussianPrior
res = fit_mbpp_bayes(obs, counts, exogenous, method="mcmc", n_chains=4)
print(res.summary())              # means, 95% credible intervals, R-hat, corr(κ,θ)

prior = GaussianPrior(means=[0.0, 0.0], sds=[1.5, 0.12])   # tight prior on log θ
res_t = fit_mbpp_bayes(obs, counts, exogenous, prior=prior)
res_h = fit_mbpp_bayes_hierarchical(obs, counts_list, exogenous_list, theta=1.0)  # pooling
```

`method="laplace"` gives a fast Gaussian approximation (inverse-Hessian covariance);
`method="mcmc"` runs adaptive (Haario) Metropolis with R-hat convergence checks.

### Operator views of the MBPP (numpy)

The solution map `s ↦ ξ` is a linear, translation-invariant operator, so it can be
solved or *learned* in interchangeable ways:

```python
from hawkes_calibration import FunctionalMBPP, ExponentialKernel, SpectralOperator
ker = ExponentialKernel(kappa=0.6, theta=0.8)
xi  = FunctionalMBPP(ker, method="ode").solve(forcing_fn, t_grid)        # exact ODE
xi  = FunctionalMBPP(ker, method="spectral").solve(forcing_fn, t_grid)   # exact Fourier
sp  = SpectralOperator(t_grid).fit(S, XI); t, phi_hat = sp.recover_kernel()  # LEARN it
```

`DeepONetOperator` (forward map, nonlinear/no-closed-form regime) and
`AmortizedInference` (inverse map `counts ↦ (κ,θ)`) round out the numpy toolkit.

### Neural PDE solver — fast, differentiable, model-agnostic (JAX / TF)

For a general non-closed-form model, a pure-Python ODE inside the optimiser is slow:
each BFGS step costs `(1 + 2·n_params)` Python time-stepping solves (finite-difference
gradient). A **physics-informed network** `N(t, p) ≈ ξ(t; p)` trained once on the MBPP
**collocation residual**

```
R(t, p) = N(t, p) − s(t; p) − ∫₀ᵗ K(t,u; p) N(u, p) du        (=0 at the solution)
```

replaces that inner cost with a single vectorised network call plus autodiff
gradients — on GPU, with no per-model derivation. Because the residual is *linear in
ξ*, its global minimum provably equals the exact MBPP solution (validated in numpy:
`solve_mbpp_residual_linear` matches the Volterra solve to 1e-9).

```python
from hawkes_calibration.operators.neural_solver import make_neural_solver, sample_covariate_paths
import numpy as np

t = np.linspace(0, 60, 128)
Z = ((np.floor(t/10) % 2) - 0.5)[:, None]                  # a regime indicator
solver = make_neural_solver(backend="jax", mode="pinn", coll_grid=t, Z_on_grid=Z, T=60., n_delta=1)
solver.train(n_steps=20000, batch=64)                       # one-off
xi = solver.solve(dict(kappa=0.4, theta=1.0, mu=2.0, delta=[1.2]), t)
f  = solver.intensity_fn()                                  # jitted, differentiable in p — drop into a JAX/optax IC-LL fit
```

`mode="pino"` trains a **Physics-Informed Neural Operator** (DeepONet + residual over a
*distribution* of covariate paths and parameters) — one operator solves any instance
in the family in a single pass, no retraining per `Z`. All four trainers accept hybrid
stabilisers `anchors=(...)`, `curriculum=True`, `reweight=True` for the stiff `κ→1`
regime, and `solver_accuracy_report` bins error by `κ` to show where it degrades.
`backend="numpy"` returns the exact (non-neural) reference behind the same interface.

### High-dimensional learning + noise-stress lab (TensorFlow, optional)

For a multivariate process `ξ(t) ∈ ℝᴹ` with an M×M kernel and large training sets,
`operators/tf.py` provides Keras operators batching over `(batch, T, M)` tensors via
`tf.data` (GPU-ready); the exact multivariate ground truth is numpy
(`solve_mbpp_ode_multivariate`).

```python
from hawkes_calibration.operators.tf import (
    FourierNeuralOperator, generate_operator_dataset, train_operator)
S, XI = generate_operator_dataset(n_samples=20000, M=20, seq_len=128)
fno = FourierNeuralOperator(seq_len=128, n_channels=20, width=64, modes=32)
train_operator(fno, S, XI, epochs=100, batch_size=128)      # learn s ↦ ξ
```

Models: `FourierNeuralOperator`, `MBPPDeepONet`, `StateSpaceMBPP` (learned stable
state-space evolution), `AmortizedKernelInference` (`counts ↦` M×M branching matrix).

`experiments/tf_lab.py` is a numpy-only synthetic-data lab for finding **where
learning breaks under noise**. It builds instances across forcing classes
(`pc / sine / impulse / smooth / bursty`), scales (`small / medium / large`), and
noise models (`gauss / mult / poisson / missing`), trains on noisy targets, and
evaluates against the clean test set. `noise_sweep(...)` returns the breaking point
(first level where clean-target error exceeds 2× the noiseless error) per forcing;
the exact linear `MultivariateSpectralOperator` is the strong reference baseline.

```python
from experiments.tf_lab import run_demo
run_demo(scale="small", noise_kind="gauss")     # numpy baseline, writes results/tf_lab_noise_sweep.png
```

A TF model plugs into `noise_sweep` via a `fit_eval(instance) -> rel_L2_error`
callback (a ready-made FNO example and a suggested 7-step protocol —
reproduce-baseline → match-clean → noise-sweep → forcing-dependence → scale →
architecture-ablations → vary-system — are in the source docstrings).

## Event-time regime (the original package)

For exact timestamps, the multivariate exponential-kernel intensity is
`λ_m(t) = exp(γ_{m,0} + γ_mᵀX(t)) + Σ_j α_{m,j} Σ_{t_{j,k}<t} e^{-β_{m,j}(t - t_{j,k})}`.
We estimate the baseline log-intercepts, covariate coefficients, and the elicitation
matrix `A = (α_{m,j})` by MLE, treating the decay matrix `β` as fixed (`α`, `β` are
weakly jointly identified). Key implementation points:

- **O(N·M) likelihood** via the per-pair kernel recursion `R_k = e^{-βΔ}R_{k-1} + (new terms)`,
  vectorised with `searchsorted` and chunked for numerical stability — ~15–20× faster
  than a per-event Python loop.
- **Analytical gradient** accumulated in the same forward pass as the log-likelihood,
  so each BFGS step is one likelihood-plus-gradient pass instead of `~2n` evaluations
  (finite-difference verified to ~1e-7).
- **Sparse high-M** estimation via FISTA with a one-sided soft-threshold prox
  (`max(z - τλ, 0)`, exact zeros), `λ` chosen by BIC — `fit_multivariate_lasso`.
- **Standard errors** from the inverse observed-information matrix.

In `exp4_highdim` (M=12, ~14k events) BIC-tuned lasso recovers edges at precision
0.74 / recall 0.94 (vs 0.61 / 0.79 unpenalised) and drives true-zero MAE from 0.016
to 0.002. The full derivation, multivariate identifiability, and generalisations
(sum-of-exp, power-law, marked/spatial, inhibition, non-parametric kernels) are in
the textbook.

## Experiments

Each writes a `.png` and a `.json`/`.npz` to `results/`.

```
exp1_1d                 univariate Hawkes — bias/variance vs horizon T
exp2_multidim           3D Hawkes — recover sparse elicitation matrix
exp3_covariates         2D Hawkes with a regime-switching baseline covariate
exp4_highdim            M=12 sparse network — MLE vs lasso, BIC λ-path
exp5_mbpp_impulse       MBPP intensity = mean Hawkes intensity (paper Figs 2–3)
exp6_ic_scenarios       recover (κ,θ) from interval-censored counts
exp7_forecast           interval-censored popularity forecasting
exp8_ic_covariates      interval-censored baseline with time-varying covariates
exp9_functional_operators   ODE / spectral / DeepONet / amortized operator views
exp10_deep              excitation covariates, multi-timescale, goodness-of-fit
exp11_bayes             Bayesian calibration — the ridge, priors, hierarchical pooling
tf_lab                  TF/operator noise-stress sweep (numpy baseline)
```

```bash
cd hawkes_calibration && python -m experiments.exp6_ic_scenarios
```

## Honest caveats (identifiability)

- **`κ` vs `θ`.** The branching ratio `κ` is well identified from counts; the decay
  `θ` is weak — a censoring bucket integrates away the timing that carries `θ`. Trust
  `κ` and covariate effects `γ`; fix or prior-constrain `θ`, and report wide intervals
  for it. The Bayesian posterior (`bayes.py`) and `paper/textbook.pdf` Ch. 8 make this
  precise (the κ–θ ridge as a Fisher-information degeneracy).
- **Excitation covariates `δ`.** The fitter is fast (exact per-regime, ~0.5 s) and
  recovers `δ` cleanly on MBPP-Poisson data; on genuine over-dispersed Hawkes data it
  is recovered on average but with real spread — the loss is non-convex and `δ` is
  weakly identified from little data.
- **Multi-timescale weights.** `fit_mbpp_ic_sumexp` recovers the *total* branching
  ratio well; the split across the θ-bank is weakly identified — use the L1 penalty
  and domain priors.
- **Over-dispersion.** True Hawkes counts are more variable than the MBPP's Poisson
  assumption. Check `dispersion()` (≈1 good, >1 clustering) and use the quasi-Poisson
  SEs (inflated by √dispersion).

## Tests

```bash
cd hawkes_calibration
PYTHONPATH=. python tests/test_interval_censored.py     # 11
PYTHONPATH=. python tests/test_operators.py             # 11
PYTHONPATH=. python tests/test_deep.py                  # 7
PYTHONPATH=. python tests/test_neural_solver.py         # 8 (numpy core of the neural solver)
PYTHONPATH=. python tests/test_bayes.py                 # 4
```

## Theory

The single mathematical reference is **[`paper/textbook.pdf`](paper/textbook.pdf)**
(`latexmk -pdf paper/textbook.tex` to rebuild). It is organised by increasing
complexity: foundations (Poisson, Hawkes) → the MBPP (solving, fitting,
identifiability, Bayesian inference) → covariates (baseline, excitation,
multi-timescale) → computation and learning (solvers, neural operators) → a
package tour, with appendices on Volterra equations, transforms/ODEs, and estimation
theory (Bregman/KL, MLE asymptotics, Bernstein–von Mises, the κ–θ ridge).
