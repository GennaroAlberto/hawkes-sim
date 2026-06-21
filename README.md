# hawkes_calibration

A small Python package for calibrating Hawkes processes in **two data regimes**:

1. **Event-time data** (exact timestamps observed). Maximum-likelihood
   calibration of multivariate Hawkes processes with exponential kernel and
   (optional) time-varying covariates in the baseline. The central object is the
   *elicitation matrix* $A=(\alpha_{m,j})$ that encodes how events of component
   $j$ trigger events of component $m$. The mathematical specification,
   likelihood derivation and identifiability discussion are in
   [`NOTES.md`](./NOTES.md).

2. **Interval-censored data** (only aggregate counts per interval observed — e.g.
   hospital admissions per day, video views per day). The Hawkes likelihood
   cannot be evaluated without event times, so we fit through the **Mean Behavior
   Poisson process (MBPP)**, following Rizoiu et al. (2022),
   *"Interval-censored Hawkes processes"* (JMLR 23(1)). A detailed, proof-level
   summary of that paper and how it maps to this code is in
   [`PAPER_SUMMARY.md`](./PAPER_SUMMARY.md).

## What this estimates

For an $M$-dimensional point process with intensity

$$
\lambda_m(t) = \exp\!\big(\gamma_{m,0} + \gamma_m^{\top} X(t)\big)
\;+\; \sum_{j=1}^{M}\alpha_{m,j}\!\!\sum_{t_{j,k}<t}\!\! e^{-\beta_{m,j}(t-t_{j,k})},
$$

the package estimates by MLE the baseline log-intercepts $\gamma_{m,0}$,
covariate coefficients $\gamma_m$, and the elicitation matrix $A$, treating
the decay matrix $\beta$ as fixed (see NOTES §3 for why).

## Package layout

```
hawkes_calibration/
    # --- event-time calibration ---
    simulate.py         exact simulation via Ogata thinning
    likelihood.py       log-likelihood + analytical gradient, vectorized per-pair
    estimate.py         MLE wrappers + asymptotic SE from observed information
    lasso.py            L1-penalized MLE via FISTA (sparse high-M elicitation)
    covariates.py       piecewise-constant covariates with closed-form baseline integral
    optim.py            minimal numpy-only BFGS, accepts analytical gradient
    # --- interval-censored calibration (MBPP) ---
    mbpp.py             Mean Behavior Poisson process: intensity & compensator,
                        closed-form (impulse response) + numerical solvers
    exogenous.py        exogenous functions: constant/rectangle/sine/Dassios,
                        multi-impulse (Def 12) and latent-Poisson LHPP (Def 14)
    ic_simulate.py      separable (cluster) simulator + interval-censoring
    interval_censored.py  IC-LL & SSE losses, MBPP fitting, forecasting (Eq. 55)
    # --- functional / operator views of the MBPP ---
    operators.py        ODE/state-space reduction (incl. multivariate) + spectral operator
    operators_nn.py     numpy DeepONet surrogate + amortized inference net
    operators_tf.py     TensorFlow (optional): high-dim multivariate FNO / DeepONet /
                        state-space evolution / amortized kernel inference
experiments/
    exp1_1d.py          univariate Hawkes -- bias / variance vs horizon T
    exp2_multidim.py    3D Hawkes, recovery of sparse elicitation matrix
    exp3_covariates.py  2D Hawkes with regime-switching covariate
    exp4_highdim.py     M=12 sparse network -- MLE vs lasso, BIC lambda path
    exp5_mbpp_impulse.py  MBPP intensity vs mean Hawkes intensity (Figs 2-3)
    exp6_ic_scenarios.py  recover (kappa,theta) from interval-censored counts
    exp7_forecast.py    ACTIVE-style interval-censored popularity forecasting
    exp8_ic_covariates.py  interval-censored baseline with time-varying covariates
    exp9_functional_operators.py  ODE / spectral / DeepONet / amortized operator views
results/
    figures and saved estimates from the experiments
NOTES.md                event-time math + identifiability + scaling + generalizations
PAPER_SUMMARY.md        full proof-level summary of the interval-censored paper
```

## Interval-censored calibration in 30 seconds

```python
import numpy as np
from hawkes_calibration import (
    LHPP, simulate_separable_hawkes, interval_censor, uniform_obs_times, fit_mbpp_ic,
)

# simulate a Hawkes cascade (immigrants + offspring kept separate)
from hawkes_calibration import Constant
imm, off = simulate_separable_hawkes(Constant(3.0, 30.0), kappa=0.5, theta=1.0, T=30.0, seed=0)

# observe ONLY interval counts (e.g. per-day volumes), not event times
obs = uniform_obs_times(30.0, 30)
off_counts = interval_censor(off, obs)            # offspring counts per interval
imm_counts = interval_censor(imm, obs)            # immigrant counts per interval

# fit the branching ratio kappa and decay theta from the counts alone
res = fit_mbpp_ic(obs, off_counts, LHPP(obs, imm_counts), loss="ic-ll", endogenous=True)
print(res.summary())                              # kappa ~ 0.5, theta ~ 1.0
```

## Running the experiments

Only numpy and matplotlib are required.

```
cd hawkes_calibration
python -m experiments.exp1_1d
python -m experiments.exp2_multidim
python -m experiments.exp3_covariates
python -m experiments.exp4_highdim       # M=12, MLE vs lasso, BIC path
python -m experiments.exp5_mbpp_impulse  # MBPP = mean Hawkes intensity (Figs 2-3)
python -m experiments.exp6_ic_scenarios  # recover kappa,theta from interval counts
python -m experiments.exp7_forecast      # interval-censored popularity forecasting
python -m experiments.exp8_ic_covariates # interval-censored baseline with covariates
python -m experiments.exp9_functional_operators  # ODE / spectral / DeepONet / amortized
```

### Functional / operator views of the MBPP

The MBPP solution map `s ↦ ξ` is a linear, translation-invariant operator, so it
can be solved or *learned* in interchangeable ways (all numpy):

```python
from hawkes_calibration import FunctionalMBPP, ExponentialKernel, SpectralOperator

ker = ExponentialKernel(kappa=0.6, theta=0.8)
# (1) exact state-space ODE reduction — any forcing, any sum-of-exp kernel
xi = FunctionalMBPP(ker, method="ode").solve(forcing_fn, t_grid)
# (2) exact Fourier operator  R(w) = 1/(1 - phi_hat(w))
xi = FunctionalMBPP(ker, method="spectral").solve(forcing_fn, t_grid)
# (3) LEARN the operator from (forcing, response) pairs, then recover the kernel
sp = SpectralOperator(t_grid).fit(S, XI);  t, phi_hat = sp.recover_kernel()
```

A `DeepONetOperator` (forward map `s↦ξ`, for the nonlinear/no-closed-form regime)
and an `AmortizedInference` net (inverse map `counts↦(κ,θ)`) round out the
operator toolkit. The state-space reduction is exact precisely when the kernel
has a rational Laplace transform (a sum of exponentials); otherwise use the
spectral operator or a sum-of-exponentials approximation. See `PAPER_SUMMARY.md`
§9 for the math.

### High-dimensional / large-data learning (TensorFlow, optional)

For a **multivariate** process — intensity vector `ξ(t) ∈ ℝᴹ` with an M×M kernel
matrix — and large training sets, `operators_tf.py` provides Keras operators that
batch over `(batch, T, M)` tensors via `tf.data` (GPU-ready). The exact
multivariate ground-truth solver is numpy:

```python
import numpy as np
from hawkes_calibration import solve_mbpp_ode_multivariate
A = np.array([[0.3,0.1,0.0],[0.0,0.25,0.15],[0.1,0.0,0.2]])   # kernel weights
B = np.ones((3,3))                                            # decays
t = np.linspace(0, 80, 8001)
xi = solve_mbpp_ode_multivariate(lambda tt: np.array([0.5,0.3,0.4]), A, B, t)  # (N, 3)
```

```python
# requires `pip install tensorflow` (kept out of __init__ so the core stays numpy-only)
from hawkes_calibration.operators_tf import (
    FourierNeuralOperator, StateSpaceMBPP, AmortizedKernelInference,
    generate_operator_dataset, train_operator,
)
M, T = 20, 128
S, XI = generate_operator_dataset(n_samples=20000, M=M, seq_len=T)   # high-dim, lots of data
fno = FourierNeuralOperator(seq_len=T, n_channels=M, width=64, modes=32)
train_operator(fno, S, XI, epochs=100, batch_size=128)              # learn s ↦ ξ
```

Models: `FourierNeuralOperator` (multivariate FNO), `MBPPDeepONet`,
`StateSpaceMBPP` (a learned, stable state-space model that integrates the
*evolution* forced by `s(t)`), and `AmortizedKernelInference` (`counts ↦` M×M
branching matrix). See `PAPER_SUMMARY.md` §10.

### Time-varying covariates in the interval-censored baseline

The interval-censored fitter supports a log-linear covariate baseline
`s(t) = exp(gamma0 + gamma^T X(t))`, recovering the covariate coefficients
jointly with the kernel from counts alone — the interval-censored analogue of the
event-time covariate model. Because a piecewise-constant covariate keeps `s(t)`
piecewise-constant, the closed-form MBPP solver applies unchanged.

```python
from hawkes_calibration import (
    PiecewiseConstantCovariate, CovariateExogenous, fit_mbpp_ic_covariates,
)
# X(t): a regime indicator (piecewise-constant), e.g. switching every 20 units
cov = PiecewiseConstantCovariate(breakpoints, values)        # values: (K, p)
res = fit_mbpp_ic_covariates(obs_times, counts, cov, loss="ic-ll")
print(res.gamma0, res.gamma, res.kappa, res.theta)           # recovered from counts
```

Each script writes a `.png` figure and either a `.json` or `.npz` file with
the raw estimates into `results/`.

## Headline results (synthetic data)

**Experiment 1 (1D)** — true $(\mu, \alpha, \beta) = (0.5, 0.4, 1.0)$, 20
replications per horizon. At $T = 5000$, $\hat\mu = 0.506 \pm 0.012$ and
$\hat\alpha = 0.389 \pm 0.016$ — bias is essentially zero already at
$T = 200$ and standard deviation contracts as $\sim 1/\sqrt T$. See
`results/exp1_1d.png`.

**Experiment 2 (3D)** — sparse elicitation matrix recovered cleanly at
$T = 5000$ from 7,606 events. Every zero entry of $A$ is estimated within
0.03 of zero and every non-zero entry within 0.03 of its true value, all
within ~1 standard error. See `results/exp2_multidim_heatmap.png` and
`results/exp2_multidim_scatter.png`.

**Experiment 3 (2D + covariates)** — regime indicator switches the baseline
every 200 time units. Both the covariate slopes ($\gamma_0 = 0.8$,
$\gamma_1 = -0.5$ recovered as $0.80$ and $-0.53$) and the elicitation
matrix are recovered jointly. See `results/exp3_covariates.png`.

**Experiment 4 (M = 12, sparse network)** — 144-entry elicitation matrix with
true density ≈ 1/3, 14k events at $T = 2500$. The unpenalized MLE (156
parameters in $\sim 12$s) recovers magnitudes well but the $2\sigma$ edge test
gives precision $0.61$ / recall $0.79$. BIC-tuned lasso at $\lambda = 80$ gives
precision $0.74$ / recall $0.94$ and drives the mean absolute error on the
true-zero entries from $0.016$ to $0.002$ — true zeros become exact zeros.
See `results/exp4_highdim_heatmap.png`, `results/exp4_highdim_scatter.png`,
and `results/exp4_highdim_lasso_path.png`.

## Scaling and generalizations

`NOTES.md` §6–§9 cover what's needed to scale (vectorized per-pair kernel
sums, analytical gradients, $\ell_1$ regularization) and how to generalize
the model: sum-of-exponentials and power-law kernels, marked / spatial
Hawkes, covariate-dependent elicitation, inhibition (non-linear Hawkes),
non-parametric kernels, Bayesian inference, and goodness-of-fit via the
time-rescaling theorem.

## Verifying the implementation

The simulator and likelihood share the same intensity formula. Two
consistency checks built into the experiments:

1. The empirical event rate in experiment 1 matches the theoretical
   stationary rate $\mu/(1 - \alpha/\beta)$ to within Monte-Carlo error.
2. The MLE log-likelihood is greater at the truth than at perturbed
   parameters (smoke-tested when developing the package).

## Extensions

`NOTES.md` §6 lists natural extensions: joint estimation of $\beta$, $\ell_1$
regularization on $A$ when sparsity in the elicitation network is expected,
non-parametric kernels, Bayesian inference, and goodness-of-fit via time
rescaling.
