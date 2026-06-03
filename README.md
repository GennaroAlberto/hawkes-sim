# hawkes_calibration

A small Python package for calibrating multivariate Hawkes processes with
exponential kernel and (optional) time-varying covariates in the baseline.
The central object is the *elicitation matrix* $A=(\alpha_{m,j})$ that
encodes how events of component $j$ trigger events of component $m$. The
mathematical specification, likelihood derivation and identifiability
discussion are in [`NOTES.md`](./NOTES.md).

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
    simulate.py         exact simulation via Ogata thinning
    likelihood.py       log-likelihood + analytical gradient, vectorized per-pair
    estimate.py         MLE wrappers + asymptotic SE from observed information
    lasso.py            L1-penalized MLE via FISTA (sparse high-M elicitation)
    covariates.py       piecewise-constant covariates with closed-form baseline integral
    optim.py            minimal numpy-only BFGS, accepts analytical gradient
experiments/
    exp1_1d.py          univariate Hawkes -- bias / variance vs horizon T
    exp2_multidim.py    3D Hawkes, recovery of sparse elicitation matrix
    exp3_covariates.py  2D Hawkes with regime-switching covariate
    exp4_highdim.py     M=12 sparse network -- MLE vs lasso, BIC lambda path
results/
    figures and saved estimates from the experiments
NOTES.md                math + identifiability + scaling + generalizations
```

## Running the experiments

Only numpy and matplotlib are required.

```
cd hawkes_calibration
python -m experiments.exp1_1d
python -m experiments.exp2_multidim
python -m experiments.exp3_covariates
python -m experiments.exp4_highdim   # M=12, MLE vs lasso, BIC path
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
