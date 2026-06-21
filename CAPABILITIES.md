# What we can model right now

A status reference for `hawkes_calibration`. "Solid" = exact/closed-form or
validated to machine precision; "Approx." = convergent numerical scheme;
"Hard" = works but with non-convex fitting / weak identifiability.

## The model we can specify

A multivariate self-exciting process with intensity, for component `m`,

```
lambda_m(t) = s_m(t)  +  sum_j  alpha_mj(t) * g_mj(t - u)  summed over past events u of j
```

with each ingredient now optionally covariate-driven:

| Ingredient | Forms available | Covariate-driven? |
|---|---|---|
| Baseline `s_m(t)` (immigrants) | constant, piecewise-constant, sine, Dassios, rectangle, multi-impulse, LHPP | **yes** — log-linear `exp(gamma0 + gamma^T X(t))` |
| Kernel shape `g_mj` | exponential (closed form), sum-of-exponentials, power-law (numeric) | — |
| Excitation strength `alpha_mj(t)` | constant `kappa*theta` | **yes** — `alpha0 * exp(delta^T Z(t))` (time-varying) |

## Observation regimes × what we can fit

| | Event times observed | Interval-censored (counts only) |
|---|---|---|
| **Univariate** | MLE of `(mu, alpha, beta)` — `fit_univariate` | IC-LL / SSE of `(kappa, theta)` — `fit_mbpp_ic` |
| **Multivariate (M-dim)** | MLE of elicitation matrix `A`, analytic gradient, SEs — `fit_multivariate`; sparse via lasso/BIC — `fit_multivariate_lasso` | IC-LL of `(kappa,theta)` per pair — `fit_mbpp_ic_multi` |
| **+ baseline covariates** | `fit_multivariate_with_covariates` | `fit_mbpp_ic_covariates` (recovers `gamma`) |
| **+ excitation covariates** | (forward model only) | `fit_mbpp_ic_excitation` (recovers `delta`) — **new** |
| **Forecasting** | — | `forecast_counts` (Eq. 55) |

## How we solve the forward model (compute the mean intensity / compensator)

| Situation | Method | Function | Status |
|---|---|---|---|
| Exponential kernel, any baseline | closed-form impulse response | `MBPP.intensity/compensator` | Solid |
| General kernel (power-law) | finite convolution sum | `MBPP(method="numeric")` | Approx. (O(dt)) |
| Sum-of-exponentials, any forcing | state-space ODE | `solve_mbpp_ode` | Solid |
| Multivariate, exponential | M^2-state ODE | `solve_mbpp_ode_multivariate` | Solid (=stationary law to 1e-15) |
| Convolution operator in frequency | transfer function `1/(1-phi_hat)` | `SpectralOperator`, `FunctionalMBPP` | Solid |
| **Excitation covariates (exp. shape)** | **linear time-varying ODE** | **`solve_mbpp_ltv`** | **Solid (=LTI to 1e-16 when delta=0)** |
| Excitation covariates (general shape) | Volterra quadrature (Nyström) | (documented, LaTeX App. A) | Planned |

## Operator learning / scaling (optional)

`SpectralOperator` (learn the operator + recover the kernel), `DeepONetOperator`
and `AmortizedInference` (numpy), and TensorFlow high-dimensional models
(`operators_tf.py`: multivariate FNO, DeepONet, state-space, amortized kernel
inference) with the `tf_lab.py` noise-stress harness.

## Honest caveats (what is *hard* to estimate)

- **`kappa` vs `theta`.** The branching ratio `kappa` is well identified from
  counts; the decay `theta` is weak (a bucket integrates away the timing). Treat
  `kappa` and covariate effects as trustworthy; fix/prior-constrain `theta`.
- **Excitation covariates `delta`.** Recoverable with enough data and covariate
  contrast (e.g. delta=1.2 recovered as ~1.15 from 12 daily sequences over 60
  days), but the loss is **non-convex** and `delta` is weakly identified from
  little data — use many sequences and restarts. Univariate is the supported
  fitter; multivariate excitation covariates add up to `p*M^2` parameters and are
  harder still (the forward solver `solve_mbpp_ltv` is multivariate).
- **Power-law + excitation covariates** needs the (planned) Volterra-quadrature
  path; the LTV-ODE route covers exponential / sum-of-exponential shapes.

See `paper/main.pdf` for the full mathematical treatment (the three regimes:
no covariates → baseline covariates → excitation covariates).
