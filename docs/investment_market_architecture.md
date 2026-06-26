# Investment-market architecture: sector Hawkes + startup ranker

This note documents the modeling decision and implementation added in PR #1. The goal is to model weekly startup-funding data without forcing firm-level self-inhibition into a signed Hawkes kernel.

## Motivation

We considered two ways to model funding events:

1. **Firm-level Hawkes process**: each startup is a dimension. A recent funding event for the same startup should reduce the chance of another funding event for a while. This creates a negative diagonal/self-effect.
2. **Sector-level marked process**: sectors are the Hawkes dimensions. A funding event in a sector increases attention and liquidity in that sector, making another sector event more likely. Once the sector event occurs, a separate ranker chooses the startup inside that sector.

The second approach is cleaner for the current data and model class. Sector self-excitation is positive and economically natural, while firm-level cooldown is handled as a covariate in the startup ranker.

The resulting factorization is:

```text
p(time, sector, startup)
  = p(time, sector) * p(startup | sector, time, history)
```

This is a marked-event model. The sector process decides **where and when capital flows**. The ranker decides **which startup receives the mark**.

## Why not negative firm-level Hawkes self-excitation?

The existing Hawkes code parameterizes excitation with positive coefficients, for example `alpha = exp(a)` in the event-time fitter. The optimizer variable `a` is unconstrained, but the realized excitation coefficient is still positive. A very negative `a` means near-zero self-excitation, not self-inhibition.

True self-inhibition would require something like:

```text
lambda_i(t) = [ baseline_i(t) - rho_i * self_history_i(t)
                + positive_cross_excitation_i(t) ]_+
```

or a log-intensity model with a negative self-history term. Those models are valid modeling choices, but they break the clean linear Hawkes / MBPP mean-field structure because expectation and the nonlinear positivity link do not commute.

Instead, we encode firm-level suppression through an observed cooldown covariate:

```text
cooldown_i,t = 1[startup i received funding in the previous K weeks]
```

and give it a negative coefficient inside the ranker. This is economically interpretable: recently funded companies are temporarily less at risk of raising again.

## Layer 1: weekly sector count model

The sector process is a discrete-time Hawkes-style Poisson GLM over weekly counts.

Let:

- `Y[s, t]` be the number of funding events in sector `s` during week `t`;
- `X[t]` be weekly exogenous market covariates;
- `L` be the number of lag weeks;
- `b[s, r, l]` be the nonnegative effect from sector `r` at lag `l` into sector `s`.

The model is:

```text
Y[s,t] ~ Poisson(Lambda[s,t])

log Lambda[s,t]
  = a[s]
    + beta[s]' X[t]
    + sum_r sum_l b[s,r,l] * Y[r,t-l]
```

By default, the lag excitation coefficients are constrained nonnegative:

```text
b[s,r,l] >= 0
```

This gives Hawkes-like sector momentum while staying aligned with weekly data.

### Structural zeros

The fitter accepts an optional sector adjacency mask. If `adjacency[s, r] = False`, then all lag coefficients from source sector `r` to receiving sector `s` are forced to zero:

```text
b[s,r,l] = 0 for all l
```

This lets us impose known sector relationships or sparse hypotheses.

## Layer 2: dynamic startup risk-set ranker

After the sector model produces an event in sector `s` at week `t`, the startup is selected from the current live risk set:

```text
R[s,t] = active startups in sector s at week t
```

Each candidate startup `i` receives a score:

```text
q[i,t]
  = (w0 + u[s])' Z[i,t]
    + eta[s] * cooldown[i,t]
```

where:

- `Z[i,t]` are startup-level features;
- `w0` is a global feature-weight vector shared across sectors;
- `u[s]` is a sector-specific deviation;
- `cooldown[i,t]` indicates whether startup `i` raised in the last `K` weeks;
- `eta[s] <= 0` is the sector-specific cooldown coefficient.

The conditional choice probability is:

```text
p(i | s,t)
  = exp(q[i,t]) / sum_{j in R[s,t]} exp(q[j,t])
```

This is the crucial design choice: the softmax normalizes over the **current sector risk set**, not over a fixed list of all startups. Therefore the number of startups per sector can change over time. New startups enter by becoming active; inactive or dead startups leave the risk set.

## Training objective for the ranker

For an observed event `(t, s, i*)`, the ranker minimizes the negative conditional log-likelihood:

```text
-loss = log p(i* | s,t)
      = q[i*,t] - log sum_{j in R[s,t]} exp(q[j,t])
```

Equivalently, the fitted loss is:

```text
NLL = sum_events [ logsumexp(q[j,t] for j in R[s,t]) - q[i*,t] ]
```

When several startups are funded in the same sector-week, this prototype treats them as repeated conditional choices over the same weekly risk set. In simulation, after a startup is sampled, its cooldown/history is updated before the next event is sampled.

## Simulation loop

A simulated path proceeds week by week:

1. Start from the observed sector and startup histories up to `train_end`.
2. For week `t`, compute sector rates from the sector count model.
3. Sample sector counts:

   ```text
   Y[:,t] ~ Poisson(Lambda[:,t])
   ```

4. For each sampled event in sector `s`, build the current risk set `R[s,t]`.
5. Score startups with the ranker and sample one startup mark.
6. Update startup history and cooldown state.
7. Move to the next week.

Exogenous covariates `X[t]` and startup features `Z[i,t]` are taken as given in the synthetic experiment. In production, future covariates should come from scenarios, forecasts, or bootstrapped historical paths.

## Files added

### `hawkes_calibration/sector_ranker.py`

Main implementation:

- `fit_sector_count_model`
- `fit_startup_ranker`
- `ranker_predict_proba`
- `evaluate_ranker`
- `simulate_synthetic_startup_market`
- `simulate_marked_paths`
- data containers for synthetic data and fitted models

### `hawkes_calibration/sector_backtest.py`

Clean end-to-end synthetic backtest wrapper:

- generates data;
- fits the sector model;
- fits the startup ranker;
- evaluates held-out sector NLL;
- evaluates held-out ranker NLL, MRR, and top-k hit rates;
- simulates marked future paths.

Important detail: post-training event histories are zeroed before simulation so simulated cooldowns do not accidentally see future held-out events.

### `experiments/exp14_sector_ranker.py`

Runnable experiment:

```bash
PYTHONPATH=. python -m experiments.exp14_sector_ranker
```

It writes metrics to:

```text
results/exp14_sector_ranker.json
```

### `tests/test_sector_ranker.py`

Coverage for:

- synthetic market shapes;
- dynamic risk sets;
- sector model fitting;
- nonnegative excitation constraints;
- ranker fitting;
- negative cooldown constraint;
- probability normalization over the risk set;
- end-to-end synthetic backtest.

### `docs/sector_ranker.md`

Short practical API reference.

## Public API

The package `__init__` exports the prototype functions:

```python
from hawkes_calibration import (
    simulate_synthetic_startup_market,
    fit_sector_count_model,
    fit_startup_ranker,
    evaluate_ranker,
    simulate_marked_paths,
    backtest_synthetic_pipeline,
)
```

## How to run

Run tests:

```bash
PYTHONPATH=. python tests/test_sector_ranker.py
```

Run the synthetic experiment:

```bash
PYTHONPATH=. python -m experiments.exp14_sector_ranker
```

## Metrics reported by the synthetic backtest

The backtest reports:

- total, train, and test event counts;
- sector held-out Poisson NLL per sector-week cell;
- historical-mean baseline sector NLL;
- improvement of the fitted sector model over baseline;
- simulation MAE for sector-week counts;
- baseline sector MAE;
- ranker held-out NLL;
- random-risk-set ranker NLL baseline;
- ranker MRR;
- ranker top-1, top-5, and top-10 hit rates.

## Current limitations

This is a first implementation pass. The main limitations are:

1. The sector process is a discrete-time Poisson GLM, not a continuous-time Hawkes likelihood.
2. The ranker currently uses a simple conditional softmax, not a full Plackett-Luce likelihood for multiple events in the same sector-week.
3. Future exogenous covariates are assumed known in the synthetic simulation.
4. The sector count model uses Poisson noise; real funding data may be overdispersed, so a negative-binomial extension is likely useful.
5. The synthetic tests use fixed random seeds and should be checked locally for runtime and robustness.
6. The ranker uses linear features. In production this can be replaced by a gradient-boosted model, neural scorer, or learned embeddings as long as it still normalizes over the current risk set.

## Next steps

Good next implementation steps:

1. Add a negative-binomial sector count model for overdispersed weekly funding counts.
2. Add Plackett-Luce training for multiple startups funded in the same sector-week.
3. Add scenario generators for future market covariates.
4. Add sector adjacency presets and sparse regularization paths.
5. Add calibration plots: sector count reliability, ranker top-k curves, and simulated event fan charts.
6. Add an adapter for real startup funding tables with columns such as `week`, `startup_id`, `sector`, `stage`, `round_size`, and `features`.
