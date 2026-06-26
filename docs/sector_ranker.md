# Sector Hawkes + startup risk-set ranker

This prototype implements the two-layer investment-market architecture:

1. **Sector process**: model weekly funding counts by sector with a positive-lag Hawkes/Poisson GLM.
2. **Startup mark model**: after a sector event occurs, choose the startup from the current live risk set with a conditional softmax ranker.

The factorisation is

```text
p(time, sector, startup) = p(time, sector) * p(startup | sector, time, history)
```

This keeps sector self-excitation positive and interpretable: a funding event in a sector can increase the chance of another funding event in the same sector. Firm-level cooldown / self-inhibition lives in the ranker, not in a signed Hawkes kernel.

## Sector count model

For weekly data the model is

```text
Y[s,t] ~ Poisson(Lambda[s,t])
log Lambda[s,t] = a[s] + beta[s]' X[t]
                + sum_r sum_l b[s,r,l] Y[r,t-l]
```

with `b[s,r,l] >= 0` by default. You can pass an adjacency mask to force structural zeros in the sector excitation matrix.

## Dynamic startup risk-set ranker

For a simulated or observed sector event `(week=t, sector=s)`, the candidate set is

```text
R[s,t] = active startups in sector s at week t.
```

Each candidate startup gets a score

```text
q[i,t] = (w0 + u[s])' Z[i,t] + eta[s] * cooldown[i,t]
```

where `cooldown[i,t]` is one if the startup was funded in the previous `K` weeks. The fitted `eta[s]` is constrained non-positive by default.

The choice probability is

```text
p(i | s,t) = exp(q[i,t]) / sum_{j in R[s,t]} exp(q[j,t])
```

Because the denominator is the current risk set, the number of startups per sector can change over time.

## Synthetic backtest

Run:

```bash
PYTHONPATH=. python -m experiments.exp14_sector_ranker
```

This creates a synthetic market, fits both layers, scores the held-out period, simulates marked paths, and writes metrics to:

```text
results/exp14_sector_ranker.json
```

## Public API

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
