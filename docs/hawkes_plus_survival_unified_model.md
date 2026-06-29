# Unified model: sector Hawkes + startup survival second stage

This document unifies the investment-market design into one modeling architecture:

```text
weekly macro + market covariates
        ↓
sector Hawkes / count process: where and when funding activity happens
        ↓
startup survival / hazard stage: which firm in that sector is funded
```

The reason for the decomposition is practical and statistical. At the sector level,
funding activity is self-exciting: one AI, fintech, biotech, or climate round can
make another sector event more likely. At the firm level, however, a recently funded
startup is usually less likely to raise again immediately. That firm-level refractory
effect is better modeled as a survival/cooldown covariate than as a negative Hawkes
kernel.

The product output is a ranked watch list:

```text
At decision week t, for horizon h, rank firms by P(funding in (t, t+h] | history).
```

## 1. Data setting

We observe positive funding events. A row exists when a company receives financing;
we do not directly observe a symmetric negative event saying that a firm actively
failed to raise. This makes the problem closer to survival / time-to-event ranking
than to ordinary classification.

The current data sources described in the project notes are:

### FactSet standardized economics

FactSet provides macroeconomic covariates. The currently used universe is the
FactSet Standardized Economics subset, covering thousands of economic series across
90+ countries. For this project the relevant macro features are inflation,
unemployment, and interest-rate series, including examples such as:

```text
FFUNDT, FFUND, CPI_RAW, CPI_YOY, RUNEMP
```

These variables are natural sector-level or market-level covariates, e.g. risk-on /
risk-off conditions, rate pressure, inflation regime, or labor-market tightness.

### PitchBook company, deals, and investors data

PitchBook provides the positive funding events and firm-level features. The fields in
scope include:

- deal identifiers and timing: `company_id`, `deal_id`, `deal_date`;
- sector and industry: `primary_industry_sector`, `primary_industry_group`,
  `primary_industry_code`, `other_industries`;
- financing status and type: `current_financing_status`, `financing_status`,
  `deal_type`, `deal_type_2`, `deal_type_3`, `deal_class`, `vc_round`,
  `vc_round_up_down_flat`, `deal_status`;
- capital and valuation fields: `deal_size`, `deal_size_status`,
  `pre_money_valuation`, `post_valuation`, `post_valuation_status`,
  `raised_to_date`, `total_raised_to_date`, `total_money_raised`,
  `last_financing_size`, `last_financing_valuation`, `last_known_valuation_date`;
- company fundamentals and scale: `number_of_employees`, `employees`, `revenue`,
  `ebitda`, `gross_profit`, `net_income`, `revenue_growth_since_last_deal`,
  `year_founded`;
- geography: `hq_location`, `company_country`, `hq_country`, `hq_global_region`,
  `hq_global_sub_region`;
- investor fields: `investor_id`, `investorname`, `investorstatus`, `investorsince`.

These are primarily used in the startup survival stage, while sector labels and deal
counts feed the sector process.

## 2. Why not a firm-level signed Hawkes process?

A direct firm-level Hawkes model would make every startup a dimension:

```text
lambda_i(t) = baseline_i(t) + sum_j excitation_{i,j}(t)
```

This is appealing because it gives a per-firm intensity. But it creates two problems.

First, the natural firm-level self-effect is negative: if startup `i` just raised, it
is usually less likely to raise again in the next few weeks or months. A classical
Hawkes kernel is nonnegative. A negative diagonal would require a nonlinear positivity
link such as rectification or log intensity. That is possible, but it breaks the clean
linear mean-field / MBPP path used by the interval-censored Hawkes framework.

Second, firm-level covariates are often only updated at financing events. For example,
`last_financing_date`, `last_financing_size`, `last_financing_valuation`, and many
capital-structure features are event-updated. Between funding events, these features
are naturally carried forward. That is exactly the survival-analysis setting: a firm
is at risk, its covariates are observed as-of time `t`, and the event is the next
funding time.

Therefore the decomposition is:

```text
sector-level Hawkes for market contagion
firm-level survival/hazard for startup selection
```

## 3. Layer 1: sector Hawkes / count process

Let there be `S` sectors, currently around 11. Let

```text
Y[s,t] = number of funding events in sector s during week t.
```

Let `M[t]` be FactSet macro covariates and other market-wide variables known at week
`t`. A weekly Hawkes-style count model is:

```text
Y[s,t] ~ Poisson(Lambda[s,t])

log Lambda[s,t]
  = a[s]
    + beta[s]' M[t]
    + sum_r sum_l b[s,r,l] Y[r,t-l]
```

where:

- `a[s]` is the sector baseline;
- `beta[s]` measures sector sensitivity to macro conditions;
- `b[s,r,l] >= 0` is lagged positive excitation from source sector `r` to receiving
  sector `s` at lag `l`.

The diagonal `b[s,s,l]` is allowed and should often be positive. At sector level,
self-excitation means that one funded company in a sector creates attention and
liquidity for other companies in the same sector. This is economically different from
saying the same firm will raise again immediately.

A negative-binomial version is a natural next step if weekly funding counts are
overdispersed:

```text
Y[s,t] ~ NegBin(mean = Lambda[s,t], dispersion = phi[s]).
```

The sector model produces either expected sector counts for week `t` or simulated
sector events.

## 4. Layer 2: startup survival / hazard stage

Conditional on a sector event in sector `s` at week `t`, define the risk set:

```text
R[s,t] = active startups in sector s at week t.
```

This risk set can change over time. A startup enters when it becomes visible/active
and exits if it is acquired, dead, outside scope, or otherwise no longer eligible.

For each candidate firm `i in R[s,t]`, construct point-in-time features:

```text
Z[i,t] = firm features known as of week t.
```

Important features include:

- age of company;
- time since last financing;
- last financing size;
- last known valuation;
- total raised to date;
- current financing status;
- VC round / stage;
- employees and revenue-scale proxies;
- sector and geography;
- investor quality / investor count;
- cooldown indicator: funded in the last `K` weeks.

The cooldown variable is:

```text
C[i,t;K] = 1[startup i was funded in weeks t-K, ..., t-1].
```

The current week is excluded to avoid leakage.

## 5. Survival stage option A: Cox risk-set model

The cleanest second-stage model is a sector-stratified Cox model. For startup `i` in
sector `s`, define a hazard:

```text
h_i(t | s) = h_0,s(t) * exp(q[i,t])
```

with score:

```text
q[i,t]
  = (w0 + u[s])' Z[i,t]
    + eta[s] * C[i,t;K].
```

Here:

- `w0` is the global startup-quality / readiness coefficient vector;
- `u[s]` is a sector-specific deviation;
- `eta[s] <= 0` is the cooldown coefficient;
- `h_0,s(t)` is a sector-time baseline hazard.

For ranking within a sector event, the baseline hazard cancels. The conditional
probability that startup `i` receives the event is:

```text
P(i is funded | event in sector s at week t)
  = exp(q[i,t]) / sum_{j in R[s,t]} exp(q[j,t]).
```

The training objective for an observed event `(t, s, i*)` is the Cox partial
likelihood:

```text
log P(i* | s,t)
  = q[i*,t] - log sum_{j in R[s,t]} exp(q[j,t]).
```

This is exactly the risk-set ranker, but expressed as survival analysis. It uses only
positive event rows while treating every other firm in the event risk set as an
implicit comparison. That is ideal when the data are positive-event observations plus
an eligible universe.

## 6. Survival stage option B: discrete-time hazard model

A second useful baseline explicitly uses firm-weeks with no funding event. For every
active firm-week, define:

```text
Y_firm[i,t] = 1[startup i receives funding in week t].
```

Then fit a weekly logistic hazard:

```text
logit P(Y_firm[i,t] = 1 | active at t)
  = c0
    + c_sector[s]
    + (w0 + u[s])' Z[i,t]
    + eta[s] * C[i,t;K].
```

This model is closer to standard supervised learning. It can learn from non-event
weeks, but it requires careful class-imbalance handling because almost every active
firm-week is negative. The implementation supports negative sampling.

Given a sector event, the conditional mark distribution is obtained by normalizing the
hazards over the sector risk set:

```text
P(i | event in sector s at t)
  = hazard[i,t] / sum_{j in R[s,t]} hazard[j,t].
```

A complementary-log-log link is also natural because it corresponds to a discrete-time
Poisson intensity:

```text
P(Y_firm[i,t]=1) = 1 - exp(-exp(score[i,t])).
```

The first implementation uses logistic hazards because they are simple and stable.

## 7. How the two layers combine in simulation

For each simulated path:

1. Start from observed histories up to the simulation origin.
2. Use the sector Hawkes/count model to sample sector counts for week `t`.
3. For each event in sector `s`, build `R[s,t]`.
4. Use the Cox or discrete-hazard stage to compute startup probabilities.
5. Sample a startup mark.
6. Update the startup's funding history, cooldown state, and any event-updated features.
7. Move to the next week.

For multiple events in the same sector-week, sample sequentially without replacement
or use a Plackett-Luce likelihood. Sequential sampling is already a reasonable first
implementation.

## 8. Why this is a good modeling choice for this data

### 8.1 We only observe positive events

A standard classifier would require constructing negative labels such as "firm did not
raise in this horizon." That can be done, but the label depends on the chosen horizon
and can create severe class imbalance.

The Cox risk-set model avoids this. It does not need to say every non-funded company
is permanently negative. It only says: at the observed event time, the chosen company
outranked the other eligible companies in the same sector risk set.

### 8.2 Covariates are point-in-time and often stopped at last positive event

Many PitchBook features are updated when a financing event occurs. Between financing
events they are carried forward. This is not a problem if we treat them as
point-in-time covariates in a survival model:

```text
Z[i,t] = latest value known before t.
```

But this creates a strict leakage rule: any variable whose value is computed using a
future financing event must be excluded or lagged. For example:

```text
last_financing_size as of t       OK
last_financing_size after t       leakage
last_known_valuation_date <= t    OK
last_known_valuation_date > t     leakage
```

The data adapter should therefore materialize as-of snapshots by week.

### 8.3 Sector Hawkes captures contagion where excitation is truly positive

At sector resolution, self-excitation has the right sign. More funding in a sector can
increase investor attention and probability of more sector funding. This allows the
Hawkes component to stay nonnegative and interpretable.

### 8.4 Firm-level cooldown belongs in survival, not Hawkes

A firm that just raised is not negatively exciting itself in the Hawkes branching
sense. It is temporarily less at risk of raising again. A negative cooldown
coefficient in the survival stage is both mathematically safe and economically clear.

## 9. Implemented in this branch

The branch now contains the first version of this architecture.

### Sector and risk-set ranker

File:

```text
hawkes_calibration/sector_ranker.py
```

Main functions:

```python
simulate_synthetic_startup_market
fit_sector_count_model
fit_startup_ranker
ranker_predict_proba
evaluate_ranker
simulate_marked_paths
```

### Survival second-stage alternatives

File:

```text
hawkes_calibration/survival_second_stage.py
```

Main functions:

```python
fit_cox_survival_stage
cox_survival_predict_proba
fit_discrete_hazard_stage
discrete_hazard_predict_proba
evaluate_survival_stage
```

The Cox stage is the mathematically explicit survival version of the conditional
risk-set ranker. The discrete-hazard stage is an alternative that learns from both
positive and sampled negative firm-weeks.

### Synthetic backtest

File:

```text
hawkes_calibration/sector_backtest.py
```

The synthetic backtest now fits and reports:

```text
sector count model
risk-set ranker
Cox survival stage
discrete-time hazard stage
```

Runnable experiment:

```bash
PYTHONPATH=. python -m experiments.exp14_sector_ranker
```

Tests:

```bash
PYTHONPATH=. python tests/test_sector_ranker.py
```

## 10. What is not yet implemented

### DeepSurv

DeepSurv replaces the linear Cox score with a neural network:

```text
q[i,t] = f_theta(Z[i,t], sector_i, history_i,t).
```

The loss remains Cox partial likelihood:

```text
sum_events [logsumexp(q[j,t] for j in R[s,t]) - q[i*,t]].
```

This is the natural nonlinear extension. It should be implemented behind an optional
PyTorch dependency rather than forced into the numpy/scipy core package.

### DeepHit / Dynamic-DeepHit

DeepHit models a full discrete distribution over future event times and possibly
competing risks:

```text
P(T_i = tau, cause = funding | history).
```

It directly optimizes ranking-style objectives and can be strong for top-k prediction.
But it is more data-hungry and less interpretable than the Cox/hazard stages. It is a
benchmark candidate, not the first production model.

### Negative-binomial sector counts

Real weekly deal counts are likely overdispersed relative to Poisson. A
negative-binomial sector layer is the next statistically important upgrade.

### Plackett-Luce multi-event sector-week likelihood

When several startups are funded in the same sector-week, the current simple approach
can be improved with a Plackett-Luce likelihood over the ordered or unordered set of
funded startups.

## 11. Recommended modeling hierarchy

The recommended implementation order is:

1. **Sector Hawkes/count model + Cox survival stage.** This is the cleanest and most
   interpretable version.
2. **Discrete-time hazard stage.** Useful supervised baseline with negative sampling.
3. **Negative-binomial sector counts.** Needed if sector counts are overdispersed.
4. **Plackett-Luce multi-event ranker.** Better for multiple same-sector events in a
   week.
5. **DeepSurv.** Add only if the linear survival stage underfits.
6. **DeepHit / Dynamic-DeepHit.** Use as a strong neural benchmark if enough data are
   available.

## 12. Evaluation protocol

Use temporal splits only:

```text
train:      weeks 0 ... T1
validation: weeks T1 ... T2
test:       weeks T2 ... T3
```

Primary metrics:

- Recall@k for firms funded within horizon `h`;
- MRR of the next funded startup;
- NDCG@k;
- conditional mark negative log-likelihood;
- calibration of horizon probabilities;
- lead time: how many weeks before the announcement the firm entered top-k.

Important baselines:

- sector-only random within risk set;
- recency / time-since-last-round heuristic;
- discrete-time hazard;
- Cox survival;
- Hawkes + Cox survival;
- later: DeepSurv / DeepHit.

## 13. Bottom line

The unified model is:

```text
sector Hawkes/count process for positive market contagion
+
startup survival/hazard stage for firm-level readiness and cooldown
```

This avoids forcing negative self-inhibition into the Hawkes kernel, respects the fact
that we mainly observe positive funding events, handles changing startup universes via
dynamic risk sets, and lets PitchBook firm covariates be treated as point-in-time
survival covariates.
