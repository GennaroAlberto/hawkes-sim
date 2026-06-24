# Calibrating self-exciting event counts with a learned forward operator

Technical report. 24 June 2026. PDF: `paper/project_overview.pdf`.

## 1. Scope

This report describes a method for fitting multivariate self-exciting (Hawkes) models
to interval-censored event data — counts aggregated into fixed time bins — together
with bin-level covariates, and the role a learned neural operator plays in that fit.
Section 2 states the data and the model. Section 3 gives the calibration procedure step
by step for a concrete case (covariates in the baseline only, constant excitation,
weekly bins). Section 4 defines the forward operator and reports its accuracy. Sections
5–6 cover parameter recovery and the ranking application. Section 8 lists limitations.
All results are on synthetic data at CPU scale.

## 2. Problem setting and model

**Data.** There are `M` event streams ("sectors"), observed over `W` time bins (e.g.
weeks) with endpoints `0 = a_0 < a_1 < … < a_W = T`. For each bin `i` and sector `m` we
observe an event count `C_{i,m}`. We also observe a covariate vector `X_i ∈ R^p` per bin
(e.g. a popularity score, a sector indicator, a macro regime). Event times within a bin
are not observed.

**Model.** Sector `m` has conditional intensity

```
λ_m(t) = s_m(t) + Σ_j Σ_{t_k^j < t} A_{m,j} e^{−θ (t − t_k^j)},
```

with a covariate-driven, log-linear baseline and a constant excitation matrix:

```
s_m(t) = exp(γ_{m,0} + γ_m^T X(t)),   X(t) = X_i for t in (a_{i−1}, a_i],
```

and `A_{m,j} ≥ 0` the excitation of sector `m` by sector `j`, decaying at rate `θ`. Here
the excitation `A` does not depend on covariates; covariates enter only through the
baseline. (Covariate-modulated excitation is a direct extension; see §8.)

**The quantity computable from counts.** The Hawkes likelihood needs event times, which
we do not have. We work instead with the Mean Behavior Poisson Process: the expected
intensity `ξ = E[λ]` is deterministic and solves the Volterra equation

```
ξ(t) = s(t) + ∫_0^t Φ(t−u) ξ(u) du,    Φ_{m,j}(τ) = A_{m,j} e^{−θτ}.            (MBPP)
```

By Watanabe's characterization the process with deterministic compensator `Ξ = ∫ ξ` is
an inhomogeneous Poisson process, so the bin counts are independent Poisson variables,

```
C_{i,m} ~ Poisson(Ξ_{i,m}),   Ξ_{i,m} = ∫_{a_{i−1}}^{a_i} ξ_m(t) dt.
```

Calibration is maximum likelihood for this Poisson model; up to a constant the negative
log-likelihood in `Θ = (γ, A, θ)` is

```
L(Θ) = Σ_i Σ_m ( Ξ_{i,m}(Θ) − C_{i,m} log Ξ_{i,m}(Θ) ).                          (NLL)
```

## 3. Calibration procedure

Evaluating `L` requires the bin compensators `Ξ_{i,m}(Θ)`, which come from solving the
MBPP equation for the current parameters. That forward solve is the only expensive step
and is repeated at every optimiser iteration. The method replaces it with a single call
to a forward operator trained once to solve the MBPP equation for any instance in the
relevant family.

**The forward operator.** `G_φ` approximates the solution map

```
G_φ : ( s(·), A ) ↦ ξ(·),
```

mapping a baseline path `s` (sampled on a fixed time grid) and an excitation matrix `A`
to the mean-intensity path `ξ` on that grid, in one forward pass. §4 describes how it is
built and trained; here it is used as a differentiable black-box solver.

**Offline (once).** Train `G_φ` over the instances the fit will query: baseline paths of
the log-linear form above spanning the plausible range of `(γ, X)`, and subcritical
excitation matrices `A`. Training minimises the residual of the MBPP equation (§4); the
trained operator is reused for every dataset.

**Online (per dataset).** Fit `Θ` by gradient descent on `L`, using `G_φ` as the forward
map. One iteration:

1. **Baseline path from covariates.** With the current `γ`, evaluate the
   piecewise-constant baseline on the grid, `s_m(t_k) = exp(γ_{m,0} + γ_m^T X(t_k))`,
   using the observed weekly covariates `X_i`.
2. **Forward solve.** `ξ = G_φ(s, A)` — a single network pass, no integral-equation solve.
3. **Bin the compensator.** `Ξ_{i,m} = ∫_{a_{i−1}}^{a_i} ξ_m`, by the trapezoid rule.
4. **Likelihood.** Evaluate `L(Θ)` against the observed counts `C_{i,m}`.
5. **Gradient and step.** Since `G_φ` is differentiable and `s` depends on `γ` in closed
   form, obtain `∇_Θ L` by automatic differentiation through the whole chain (no finite
   differences, no solver loop) and take a descent step.

The output is the fitted baseline coefficients `γ` (the covariate effects), the
excitation matrix `A`, and the decay `θ`. Standard errors follow from the observed
information; a Bayesian posterior is available by sampling `L` plus a prior.

**What the operator does and does not change.** It changes only how the forward solve in
step 2 is computed; the statistical model and the estimator are unchanged. For the
baseline-only case here the forward solve also has an exact, inexpensive numerical
solution (the kernel is a convolution), so the operator is used for speed and clean
autodiff gradients; its decisive advantage appears in the covariate-modulated-excitation
case (§8), where the equation is no longer a convolution and the per-iteration solve is
the bottleneck.

## 4. The forward operator

**Architecture.** `G_φ` is a DeepONet: a branch network encodes the instance (the
baseline path `s` sampled on the grid and the excitation matrix `A`) and a trunk network
encodes the query time; their inner product gives `ξ(t)`. The operator must be
conditioned on the instance it solves: a network that does not receive `A` (or receives
only the forcing while the system varies) is not a function on the family — the same
input corresponds to many correct outputs — and cannot be fit below roughly 35–50%
relative error. Conditioning on `(s, A)` removes this.

**Training objective.** On a grid, the discretised MBPP equation is `ξ = s + A(Gξ)`,
with `G` the lower-triangular matrix implementing `∫_0^t e^{−θ(t−u)}(·) du`. The residual
`R[ξ] = ξ − s − A(Gξ)` is linear in `ξ`, so it has a unique zero at the exact solution.
The operator is trained to drive `R` to zero, minimising `E_{(s,A)}[ ‖R‖² / ‖ξ‖² ]` plus
a supervised term on a small set of exact anchor solutions. The relative normalisation
keeps large-intensity instances from dominating; no labelled solutions are needed for
the residual term.

**Accuracy.** Evaluated against the exact multivariate solver on 400 held-out instances
per dimension:

| M (sectors) | instance dim (M+M²) | rel-L2 mean | median | p90 | training time |
|---|---|---|---|---|---|
| 3 | 12 | 0.55% | 0.47% | 0.87% | 118 s |
| 5 | 30 | 1.52% | 1.36% | 2.40% | 161 s |
| 8 | 72 | 2.03% | 1.85% | 3.11% | 242 s |

Error is approximately constant across the branching ratio and uniform across sectors;
one evaluation takes about 3 ms. Trained on noise-corrupted targets and tested against
the clean solver, the error remains within twice its noiseless value up to about 25%
multiplicative noise. Source: `experiments/exp17_pino_jax.py`,
`results/exp17_pino_jax_summary.png`.

## 5. Parameter recovery

Two routes recover the parameters. The first is the likelihood fit of §3, returning
`(γ, A, θ)` with standard errors. The second is direct inversion: minimise
`‖G_φ(s,A) − ξ_obs‖² / ‖ξ_obs‖²` over the parameters by gradient descent through the
operator. On clean data both the covariate coefficients and the excitation matrix are
recovered to within a few percent (coefficients ≤0.1%, excitation ≤2.5% at M≤5). Under
observation noise the covariate coefficients remain the well-determined part (within 5%
at noise σ=0.02 on a single path), while the excitation matrix is more sensitive and is
recovered by averaging `R` repeated observations, with error scaling as `σ/√R`. A
one-shot regression network from counts to parameters is fast but only weakly accurate
(entry correlation ≈0.38, stability correlation ≈0.48) and is suitable as an
initialisation, not a final estimate. Source: `experiments/exp18_covariate_inverse.py`.

## 6. Application: ranking the next event

Given a fitted model, the rank score for stream `m` is the probability of at least one
event in `(t, t+h]`,

```
p_m(t, h) = 1 − exp( −∫_t^{t+h} ξ_m(u) du ),
```

the compensator increment, which the forward operator already produces.

An alternative model class for the same task is survival analysis with a ranking
objective: model the time to the next event with a discrete-time competing-risks hazard,
fit by the survival likelihood (a masked Bernoulli over time bins) plus a pairwise
ranking loss on the cumulative-incidence function, with the Kaplan–Meier curve as the
population baseline. On a synthetic firm population the fitted model attains a
concordance index of 0.687 against 0.617 for a single-covariate baseline (0.5 is
random), with calibrated survival curves and a Recall@k improvement over the baseline. A
cause-specific hazard is a per-stream intensity; survival lets it depend on covariates
but not on other streams' events, which the Hawkes excitation adds. Detail:
`SURVIVAL_RANKING.md`, `experiments/exp16_survival.py`.

## 7. Software and reproducibility

The model, fitters, exact solvers, operator, and experiments are in the
`hawkes_calibration` package. The counts-based fitters are `fit_mbpp_ic_covariates`
(baseline covariates) and `fit_mbpp_ic_excitation_multi` (covariate-modulated excitation,
multivariate); the forward operator is in `hawkes_calibration/operators`; goodness-of-fit
(time-rescaling) and a Bayesian posterior are provided. The automated test suite passes
(53/53). Reproduction commands accompany each experiment script.

## 8. Limitations

- Results are on synthetic data at CPU scale. Validation at larger scale and on real
  binned-count series is required before drawing conclusions about absolute accuracy.
- Covariate-modulated excitation `A_{m,j}(t) = A_{m,j} exp(δ^T Z(t))` uses the same
  procedure as §3 with the operator additionally conditioned on `(δ, Z)`; this is the
  case where the per-iteration solve is most costly and the operator most useful. It is
  implemented in the exact solver and fitter; the operator for this case is the next step.
- The branching ratio (overall stability) is well determined from counts; the decay `θ`
  and individual cross-excitation entries are weakly identified, the more so the coarser
  the bins.
- For `M` beyond roughly 15, the `M²` excitation matrix should be encoded through a
  low-rank parameterisation rather than passed entry-wise to the operator.

## 9. References

Checked against source on 24 June 2026.

**Model and estimation.**
Rizoiu et al. (2022), *Interval-censored Hawkes processes*, JMLR 23(1) — <https://www.jmlr.org/papers/v23/21-0917.html>;
Hawkes (1971), Biometrika 58(1);
Hawkes & Oakes (1974), J. Appl. Probab. 11(3);
Daley & Vere-Jones (2003), *An Introduction to the Theory of Point Processes*, Vol. I, Springer;
Banerjee, Merugu, Dhillon, Ghosh (2005), *Clustering with Bregman divergences*, JMLR 6.

**Neural operators and point processes.**
Li et al. (2021), *Physics-Informed Neural Operator* — <https://arxiv.org/abs/2111.03794>;
Lu, Jin, Pang, Zhang, Karniadakis (2021), *Learning nonlinear operators via DeepONet*, Nature Machine Intelligence 3;
Du et al. (2016), *Recurrent Marked Temporal Point Processes*, KDD — <https://www.kdd.org/kdd2016/papers/files/rpp1081-duA.pdf>;
Mei & Eisner (2017), *The Neural Hawkes Process*, NeurIPS — <https://arxiv.org/abs/1612.09328>;
Zuo et al. (2020), *Transformer Hawkes Process*, ICML — <http://proceedings.mlr.press/v119/zuo20a/zuo20a.pdf>;
Zhang, Lipani, Kirnap, Yilmaz (2020), *Self-Attentive Hawkes Processes*, ICML — <https://arxiv.org/abs/1907.07561>;
Shchur, Biloš, Günnemann (2020), *Intensity-Free Learning of Temporal Point Processes*, ICLR — <https://arxiv.org/abs/1909.12127>;
Trivedi, Dai, Wang, Song (2017), *Know-Evolve*, ICML — <https://arxiv.org/abs/1705.05742>;
Trivedi, Farajtabar, Biswal, Zha (2019), *DyRep*, ICLR — <https://openreview.net/forum?id=HyePrhR5KX>;
Xue et al. (2024), *EasyTPP*, ICLR — <https://arxiv.org/abs/2307.08097>.

**Survival analysis and ranking.**
Kaplan & Meier (1958), JASA 53;
Cox (1972), JRSS-B 34;
Fine & Gray (1999), JASA 94:496–509;
Graf, Schmoor, Sauerbrei, Schumacher (1999), Statistics in Medicine 18(17–18);
Antolini, Boracchi, Biganzoli (2005), Statistics in Medicine 24(24) — <https://onlinelibrary.wiley.com/doi/10.1002/sim.2427>;
Katzman et al. (2018), *DeepSurv*, BMC Med. Res. Methodol. 18 — <https://link.springer.com/article/10.1186/s12874-018-0482-1>;
Lee, Zame, Yoon, van der Schaar (2018), *DeepHit*, AAAI — <https://cdn.aaai.org/ojs/11842/11842-13-15370-1-2-20201228.pdf>;
Lee, Yoon, van der Schaar (2019), *Dynamic-DeepHit*, IEEE TBME 67(1);
Ross, Das, Sciro, Raza (2021), *CapitalVX*, J. Finance and Data Science — <https://www.sciencedirect.com/science/article/pii/S2405918821000040>.

Reference implementations: `pycox`, `scikit-survival` (DeepHit, Cox, Brier, concordance);
`EasyTPP` (RMTPP / NHP / THP baselines).

## Related documents
`paper/investment_case_study.pdf` (the model and proofs); `SURVIVAL_RANKING.md` (survival
method in full); `NEXT_FUNDING_DESIGN.md` (the ranking application); `README.md` (the package).
