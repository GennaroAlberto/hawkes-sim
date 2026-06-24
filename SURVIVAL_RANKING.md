# Survival analysis with ranking for "who is funded next"

The alternative to the Hawkes/MBPP approach. Where Hawkes models the *event
intensity*, survival analysis models the *time to the next event* per firm and
ranks firms by their probability of funding within a horizon. This document gives
the references, the math model, how to build the training set, and exactly what we
optimize. It is grounded by a runnable demo (`experiments/exp16_survival.py`,
numbers in §6).

---

## 1. Why this approach, and how it relates to Hawkes

The business question — *rank the firms most likely to raise in the next `h`
weeks* — is a **time-to-event ranking** problem with **right censoring** (firms
that have not raised yet) and **competing risks** (a firm acquired or shut down
leaves the funding risk set). Survival analysis is the native framework, and its
modern deep variants (DeepHit) optimize a **ranking** loss directly aligned with
our watch-list metric.

The bridge to Hawkes is exact (§7): a per-firm **cause-specific hazard** *is* a
per-firm conditional intensity. Classical survival lets the hazard depend on
covariates but not on the event history of *other* firms; Hawkes adds that mutual
excitation. We therefore reintroduce the self-exciting signal as a **recency /
maturity covariate** (time since last round), which the empirical VC literature
finds is the dominant predictor of the next round (§8). Survival-with-ranking is
thus a strong, metric-aligned discriminative model and the right baseline; Hawkes
is the more faithful generative model.

---

## 2. References

- **Kaplan & Meier (1958)** — the non-parametric product-limit estimator of `S(t)`.
- **Nelson (1972) / Aalen (1978)** — cumulative-hazard estimator.
- **Cox (1972)** — proportional-hazards regression; **DeepSurv** (Katzman et al.
  2018) is its neural version.
- **Fine & Gray (1999)** — regression on the subdistribution hazard for **competing
  risks** (the CIF).
- **Graf et al. (1999)** — the **integrated Brier score** with inverse-probability-
  of-censoring weighting (IPCW) for calibration/accuracy under censoring.
- **Antolini et al. (2005)** — the **time-dependent concordance index** `C_td`, the
  standard ranking metric for survival models.
- **DeepHit** (Lee et al., AAAI 2018) — discrete-time, distribution-free deep model
  for competing risks with a **ranking loss**; **Dynamic-DeepHit** (Lee et al. 2019)
  adds longitudinal (time-varying) covariates.
- **Interval-censored Hawkes / MBPP** (Rizoiu et al. 2022) — the generative
  alternative (this repo's `paper/investment_case_study.pdf`).

(Full links in §9.)

---

## 3. The math model

### 3.1 Set-up
For firm `i` observed from a decision time (origin) `τ`, let `T_i ≥ 0` be the time
until its next **funding** event and `c_i ∈ {funding, acquired, …}` the cause. We
observe `(\tilde T_i, δ_i, c_i)` where `\tilde T_i = min(T_i, C_i)` and `C_i` is the
censoring time (administrative end of data, or a competing event); `δ_i = 1` iff
funding is observed (`T_i ≤ C_i`, `c_i = funding`).

### 3.2 Hazard, survival, CIF (continuous time)
The **cause-specific hazard** of funding is
```
λ_f(t | x) = lim_{dt→0} (1/dt) P(t ≤ T < t+dt, c = funding | T ≥ t, x).
```
The overall survival (no event of any cause yet) is
`S(t|x) = exp(−∫_0^t Σ_e λ_e(u|x) du)`, and the **cumulative incidence function**
(the quantity we actually rank on) is
```
F_f(t | x) = P(T ≤ t, c = funding | x) = ∫_0^t S(u|x) λ_f(u|x) du.
```
`F_f(h|x)` = *probability firm i is funded within the horizon `h`* = the ranking
score.

### 3.3 Discrete-time (DeepHit) parametrization — what we implement
Partition `[0,h]` into `K` bins with edges `0=a_0<…<a_K=h`. A network maps the
covariates to a **discrete hazard** per bin,
```
h_k(x) = P(event in bin k | survived to bin k, x) = σ(o_k(x)),   o(x) = MLP(x) ∈ R^K,
```
from which
```
S_k(x) = ∏_{j≤k} (1 − h_j(x)),         (survival through bin k)
F_k(x) = 1 − S_k(x).                    (CIF; risk = F_{K}(x) = funding within h)
```
This logistic-hazard form is exact-likelihood, needs no proportional-hazards
assumption, and extends to competing risks by predicting a hazard per (bin, cause)
and softmax-normalizing.

### 3.4 Kaplan–Meier (the non-parametric baseline + calibration target)
With no covariates, the product-limit estimator is
```
Ŝ(t) = ∏_{t_l ≤ t} (1 − d_l / n_l),
```
`d_l` = funding events at `t_l`, `n_l` = number at risk just before `t_l`. We use it
as the population curve and as a **calibration check**: a well-fit model's average
`S(t|x)` over a stratum should track the stratum's KM curve (demo panel (a)).

---

## 4. Building the training set

The single most important — and most error-prone — step. From the raw event stream
we manufacture survival examples by **sampling decision-time origins**:

1. **Sample origins.** Choose decision times `τ` (a grid, every event time, or random
   times). At each `τ`, the *risk set* is the firms currently "alive" (already
   founded, not yet exited). For each at-risk firm `i` emit one example.
2. **Label (point-in-time).** `\tilde T_i = min(next funding − τ, competing exit − τ, h)`;
   `δ_i = 1` iff a funding event occurs in `(τ, τ+h]` before any competing exit.
   Bin it: `κ_i = ` index of `\tilde T_i`.
3. **Covariates as-of `τ`** (no leakage): static (sector, size) **and dynamic** —
   crucially **time since last round** (the recency/maturity feature that carries the
   self-exciting signal), number of past rounds, popularity score at `τ`. Anything
   computed must use only information available strictly before `τ`.
4. **Censoring & truncation.** Firms not yet founded at `τ` are excluded
   (left-truncation); firms acquired/dead before their next funding are **censored**
   at the exit time (competing risk); everyone still un-funded at `τ+h` is censored
   administratively at `h`.
5. **Multiple origins per firm** give more examples and expose the model to the same
   firm at different maturities; cluster-robust validation (split by firm, not by
   example) avoids leakage across the train/test boundary.

The demo's `build_dataset` implements exactly this (origin sampling, recency
feature, horizon/competing censoring).

---

## 5. What we optimize (the objective)

DeepHit's **total loss = likelihood + ranking**:

### 5.1 Survival likelihood (fits the curves)
For the discrete model this is the negative log-likelihood, which reduces to a
**masked Bernoulli cross-entropy** over bins: with target `y_{i,j}=1` only at the
event bin of an uncensored example, and a mask `m_{i,j}=1` for all bins up to the
observed bin,
```
L_like = − (1/Σm) Σ_i Σ_j m_{i,j} [ y_{i,j} log h_j(x_i) + (1−y_{i,j}) log(1−h_j(x_i)) ].
```
Its gradient w.r.t. the logits is simply `m·(h − y)` — this is what the demo
backpropagates. Minimizing `L_like` alone already yields calibrated `S(t|x)` curves
(it is "Kaplan–Meier with covariates").

### 5.2 Ranking loss (sharpens the ordering — the watch-list metric)
DeepHit adds a pairwise **concordance surrogate** on the cause-specific CIF. For
comparable pairs `(i,j)` where `i` is funded at `\tilde T_i` and `j` is still at risk
then (`\tilde T_j > \tilde T_i`),
```
L_rank = (1/|P|) Σ_{(i,j)∈P} exp( −(F_i(\tilde T_i) − F_j(\tilde T_i)) / σ ),
```
which is small when the funded firm's risk exceeds the still-at-risk firm's risk —
i.e.\ it directly rewards correct ranking. Because `R_i = F_K(x_i) = 1−∏(1−h_j)` has
the clean derivative `∂R_i/∂o_k = (1−R_i) h_{i,k}`, the ranking gradient maps to the
logits in closed form (implemented in the demo's `rank_weight` path). The total
objective is `L = L_like + α L_rank`.

### 5.3 What "KM-like curves + ranking" means concretely
- **Curves:** the model outputs an individualized survival/CIF curve per firm; we
  validate them against Kaplan–Meier by stratum (calibration) and read off
  `F_f(h|x)` = funding-within-horizon probability.
- **Ranking:** we sort firms by `F_f(h|x)` to get the watch-list, and optimize/score
  that ordering with the ranking loss and the concordance index.

---

## 6. The runnable demo and its results (`experiments/exp16_survival.py`)

A 600-firm synthetic population (covariate- and maturity-driven funding, 25% subject
to a competing acquisition risk) → 3,000 origin-sampled survival examples (58% fund
within the horizon) → Kaplan–Meier + a discrete-hazard model (likelihood, and
likelihood+ranking) → evaluation:

| model | concordance index | Recall@10% |
|---|---|---|
| popularity baseline (rank by one covariate) | 0.617 | 0.143 |
| survival, likelihood | 0.684 | 0.159 |
| **survival, likelihood + ranking** | **0.687** | 0.161 |
| random | 0.500 | — |

The survival model clearly beats the single-covariate baseline; the ranking loss
gives a small, honest improvement in concordance (it matters more under heavier
censoring / when the likelihood ordering is suboptimal). `results/exp16_survival.png`
shows: (a) model survival curves by popularity stratum bracketing the Kaplan–Meier
baseline (calibration), (b) the concordance bars, (c) predicted risk separating
funders from non-funders, (d) the Recall@k watch-list lift over the baseline.

Evaluation metrics to report on real data: **time-dependent concordance** (Antolini),
**integrated Brier score with IPCW** (Graf — censoring-corrected calibration), KM
calibration by stratum, and the business **Recall@k / lead time**.

Run: `PYTHONPATH=. python -m experiments.exp16_survival`.

---

## 7. Survival vs Hawkes — and a hybrid

- **Use survival-with-ranking** when you want to optimize the ranking metric
  directly, handle competing risks cleanly, and you have good covariates; it is
  discriminative and metric-aligned but discards the generative excitation structure
  and needs event times (not counts).
- **Use Hawkes/MBPP** when mutual excitation and contagion matter, when you only have
  aggregate counts (the MBPP path), or when you need an interpretable branching
  matrix and calibrated generative probabilities.
- **Hybrid (recommended to test):** (i) feed **Hawkes-derived features** (the fitted
  per-firm intensity, recency, branching exposure) as covariates into the survival
  model; or (ii) read the survival score off the **Hawkes intensity** directly via
  `F_f(h) = 1 − exp(−∫_t^{t+h} λ_i)` and fine-tune it with the DeepHit ranking loss.
  This combines generative fidelity with metric alignment.

---

## 8. Domain note
The empirical VC literature (CapitalVX; Crunchbase studies) consistently finds that
**maturity and financing recency dominate "next round within 12 months"** —
i.e.\ momentum in investor attention. That is exactly the recency covariate of §4 and
the self-excitation of the Hawkes kernel, so both models should encode it; a survival
model *without* a recency feature will underperform.

---

## 9. Reference links
- E. Kaplan, P. Meier (1958). Nonparametric estimation from incomplete observations. *JASA* 53.
- D. R. Cox (1972). Regression models and life-tables. *J. R. Stat. Soc. B* 34.
- J. Fine, R. Gray (1999). A proportional hazards model for the subdistribution of a competing risk. *JASA* 94.
- E. Graf et al. (1999). Assessment and comparison of prognostic classification schemes for survival data. *Statistics in Medicine* 18.
- L. Antolini et al. (2005). A time-dependent discrimination index for survival data. *Statistics in Medicine* 24.
- J. Katzman et al. (2018). [DeepSurv](https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-018-0482-1). *BMC Med. Res. Methodol.*
- C. Lee et al. (2018). [DeepHit: A Deep Learning Approach to Survival Analysis with Competing Risks](https://cdn.aaai.org/ojs/11842/11842-13-15370-1-2-20201228.pdf). *AAAI*.
- C. Lee et al. (2019). Dynamic-DeepHit. *IEEE TBME*.
- pycox / scikit-survival — reference implementations of the above for real-data work.
- M.-A. Rizoiu et al. (2022). [Interval-censored Hawkes processes](https://www.jmlr.org/papers/v23/21-0917.html). *JMLR* 23(1). (The generative alternative; this repo.)
- G. Ross et al. (2021). [CapitalVX](https://www.sciencedirect.com/science/article/pii/S2405918821000040). *J. Finance and Data Science*.
