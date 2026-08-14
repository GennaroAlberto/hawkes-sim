# Ranking Individuals by Next-Event Risk from an Event-Only Log

A modeling guide for recurrent-event data where covariates update at events, most individuals have 1–2 events, and the population roster is unknown (only events are observed).

---

## 1. Problem setup

The natural mathematical object is a **marked point process** with conditional intensity

λᵢ(t | 𝓕ₜ₋)

where 𝓕ₜ₋ is everything known *strictly before* t. Two features of the data make this hard:

**Event-driven covariates.** Part of each individual's covariate vector only updates when they have an event. Between events it is frozen — increasingly stale — while another part (shared/public covariates) evolves continuously.

**Event-only observation.** The dataset contains only events. An individual is unknown until their first event, so the risk set has no denominator. This has two unavoidable consequences:

1. *You cannot rank individuals you have never seen.* First events per-individual are unidentifiable; at most you can model the aggregate arrival rate of new individuals. The ranking task is legitimately: **among individuals observed at least once, who is most likely to have the next event?**
2. *Silent dropout is invisible.* An individual who left the population and one with low intensity produce identical data — a long trailing gap. Section 6 (Pareto/NBD) addresses this.

Individuals therefore **enter the risk set at their first event** — left truncation at first appearance. Handled correctly, this is fine; ignored, it biases everything.

One simplification follows: since everyone in the modeling table has at least one event, everyone has private covariates. There is no "never-observed covariates" case — staleness becomes the entire problem.

---

## 2. Encoding stale covariates

For each event-updated covariate Z, carry:

- `Z_last` — the value at the last update, always taken as the **left-limit** Z(t−): the value from the *previous* event, never the event defining the current prediction target. Covariates measured *at* an event are contaminated by that event; using them is the most common silent leak in these models.
- `staleness` — time since the last update.

Crucial structural fact: since Z updates only at events, **staleness ≡ time since last event ≡ the gap-time scale of the hazard itself**. Covariate decay and duration dependence are the same number and cannot be separated. Do not fight this. Feed the gap in flexibly (splines, binning, or let trees handle it) and interact it with `Z_last`; interpretation is merged but prediction is unaffected.

The staleness-robust workhorse feature is the **decayed event count** at several half-lives τ:

H_τ(t) = Σⱼ exp(−(t − tⱼ)/τ)  over past events tⱼ

Use 3–4 values of τ (e.g., 1 week, 1 month, 6 months). This is a discretized Hawkes (self-exciting) summary: it fades smoothly instead of freezing, which is exactly the behavior a latent-state model would give you without fitting one.

---

## 3. Semiparametric option: PWP-stratified Cox

### 3.1 The recurrent-event Cox family, briefly

Extending Cox regression to recurrent events requires deciding (a) who is at risk for the k-th event and when, and (b) whether all events share one baseline hazard. The three classical answers:

**Andersen–Gill (AG).** Every individual under observation is at risk for "an event" at all times; one common baseline hazard; events are conditionally independent given covariates (Poisson-like). Simple, but assumes the 3rd event behaves like the 1st, which is rarely true.

**Wei–Lin–Weissfeld (WLW).** Marginal models: individual i is "at risk" for event k from time zero even before having event k−1. Useful for marginal treatment effects, wrong for your purpose — it deliberately ignores the sequential structure.

**Prentice–Williams–Peterson (PWP, 1981).** The sequential model:

- **Conditional risk sets:** individual i is at risk for their k-th event only after their (k−1)-th event has occurred. The risk set for "2nd events" at time t contains only individuals who currently have exactly one prior event. This respects the actual information flow.
- **Stratified baseline:** each event number k gets its own baseline hazard λ₀ₖ(·). The hazard for individual i's k-th event is

  λᵢₖ(t) = λ₀ₖ(t) · exp(β′Xᵢ(t))

  with β either common across strata or stratum-specific (β_k) if you have enough repeat events to support it.
- **Two time scales:**
  - *PWP-Total Time:* t is time since the individual's origin. Baseline captures "hazard of k-th event as a function of age in the system."
  - *PWP-Gap Time:* t resets to zero at each event, so the baseline is over time-since-last-event. **Gap time is the right choice here**, because (Section 2) time-since-last-event is simultaneously your staleness clock — the stratified baseline over gap time absorbs the mechanical staleness/duration effect, and covariates explain what remains.

### 3.2 Why PWP fits this problem

With most individuals contributing 1–2 events, the strata are wildly unbalanced: a huge stratum of individuals awaiting their 2nd event, a small stratum awaiting the 3rd, a tail beyond. An unstratified (AG-style) model would let the majority stratum's baseline shape distort the repeat-event minority and vice versa. PWP gives each its own baseline. Practically: strata {1 prior event, 2 prior events, ≥3 prior events} — collapse the tail, since strata with few events yield unstable baselines.

### 3.3 Adaptations for this data

- With an event-only log there is no "awaiting first event" stratum — entry is at the first event, so the model starts at the 2nd-event stratum. This is left truncation, handled naturally by the counting-process (start, stop] data format.
- Use cluster-robust (sandwich) standard errors by individual.
- **Skip frailty terms.** Gamma frailty (a per-individual random effect multiplying the hazard) is tempting but with 1–2 events per individual it is barely identified — the frailty and the baseline fight over the same variation. The decayed-count features of Section 2 capture "this individual runs hot" observably instead.

Software: `coxph(Surv(gap_start, gap_stop, status) ~ X + strata(event_number) + cluster(id))` in R's `survival`, or `lifelines` / `sksurv` in Python with the (start, stop] format.

---

## 4. Prediction-first options

Ranking with time consistency is a prediction target, so it is usually better to optimize it directly than to fit an intensity and hope.

### 4.1 Landmark supermodel (van Houwelingen)

Pick landmark times s₁,…,s_K. At each s: take everyone at risk, snapshot features as of s−, and model P(event in (s, s+w] | at risk at s) for fixed horizon w. Pool all landmarks into one dataset with s as a covariate (coefficients varying smoothly in s). One model, no covariate-process model needed, ranking = fitted probability, and time consistency comes free because s is a feature and w is fixed.

### 4.2 Discrete-time ML (person-period table)

The fully general version, best suited to gradient boosting. Construction, given the event-only log:

1. **Roster:** all individuals with ≥1 event. Individual i contributes rows only for bins with bin_start ≥ first_event_time(i) — no rows before entry (that is the truncation, handled correctly).
2. **Rows:** one per (individual, calendar bin) from entry until end of data. Choose the bin width so the per-bin event rate is roughly 0.1–5%. Label y = 1 if an event falls in the bin.
3. **Features**, all computed strictly from information before bin_start:

   | Feature | Notes |
   |---|---|
   | `gap` = bin_start − last event time | ≡ staleness; the single most important feature |
   | `Z_last` (private covariates) | left-limit: from the previous event only |
   | `Z_last × f(gap)` interactions | or let trees learn them |
   | Decayed counts H_τ, 3–4 half-lives | Hawkes-style smooth history |
   | `n_prior_events`, last gap, mean past gap | gap stats only exist for ≥2-event individuals |
   | Public covariates at bin_start | the continuously evolving part |
   | Calendar features, time since first appearance | seasonality and drift |

4. **Subsample the zeros.** The table is ~99% zeros. Keep all y = 1 rows, sample zero rows at rate p, and add offset log(1/p) (or recalibrate after fitting). Ranking is unaffected by subsampling; calibration needs the correction.
5. **Split by calendar time** — train on the past, validate on a later window. Never split rows randomly: rows within an individual are serially dependent and random splits leak.

### 4.3 Neural temporal point processes

An RNN/transformer over the event sequence emitting λ(t) is the principled version of "covariates update at events." With 1–2 events per individual there is almost no sequence to learn from; expect it to lose to the landmark or boosted model unless data volume is very large. Try last, if at all.

---

## 5. The dropout confound

Long gaps earn low predicted hazard partly because those individuals are *gone*, not merely quiet. For pure ranking this is often acceptable — a departed individual should rank low. It becomes a problem when you need "still-active but quiet" ranked above "departed," which the gap alone cannot distinguish. That is exactly what the Pareto/NBD family was built for.

---

## 6. Pareto/NBD and the "Buy Till You Die" family

### 6.1 Origin and setup

Schmittlein, Morrison & Colombo (1987) posed the customer-base problem: given only a purchase log — no roster, no churn notifications — which customers are still "alive," and how many future purchases should each be expected to make? This is structurally identical to your data situation.

The model's assumptions, per individual:

1. **While alive**, events follow a Poisson process with individual rate λ (exponential inter-event gaps).
2. **Lifetime** is exponential with individual dropout rate μ; death is silent and unobserved.
3. **Heterogeneity:** across the population, λ ~ Gamma(r, α) and μ ~ Gamma(s, β).
4. λ and μ are independent across and within individuals.

The gamma-mixed Poisson gives the "NBD" (negative binomial distribution) part of the name; the gamma-mixed exponential lifetime gives a Pareto — hence Pareto/NBD.

### 6.2 Why it is remarkable here

The individual-level likelihood depends on the data **only through three numbers**: x (number of repeat events), t_x (recency — time of the last event), and T (total observation time since first appearance). Frequency and recency are sufficient statistics. From four fitted population parameters (r, α, s, β), the model yields in closed(ish) form:

- **P(alive | x, t_x, T)** — the probability the individual is still active. This is the quantity nothing in Sections 3–4 can give you, because it disentangles "quiet" from "gone." The model's logic is intuitive: a historically frequent individual who goes silent becomes *probably dead* quickly, while an infrequent one with the same silence is *probably just quiet*.
- **E[future events in (T, T + w] | history]** — a direct ranking score.

### 6.3 Variants

- **BG/NBD** (Fader, Hardie & Lee 2005): replaces the exponential lifetime with "after each event, drop out with probability p" (geometric, p ~ Beta). Nearly identical fit, much easier estimation; the standard default. Quirk: individuals can only die at events, so a one-event individual can never be classified as dead — check whether this matters for your tail.
- **Pareto/GGG** and Weibull-gap variants: relax exponential gaps (allow regularity/clumpiness in timing).
- **Covariate extensions:** the gamma parameters can be regressed on covariates, though the classical models are covariate-free.

Software: `BTYD` / `BTYDplus` (R), `lifetimes` (Python).

### 6.4 How to use it in this pipeline

The basic model ignores your covariates, so use it two ways rather than as the final model:

1. **Baseline:** Pareto/NBD or BG/NBD expected-events is the bar any covariate model must beat. It is embarrassingly strong on event-only data.
2. **Feature:** feed P(alive | x, t_x, T) into the discrete-time ML model. This injects the latent-aliveness structure that gap features alone cannot express, directly targeting the dropout confound.

The ML analog, if you want covariates inside the aliveness structure itself: a two-part model, P(alive | gap, history, X) × λ(event | alive, X), trained jointly or via EM.

---

## 7. Evaluation

**Ranking.** At each held-out bin/landmark s: rank the full risk set by predicted P(event in (s, s+w]); compute AUC (or Uno's IPCW C-index in the survival formulation) and recall@k *per landmark*, then average. A single global C-index is dominated by easy long-horizon comparisons and will flatter the model.

**Time consistency.** Two separate checks:

1. *Calibration per landmark* — calibration-in-the-large and slope at each s; drift shows up in the intercept.
2. *Rank stability* — rank correlation of scores across consecutive landmarks for individuals with no new events. It should be high; if scores jump when nothing happened, the staleness handling is wrong. Also check that each individual's score vs. gap curve is smooth and stable across calendar time.

**Sharpness.** IPCW Brier score / IBS at horizon w.

**Baselines to beat, in order:** (1) rank by decayed event count alone; (2) BG/NBD expected events. If the covariate model barely beats these, `Z_last` is not adding signal and the machinery is not paying for itself.

---

## 8. Pitfalls checklist

- [ ] Every feature in every row computable strictly before the row's time origin (audit a random sample by hand).
- [ ] `Z_last` taken from the *previous* event, never the labeled one (left-limit).
- [ ] No rows before an individual's first event (left truncation respected).
- [ ] Train/validation split by calendar time, never by random rows.
- [ ] Zero-row subsampling corrected with an offset (or post-hoc recalibration) if calibrated probabilities are needed.
- [ ] PWP strata collapsed where sparse (≥3 events → one stratum); gap-time scale used.
- [ ] No frailty terms with 1–2 events per individual.
- [ ] Dropout confound addressed (P(alive) feature or two-part model) if "quiet vs. gone" matters.
- [ ] Model beats decayed-count and BG/NBD baselines before being trusted.

---

## References

- Prentice, Williams & Peterson (1981), *On the regression analysis of multivariate failure time data*, Biometrika.
- Andersen & Gill (1982), *Cox's regression model for counting processes*, Ann. Statist.
- Schmittlein, Morrison & Colombo (1987), *Counting your customers*, Management Science.
- Fader, Hardie & Lee (2005), *"Counting your customers" the easy way: An alternative to the Pareto/NBD model*, Marketing Science.
- van Houwelingen (2007), *Dynamic prediction by landmarking in event history analysis*, Scand. J. Statist.
