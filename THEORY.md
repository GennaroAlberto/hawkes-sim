# THEORY.md — the escalation ladder, treated mathematically

Companion to `STRATEGY.md` (which states the ladder and the deployment gates)
and `MATH_NOTES.md` (repo-specific derivations M1–M8, cross-referenced here).
This document gives the deep treatment: each rung of the ladder as a formal
statistical model — likelihood, estimation theory, the assumptions it needs,
what the data must show for those assumptions to be checkable, and the
statistical problems that arise when calibrating it, with particular care for
**censoring and truncation** (what changes if the data are right-censored, and
what never stops being censored in this problem). Rung 0 (XGBoost) is a
benchmark, not a probability model, and is not treated. The annotated
literature map is §7.

**Notation** (matches `MATH_NOTES.md` and `paper/complete_account.tex`).
Firms $i \in \mathcal I$ with sector map $s(i)$; sectors $s = 1,\dots,S$;
funding stages $k \in \mathcal K = \{\text{Seed}, A, B, C, D, E, \dots\}$ plus
absorbing exits $\{\text{IPO/M\&A}, \text{fail}\}$. Calendar time $t \in
[0,T]$. Macro/context covariates $X(t)$ (external), firm features $Z_{i,t}$
(last-observation-carried-forward), weekly counts $Y_{s,t}$. Internal history
$\mathcal H_t$. Ground intensity $\Lambda_s(t)$, mark kernel
$P(i \mid s, t, \mathcal H_t)$, and the exact factorization
$\lambda_i(t) = \Lambda_{s(i)}(t)\, P(i \mid s(i), t, \mathcal H_t)$ (M1),
which holds at every rung.

---

## 0. The observation scheme, formally

Everything downstream depends on being precise about *what is observed*. This
section fixes the probabilistic setup once; each rung then only has to say
which parts it uses.

### 0.1 Counting processes and conditional intensities

Work on a filtered probability space $(\Omega, \mathcal F, \{\mathcal F_t\},
\mathbb P)$ where $\mathcal F_t$ contains the internal history of the marked
point process and the (external) covariate paths. For each firm $i$ and each
admissible transition $k \to j$, let $N_{i,k\to j}(t)$ count observed
transitions. A **conditional intensity** $\lambda_{i,k\to j}(t)$ is a
nonnegative predictable process such that

$$M_{i,k\to j}(t) \;=\; N_{i,k\to j}(t) - \int_0^t Y_{i,k}(u)\,
\lambda_{i,k\to j}(u)\, du$$

is a local martingale (Doob–Meyer; Andersen–Borgan–Gill–Keiding [ABGK, ch.
II]). Here $Y_{i,k}(t) \in \{0,1\}$ is the **at-risk indicator**: firm $i$ is
in stage $k$, under observation, and eligible at $t-$. All likelihoods below
are instances of Jacod's formula: for a realization $\{(t_m, e_m)\}$ of events
with types $e$,

$$\log L(\theta) \;=\; \sum_m \log \lambda_{e_m}(t_m; \theta)
\;-\; \sum_e \int_0^T Y_e(u)\, \lambda_e(u; \theta)\, du .
\tag{0.1}$$

The martingale structure is not decoration: it is what delivers unbiased
scores ($\mathbb E\,\partial_\theta \log L = 0$ at the truth), the Fisher
information as the predictable variation of the score, and asymptotic
normality via Rebolledo's martingale CLT — the entire estimation theory of
every rung is this one theorem applied with different $\lambda$'s
(Andersen–Gill 1982; ABGK ch. VI).

### 0.2 The four observation defects of funding data

**(C1) Right censoring at $T$ (administrative).** Every firm alive at the end
of the window has its next round unobserved. This is the benign defect. Under
**independent censoring** — formally: censoring is a predictable $\{0,1\}$
process $C_i(t)$ and the intensity of $N_i$ with respect to the *enlarged*
filtration (internal history + censoring information) is still
$\lambda_i(t)$ [ABGK III.2.2] — the observed process $\int C_i\, dN_i$ has
intensity $C_i(t)\lambda_i(t)$, and (0.1) applied to the observed data is a
genuine (partial) likelihood. Administrative censoring at a fixed $T$ is
independent by construction. The censored spell contributes exactly its
survival factor $\exp(-\int_{\text{spell}} \lambda)$: **no term is dropped,
no term is invented.**

*What if the data were not right-censored?* Only if every firm's history were
run to an absorbing state (all resolved by failure/exit). Then every spell
ends in an observed event and (0.1) becomes a complete-data likelihood.
Funding data is never in this regime — treat "no censoring" as a fiction used
only to see what censoring costs: censored spells still contribute
$\exp(-\int\lambda)$, i.e. *pure information about the integral of the
hazard, none about its shape at the jump*. The practical corollary: **never
drop open spells.** Discarding censored spells converts right censoring into
*right truncation* at $T$ — selection on the event occurring in-window —
which oversamples short gaps and biases every hazard upward. (Do not call
this length bias: classical length-biased sampling is the opposite
selection, oversampling *long* spells in prevalent-cohort designs; the
right pointer is Kalbfleisch–Prentice's treatment of sampling schemes and
truncation.)

**(C2) Left truncation (delayed entry).** A firm is invisible before its
first funding: it enters observation at $V_i$ = first-round date, and enters
it *because of an event*. Two consequences. (i) The mechanical one is
standard: risk sets are $Y_i(t) = \mathbf 1\{V_i \le t < U_i\}$ and every
integral in (0.1) starts at $V_i$ — delayed entry in the sense of Keiding
(1990). (ii) The statistical one is not: the *entry process itself* is an
outcome (a funding event), so the population of observed firms is a
selection. Anything claimed about "firms" is really about "firms conditioned
on having raised at least once before $T$". Rung 5 is the place where this
selection is modeled rather than ignored.

**(C3) Unlabeled exits (informative censoring risk).** Failure and quiet
wind-downs are mostly unrecorded. A dead firm keeps $Y_i(t) = 1$ in any naive
risk set — the **immortal-firm problem**. Keep three facts distinct. (i) The
contamination bias — dead firms held at risk deflate every hazard (§2.4) —
arises under *any* unlabeled-exit mechanism, even exit independent of
funding propensity: it is a misrecorded $Y_i$, not a dependence effect.
(ii) Because exit additionally correlates with the same frailty that drives
funding, the defect is *uncorrectable from observables*: no estimated exit
model can reconstruct the correct risk set — our synthetic rehearsal
measured the resulting inversion of ranking economics, and an exit model
with AUC 0.877 was not enough to repair it (E2 campaign). (iii) If exits
*were* labeled, removing firms at exit is valid for cause-specific funding
intensities with no independence assumption at all (the standard
competing-risks fact, §2.3): the problem is that exits are unobserved, not
that they are dependent. This is the single most dangerous defect in the
data.

**(C4) Interval censoring by aggregation.** Weekly bucketing replaces event
times by bin counts. The likelihood becomes the integrated (MBPP) one, and
the information loss is quantifiable and *selective*: branching-scale
parameters survive aggregation, timescale parameters do not (the κ–θ ridge,
M3; Fisher information in the decay $\to 0$ as bin width / half-life grows).
Turnbull (1976) is the classical ancestor for likelihoods under grouped and
censored observation.

### 0.3 Positive-only labels

We observe fundings, never rejections (M8). There is no firm-level negative
class; the natural firm-level object is the **conditional (partial)
likelihood** "given a deal in $(s,t)$, which firm" — a Cox-type partial
likelihood, algebraically a conditional logit (McFadden 1974; Cox 1975). The
sector-time baseline cancels in the ratio, which is exactly why the mark
factor needs no timing model (M1, M8). Every rung's firm-level component
inherits this.

---

## 1. Rung 1 — Inhomogeneous Poisson with covariate-driven intensity

### 1.1 Model and likelihood

$$N_s \sim \text{Poisson process on } [0,T] \text{ with intensity }
\Lambda_s(t) = \exp\!\big(a_s + \beta_s^\top X(t)\big),$$

$X(t)$ **external** (macro series: their evolution does not depend on the
deal stream — the Kalbfleisch–Prentice internal/external distinction, which
becomes load-bearing at rung 3). Log-likelihood, from (0.1):

$$\ell_s(a_s, \beta_s) = \sum_{m: s_m = s} \big(a_s + \beta_s^\top X(t_m)\big)
- \int_0^T e^{a_s + \beta_s^\top X(u)}\, du .
\tag{1.1}$$

**Concavity.** The first sum is linear in $(a_s,\beta_s)$; the integrand
$e^{\eta}$ is convex in the linear predictor, hence the integral is convex
and $\ell_s$ is globally concave. Strictly concave iff the covariate paths
are not confined to an affine subspace: the Hessian is
$-\int_0^T \tilde X(u) \tilde X(u)^\top \Lambda_s(u)\, du$ with
$\tilde X = (1, X)$, negative definite exactly when
$\{\tilde X(u)\}$ spans. The MLE exists and is unique under that spanning
condition plus no direction of separation: no $v$ with
$v^\top \tilde X(t_m) \ge 0$ at every event (strictly at one) while
$v^\top \tilde X(u) \le 0$ for a.e. $u \in [0,T]$ — along such a ray the
event term grows while the compensator stays bounded, so the likelihood
escapes to its supremum at infinity; the point-process analogue of logistic
separation, occurring with near-constant covariates and few events.

**Asymptotics.** The score $U(\theta_0) = \int \tilde X\, (dN_s -
\Lambda_s dt)$ is a martingale with predictable variation
$I(\theta_0) = \int \tilde X \tilde X^\top \Lambda_s\, dt$ — the Fisher
information. Rebolledo's CLT gives
$\hat\theta \approx \mathcal N(\theta_0, I^{-1})$ as information accumulates
(long window and/or many sectors); this is Ogata's (1978) program
specialized to the concave case. Weekly discretization with
piecewise-constant covariates changes nothing structurally:
$Y_{s,t} \sim \text{Poisson}(w\, e^{a_s + \beta_s^\top X_t})$ with the bin
width absorbed by the intercept — the log link is aggregation-coherent as
long as covariates are constant within bins (if they vary within-bin, a
Jensen gap $\log \int e^\eta \ge \overline{\eta} + \log w$ opens; at weekly
bins with monthly macro series it is nil).

### 1.2 Censoring at this rung

Right censoring (C1) is invisible: an NHPP likelihood on $[0,T]$ *is* the
censored likelihood — the window edge contributes the survival factor
automatically. Left truncation (C2) and unlabeled exits (C3) do not bite the
*sector-level* ground process because sectors do not die; they bite the
moment we go firm-level (rung 2). This is a genuine advantage of starting at
sector resolution: **rung 1 is the one rung whose likelihood is exactly
right under the observation scheme with no extra assumptions.**

### 1.3 Goodness of fit: what "calibrated" means here

- **Time-rescaling theorem** (Meyer's time change; Brown et al. 2002 for the
  statistical use). If $\Lambda_s$ is the true intensity, then
  $\tau_m = \int_0^{t_m} \Lambda_s(u)\, du$ is a unit-rate Poisson process;
  gaps $\tau_m - \tau_{m-1} \sim \text{Exp}(1)$ i.i.d., so
  $u_m = 1 - e^{-(\tau_m - \tau_{m-1})} \sim U(0,1)$ — KS-testable. This is
  the event-time GOF instrument at every rung; misfit *localizes* (which
  weeks, which sectors).
- **Randomized PIT for counts** (Smith 1985; Brockwell 2007): for discrete
  predictive distributions, draw $u \sim U(F(y-1), F(y))$; uniformity =
  calibration. Czado–Gneiting–Held (2009) is the count-data calibration
  reference and proposes the *non-randomized* variant of the same check.
  This is the weekly-resolution analogue and our standing diagnostic.
- **Dispersion.** Poisson forces mean = variance. The quasi-Poisson
  dispersion $\hat\phi = \frac{1}{n-p}\sum (y - \hat\mu)^2/\hat\mu$ is the
  cheapest sufficient statistic for "history or heterogeneity is missing":
  a gamma-mixed Poisson (negative binomial) has $\phi > 1$, and the mixing
  interpretation matters — **overdispersion is exactly what both frailty and
  contagion produce**, which is why $\hat\phi$ is the escalation trigger but
  never evidence *for excitation specifically* (§4.5).

### 1.4 Statistical problems in calibration

1. **Collinearity of macro covariates.** Macro series are few, smooth and
   correlated; $I(\theta)$ is near-singular along collinear directions. The
   fix is ridge ($\ell_2$) with the penalty reported, and inference on
   *linear combinations* that are well-identified rather than on individual
   coefficients.
2. **Errors in variables.** As-of macro data are revised; features are noisy.
   Classical measurement error $W = X + U$ attenuates: in the linear-model
   limit $\text{plim}\,\hat\beta = \beta\,\sigma^2_X / (\sigma^2_X +
   \sigma^2_U)$, and the same attenuation holds to first order in the
   log-linear Poisson model (Carroll–Ruppert–Stefanski–Crainiceanu 2006).
   Measured in our rehearsal (exp22): 20% relative covariate noise
   attenuated the fitted choice weights by ×0.88 (×0.93 at 15%) and eroded
   the stage-1 $\beta$ correlation from 0.67 to 0.53, while leaving
   *prediction* nearly intact — attenuation hits interpretation before it
   hits forecasting.
3. **Covariate support overlap.** Temporal validation means the test window's
   $X$ must lie inside (or near) the train window's support; extrapolating a
   log-linear intensity outside observed covariate ranges is exponential
   extrapolation. Diagnostic: report train/test covariate ranges next to any
   NLL.
4. **Nonstationarity.** $a_s$ absorbs the level; secular drift not carried by
   $X$ shows up as PIT drift over calendar time — which is a *feature*: it is
   the cheapest detector of coverage change in the data vendor (S4).

### 1.5 What we need to see in the data

Covariate variation within the training window (spanning condition);
event counts per sector-week compatible with the asymptotics (pooled
information, not per-cell counts, is what matters for a $p$-dimensional
$\beta_s$); stable vendor coverage or a coverage proxy to include in $X$;
PIT/dispersion diagnostics *after* the fit deciding whether rung 2+ is
needed at all.

---

## 2. Rung 2 — Multi-state model with stage transitions

This is where survival analysis stops being an analogy and becomes the exact
formalism: a funding history is an event history on a finite state space, and
the model is a **multi-state counting process** (ABGK; Andersen–Keiding 2002;
Putter–Fiocco–Geskus 2007). The closest applied cousin is credit-rating
migration estimated by transition intensities (Lando–Skødeberg 2002) — same
mathematics, same censoring pattern, same sparse-cell problems.

### 2.1 State space and admissibility

$\mathcal K = \{\text{Seed}, A, B, C, D, E, \dots\} \cup
\{\text{IPO/M\&A}, \text{fail}\}$ with the partial order $\succeq$:
from stage $k$ the admissible next rounds are $j \succeq k$ — $k$ itself
(extension/bridge: a second C), any forward stage including skips
(C → E without D), and the absorbing exits; **never backward** (from C,
no A or B). The transition graph is a DAG plus self-loops; each firm's
history is a monotone path. Admissibility enters the model as a structural
zero:

$$\lambda_{i, k\to j}(t) \;=\; \mathbf 1\{j \succeq k\}\cdot
\exp\!\big(\alpha_{k\to j} + \beta_{k\to j}^\top X(t)
+ \delta_{k\to j}^\top Z_{i,t}\big).
\tag{2.1}$$

Structural zeros are free information in the precise sense that the
likelihood assigns them probability zero with **zero parameters spent**, and
every unit of probability mass a flat model would waste on C → A is
reallocated to admissible cells.

### 2.2 Likelihood: exact factorization over transitions

From (0.1), with at-risk indicators $Y_{i,k}(t)$:

$$\log L = \sum_{(k \to j)\ \text{adm.}} \underbrace{\Big[
\sum_{m \in \mathcal T_{k\to j}} \log \lambda_{i_m, k\to j}(t_m)
- \sum_i \int_0^T Y_{i,k}(u)\, \lambda_{i, k\to j}(u)\, du
\Big]}_{\ell_{k\to j}(\alpha_{k\to j},\, \beta_{k\to j},\, \delta_{k\to j})} .
\tag{2.2}$$

Because the parameter blocks are **variation-free across transitions**, the
maximization separates: each $\ell_{k\to j}$ is a concave exp-link fit (same
argument as §1.1) with its own risk set, and the joint MLE is the collection
of per-transition MLEs. No new estimation theory is required — this is the
formal content of "the multi-state model decomposes into independent Poisson
fits" and it is a theorem, not a heuristic (ABGK ch. VII). If sparsity forces
partial pooling across transitions (§2.5), the separation is deliberately
broken by the prior and only then.

**Clock choice.** (2.1) runs on calendar time (Markov given covariates —
"clock forward"). The semi-Markov alternative makes intensities functions of
sojourn time $t - T^{\text{entry}}_{i,k}$ ("clock reset";
Prentice–Williams–Peterson 1981 is the canonical treatment of the two clocks
for ordered recurrent events). The pragmatic hybrid — calendar clock plus
sojourn-time covariate inside $Z_{i,t}$ — keeps concavity and lets the data
choose; it is also precisely where rung 3's gap covariate will live, so the
clock question is answered once and reused.

**Prediction objects.** With $\hat\lambda$'s in hand, transition probability
matrices come from the product-integral (Aalen–Johansen 1978):
$\mathbf P(s, t] = \prod_{(s,t]} \big(\mathbf I + d\mathbf A(u)\big)$, with
$\mathbf A$ the cumulative transition hazards. Under the Markov assumption
this is the transition matrix; if Markov fails, the same estimator still
consistently estimates **state occupation probabilities** (Datta–Satten
2001) — a usable robustness for stage-mix forecasts, but a *conditional*
one: the theorem needs censoring independent of the process (exactly what
C3 threatens) and observation from the time origin (delayed entry weakens
it), and it covers occupation probabilities seeded at the origin — not
landmark transition matrices $\mathbf P(s, t]$ from a later $s$, which
remain Markov-dependent.

### 2.3 Competing risks, done at the right level

From stage $k$, the admissible targets compete: C → C, C → D, C → E, exit.
Two hazard concepts exist and they answer different questions
(Putter et al. 2007):

- **Cause-specific intensity** $\lambda_{k\to j}$ — what (2.1) models. It is
  the object that *simulates* (thinning over competing intensities) and the
  object whose likelihood factorizes. All modeling happens here.
- **Subdistribution hazard** (Fine–Gray 1999) — the hazard "of the cumulative
  incidence", whose risk set unnaturally retains firms already departed to
  competing states. It answers "how does $X$ shift the eventual fraction
  taking C → D" directly, and it is the right *reporting* transform when a
  stakeholder asks about cumulative incidence — but it is not an intensity,
  does not simulate, and must never be plugged into (2.2).

The cumulative incidence itself is
$F_{k\to j}(t) = \int_0^t S_k(u^-)\, \lambda_{k\to j}(u)\, du$ with
$S_k(u) = \exp(-\sum_{j'} \int_0^u \lambda_{k \to j'})$ — compute it from
cause-specific fits; don't refit for it.

### 2.4 Censoring and truncation: where this rung earns survival analysis

**(C1) again — benign.** A firm sitting in stage C at $T$ contributes
$\exp(-\sum_j \int \lambda_{C\to j})$ over its observed sojourn: standard,
automatic, correct.

**(C3) — the immortal-firm bias, now derivable.** Suppose a fraction of the
stage-$k$ risk set is secretly dead ("zombies", $Z$-set) with true funding
intensity 0, and the naive risk set is alive ∪ zombies. The estimating
equation for the intercept solves
$\sum_m 1 = \int \sum_{i \in \text{alive} \cup Z} \hat\lambda\,dt$, so
$\hat\lambda \approx \lambda \cdot \frac{|\text{alive}|}{|\text{alive}| +
|Z|}$: **all hazards deflate** by the contamination fraction. Worse, the
mark-factor softmax spreads probability over zombies with stale features, so
covariates correlated with staleness (recency!) get *inverted* weight — the
Bayes-optimal response to a contaminated pool is to bet against staleness,
which is exactly the inversion we measured (E2). Because exit correlates
with the frailty that also drives funding, this is **informative** censoring:
no reweighting by an estimated exit model with observables fixes it (we
measured AUC 0.877 to be insufficient — the residual confounding is
structural). Honest treatments, in order of cheapness:
1. **Exposure windows**: administratively censor any firm $c$ years after
   its last observed activity. Be precise about what this does: the window
   rule is a predictable function of the observed history, so it is
   *exactly* independent censoring (ABGK III.2.2) — its benefit is not
   repairing the censoring mechanism but **capping each dead firm's
   immortal exposure** at $c$ years instead of $T - \text{death}$. The
   deflation is bounded, not removed; smaller $c$ trades residual zombie
   exposure against censoring genuine long-gap re-raises (and, where
   hazards depend on the gap, restricts the estimand to gaps $< c$);
2. **Mover–stayer / cure mixture** (Farewell 1982): a latent "will never
   raise again" class with logistic membership; identified mainly by the
   long-gap tail, so fragile at short $T$;
3. **Sensitivity intervals**: report hazards under risk-set rules with and
   without the window (better: over a grid of $c$'s). Only the no-window
   side carries a one-sided bound interpretation — extra zero-intensity
   exposure can only deflate; the windowed estimate is *not* a guaranteed
   upper bound, since up to $c$ years of zombie exposure per dead firm
   survives inside it. What is *not* honest is reporting a single number
   from the naive risk set.

**(C2) — delayed entry.** Every stage-$k$ spell starts at an observed event
(entry into $k$), so within-stage truncation is handled by risk sets
mechanically. The residual issue is selection *into* observation (first
funding), deferred to rung 5 — with one interaction worth naming: stage-Seed
risk sets are the entry cohort, so Seed-stage estimates inherit the entry
selection most strongly.

**Ties and discreteness.** Funding dates are days and cluster; weekly work
makes ties endemic. Options, in increasing fidelity: Breslow (1974) / Efron
(1977) tie corrections in the partial likelihood; or go genuinely
discrete-time — the grouped-PH result (Prentice–Gloeckler 1978) says that if
the continuous-time model is proportional hazards, the induced discrete-time
hazard obeys $\text{cloglog}\, h_{j} = \alpha_j + \beta^\top x$ with the
*same* $\beta$: complementary log-log, not logit, is the
aggregation-coherent link (logit ≈ cloglog only in the rare-event limit —
our `sector_hazard` logit is fine at weekly deal rates, and the statement of
*why* it is fine is exactly this limit; Allison 1982 for the discrete-time
event-history frame).

### 2.5 Statistical problems in calibration

1. **Sparse transitions.** C → E, D → D are rare; some admissible cells may
   have single-digit counts. Consequences: separation (MLE at infinity) and
   noisy $\hat\alpha_{k\to j}$. Remedies: Firth (1993) penalization
   per-transition, or partial pooling — hierarchical shrinkage of
   $(\alpha_{k\to j}, \beta_{k\to j})$ toward stage-level means, which
   deliberately breaks the factorization of (2.2) in exchange for variance.
   Report per-cell event counts next to every transition estimate — the
   count *is* the credibility.
2. **Stage mislabels.** With a row-stochastic confusion matrix $Q$
   (probability of recording stage $j'$ for true $j$), the observed
   transition flows are pre/post-multiplied by $Q$: mass leaks into
   *inadmissible* cells. This inverts into a diagnostic: the empirical
   C → A cell must be ≈ 0; its size estimates label noise (we ship with 3%
   sector mislabels in the synthetic data and expect stage noise of the
   same order in real vendors). Under noise the clean-cell estimates
   attenuate like the misclassification rate — full correction requires a
   hidden multi-state model (stage as latent, labels as emissions), which
   is rung-2's own escalation path if the C → A cell is materially nonzero.
3. **Self-loop semantics.** "C again" mixes true extensions with vendor
   round-relabeling; the self-loop intensity is only as meaningful as the
   deal-type field. Data validation, not modeling, is the fix.
4. **Absorbing-state incompleteness** is (C3) — see above; the exit states
   exist in the model precisely so that *labeled* exits stop accumulating
   hazard, which removes the measured rich-get-richer failure (winners
   never leaving the pool).

### 2.6 What we need to see in the data

Monotone stage sequences (audit the raw histories against $\succeq$ —
violation rate = label noise estimate); per-transition event counts (the
model's resolution is bounded by the thinnest admissible cell); the deal-type
field's provenance for self-loops; exit labels where they exist (IPO/M&A are
mostly labeled — use them; failure mostly isn't — exposure windows); and
pool-size trajectories per stage (for rung-5 interaction).

---

## 3. Rung 3 — History as covariates: the self-modulated GLM

The distinction this rung formalizes: **the predictive content of Hawkes does
not require estimating a Hawkes process.** Fix the timescales, put history
into the intensity as covariates, keep concavity, keep sign freedom.

### 3.1 Model, and why it is still a legitimate point process

Augment (2.1) with two history functionals:

$$\lambda_{i,k\to j}(t) = \mathbf 1\{j \succeq k\}\,
\exp\!\Big(\alpha_{k\to j} + \beta^\top X(t)
+ \phi^\top H_{s(i)}(t) - \psi\, g\big(t - T_i^{\text{last}}\big)\Big),
\tag{3.1}$$

where $H_s(t) = \big(H_s^{(\tau_1)}(t), \dots\big)$ are exponentially
filtered sector deal streams at **fixed** half-lives $\tau_r$,

$$H_s^{(\tau)}(t) = \sum_{t_m < t,\ s_m = s} e^{-(t - t_m)/\tau}
\quad\Longleftrightarrow\quad
dH_s^{(\tau)} = -\tfrac{1}{\tau} H_s^{(\tau)}\, dt + dN_s(t),$$

and $g$ is a fixed monotone gap transform (our $\log(1 + \text{days}/30)$;
M6 derives the induced hazard-vs-gap shapes and the binned-dummy check).

Two mathematical readings:

**(i) Markov embedding.** The ODE form shows $(X, H_s, t - T_i^{\text{last}})$
is a finite-dimensional state vector, so (3.1) is a *bona fide* conditional
intensity — predictable, adapted, with likelihood (0.1) — depending on
$\mathcal H_t$ only through a finite summary. Kernels with rational Laplace transform (matrix-exponential kernels — sums
of exponentials as the workhorse case, with Erlang and damped-oscillatory
kernels also in the class) are the ones with finite Markov embeddings;
power-law kernels have none — which is why fixed-decay exponential filters,
and not fixed-decay power laws, are the free lunch.
Given fixed $\tau$'s, all parameters enter the linear predictor ⇒ **the
log-likelihood is concave** (same proof as §1.1). The entire fragility of
Hawkes estimation lives in the decay; freeze it and the estimation problem
returns to GLM-land.

**(ii) Discrete time.** Weekly, (3.1) is a log-linear count autoregression:
$\log \mu_{s,t} = a + \beta^\top X_t + \phi\, \tilde H_{s,t}$ with
$\tilde H$ an EWMA of past counts — the Fokianos–Tjøstheim (2011) log-linear
Poisson autoregression. Their geometric-ergodicity theory gives contraction
conditions of the form $|b| < 1$, $|b + c| < 1$ on the feedback
coefficients: **log-link count feedback is only conditionally stable**, and
feeding *raw* lagged counts into an exponent violates the Lipschitz
structure the theory needs — that is Mechanism B, and it is not hypothetical
(we measured a 169k-event metastable excursion in a world with spectral
radius nominally ≪ 1). Design rule that follows: history enters as *bounded
or slowly-varying transforms* — EWMA levels, $\log(1 + \cdot)$ counts —
never raw counts in the exponent.

### 3.2 Internal covariates: the price sticker

$H_s$ and the gap are **internal** covariates (they are functions of the
process being modeled). Kalbfleisch–Prentice's classical warning applies
with full force: with internal covariates the *intensity* keeps its
interpretation, but survival-type marginal statements
("$P(\text{no deal in } [t, t+90d])$" computed by plugging in today's
$H_s$) are **not** probabilities of anything — the covariate path would
itself evolve with the process. Consequences:
- one-step (weekly) prediction is clean and needs nothing extra;
- multi-step prediction and any scenario analysis must **simulate the
  system jointly** (draw events, update $H_s$, repeat) — at which point the
  stability conditions of §3.1(ii) are load-bearing, not pedantry;
- this is also the honest boundary of rung 3: it *predicts* with history but
  cannot *propagate* a shock with calibrated amplification. Amplification
  semantics is precisely what rung 4 buys and the only thing it uniquely
  buys (branching interpretation, cluster sizes, endogenous share).

### 3.3 Statistical problems in calibration

1. **The κ–θ ridge, inherited.** Jointly free decay is not identifiable at
   weekly resolution and noisy at day resolution without pooled volume (M3:
   the Fisher information in the decay collapses as bin width / half-life
   grows; measured in exp24 — bucket ≤ 1 half-life identifies timing, ≥ 4
   half-lives destroys it). Fixing $\tau$ converts an ill-conditioned
   direction of the likelihood into a *modeling choice*, which is the
   entire point of this rung.
2. **Filter collinearity.** EWMAs at nearby half-lives are strongly
   correlated ($H^{(30d)}, H^{(90d)}, H^{(180d)}$ pairwise corr typically
   > 0.9 on smooth deal flow); the $\phi$'s individually are then
   ridge-identified only. Use 2–3 *well-separated* half-lives; interpret the
   fitted *curve* $\sum_r \phi_r e^{-u/\tau_r}$ (the implied kernel), never
   individual $\phi_r$'s. This is a small basis expansion of the kernel — the
   fixed-grid cousin of nonparametric Hawkes estimation.
3. **Frailty masquerading as heat (spurious excitation).** If a latent
   positive-autocorrelated factor $\xi(t)$ multiplies the truth and the
   model omits it, the score of $\phi$ at $\phi = 0$ has expectation
   $\propto \int \text{Cov}(H_s(t), \xi(t))\, dt > 0$, because $H_s$ is
   built from past events which loaded on $\xi$: **the fitted heat
   coefficient absorbs the common factor.** Measured: freely-fitted
   excitation radius 2.4× truth under a latent factor (campaign, S1). The
   coefficient $\phi$ is therefore a *predictive* loading, reported as an
   upper bound on contagion — the same discipline as rung 4, one rung
   early. The classical frame is Heckman–Borjas (1980): occurrence
   dependence vs unobserved heterogeneity is not separable from a single
   realization without restrictions.
4. **Coverage drift masquerading as heat.** $H_s$ counts *observed* deals;
   a vendor coverage expansion raises $H_s$ with no market change (S4).
   Guard: coverage proxies inside $X(t)$, PIT drift as tripwire.
5. **Inhibition shape risk.** $\psi$ and $g$ fix a parametric cooldown; the
   E2.1 binned-dummy protocol (M6) estimates the shape once, then freezes
   the transform. The measured self-inhibition *ordering* is robust (rank
   corr 0.937 under misspecification), the shape is not.

### 3.4 What we need to see in the data

Residual clustering *after* rung 2 (PIT autocorrelation, dispersion) — this
rung exists to absorb it; a stable inhibition sign across stages ($\psi > 0$
uniformly); the implied kernel curve concentrated at horizons the deal cycle
makes plausible (months, not hours); and the escalation gate to rung 4:
whatever clustering survives (3.1).

---

## 4. Rung 4 — Full Hawkes: branching structure, cascades of Poisson processes

### 4.1 Model

$$\Lambda_s(t) = \mu_s(t) + \sum_{r=1}^S \sum_{t_m < t,\ s_m = r}
A_{sr}\, \kappa_\theta(t - t_m), \qquad \kappa_\theta \ge 0,\
\int_0^\infty \kappa_\theta = 1,
\tag{4.1}$$

with covariate-driven immigration $\mu_s(t) = \exp(a_s + \beta_s^\top X(t))$
and branching matrix $A \ge 0$. Subcriticality $\rho(A) < 1$ is the standing
constraint (enforced in the repo via row-sum bounds in
`fit_sector_count_model`; nonlinear/exp-link variants instead need
Brémaud–Massoulié (1996) Lipschitz conditions, and their log-link count
cousins are exactly the Mechanism-B territory of §3.1).

### 4.2 The cluster representation (Hawkes–Oakes 1974) — the theorem that organizes everything

**Statement.** A subcritical linear Hawkes process is *equal in law* to a
Poisson cluster process: immigrants arrive as an inhomogeneous Poisson
process with rate $\mu_s(t)$; each event (immigrant or not) of type $r$
independently generates direct offspring in type $s$ as an inhomogeneous
Poisson process with rate $A_{sr}\kappa_\theta(u)$, $u$ = age; the process is
the superposition of all generations.

**Consequences, each load-bearing for us.**
- *Stationarity and explosion*: the offspring cascade is a multitype
  branching process with mean-offspring matrix $A$; it is a.s. finite for
  $\rho(A) \le 1$ (non-degenerate case — even critical cascades die out),
  but the **expected** cluster size is finite, and a stationary Hawkes
  version exists, iff $\rho(A) < 1$; there the expected total progeny of
  one type-$r$ event is the $r$-th column sum of $(I - A)^{-1}$.
  Subcriticality is therefore the operative constraint. This is the quantitative content of
  every stability guard in the repo, and $(I-A)^{-1}$ is the object scenario
  amplification actually uses.
- *Endogeneity share*: the stationary fraction of events that are offspring
  (not immigrants) is $1 - \bar\mu / \bar\Lambda$; reporting it is the
  honest summary of "how much of deal flow is contagion" — with the S1
  caveat that it is an upper bound (§4.5).
- *The immigrant/offspring split is rung 5's skeleton*: newcomers enter
  through $\mu$, incumbents' repeat rounds live in the offspring channel
  (N4). The two-step of rung 5 is this representation read at the mark
  level.

### 4.3 Cascades of Poisson processes: Simma–Jordan (2010) and EM

Simma & Jordan, *Modeling Events with Cascades of Poisson Processes* (UAI
2010), work directly with the cluster representation as a **latent-variable
model**: each event $m$ carries a latent parent $u_m \in \{0\ (\text{immigrant}),
1, \dots, m-1\}$. The complete-data log-likelihood

$$\log L_c = \sum_m \Big[ \mathbf 1\{u_m = 0\} \log \mu(t_m)
+ \sum_{l < m} \mathbf 1\{u_m = l\}\, \log \big(A_{s_m s_l}
\kappa_\theta(t_m - t_l)\big) \Big]
- \int_0^T \mu - \sum_l \int_{t_l}^{T} A_{\cdot s_l} \kappa_\theta(u - t_l)\, du$$

is *separable given the parents* — concave in the branching and (linearly
parametrized) immigration blocks, while the decay block admits a closed-form
update but is not concave in general (it depends on the parametrization,
and the right-truncation exposure correction below breaks it) — so the
augmentation restores most, not all, of the tractability the marginal
likelihood lacks. EM:

- **E-step** (exact, by Bayes on the competing Poisson thinnings):
  $p_{ml} \propto A_{s_m s_l} \kappa_\theta(t_m - t_l)$ for $l \ge 1$,
  $p_{m0} \propto \mu(t_m)$, normalized over $l \in \{0, \dots, m-1\}$.
- **M-step** (closed-form for exponential kernels): branching entries update
  as expected offspring counts over exposure,
  $\hat A_{sr} \propto \sum_{m: s_m = s} \sum_{l: s_l = r} p_{ml}$; the decay
  updates as a $p$-weighted mean of parent–child gaps. Right-window
  truncation is handled *inside* the M-step by replacing each parent's unit
  exposure with $\int_0^{T - t_l} \kappa_\theta$ — which is exactly the
  boundary correction of §4.4.

The same EM is Veen–Schoenberg (2008) in seismology (ETAS) and, with a
histogram kernel, the nonparametric EM of Lewis–Mohler (2011); Simma–Jordan's
contribution is the general cascade formulation at scale with flexible
kernels. Two practical reasons this route matters to us beyond aesthetics:
(i) EM is monotone and keeps $A \ge 0$ automatically — but subcriticality is
**not** automatic: the parent-mass accounting bounds only an event-weighted
*average* of column sums below 1, and individual columns (hence $\rho(A)$)
can exceed 1 when a rare source type is attributed many children, so
$\rho(\hat A) < 1$ must be checked and, as in the repo's fitters, enforced; (ii) the **posterior parentage matrix $p_{ml}$ is the
diagnostic**: expected endogeneity share, expected cluster depth, and — the
sharpest available probe of frailty-vs-contagion — whether attributed
parent–child pairs concentrate at the kernel's timescale or spread evenly
(a common factor inflates $A$ but produces parentage attributions with no
timescale structure).

**Direct MLE** is the alternative: for exponential kernels the recursion
$R_s(m) = e^{-(t_m - t_{m-1})/\theta}\big(R_s(m-1) + \mathbf 1\{s_{m-1} =
s\}\big)$ evaluates (0.1) in $O(n)$ per pass. Jointly in $(A, \theta)$ the
likelihood is non-concave; **for fixed decay it is concave when the
immigration part is linearly parametrized** ($\log(\mu + \text{linear in }
A)$ concave, compensator linear — Ogata's setting). With the exp-link
immigration of (4.1), even the fixed-decay problem is only *block*-concave
($A$ given the baseline, baseline given $A$), so the honest algorithm is
profile likelihood over a $\theta$-grid with alternating maximization
inside each grid point. The unification with rung 3 is an **analogy, not an
identity**: both reuse the same fixed-decay kernel-sum features over a
$\theta$-grid, but rung 3 puts them *inside the exponent* (log-linear)
while (4.1) adds them *linearly* to the intensity — the rung-3 fits do not
compute (4.1)'s profile likelihood, they fit a neighboring family.
Asymptotics for the MLE are Ogata (1978); GOF is time-rescaling with the
full $\hat\Lambda_s$, which now also tests the kernel shape.

### 4.4 Censoring: what (C1)/(C4) do to a branching process

- **Edge effects (right censoring of clusters).** An event near $T$ has
  unobserved offspring: naive fitting deflates $A$ (its children are
  missing) and inflates $\mu$ (orphaned mid-window events whose parents
  predate the window get called immigrants at the left edge). The
  right-edge correction is exact within EM (exposure $\int_0^{T -
  t_l}\kappa$ per parent — the offspring process of an observed parent
  really is Poisson observed on $[t_l, T]$); the left edge is only
  *mitigated* by burn-in — in-window events with pre-window parents stay
  misattributed to immigration unless the unobserved pre-window history is
  modeled (e.g. under an explicit stationarity assumption). Neither is
  optional at our windows: with 90-day kernel mass and a 10-year window the
  bias is small, but per-sector fits on short subwindows can lose
  double-digit percent of $A$.
- **Interval censoring (C4).** Binning makes the exact Hawkes likelihood
  intractable: conditional on the pre-bin history, a bin count is an
  immigrants-plus-within-bin-cascade total — not Poisson (and a Hawkes
  process is not a Cox process, so "Poisson given the intensity path" is
  not even well posed). The MBPP likelihood (the repo's `mbpp/`) is the
  tractable *replacement*: bin counts treated as conditionally Poisson
  given pre-bin history — exact at rung 1, an approximation at rung 4 whose
  error grows with within-bin excitation — and the κ–θ ridge (M3) is the
  statement of what the bin sums cannot see regardless of the likelihood
  used.
  The rule of thumb it licenses: **bin ≤ one kernel half-life to estimate
  timing; at ≥ 4 half-lives only the branching scale survives.** This is
  the quantitative reason rung 3 fixes decays: weekly data was never going
  to identify them anyway.
- **(C3) enters through the mark factor** as before; at the sector-count
  level zombies do not exist, which is why rung 4's *sector* layer shares
  rung 1's immunity.

### 4.5 The deep identifiability problem: frailty vs contagion

Clustered, overdispersed event flow has two structural explanations:
**contagion** (events beget events — Hawkes) and **frailty** (a latent
common factor modulates everyone's rate — doubly-stochastic/Cox process).
Both produce overdispersion, autocorrelated counts and cross-sector
comovement; from one realization of moderate length, without further
structure, they are close to observationally equivalent (the duration-
analysis version of this is Heckman–Borjas 1980's state-dependence vs
heterogeneity; the point-process version is the shared wisdom of the credit-
risk literature). What actually separates them:
1. **Covariates first** (Duffie–Saita–Wang 2007): put every observable
   common driver into $\mu(t)$; whatever comovement covariates explain was
   never contagion. Our ladder does this by construction (rungs 1–3 precede
   4).
2. **Cross-sectional restrictions** (Duffie–Eckner–Horel–Saita 2009, frailty
   correlated default): a common factor loads *simultaneously* on all
   sectors; contagion propagates along the event stream with the kernel's
   lag structure. With many sectors and a factor model for $\xi$, the two
   have different fingerprints — this is genuinely estimable but is a
   latent-factor model (particle MCMC / EM over $\xi$), an escalation
   *beyond* rung 4 that we have deliberately not built.
3. **Ancestry information** — never available in market data.
4. **Both can be present** (Azizpour–Giesecke–Schwenkler 2018 find exactly
   that for corporate defaults after conditioning on covariates): the
   operational conclusion is not "pick one" but *report the fitted
   branching as an upper bound on contagion* — the standing S1 discipline,
   with our measured factor-free vs latent-factor gap (fitted radius 2.4×
   truth) as the calibration of how bad the bound can be.

### 4.6 What we need to see in the data

Event-time (day) resolution for any timing claim (C4 rule); volume: the
information about $\theta$ scales with the number of *parent–child pairs
within a half-life*, i.e. with clustering density, not raw counts; residual
clustering surviving rung 3 (the gate); covariates already absorbing the
macro cycle (else §4.5.1 says the fit is confounded by construction); and
for any simulation use, the stability guards on (Mechanism A: $\rho(A)$;
Mechanism B: no raw counts in exponents).

---

## 5. Rung 5 — The open universe: entry model + conditional choice

The mark space is not closed: a deal in $(s, t)$ may go to a firm outside
the risk set — a true entrant (invisible before its first round: (C2) as a
*population* phenomenon) or an untracked incumbent (vendor coverage). Rung 5
is the mark-level model of that opening.

### 5.1 The two-step as an exact factorization

Extend the mark space to $\mathcal M_{s,t} = R_s(t) \cup \{\text{new}\}$ and
decompose the mark kernel:

$$P(m \mid s, t, \mathcal H_t) =
\underbrace{\pi_{s,t}^{\,\mathbf 1\{m = \text{new}\}}
(1 - \pi_{s,t})^{\,\mathbf 1\{m \in R_s(t)\}}}_{\text{step 1: binary}}
\cdot
\underbrace{\Big[\frac{e^{q_{m,t}}}{\sum_{i \in R_s(t)} e^{q_{i,t}}}
\Big]^{\mathbf 1\{m \in R_s(t)\}}}_{\text{step 2: choice}},
\qquad \pi_{s,t} = \sigma\big(b_0 + \gamma^\top g_{s,t}\big),
\tag{5.1}$$

with $g_{s,t}$ **global** covariates only (sector deals last 90d, historical
first-financing share, macro as-ofs, log pool size — nothing firm-level
exists on the "new" branch by construction).

**Theorem (exact separation).** If $(b_0, \gamma)$ and the choice parameters
are variation-free, then
$\mathrm{NLL} = \mathrm{NLL}_{\text{binary}}(b_0, \gamma) +
\sum_{\text{incumbent deals}} \mathrm{NLL}_{\text{choice}}$, and maximizing
the two terms separately **is** the joint MLE. *Proof*: take logs in (5.1)
and sum over events; the two terms share no parameters, so the joint
maximizer is the pair of separate maximizers. ∎ — No approximation is made
by fitting the newcomer model and the choice model independently. The
empirical companion (exp28): N3's incumbent weights sit within ~0.03 (raw
weights) of the one-shot N1 fit — close but not identical, and the theorem
does not predict identity there, because N1 is a *different* likelihood:
its newcomer alternative shares the softmax denominator with the
incumbents, which the factorization in (5.1) deliberately does not.

**Composition upward (the thinning view).** Given the sector stream with
intensity $\Lambda_s(t)$ and a predictable newcomer probability
$\pi_{s,t}$, the marking theorem for conditional intensities gives the
newcomer substream the intensity $\pi_{s,t} \Lambda_s(t)$ — position-
dependent thinning. So step 1 composes with *any* rung's ground model into
a coherent **entrant-arrival intensity**, and in rung 4's language this is
precisely the immigrant channel: rung 5 is the Hawkes–Oakes
immigrant/offspring split (§4.2) read at the mark level, not a bolted-on
classifier. (The measured irony from N4 stands: fitted "excitation" radius
was *larger* on the immigrant stream, 0.30 vs 0.08–0.11 — entry waves are
where the latent factor lives, which is coherent with §4.5: common factors
masquerade as clustering most where covariates are thinnest.)

### 5.2 The entrant profile and the selection it carries

Optionally, the "new" branch gets a profile: entrant features at first
funding $\sim \mathcal N(\mu_s, \Sigma_s)$ (estimated on the train window),
and the exact integrated utility of a representative entrant is the Gaussian
log-MGF $w^\top \mu_s + \tfrac12 w^\top \Sigma_s w$ (M7). Two honesty
clauses, both consequences of the observation scheme:
- $(\mu_s, \Sigma_s)$ is a **selection distribution** — features of firms
  *conditional on having entered* (C2). It supports prediction of the next
  entrant's profile; it says nothing about the latent population of
  not-yet-funded firms (right truncation of entry: firms that will first
  raise after $T$ are invisible entirely).
- The pool-size coefficient inside the newcomer utility is **not identified
  separately from the intercept unless pool size varies over time** (M7) —
  with static pools, $\log|R_s|$ is collinear with the constant. Measured:
  N2b. Do not interpret $b_0$ structurally.

### 5.3 Statistical problems in calibration

1. **Entrant vs untracked incumbent is an unidentified mixture.** Both land
   in "new"; the mixing weight is not estimable without auxiliary coverage
   data (a vendor census, or a second vendor). Likelihood cost of
   conflation: nearly zero (+0.003 nats with oracle labels, exp28);
   *semantic* cost: total — say "newcomer probability", never "entrant
   probability", on real data. If a coverage series exists, it belongs in
   $g_{s,t}$ and partially de-confounds.
2. **Rare-event logistic bias.** If the newcomer share is small, the MLE
   intercept is biased (King–Zeng 2001); Firth (1993) fixes it. At our
   measured shares (~27% of deals overall, 23–25% in test windows — of
   which untracked incumbents account for ~12% of all deals) this is minor
   but costs one line to include.
3. **Calibration over time is the real gate.** The operational requirement
   is that predicted quarterly newcomer *shares* track realized ones
   (measured: mean absolute error 3–4pp with context covariates vs 6–9pp
   with a constant, exp28/N1). A drifting share with stable $g_{s,t}$ is
   the standing tripwire for coverage change (S4) — the model doubles as a
   data-quality monitor.
4. **Step-1 effort allocation.** Entrant quality explains only 13–15% of
   the newcomer utility (exp28); the share is carried by context and pool
   size. Invest in $g_{s,t}$, not in richer entrant profiles — a measured
   design instruction.
5. **Sampled softmax on the incumbent side** stays consistent under uniform
   sampling with the winner retained (McFadden 1978; M4) — the incumbent
   fits scale to large pools without changing the estimand.

### 5.4 What we need to see in the data

Pool-size variation over time (else drop the pool-size term, M7); a
first-financing flag with known provenance (it defines the entrant label);
any coverage/backfill metadata the vendor exposes; quarterly newcomer-share
history (the calibration target); and stage mix of entrants (Seed-heavy
entry interacts with rung 2's Seed risk sets, §2.4).

---

## 6. The censoring ledger: right-censored or not, in one place

The question "what if we have right-censored data or not" deserves a flat
answer per rung. The unifying fact is (0.1) + independent censoring: **a
right-censored spell contributes its survival factor
$\exp(-\int_{\text{observed spell}} \lambda)$ and nothing else** — censoring
handled correctly costs nothing but the lost information itself; censoring
handled incorrectly (dropping open spells, or keeping zombies at risk) has a
derivable bias direction.

| rung | if right-censored (real life) | if magically complete | what censoring can break |
|---|---|---|---|
| 1 (NHPP) | nothing to do — the window-likelihood *is* the censored likelihood | identical | nothing (sectors don't die) |
| 2 (multi-state) | open sojourns contribute $e^{-\sum_j \int \lambda_{k\to j}}$; risk sets do the work | every path ends in an absorbing state; (2.2) is complete-data | dropping open spells → right truncation, hazards inflate; zombies (C3) → hazards deflate by the contamination fraction and recency weights invert; both directions derived in §2.4 |
| 3 (+ history cov.) | as rung 2; additionally $H_s$ must use only *observed* events | as rung 2 | coverage drift enters through $H_s$ and reads as heat (S4) |
| 4 (Hawkes) | cluster edge effects: offspring censored at $T$ deflate $\hat A$, orphans at the left edge inflate $\hat\mu$; exact EM exposure corrections §4.3–4.4 | no edge corrections needed | short subwindows lose double-digit % of $A$ without the correction; binning (C4) removes decay information regardless |
| 5 (two-step) | step 1 unaffected (per-event Bernoulli); entrant profile is a selection distribution under (C2) | entry still selective unless the *unfunded* population is observed — completeness of the event record does not close the universe | right truncation of entry: post-$T$ first-raisers invisible; mixture with untracked incumbents unidentified either way |

Three ledger notes worth stating once:

1. **The hierarchy of defects.** (C1) is free if respected; (C4) is a known,
   quantified information loss (κ–θ ridge) you choose by choosing
   resolution; (C2) is handled mechanically by delayed-entry risk sets and
   *statistically* by rung 5; **(C3) is the only defect that biases
   silently** — it violates the independent-censoring condition itself, and
   its mitigations (exposure windows, cure mixtures, bounds) are all
   partial. Ranked by danger: C3 ≫ C2 > C4 > C1.
2. **Positive-only is not censoring — and two distinct cautions live
   here.** For the *mark factor*, the absence of rejection labels (M8)
   means there is no firm-level negative class: use conditional/partial
   likelihood, never fabricate "rejected firm" rows. For the *timing
   factor*, firm-week person-period files with 0/1 event indicators are a
   perfectly valid discrete-time hazard likelihood (Allison 1982;
   Prentice–Gloeckler — §2.4 uses exactly this frame): a week without a
   deal is an observed non-event *provided the at-risk indicator is
   correct*. The danger is not the person-period construction; it is C3
   feeding it contaminated risk sets.
3. **Every rung reports its censoring treatment.** The standing discipline:
   any hazard-like number ships with (a) the risk-set rule that produced
   it, (b) the exposure-window choice if any, and (c) for rung 4, whether
   edge corrections were applied. These are reproducibility metadata of the
   same rank as the split dates.

---

## 7. Annotated literature map

Grouped by what each source is *for* in this project. One line of honesty
per entry beats a wall of citations.

**Point processes, foundations.**
- Daley & Vere-Jones (2003/2008), *An Introduction to the Theory of Point
  Processes* I–II — the factorization (M1), Jacod likelihood, marked
  processes; the book this document's (0.1) lives in.
- Andersen, Borgan, Gill & Keiding (1993), *Statistical Models Based on
  Counting Processes* [ABGK] — the martingale estimation theory, independent
  censoring (III.2), multi-state likelihoods (VII); the single most
  load-bearing reference for rungs 2–3.
- Fleming & Harrington (1991), *Counting Processes and Survival Analysis* —
  the same machinery, more gently.
- Aalen, Borgan & Gjessing (2008), *Survival and Event History Analysis: A
  Process Point of View* — the modern process-first survival text; closest
  in spirit to how this repo is built.

**Hawkes and cascades.**
- Hawkes (1971), Biometrika — the model.
- Hawkes & Oakes (1974), J. Appl. Prob. — the cluster representation (§4.2);
  the theorem that makes stability, endogeneity share, and rung 5's
  immigrant reading one object.
- **Simma & Jordan (2010), *Modeling Events with Cascades of Poisson
  Processes*, UAI** — the cascade/latent-parent formulation and EM at scale
  (§4.3); our preferred estimation route for rung 4 because its E-step
  by-product (posterior parentage) is the honest endogeneity diagnostic.
- Veen & Schoenberg (2008), JASA — the same EM in seismology (ETAS), with
  the boundary corrections we adopt.
- Lewis & Mohler (2011), preprint — nonparametric (histogram-kernel) EM;
  the escalation if the exponential kernel misfits.
- Ogata (1978), Ann. Inst. Statist. Math. — MLE asymptotics for point
  processes; (1981), IEEE IT — thinning simulation; (1988), JASA — ETAS +
  residual analysis practice.
- Brémaud & Massoulié (1996), Ann. Prob. — stability of nonlinear Hawkes;
  the theory behind our exp-link caution.
- Bacry, Mastromatteo & Muzy (2015), *Hawkes processes in finance* — survey;
  context for branching-ratio ("endogeneity") reporting conventions.

**Survival analysis, censoring, truncation.**
- Cox (1972), JRSS-B; Cox (1975), Biometrika — partial likelihood; with
  McFadden (below), the two parents of our mark factor.
- Kalbfleisch & Prentice (2002), *The Statistical Analysis of Failure Time
  Data* — internal vs external covariates (§3.2's warning), grouped data,
  length bias.
- Andersen & Gill (1982), Ann. Statist. — Cox for counting processes /
  recurrent events; the large-sample backbone.
- Prentice, Williams & Peterson (1981), Biometrika — clocks and risk-set
  conventions for ordered recurrent events (§2.2's clock choice).
- Prentice & Gloeckler (1978), Biometrics — grouped PH ⇒ cloglog; the
  aggregation-coherent discrete-time link (§2.4).
- Allison (1982), Sociological Methodology — discrete-time event history;
  the frame our weekly hazards sit in.
- Turnbull (1976), JRSS-B — arbitrary grouping/censoring/truncation; the
  ancestor of the MBPP viewpoint (C4).
- Keiding (1990), Phil. Trans. R. Soc. A — delayed entry / Lexis diagram
  (C2).
- Farewell (1982), Biometrics — mixture cure; our mover–stayer option for
  (C3).
- Cook & Lawless (2007), *The Statistical Analysis of Recurrent Events* —
  the applied companion for repeat funding rounds.
- Hougaard (2000), *Analysis of Multivariate Survival Data*; Vaupel, Manton
  & Stallard (1979), Demography — frailty: the "other explanation" of
  clustering (§4.5).

**Multi-state and competing risks.**
- Aalen & Johansen (1978), Scand. J. Statist. — the product-integral
  transition-probability estimator (§2.2).
- Andersen & Keiding (2002), Stat. Methods Med. Res.; Putter, Fiocco &
  Geskus (2007), Stat. Med. — multi-state practice, cause-specific vs
  subdistribution.
- Fine & Gray (1999), JASA — subdistribution hazards; reporting transform,
  not a simulation object (§2.3).
- Datta & Satten (2001), Statist. Prob. Letters — Aalen–Johansen consistency
  for state occupation without Markov; the robustness we lean on for
  stage-mix forecasts.
- Lando & Skødeberg (2002), J. Banking & Finance — rating migration via
  transition intensities; rung 2's closest applied cousin.
- Firth (1993), Biometrika — penalized likelihood for separation/sparse
  cells (§2.5, §5.3).

**Discrete choice (the mark factor).**
- McFadden (1974) — conditional logit; algebraically our softmax mark
  kernel.
- McFadden (1978) — sampling of alternatives; consistency of sampled
  softmax (M4).
- Train (2009), *Discrete Choice Methods with Simulation* — the reference
  for extensions (nested/mixed logit) if firm-level heterogeneity escalates.

**Count time series (rung 3's discrete face).**
- Fokianos, Rahbek & Tjøstheim (2009), JASA — Poisson autoregression.
- Fokianos & Tjøstheim (2011), J. Multivariate Anal. — the log-linear
  variant and its ergodicity conditions; the theory behind Mechanism B
  discipline (§3.1).

**Frailty vs contagion, applied (the credit-risk cousins).**
- Duffie, Saita & Wang (2007), JFE — doubly-stochastic default intensities
  with covariates: rungs 1–3 for defaults, and the covariates-first
  doctrine (§4.5.1).
- Duffie, Eckner, Horel & Saita (2009), J. Finance — frailty correlated
  default: the latent-factor escalation beyond rung 4 (§4.5.2).
- Azizpour, Giesecke & Schwenkler (2018), JFE — contagion *and* frailty
  coexist in defaults after covariates; the template for our upper-bound
  reporting.
- Heckman & Borjas (1980), Economica — state dependence vs heterogeneity;
  the original statement of §4.5's identification problem, in duration
  language.

**Calibration & diagnostics.**
- Brown, Barbieri, Ventura, Kass & Frank (2002), Neural Computation — the
  time-rescaling theorem as a GOF tool (§1.3).
- Smith (1985), J. Forecasting; Brockwell (2007), Statist. Prob. Letters —
  the randomized PIT for discrete data; Czado, Gneiting & Held (2009),
  Biometrics — count-data calibration assessment and the non-randomized PIT
  variant. Together, our standing weekly diagnostic.
- King & Zeng (2001), Political Analysis — rare-events logistic correction
  (§5.3).
- Carroll, Ruppert, Stefanski & Crainiceanu (2006), *Measurement Error in
  Nonlinear Models* — attenuation under errors-in-variables (§1.4, exp22).

---

## Coda: the one-paragraph summary

Every rung is the likelihood (0.1) with a different intensity, estimated by
the same martingale theory, diagnosed by the same two instruments
(time-rescaling at event resolution, randomized PIT at bin resolution), and
guarded by the same censoring ledger. The ladder ascends by *adding
structure to $\lambda$* — covariates (1), states (2), fixed-timescale
history (3), estimated branching (4), an open mark space (5) — and each
addition is priced: what it assumes, what the data must show, and which
identification problem it wakes up (collinearity at 1; sparse cells and
immortal firms at 2; ridge and spurious excitation at 3; frailty vs
contagion at 4; unidentified mixtures at 5). The mathematics is old and
solid — Cox, Aalen, Hawkes–Oakes, McFadden — and the discipline is new only
in being enforced: no hazard without its risk-set rule, no branching ratio
without its upper-bound caveat, no newcomer probability sold as an entrant
probability.
