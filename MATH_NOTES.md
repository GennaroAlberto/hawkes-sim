# MATH_NOTES.md — derivations M1–M8

Deliverable for REVIEW.md §M. One section per item; each gives the **statement**, a
**derivation**, **what this licenses in our design**, and the **measured evidence**
(numbers from `docs/campaign_E1_E2_results.md`, `docs/stability_and_explosion_report.md`,
`EXPERIMENTS.md`, cross-checked against `paper/complete_account.tex` and `paper/book/`
rather than assumed). No code is changed by this document.

**Notation.** Sectors $s=1,\dots,S$ ($S{=}12$), firms $i$ with sector map $s(i)$, weeks
$t=0,\dots,519$ (days $d$, `TRAIN_END` = week 416). Events $(t_k, s_k, i_k)$; internal
history $H_t$. Ground intensity $\Lambda_s(t)$; mark kernel $P(i\mid s,t,H_t)$; weekly
counts $Y_{s,t}$; macro covariates $X_t$ (FFUND, CPI_YOY, RUNEMP); firm features
$Z_{i,t}$ (LOCF); cooldown $C_{i,t}$; choice utilities
$q_{i,t}=(w_0+u_s)^\top Z_{i,t} + \eta_s C_{i,t}$.

---

## M1 — The marked-point-process factorization (why "two-stage" is lossless)

**Statement (Daley–Vere-Jones).** Let $N$ be a simple marked point process on $[0,T]$
with marks in the finite firm set $\mathcal I$, admitting a conditional intensity
$\lambda(t,i\mid H_t)$ (predictable; $N([0,t]\times\{i\})-\int_0^t\lambda(u,i)\,du$ a
martingale for each $i$). Define the ground intensity
$\lambda_g(t)=\sum_i\lambda(t,i)<\infty$. Then:

1. $\lambda(t,i)=\lambda_g(t)\cdot P(i\mid t,H_t)$ with
   $P(i\mid t,H_t):=\lambda(t,i)/\lambda_g(t)$ a probability kernel on $\mathcal I$
   (well defined off $\{\lambda_g=0\}$, where a.s. no event occurs);
2. the likelihood of a realization $\{(t_k,i_k)\}$ factorizes **exactly**:

$$
L \;=\; \underbrace{\Big[\prod_k \lambda_g(t_k)\Big]e^{-\int_0^T\lambda_g}}_{\text{ground (timing)}}\;\cdot\;
\underbrace{\prod_k P(i_k\mid t_k,H_{t_k})}_{\text{marks (which firm)}} .
$$

Grouping marks by sector (each firm belongs to one sector, so the mark
$i$ refines the mark $s$), $\lambda_i(t)=\Lambda_{s(i)}(t)\cdot P(i\mid s(i),t,H_t)$
with $\Lambda_s(t)=\sum_{i:s(i)=s}\lambda_i(t)$, and the ground factor further splits
into $S$ sector timing factors:

$$
\log L=\sum_s\Big[\sum_{k:s_k=s}\log\Lambda_s(t_k)-\int_0^T\Lambda_s\Big]
\;+\;\sum_k\log P(i_k\mid s_k,t_k,H_{t_k}).
$$

**Derivation (sketch).** Part 1 is a definition plus the observation that on
$\{\lambda_g(t)=0\}$ the compensator does not grow, so no event lands there. Part 2:
the point-process likelihood (Jacod's formula; Daley–Vere-Jones 2003) is
$L=\prod_k\lambda(t_k,i_k)\cdot\exp(-\sum_i\int_0^T\lambda(u,i)\,du)$. Substitute
$\lambda=\lambda_g P$; the exponent equals $\exp(-\int\lambda_g)$ because
$\sum_i\lambda(u,i)=\lambda_g(u)$ (Tonelli, all terms nonnegative), and the product
splits since $\prod_k\lambda_g(t_k)P(i_k|\cdot)$ is term-by-term separable. Nothing is
approximated: the "two-stage model" is an identity of likelihoods, and any joint model
of the marked process can be written this way. The chained refinement
$P(i\mid t,H)=P(s\mid t,H)\,P(i\mid s,t,H)$ is the same fact applied to the coarser
mark $s$.

**What our parametrizations restrict (the approximations live *inside* the factors).**

Ground factor — modeled as $\log\Lambda_{s,t}=a_s+\beta_s^\top X_t
+\sum_{r,\ell}b_{sr\ell}Y_{r,t-\ell}$, $b\ge0$ (`fit_sector_glm_fast`):

* **[G1] weekly aggregation.** The continuous-time compensator $\int_{\text{week}}\Lambda_s$
  is replaced by a per-week constant rate; within-week feedback is dropped (history
  enters only through *completed* past weeks).
* **[G2] count-sufficient history.** $H_t$ enters only via the count matrix
  $\{Y_{r,t-\ell}\}_{\ell\le L}$ — not via event timing inside weeks, nor via *which*
  firms won (identity-blind ground). True in regime A by construction; false in
  regime B, where firm-level fields feed back into the intensity.
* **[G3] nonnegative finite-lag excitation.** $b\ge0$, $L$ lags; inhibition cannot live
  here (it is delegated to the mark factor; see M6, M8).
* **[G4] log link on raw counts.** The exp-of-counts recursion is the fragile
  Mechanism-B specification (M5).
* **[G5] no latent factor.** The DGP's risk-appetite factor $F_t$ is unmodeled; it is
  absorbed into spurious excitation. Measured: fitted effective radius **0.355 vs true
  0.150** (E1, `docs/campaign_E1_E2_results.md`), i.e. treat fitted "contagion" as an
  upper bound.

Mark factor — modeled as softmax of linear utilities over a constructed candidate set
(`build_choice_sets` + `fit_choice_fast`):

* **[P1] linear utilities** $q=(w_0+u_s)^\top Z + \eta_s C$; no interactions.
* **[P2] LOCF features.** $Z_{i,t}$ is carried forward from deals *strictly before* the
  event day; staleness is an approximation of the true state.
* **[P3] risk-set rule.** Pool = tracked, active, funded-at-least-once firms of the
  sector, minus the last-funded firm (hard refractory); a restriction of the mark
  space, applied identically to data and model.
* **[P4] aggregate newcomer.** All first-time and out-of-universe winners collapse into
  a single alternative (M7); ~27% of events.
* **[P5] sampled softmax.** Denominator subsampled to 64 candidates + newcomer (M4).
* **[P6] cooldown as a feature.** True inhibition (indicator in A, exponential in B) is
  proxied by $\log(1+\text{gap}/30\text{d})$ (M6).

**What this licenses.** The report may claim the two stages are the ground/mark
decomposition — a lossless identity — with all approximations named and localized in
one factor each. Improving one factor never requires touching the other *as long as
parameters are not shared* (the precise condition is M8).

**Evidence.** The mark factor at the oracle ceiling: exp25 ladder, logit top-5 0.29 vs
oracle 0.29 (regime A, oracle pools); choice baselines A: test NLL 3.346 vs random
4.011, top-5 0.446; B: 3.796 / 4.172 / 0.316.

---

## M2 — Concavity of the event-time block-Hawkes MLE

**Statement.** For the one-layer model (`complete_account.tex` §blockhawkes)

$$
\ell(\Theta)=\sum_k \eta_{i_k}(t_k)-\sum_i\int_{b_i}^{d_i} e^{\eta_i(t)}\,dt,\qquad
\eta_i(t)=a_{s(i)}+\beta_{s(i)}^\top X_i-\rho_{s(i)}R_i(t)+\sum_b A_{s(i),b}E_{i,b}(t),
$$

with **fixed decays** $\omega_{\text{self}},\omega_{\text{cross}}$ (so
$\Theta=(a,\beta,\rho,A)\mapsto\eta_i(t)$ is affine), at-risk windows $[b_i,d_i)$
independent of $\Theta$, and $R_i,E_{i,b}$ bounded with $\int e^{\eta_i}<\infty$:
$\ell$ is concave on the convex feasible set $\{\rho\ge0,\ A\ge0\text{ on the mask}\}$,
strictly concave iff no direction $v\neq0$ has $v^\top\nabla\eta_i(t)=0$ for a.e.
observed $(i,t)$. The MLE is then the unique global maximizer.

**Derivation.** The event sum is linear in $\Theta$. Each integrand
$e^{\eta_i(t)}=\exp(\text{affine}(\Theta))$ is convex in $\Theta$ (convex increasing
$\exp$ composed with affine — the standard log-link exponential-family argument);
integrals of pointwise-convex functions over parameter-free domains are convex, so
$-\sum_i\int e^{\eta_i}$ is concave, and linear + concave = concave. Equivalently, the
Hessian $\nabla^2\ell=-\sum_i\int \nabla\eta_i\,\nabla\eta_i^\top e^{\eta_i}\,dt\preceq0$.
The *negative* inhibition term poses no problem: concavity needs $\eta$ affine, not the
coefficients signed — this is exactly why the exponential link admits self-inhibition
while a linear (additive) Hawkes cannot.

**Conditions, checked.**

* *Boundedness/integrability*: $R_i(t)\le N_i(T)$ and
  $E_{i,b}(t)\le N_b(T)/n_b$ (finitely many events, decaying exponentials), $X_i$
  finite, so $\eta_i$ is bounded on $[0,T]$ and $\int e^{\eta_i}<\infty$;
  differentiation under the integral is justified. If features were unbounded the
  integral could be $+\infty$ on part of parameter space — concavity survives on the
  effective domain but the MLE could sit at infinity (monotone likelihood); bounded
  fields rule this out.
* *Fixed decays are essential*: if $\omega$ is estimated, $\eta$ is no longer affine
  and concavity is lost — the analogue of the non-convex covariate-excitation MBPP.
  This is the mathematical basis for the "fixed decay banks" design.
* *At-risk sets must not depend on $\Theta$*: entry/exit are data. (They may depend on
  *history* — see M8.)

**The nonnegativity boundary.** The constraints define a convex feasible set, so the
constrained problem is still a concave program with a unique maximum (if strict).
The optimum need not be interior: KKT stationarity for $\rho_s\ge0$ reads
$\partial\ell/\partial\rho_s=-\mu_s\le0$ at $\rho_s=0$ — the score may be strictly
negative at an active constraint (the data "want" the wrong sign, e.g. a genuinely
inhibitory cross-sector pair pinned at $A_{sr}=0$). Consequences: (i) fitted zeros
accumulate at the boundary; (ii) Wald standard errors from the inverse Hessian are
invalid for boundary coordinates (nonstandard asymptotics — chi-bar-square mixtures,
Self & Liang 1987); reporting held-out likelihood instead, as the repo does, is the
correct response; (iii) uniqueness of the *maximizer* does not imply good
conditioning: $R_i$ and $E_{i,s(i)}$ are strongly correlated (a firm's own recent
activity tracks its sector's), so the $\rho_s$-vs-$A_{ss}$ split sits on a flat ridge.

**What this licenses.** Fit the event-time layer by any local method (L-BFGS-B,
Berman–Turner Poisson regression) with a global-optimum guarantee; no restarts needed;
block-sparsity masks, ridge, and group-lasso are convex additions that preserve this.

**Evidence.** exp23/E3 on regime B: self-inhibition *ordering* recovered with rank
corr **0.937** while individual contagion entries have corr $\approx0$ — global
optimum found, ridge direction unidentified, exactly as the geometry predicts. exp20:
$\gamma,\alpha$ recovered to ~0.05 max abs error from ~20k exact event times.

---

## M3 — The κ–θ ridge: Fisher information from event times vs interval counts

**Setup.** Scalar exponential Hawkes, kernel $\phi(\tau)=\kappa\theta e^{-\theta\tau}$
(branching ratio $\kappa=\int\phi$, decay $\theta$), baseline $\mu$; effective
relaxation rate of the mean intensity $r=\theta(1-\kappa)$ (from the segment ODE
$y'=\mu-r y$ of `complete_account.tex` eq. (seg_ode)). Data: (a) event times on
$[0,T]$; (b) bucket counts $C_w$ of width $\Delta$, $W=T/\Delta$ buckets.

**Step 1 — the stationary bucket mean carries zero θ-information.** At stationarity
$\Xi_w=\bar\xi\Delta=\mu\Delta/(1-\kappa)$: $\partial\Xi_w/\partial\theta=0$
identically, while $\partial\Xi_w/\partial\kappa=\mu\Delta/(1-\kappa)^2$ is order-one
and monotone. So *all* θ-information in counts lives in temporal structure
(transients and autocovariances), while κ is pinned by first moments (plus
overdispersion). This is the exact content of the book's ch. 7 whybox ("counts hide
the timescale") made quantitative.

**Step 2 — the θ-carrying statistic and its sensitivity.** For the stationary
exponential Hawkes, bucket-count autocovariances decay geometrically:
$\mathrm{Cov}(C_w,C_{w+h})=a(\kappa,\theta,\Delta)\,\varphi^{\,h-1}$ for $h\ge1$ with
AR root $\varphi=e^{-r\Delta}$ (Da Fonseca–Zaatour-type closed forms; only the
structure matters here). Thus θ enters the count law only through $\varphi$ (and
through amplitudes shared with κ). By the delta method,

$$
I_\theta(\Delta)\;=\;\Big(\frac{\partial\varphi}{\partial\theta}\Big)^2 I_\varphi
\;=\;(1-\kappa)^2\Delta^2\,e^{-2r\Delta}\;I_\varphi .
$$

The factor $\Delta^2 e^{-2\theta(1-\kappa)\Delta}$ is the **ridge-collapse rate**: the
sensitivity of the only θ-carrying statistic decays *exponentially* in the bucket
width (stronger than any polynomial $O((\theta\Delta)^{-p})$ bound).

**Step 3 — variance scaling.** Estimating an AR root from $W$ buckets gives
$\mathrm{Var}(\hat\varphi)\gtrsim(1-\varphi^2)/W$ (AR(1) benchmark; Poisson
observation noise inflates this further, increasingly so at large $\Delta$ where the
autocovariance amplitude saturates while the per-bucket variance grows $\propto\Delta$).
Substituting, with $x=r\Delta$:

$$
\mathrm{Var}(\hat\theta)\;\gtrsim\;\frac{r}{(1-\kappa)^2\,T}\cdot J(x),\qquad
J(x)=\frac{(1-e^{-2x})\,e^{2x}}{x}.
$$

* $x\ll1$ (fine buckets): $J\to2$ — a Δ-independent plateau; refining buckets below
  the relaxation time buys little through second moments, and the event-time
  likelihood (which also reads within-cluster timing) is the floor. As $\Delta\to0$
  the count likelihood converges to the event-time likelihood and the ridge opens up —
  the E1.2 rationale for the day-resolution track.
* $x\gg1$ (coarse buckets): $J\sim e^{2x}/x$ — exponential information destruction;
  the estimator slides along the $\kappa/\theta$ ridge and is dominated by
  prior/profile shape, i.e. *bias*, not noise.

**Step 4 — plug in the measured world.** exp24: $\theta=\ln2/7\approx0.099$/day,
$\kappa\approx0.35\Rightarrow r\approx0.064$/day, $1/r\approx15.5$ d.

| bucket | $x=r\Delta$ | $J(x)$ | predicted SE vs 1d | measured (16k events) |
|---|---|---|---|---|
| 1 d | 0.064 | 2.13 | 1.0 | $\hat\theta=0.112\pm0.030$ (unbiased) |
| 7 d | 0.45 | 3.25 | ×1.2 | $\approx$ same as 1 d ("loses almost nothing") |
| 28 d | 1.80 | 19.8 | ×3.0 + noise inflation + ridge bias | $0.081\pm0.014$: **biased low ~20%** |

At 28 d the quadratic (variance) picture breaks down and the ridge takes over: the
28-d estimate is *systematically* low with a deceptively small spread — the classic
signature of an unidentified direction collapsing onto a ridge, aggravated by the
within-bucket uniform imputation (smearing events across 28 d flattens short-lag
clustering, mimicking a slower decay). κ, read from first moments, is robust at every
width (0.28–0.38 around 0.35).

**What this licenses.** Rule of thumb, now derived: bucket width ≤ one kernel
half-life ($x\lesssim\ln2(1-\kappa)\approx0.45$) is safe; ≥ 4 half-lives
($x\gtrsim1.8$) destroys timing. Report κ and covariate effects as primary from
counts; treat θ as estimable only from timestamps at sufficient volume (even at
$\Delta=1$ d, 4k events leave a ±0.05 spread — the ridge is also a *volume*
phenomenon, since θ-information scales with the number of parent–offspring pairs
$\sim\kappa N$); fix or prior-constrain θ otherwise (fixed decay banks).

---

## M4 — Sampled-softmax consistency (McFadden 1978)

**Statement.** Let the true choice probability over the full set $\mathcal C$ be
$P(i\mid\mathcal C)=e^{q_i}/\sum_{j\in\mathcal C}e^{q_j}$. Sample a subset
$D\ni i^*$ (the chosen alternative is always kept) with protocol $\pi(D\mid i^*)$.
Then the conditional likelihood of the choice given the sampled set is

$$
P(i^*\mid D)\;=\;\frac{e^{\,q_{i^*}+\ln\pi(D\mid i^*)}}{\sum_{j\in D}e^{\,q_j+\ln\pi(D\mid j)}},
$$

and if $\pi$ satisfies the **uniform conditioning property** —
$\pi(D\mid i)=\pi(D\mid j)$ for all $i,j\in D$ — the correction terms cancel and the
plain conditional logit on $D$ is a valid conditional likelihood, hence consistent
(its score is conditionally mean-zero at the truth).

**The two-line cancellation.** The joint law of (choice, sampled set) is
$P(i\mid\mathcal C)\,\pi(D\mid i)$. Conditioning on the drawn set $D$:

$$
P(i\mid D)=\frac{P(i\mid\mathcal C)\,\pi(D\mid i)}{\sum_{j\in D}P(j\mid\mathcal C)\,\pi(D\mid j)}
=\frac{e^{q_i}\pi(D\mid i)}{\sum_{j\in D}e^{q_j}\pi(D\mid j)},
$$

the full-set normalizer $\sum_{\mathcal C}e^{q}$ cancelling between numerator and
denominator (line 1); under uniform conditioning the $\pi$'s cancel too (line 2).
Uniform sampling of $K-1$ non-chosen alternatives without replacement satisfies the
property: $\pi(D\mid i)=1/\binom{n-1}{K-1}$ for every $i\in D$ ($n=|\mathcal C|$).

**Where it FAILS.** For non-uniform sampling (popularity-proportional, importance
sampling), $\pi(D\mid i)$ varies over $i\in D$ and the uncorrected estimator maximizes
a mis-specified objective — utilities are biased by the omitted offset
$\ln\pi(D\mid i)$ (the origin of the $q_j-\ln q(j)$ corrections in the ML
sampled-softmax literature). Keeping the winner but sampling the rest by anything
other than a scheme with the uniform-conditioning property requires the explicit
$\ln\pi$ terms.

**Our implementation, checked against `synthetic/loaders.py`.** When the winner is an
incumbent, `build_choice_sets` keeps it and draws 63 others uniformly without
replacement — uniform conditioning holds *within the incumbent block*, so incumbent
weights are consistently estimated. One genuine caveat: the newcomer alternative is
always included, and when the *newcomer* wins, **64** incumbents are drawn from $n$
rather than 63 from $n-1$; then $\pi(D\mid\text{new})/\pi(D\mid i)
=\binom{n-1}{63}/\binom{n}{64}=64/n$. The exact likelihood therefore carries a
$\ln(K/n_e)$ offset on the newcomer score that the uncorrected fit absorbs into the
ASC: $\hat b_0\approx b_0+\mathbb E[\ln(K/n_e)]$, with its pool-size variation
unmodeled. Sharp, testable prediction for E2.3: for pools with $n\gg K$, the fitted
ASC should rise by $\ln 2\approx0.69$ per doubling of `max_candidates`, while the
incumbent weights stay put. (This also means the measured ASCs — 6.86 on A, ~4.7 on
B — are $\ln(n/64)$-understatements of the full-pool ASC; see M7.)

**What this licenses.** `max_candidates=64` is a computational choice, not a modeling
compromise, for everything except the level of the newcomer ASC; weight stability
across $K$ (E2 sub-experiment 3) is the empirical certificate to attach.

---

## M5 — The log-link vs MBPP/additive gap, and the two explosion mechanisms

**Statement.** The MBPP surrogate is exact only for intensities *linear in history*
(Lemma "mean-intensity Volterra equation" in `complete_account.tex`: the proof needs
$\mathbb E[\text{linear}(H)]=\text{linear}(\mathbb E[H])$). The sector GLM is
$\Lambda_{s,t}=\exp(a_s+\beta_s^\top X_t+\sum b\,Y)$ — nonlinear in history — so its
mean does not satisfy the model equation: by Jensen,
$\mathbb E[e^{b^\top Y}]>e^{b^\top\mathbb E[Y]}$. For one lag and Poisson $Y_t$
the gap is exact via the Poisson MGF:

$$
\mathbb E[\Lambda_{t+1}\mid\Lambda_t]=e^{a}\,\mathbb E[e^{bY_t}\mid\Lambda_t]
=\exp\!\big(a+\Lambda_t(e^{b}-1)\big)
=\underbrace{\exp(a+b\Lambda_t)}_{\text{naive}}\cdot
\underbrace{\exp\!\big(\Lambda_t(e^{b}-1-b)\big)}_{\text{Jensen gap}},
$$

with the gap factor growing exponentially in $b$ and *in the count level* $\Lambda_t$.

**Stability of the log-link recursion.** Take the (exact-conditional-mean) map
$m\mapsto f(m)=\exp(a+c\,m)$, $c=e^b-1>0$; multivariate: $f(m)=\exp(a+Bm)$
componentwise. $f$ is convex increasing, so:

* **Fixed points exist iff $c\,e^{a+1}\le1$** (tangency $f(m)=m$, $f'(m)=1$ at
  $m=1/c$). If they exist there are two: a lower stable $m_-$ and an upper unstable
  $m_+$.
* **Local stability at $m_-$**: $f'(m_-)=c\,m_-<1$; multivariate Jacobian
  $J=\mathrm{diag}(m^*)B$, condition $\rho(\mathrm{diag}(m^*)B)<1\approx \bar m\,\rho(B)$.
  Stability depends on the **operating level of counts**, not on the coefficients
  alone: this is precisely the generator's knob
  `effective_radius = ρ(G) × mean_count` (`synthetic/regime_a.py`), and it is why
  $\rho(b)<1$ on raw coefficients — the correct condition for the *identity-link*
  INGARCH $\Lambda_t=a+\sum bY$ — is neither necessary nor sufficient here.
* **Metastability.** Even with $m_-$ locally stable, Poisson noise eventually carries
  $Y$ past the basin boundary $m_+$; the escape probability per week is a Chernoff
  tail $\approx\exp\{-[\,m_+\ln(m_+/m_-)-(m_+-m_-)\,]\}$, after which the
  deterministic map diverges double-exponentially ($\log m_{t+1}\approx c\,m_t$).
  Explosion is an *absorbing rare event*, not a gradual drift.

**Relation to Fokianos–Tjøstheim.** The stable log-linear Poisson autoregression
feeds back $\log(1+Y_{t-1})$ and/or lagged log-intensities,
$\nu_t=d+a_1\nu_{t-1}+b_1\log(1+Y_{t-1})$, with $|a_1+b_1|<1$: feedback that is
*linear in the log domain* gives a contraction and geometric ergodicity. Raw counts
inside $\exp(\cdot)$ — our specification — is the explosive variant; the stable
log-AR condition ("$|b|<1$ in the log domain") is on the log-count coefficients, not
$\rho(b)$ on raw counts.

**Mechanism A vs Mechanism B** (cross-checked against `complete_account.tex`
§stability and `docs/stability_and_explosion_report.md`):

| | Mechanism A (additive Hawkes/MBPP) | Mechanism B (log-link counts) |
|---|---|---|
| condition | $\rho(G)<1$, exact, level-free | $\rho(\mathrm{diag}(m^*)B)<1$ **and** fluctuations within basin |
| failure mode | gradual: $\xi^*=(I-G)^{-1}s$ diverges as $\rho\to1$ | abrupt tipping past $m_+$, double-exponential runaway |
| diagnostic | spectral radius of $\alpha/\beta$ | effective radius = $\rho(B)\times$ count level |

**Measured evidence.** Unconstrained fit reached $\rho\approx3.7$ → forward simulation
hit the $\exp(20)\approx4.85\times10^8$ rate clip → ~half-billion-iteration loop, red
CI (stability report §2). Effective radius **0.35** left the target fixed point only
*metastable*: one lucky burst tipped a ~50k-event world to **169k events** (the
briefing's measured episode; `regime_a.py` docstring). Design default 0.15. On
A-strong, the lag-feedback channel actively hurts: GLM-full held-out NLL/cell
**2.696** vs covariates-only **1.549**, EWMA **1.448** — the feedback overfits the
latent factor as covariate signal grows; on B the unstructured ridge arm wins
(2.326 vs GLM-full 2.467). Count self-excitation must be kept tiny; covariates and
the ranker carry the value.

---

## M6 — Cooldown functional forms and the E2.1 prediction

**The three forms.** Everything is a statement about the *relative log-hazard* (choice
utility) as a function of the gap $g$ (days) since the firm's last funding:

* **Truth, regime A** (`regime_a.py`): indicator cooldown,
  $\Delta\log h_A(g)=\eta_s\,\mathbb 1\{7\le g\le182\}$, $\eta_s\sim-U(2.2,3.2)$
  (26 weeks = 182 d). A step: hazard suppressed by factor
  $e^{\eta}\approx e^{-2.7}\approx0.07$ inside the window, exactly 1 outside.
* **Truth, regime B** (`regime_b.py`): exponential inhibition through
  $-\rho_s R_i(t)$, $R_i=\sum_{\text{own past}}e^{-\omega_{\text{self}}\Delta t}$,
  $\omega_{\text{self}}=\ln2/30\approx0.0231$/day. For a single prior event,
  $\Delta\log h_B(g)=-\rho_s\,e^{-\omega_{\text{self}}g}$, $\rho_s\sim U(1.2,2.2)$:
  smooth recovery with 30-day half-life; multiple recent events stack
  (deeper suppression for serial raisers).
* **Fitted** (`loaders.py` feature 4): coefficient $w_g$ on $\log(1+g/30)$, i.e.
  hazard $\propto(1+g/30)^{w_g}$ — a **power law**, monotone, unbounded, no
  saturation. Measured: $w_g=+1.15$ (A), $+0.44$ (B); positive sign = correct
  cooldown direction in both worlds.

**What the single-coefficient fit is.** $\hat w_g$ is the population projection of the
true $\Delta\log h(g)$ onto $\{1,\log(1+g/30)\}$ under the candidate gap
distribution. Both truths are increasing in $g$, so $w_g>0$; but the power law keeps
rising where both truths saturate (A after 182 d, B after ~4 half-lives ≈ 120 d), so
the log-gap model overstates the hazard of very stale firms in both regimes, and its
coefficient is DGP-specific (exp23: "the gap weight reflects the 30d inhibition, not
a 26w cooldown").

**Derived E2.1 prediction for binned dummies** ({<1m, 1–3m, 3–6m, 6–12m, >12m},
reference = >12m; bin midpoints 15, 60, 135, 272 d):

* **World A — a staircase with one cliff.** Bins <1m, 1–3m, 3–6m lie entirely inside
  the 182-d window; 6–12m is (essentially) outside (only 180–182 d overlaps). Expected
  coefficients $\approx[\eta,\eta,\eta,0,0]$ with $\eta\approx-2.7$: three equal-depth
  steps, then a cliff of ~2.7 nats between 3–6m and 6–12m, then flat.
* **World B — geometric decay.** Expected coefficients
  $\approx-\bar\rho\,e^{-\omega_{\text{self}} g_{\text{mid}}}$ with
  $\bar\rho\approx1.7$: $[-1.20,\,-0.43,\,-0.08,\,-0.003]$ — most of the action in the
  first two bins, half the effect gone by 30 d, smooth concave recovery, no cliff.
  (Slight over-deepening of the first bin is expected from multi-event stacking.)

The binned fit is therefore a *shape discriminator between the worlds*: cliff-at-6m
vs geometric-decay-from-day-0. Neither is representable by the single log-gap
coefficient, which is why E2.1 compares held-out NLL across
{log-gap, bins, $e^{-g/\tau}$, $\tau\in\{30,90,180\}$} — in world B the $\tau=30$
exponential is the correctly specified form and should win; in world A the bins
should win (they can express the step).

**What this licenses.** Keep the hard exclude-last rule (a separate, infinite-depth
refractory at $g\to0$) plus a *flexible* soft cooldown; report the implied
hazard-vs-gap curve, not just the coefficient, since the coefficient does not
transfer across DGPs.

---

## M7 — The newcomer alternative as a log-MGF (integrated-out entrant)

**Statement.** Suppose the winner may be one of $N_{\text{pool}}$ exchangeable
unobserved entrants, each with features $z\sim\mathcal N(\mu_s,\Sigma_s)$ i.i.d. and
utility $w^\top z$. Then the aggregate "newcomer" option enters the softmax
denominator as a single alternative with score

$$
q_{\text{new},s}=\ln N_{\text{pool}}+w^\top\mu_s+\tfrac12\,w^\top\Sigma_s w .
$$

**Derivation.** The probability the winner is *some* entrant is
$\sum_{j=1}^{N}e^{w^\top z_j}\big/\big(\sum_{\text{inc}}e^{q}+\sum_j e^{w^\top z_j}\big)$.
Replace the random entrant mass by its expectation:
$\mathbb E\big[\sum_j e^{w^\top z_j}\big]=N\,\mathbb E[e^{w^\top z}]
=N\exp(w^\top\mu_s+\tfrac12 w^\top\Sigma_s w)$ — the Gaussian MGF, hence "log-MGF
formula": $\ln\mathbb E[e^{w^\top z}]=w^\top\mu_s+\tfrac12w^\top\Sigma_s w$. Two named
approximations: (i) expectation-for-realization, accurate for large $N$ by the LLN
(relative fluctuation $N^{-1/2}\sqrt{e^{w^\top\Sigma w}-1}$ — the log-normal CV);
(ii) entrant features independent of the event history $H_t$.

**Identification (the pool-size term is NOT separately identified).** The likelihood
depends on $(N_{\text{pool}},\mu_s,\Sigma_s)$ only through the scalar
$q_{\text{new},s}$: the map
$(\ln N,\;w^\top\mu+\tfrac12w^\top\Sigma w)\mapsto q_{\text{new}}$ has rank 1, so any
reallocation between "pool size" and "mean entrant quality" leaving the sum fixed is
likelihood-invariant. A free ASC (our `is_newcomer` weight, plus per-sector deviation
$u_{s}$) estimates exactly this sum and nothing finer. Separate identification
requires exogenous *time variation* in $N_{\text{pool},s,t}$ with known values: then
$q_{\text{new},s,t}=\ln N_{\text{pool}}(s,t)+\text{const}_s$ and one can enter
$\ln N_{\text{pool}}$ as an offset (coefficient fixed at 1) or free (coefficient
$\approx1$ is a test of the exchangeable-entrant story). With constant $N$ the ASC
absorbs it — say so in the report rather than interpreting the ASC level.

**Reading the measured ASC.** Fitted newcomer ASC ≈ **6.86** (A) / **4.7** (B) against
a ~27% newcomer share. In the fitted parametrization the newcomer row is all-zero
except the indicator, so the ASC must cover (i) $\ln N_{\text{pool}}$ (thousands of
never-yet-funded tracked firms plus ~12% out-of-universe winners — note the newcomer
option *conflates* true entrants with untracked incumbents, REVIEW §N caveat),
(ii) the mean entrant utility deficit $w^\top\mu_s$ relative to the incumbents'
nonzero features, (iii) the Gaussian variance bonus $\tfrac12w^\top\Sigma_s w$, and
(iv) the sampled-softmax offset $\mathbb E[\ln(K/n_e)]<0$ from M4. The N2b design is
precisely the attempt to move (ii)+(iii) out of the ASC using the observed
at-first-funding feature distribution $(\hat\mu_s,\hat\Sigma_s)$; on synthetic data
the residual ASC then estimates $\ln N_{\text{pool}}$ up to the M4 offset, checkable
against the true pool in `ground_truth.npz` + `private_deal_map.csv`.

**What this licenses.** N2(b)'s utility $w^\top\hat\mu_s+\tfrac12 w^\top\hat\Sigma_s w
+ c_s$ is not a heuristic — it is the exact aggregate of a Gaussian entrant pool with
$c_s=\ln N_{\text{pool},s}$; and the reliability target of N1 (time-varying newcomer
share) is the identified object, while the ASC *level* decomposition is identified
only in the synthetic world where the pool is known.

---

## M8 — Positive-only observation: why the mark likelihood needs no failures

**Statement.** From M1, $\log L=\ell_g(\Lambda)+\ell_m(P)$ with
$\ell_m=\sum_k\log P(i_k\mid s_k,t_k,H_{t_k})$. If the parametrization is
**variation-free** — $(\theta_g,\theta_m)\in\Theta_g\times\Theta_m$ with $\Lambda$
depending only on $\theta_g$ and $P$ only on $\theta_m$ — then the joint MLE of
$\theta_m$ is obtained by maximizing $\ell_m$ alone: the two stages *separate*.
Conversely, if a parameter enters both factors, maximizing $\ell_m$ alone remains a
valid **partial-likelihood** estimator of $\theta_m$ (consistent, conditionally
mean-zero score) but discards the information about it in $\ell_g$ — separation holds
*iff* no parameter is shared.

**Derivation.** Separation: with variation-free parameters,
$\arg\max_{\theta}\ell=(\arg\max_{\theta_g}\ell_g,\arg\max_{\theta_m}\ell_m)$ —
immediate since each summand depends on its own block. Why no non-event data are
needed: all timing information (how long between events, weeks with no events) sits in
the ground factor's compensator $\int\Lambda_s$; the mark factor is a product of
probabilities *conditional on an event having occurred*, each summing to 1 over the
risk set — so "firm $i$ was not funded in week $t$" enters only as $i$'s membership
in the risk sets of events won by others, never as a manufactured negative label.
Validity as a partial likelihood (Cox 1975): each factor
$P(i_k\mid s_k,t_k,H_{t_k})$ conditions on the full history up to $t_k$; the score
increments $\nabla_{\theta_m}\log P(i_k\mid\cdot)$ therefore have zero conditional
mean at the truth, the summed score is a martingale, and standard estimating-equation
asymptotics apply — regardless of how complicated (history-dependent) the
conditioning objects are, *provided the conditional mark probabilities are correctly
specified*.

**Where we DO couple the factors — and why it is still valid.**

1. **Cooldown built from past events.** $C_{i,t}$ (and the gap feature) are functions
   of $H_t$ — the same events whose timing the ground factor models. This is coupling
   through *data*, not through *parameters*: $C_{i,t}$ is $H_t$-measurable and enters
   $\ell_m$ as a covariate, so the conditional score argument is untouched. Coupling
   would break separation only if a shared parameter (e.g. one $\rho$ governing both a
   firm-level term in $\Lambda_s$ and the utility) appeared in both factors; our
   ground model is sector-level with no firm parameters, so variation-freeness holds
   in the fitted model.
2. **The funded-pool rule makes risk sets history-dependent.** The pool (firms funded
   at least once, minus the last-funded firm) is again an $H_t$-measurable set; each
   event's likelihood term conditions on it. The exclude-last rule additionally
   *truncates* the mark space; events whose winner is the excluded firm are dropped
   and counted (`n_dropped_repeat` = 35 (A) / 2 (B)) — a selection that is applied
   identically to model and data and reported, so the conditional likelihood over the
   retained events stays coherent.
3. **Where the fitted model is *not* an exact factor.** In regime B the truth couples
   parameters: $\Lambda_s(t)=\sum_i\lambda_i(t)$ and
   $P(i\mid s,t)=\lambda_i/\Lambda_s$ share $(\rho_s,A,\beta_s)$. The two-stage fit is
   then a quasi/partial-likelihood *projection*: still consistent for the
   best-in-class mark model (measured: gap weight +0.44 correctly carries the 30-d
   inhibition; the sector layer's fitted "excitation" radius 0.61 is aggregation
   artifact), but the factors' parameters no longer mean what the DGP's do.

**The assumption that actually bites.** The partial-likelihood logic requires the
risk set to be the *truthful* at-risk set. It is an observation-model assumption, not
a theorem, and its failure is the measured headline of E2-v1: with the naive observed
pool the fitted weights invert (raised **+1.94** vs truth 0.0), held-out top-5 falls
to **0.144** vs oracle **0.288**, and no observable pruning/reweighting closes the
gap — an exit model with AUC **0.877** still cannot reconstruct the oracle mask. The
conditional-choice machinery was working correctly in that experiment; it was
consistently estimating the wrong estimand, because conditioning on a contaminated
risk set changes the question. Under v2 the risk set is truthful by construction and
the same machinery sits at the oracle ceiling (exp25: logit top-5 0.29 = oracle 0.29;
a hazard model trained *with* explicit negative firm-weeks does no better, 0.28).

**What this licenses.** Train stage 2 on positives + risk sets only; report
$P(\text{newcomer}/\text{outside})$ as a monitored output; spend real-data effort on
exit/graduation labeling (the risk-set truthfulness assumption), not on richer
negatives — and quote the observed-vs-best-mask metric spread (0.14 → 0.29 here) as
the price of the observation gap.

---

*Cross-references.* M1↔R2 (losslessness), M2↔E3, M3↔E1b/exp24, M4↔E2.3, M5↔stability
report + exp25, M6↔E2.1, M7↔N1/N2, M8↔E2-v1/exp25.
