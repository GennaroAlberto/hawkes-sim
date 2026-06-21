# Interval-censored Hawkes processes — a detailed summary

**Paper.** Marian-Andrei Rizoiu, Alexander Soen, Shidi Li, Pio Calderon, Leanne J. Dong,
Aditya Krishna Menon, Lexing Xie. *"Interval-censored Hawkes processes."*
Journal of Machine Learning Research **23** (2022) 1–84. (Manuscript 21-0917.)

This document summarizes the paper in full technical detail — what the authors
do, how they do it, and the proofs of the central results — and then maps every
construction onto the code added to this package (`mbpp.py`, `exogenous.py`,
`ic_simulate.py`, `interval_censored.py`, and experiments `exp5`–`exp7`).

Throughout, observed/data quantities are written in upright font
($\mathsf{C}, \mathsf{S}, \mathsf{F}$) and modelled/random quantities in italics.

---

## 0. The one-paragraph version

A Hawkes process is the standard model for *self-exciting* events (earthquakes,
trades, tweets), but its likelihood needs the **exact event times**. In many
applications we only see **interval-censored** data: the *count* of events in
each time bucket (patients admitted per day, views per day), not when each event
happened. The Hawkes process cannot be fit to such data because (i) its
log-likelihood is undefined without timestamps and (ii) it lacks the
*independent-increments* property that would let us use a Poisson likelihood on
the counts. The paper's solution is to introduce a **companion process**, the
**Mean Behavior Poisson process (MBPP)**: the inhomogeneous Poisson process whose
deterministic intensity $\xi(t)$ equals the *expectation of the Hawkes
intensity* over all realizations. The MBPP shares its parameters one-to-one with
the Hawkes process, but — being Poisson — *does* have independent increments, so
its interval-censored likelihood is a tractable product of Poisson terms. Fitting
the MBPP on the counts therefore recovers (approximations of) the Hawkes
parameters. The paper supplies: a closed-form/numerical solver for $\xi$
(Theorems 1–2), the interval-censored log-likelihood IC-LL and its interpretation
as a Bregman/KL divergence (Sections 4–5), two "exogenous functions" for when the
external drivers are themselves observed (Section 6), synthetic validation across
six data-observability scenarios (Section 7), and a real-world popularity-
prediction experiment that beats the prior state of the art HIP (Section 8).

---

## 1. Setting and notation

A **temporal point process** is a random set of event times $\{t_i\}$, equivalently
a counting process $N(t)=\#\{i: t_i\le t\}$. Its **conditional intensity** is

$$
\lambda(t)\;=\;\lim_{h\downarrow 0}\frac1h\,\mathbb E\!\big[N(t{+}h)-N(t)\mid \mathcal H_{t^-}\big],
$$

the instantaneous event rate given the history $\mathcal H_{t^-}$. The
**compensator** is the integrated intensity $\Lambda(s,t)=\int_s^t\lambda(u)\,du$,
and we write $\Lambda(t)=\Lambda(0,t)$.

**Poisson process.** A process is (inhomogeneous) Poisson with compensator
$\Lambda$ if disjoint intervals give independent Poisson counts:
$N(A_i)\sim\text{Poisson}(\Lambda(A_i))$ independently. Two consequences are used
repeatedly: it has **independent increments**, and $\mathbb E[N(t)]=\Lambda(t)$.

**Hawkes process** (Hawkes, 1971). A self-exciting process with stochastic
intensity

$$
\lambda(t)\;=\;s(t)\;+\;\int_0^{t^-}\!\!\phi(t-u)\,dN(u)
\;=\;\underbrace{s(t)}_{\text{exogenous}}\;+\;\underbrace{\sum_{t_i<t}\phi(t-t_i)}_{\text{endogenous}}.
\tag{3}
$$

$s(t)\ge 0$ is the **exogenous** (background) intensity generating *immigrant*
events; $\phi\ge 0$ is the **triggering kernel**, by which each past event raises
the future rate, generating *offspring*. The **exponential kernel** is

$$
\phi(t)\;=\;\kappa\theta\,e^{-\theta t}\,\mathbb 1[t>0],\qquad
\int_0^\infty\phi(t)\,dt=\kappa,
\tag{13}
$$

so $\kappa\in(0,1)$ is the **branching ratio** (the mean number of direct
offspring per event; the subcriticality condition is $\kappa<1$) and $\theta>0$
the decay rate. The **power-law kernel** $\phi(t)=\kappa(t+c)^{-(1+\theta)}$ is
used for heavy-tailed excitation.

> **Code conventions.** This package's event-time modules use
> $\phi(t)=\alpha e^{-\beta t}$. The paper's $(\kappa,\theta)$ map to this via
> $\alpha=\kappa\theta,\ \beta=\theta$, i.e. $\kappa=\alpha/\beta$ is exactly the
> branching ratio. `mbpp.kappa_theta_to_alpha_beta` / `alpha_beta_to_kappa_theta`
> convert between them.

**Why the Hawkes process is not Poisson.** Because $\lambda(t)$ depends on the
realized history, the number of events in an interval depends on the number in
earlier intervals — the process is non-Markovian and has *no* independent
increments. The MLE objective, the **point-process log-likelihood (PP-LL)**,

$$
\mathcal L(\theta;T)\;=\;-\sum_{t_i\le T}\log\lambda(t_i;\theta)\;+\;\int_0^T\lambda(u;\theta)\,du,
\tag{4}
$$

explicitly needs the times $t_i$. Both facts block direct interval-censored
fitting and motivate the MBPP.

**Prior work the paper improves on — HIP** (Rizoiu et al., 2017b). The Hawkes
Intensity Process fits counts by least squares against the *expected intensity*
$\xi(t)=\mathbb E[\lambda(t)]$:
$\min_\theta\sum_i\big(\mathsf C(o_{i-1},o_i]-\xi(o_i)\big)^2$. The paper's
critique (Section 2.4): (L1) it matches a *count* to a *rate*; (L2) it is not the
likelihood of any process, so its statistical meaning is unclear; (L3) it relies
on a discrete approximation of $\xi$ of uncertain quality. The MBPP framework
fixes all three and recovers HIP as a special case (Section 5).

---

## 2. The Mean Behavior Poisson process (MBPP)

### 2.1 Definition

Given a Hawkes process with exogenous $s$ and kernel $\phi$, define the MBPP as
the **inhomogeneous Poisson process** whose intensity is the expected Hawkes
intensity:

$$
\xi(t)\;:=\;\mathbb E_{\mathcal H_t}\big[\lambda(t)\big]
\;=\;s(t)+\int_0^t\xi(\tau)\,\phi(t-\tau)\,d\tau
\;=\;s(t)+(\xi*\phi)(t).
\tag{9}
$$

The middle equality is obtained by taking expectations in (3) and using that, by
the tower property, the expected endogenous contribution of events near $\tau$ is
$\mathbb E[dN(\tau)]=\xi(\tau)\,d\tau$. Equation (9) is a **Volterra integral
equation of the second kind**: $\xi$ appears on both sides. Crucially the same two
ingredients $(s,\phi)$ define both processes, so **Hawkes and MBPP are in
one-to-one parameter correspondence**. Empirically (Fig. 2), the average of many
simulated Hawkes intensities lies right on top of $\xi(t)$.

> **Code.** `mbpp.MBPP(kernel, exogenous)` is this object; `exp5_mbpp_impulse.py`
> regenerates Figs 2–3, overlaying the Monte-Carlo Hawkes mean on the closed-form
> $\xi$ (max abs. error $\approx$ Monte-Carlo noise, $\sim$0.02–0.06).

### 2.2 The compensator obeys the *same* equation

Define $\Xi(t):=\mathbb E[\Lambda(t)]$. Then $\Xi$ is the MBPP compensator and
satisfies a self-consistent equation identical in form to (9):

$$
\Xi(t)\;=\;\mathsf S(t)+\int_0^t\Xi(t-y)\,\phi(y)\,dy\;=\;\mathsf S(t)+(\Xi*\phi)(t),
\qquad \mathsf S(t):=\int_0^t s(u)\,du.
\tag{10}
$$

> **Proof (Eq. 10).** Starting from $\Xi(t)=\int_0^t\xi(z)\,dz$ and substituting
> (9),
> $$
> \Xi(t)=\int_0^t\!\Big(s(z)+\int_0^z\xi(z-y)\phi(y)\,dy\Big)dz
> =\mathsf S(t)+\int_0^t\!\!\int_0^z\xi(z-y)\phi(y)\,dy\,dz .
> $$
> Swap the order of integration over the triangle $\{0\le y\le z\le t\}$ (Fubini,
> valid since $\lambda$ is bounded so all integrands are integrable):
> $$
> =\mathsf S(t)+\int_0^t\phi(y)\!\int_y^t\xi(z-y)\,dz\,dy .
> $$
> Substitute $x=z-y$ in the inner integral ($dz=dx$, limits $0\to t-y$):
> $$
> =\mathsf S(t)+\int_0^t\phi(y)\!\int_0^{t-y}\xi(x)\,dx\,dy
> =\mathsf S(t)+\int_0^t\phi(y)\,\Xi(t-y)\,dy ,
> $$
> using $\Xi(t-y)=\int_0^{t-y}\xi$. This is exactly (10). $\;\blacksquare$

The practical payoff: **the same solver delivers both $\xi$ and $\Xi$** — apply it
to $s$ to get $\xi$, and to $\mathsf S$ to get $\Xi$.

> **Code.** `MBPP.intensity` solves (9); `MBPP.compensator` solves (10) by feeding
> the cumulative exogenous $\mathsf S$; `MBPP.compensator_interval` returns the
> increments $\Xi(o_{i-1},o_i]=\Xi(o_i)-\Xi(o_{i-1})$ needed by the likelihood.

### 2.3 Solving the MBPP equation

Two solution routes are given.

**(a) Laplace transform (Theorem 1).** Transforming (9) and using the convolution
theorem $\widehat{(\xi*\phi)}=\widehat\xi\,\widehat\phi$ gives
$\widehat\xi=\widehat s+\widehat\xi\,\widehat\phi$, hence

$$
\xi(t)=\mathcal L^{-1}\!\left\{\frac{\mathcal L\{s\}(\omega)}{1-\mathcal L\{\phi\}(\omega)}\right\}(t).
\tag{11}
$$

This is exact *when the transforms exist*, which fails for the Dirac-comb
exogenous (Section 5) and for power-law/Rayleigh kernels — and must be recomputed
whenever $s$ changes. These limitations motivate route (b).

**(b) Impulse-response solution (Theorem 2 + Corollary 3).** Equation (9) defines
a **causal, linear, time-invariant (LTI) system** with input $s$ and output $\xi$.
Such a system is completely characterized by its **impulse response** $E$ (its
output to a unit impulse $\delta$), and the output to any input is the convolution
$\xi=E*s$.

> **Theorem 2.** If $\int_0^\infty\phi<1$, the MBPP system's impulse response is
> $$
> E(t)=\delta(t)+h(t),\qquad h(t)=\sum_{n=1}^{\infty}\phi^{*n}(t),
> $$
> where $\phi^{*n}$ is the $n$-fold convolution of $\phi$ with itself.

> **Proof.** *LTI:* linearity of (9) is immediate (scale $s$, $\xi$ scales).
> Time-invariance: for input $s(t-t_0)$, change variables $t'=t-t_0$ in the
> convolution and use causality $\phi(\cdot)=0$ on the negatives to show the
> output is $\xi(t-t_0)$; together with causality of $\xi$ this makes the system
> causal LTI. *Impulse response:* set $s=\delta$ in (9). With $E=\delta+h$,
> $$
> E=\delta+E*\phi=\delta+(\delta+h)*\phi=\delta+\phi+h*\phi,
> $$
> using $\delta*\phi=\phi$. Matching the non-impulsive parts gives the
> fixed-point $h=\phi+h*\phi$. Iterating,
> $$
> h=\phi+(\phi+h*\phi)*\phi=\phi+\phi^{*2}+h*\phi^{*2}=\cdots=\sum_{n=1}^{\infty}\phi^{*n}.
> $$
> *Convergence:* by Young's inequality $\|\phi^{*n}\|_1\le\|\phi\|_1^{\,n}$, and
> $\|\phi\|_1=\int_0^\infty\phi<1$, so $\|\phi^{*n}\|_1\to 0$; the series converges
> in $L^1$ and $\phi^{*n}\to 0$. $\;\blacksquare$

> **Corollary 3.** For any (generalized) input $s$, $\xi=E*s=s+h*s$ solves (9).
> *Proof:* $\xi=(\delta+h)*s=s+h*s$, and $h=E*\phi$ gives
> $\xi=s+(E*s)*\phi=s+\xi*\phi$. $\;\blacksquare$

> **Corollary 5 (compensator).** Likewise $\Xi(x,y]=(E*\mathsf S)(y)-(E*\mathsf S)(x)$.

**The exponential-kernel collapse.** For $\phi(t)=\kappa\theta e^{-\theta t}$ the
$n$-fold convolution is a Gamma-shaped term,
$\phi^{*n}(t)=\kappa^n\theta^n e^{-\theta t}\,\frac{t^{n-1}}{(n-1)!}$, and the
infinite sum is a closed form:

$$
h(t)=\sum_{n\ge1}\phi^{*n}(t)=\kappa\theta\,e^{-\theta t}\sum_{n\ge0}\frac{(\kappa\theta t)^n}{n!}
=\kappa\theta\,e^{(\kappa-1)\theta t},\qquad t>0.
\tag{14}
$$

So $E(t)=\delta(t)+\kappa\theta e^{(\kappa-1)\theta t}$. Because $\kappa<1$ the
exponent $(\kappa-1)\theta<0$: the response decays, exactly as a stable system
should. For a single immigrant at $a$ (input $s=\delta(\cdot-a)$),

$$
\xi(t)=\delta(t-a)+\kappa\theta\,e^{(\kappa-1)\theta(t-a)}\mathbb 1[t>a],
$$

whose total endogenous mass $\int_a^\infty h(t-a)\,dt=\dfrac{\kappa}{1-\kappa}$ is
exactly the expected total progeny $\kappa+\kappa^2+\cdots$ of one event.

> **Code.** `ExponentialKernel.impulse_response_h` is (14). The closed-form
> $\xi,\Xi$ for every exogenous in `exogenous.py` are assembled as $s+h*s$ and
> $\mathsf S+h*\mathsf S$ (Table 10 of the paper). For kernels without a closed
> form (`PowerLawKernel`) `MBPP(..., method="numeric")` truncates
> $h\approx\sum_{n=1}^{N}\phi^{*n}$ and convolves on a grid; this is the paper's
> "approximate MBPP", and it converges to the closed form at rate $O(\Delta t)$.

---

## 3. The interval-censored setting and its likelihood (IC-LL)

**The data.** Observation times $\mathcal O=\{o_0=0<o_1<\dots<o_m=T\}$ partition
$(0,T]$ into intervals; for each we observe only the **count**
$\mathsf C(o_{i-1},o_i]$. We want

$$
\max_\theta\;\mathbb P\big(N(o_{i-1},o_i]=\mathsf C(o_{i-1},o_i],\ i=1,\dots,m\big).
\tag{17}
$$

For a Hawkes process this is intractable: without independent increments the
joint law (17) factorizes only through generation-by-generation Borel
distributions, which have no closed form at finite $T$ and scale poorly by Monte
Carlo.

**The MBPP makes it tractable.** Replace the Hawkes process by its MBPP. Being
Poisson, the MBPP has independent increments, so (17) factorizes into independent
Poisson terms with rates $\Xi(o_{i-1},o_i]$. The negative log-likelihood is

$$
\begin{aligned}
\mathcal L(\theta)&=-\log\prod_{i=1}^{m}\frac{\Xi(o_{i-1},o_i]^{\,\mathsf C(o_{i-1},o_i]}}{\mathsf C(o_{i-1},o_i]!}\,e^{-\Xi(o_{i-1},o_i]}\\
&=\sum_{i}\Xi(o_{i-1},o_i]-\sum_i\mathsf C(o_{i-1},o_i]\log\Xi(o_{i-1},o_i]+\underbrace{\sum_i\log \mathsf C(o_{i-1},o_i]!}_{\text{const in }\theta}.
\end{aligned}
$$

Dropping the constant gives the **interval-censored log-likelihood**:

$$
\boxed{\;\mathcal L_{\text{IC-LL}}(\theta)=\sum_{i=1}^{m}\Xi(o_{i-1},o_i;\theta]-\sum_{i=1}^{m}\mathsf C(o_{i-1},o_i]\,\log\Xi(o_{i-1},o_i;\theta]\;}
\tag{19}
$$

Note this loss compares a **count** $\mathsf C$ to an **expected count** $\Xi$
(both extensive quantities) — fixing HIP's limitation L1.

> **Code.** `interval_censored.ic_ll(counts, Xi)` is (19).

### 3.1 Computing the compensator $\Xi(o_{i-1},o_i]$

* **Analytical.** When the impulse response has closed form (exponential kernel),
  Corollary 5 gives $\Xi$ in closed form for each exogenous (implemented per
  exogenous as `endo_compensator_exp`). Verified internally: the analytic
  endogenous compensator matches a fine quadrature of the analytic endogenous
  intensity to $\sim10^{-8}$.
* **Numerical lower bound (Propositions 6–7).** When no closed form exists, place
  approximation points $\{d_j\}$ and lower-bound the counting process by a
  piecewise-constant $M_D^-(t)=\sum_j M(d_{j-1},d_j]\,\mathbb 1[d_j<t]$, which
  $\to M(t)$ as $D\to\infty$ (Prop. 6). Substituting into (10) and using
  $\Xi=\mathbb E[M]$ yields the computable lower bound (Prop. 7)
  $$
  \Xi_D^-(t)=\mathsf S(t)+\sum_{j:\,d_j<t}\mathbb E\big[M(d_{j-1},d_j]\big]\int_{d_j}^{\min(t,d_D)}\!\phi(y)\,dy .
  $$
  Its decisive feature: the per-interval expected counts can be replaced by
  **observed** counts — the basis of forecasting (Section 6).

> **Code.** `MBPP.compensator_lower_bound` implements Prop. 7;
> `MBPP._kernel_integral` supplies $\int\phi$ in closed form for the exp/power-law
> kernels.

---

## 4. IC-LL is a Bregman divergence; HIP is a special case

A **Bregman divergence** for a strictly convex generator $\varphi$ is
$B_\varphi(x,y)=\varphi(x)-\varphi(y)-(x-y)^{\top}\nabla\varphi(y)$. Two choices:

* generator $\varphi(x)=\sum_i x_i\log x_i$ gives the generalized **KL divergence**;
* generator $\hat\varphi(x)=\|x\|^2$ gives the **squared error** $\|x-y\|^2$.

> **Proposition 8.** With $\mathsf C=(\mathsf C(o_{i-1},o_i])_i$ and
> $\Xi(\theta)=(\Xi(o_{i-1},o_i;\theta])_i$,
> $$
> \mathcal L_{\text{IC-LL}}(\theta)=\mathrm{KL}\big(\mathsf C,\Xi(\theta)\big)+\Gamma(\mathsf C),
> $$
> with $\Gamma$ independent of $\theta$; hence $\arg\min_\theta\mathcal L_{\text{IC-LL}}=\arg\min_\theta \mathrm{KL}(\mathsf C,\Xi(\theta))$.

> **Proof.** Generalized KL is
> $\mathrm{KL}(\mathsf C,\Xi)=\sum_i \mathsf C_i\log\frac{\mathsf C_i}{\Xi_i}-\sum_i(\mathsf C_i-\Xi_i)$.
> Expand: $\sum_i \mathsf C_i\log \mathsf C_i-\sum_i \mathsf C_i\log\Xi_i-\sum_i \mathsf C_i+\sum_i\Xi_i$.
> The terms $\sum_i \mathsf C_i\log \mathsf C_i-\sum_i \mathsf C_i=:\Gamma(\mathsf C)$ are
> constant in $\theta$; the rest is exactly (19). $\;\blacksquare$

Swapping the KL generator for the squared one gives the **SSE loss**

$$
\mathcal L_{\text{SSE}}=\sum_{i=1}^{m}\big(\mathsf C(o_{i-1},o_i]-\Xi(o_{i-1},o_i;\theta]\big)^2 .
\tag{36}
$$

Via the Bregman↔exponential-family bijection, IC-LL assumes each interval is
**Poisson** and SSE assumes each interval is **Gaussian (constant variance)**;
they agree in the large-sample limit (Poisson $\to$ Gaussian). In the experiments
IC-LL is the more robust of the two, especially near criticality.

**Recovering HIP (Theorems 10–11).** HIP equals SSE under three extra
assumptions: unit-length intervals, a constant MBPP intensity per interval, and a
discrete-convolution approximation of $\xi$.

> **Lemma 9.** If $\xi$ is constant on a unit interval $(i{-}1,i]$ then
> $\Xi(i{-}1,i]=\xi(i)$. *Proof:* $\int_{i-1}^{i}\xi = \xi(i)\cdot 1$. $\;\blacksquare$

So under those assumptions SSE in (36) becomes $\sum_i(\mathsf C(i{-}1,i]-\xi(i))^2$,
the HIP objective (Theorem 10), and with a power-law kernel + discrete
convolution one recovers the full HIP model (Theorem 11). This *derives* HIP as a
restricted MBPP, repairing its limitations L2–L3 and, as a bonus, giving HIP a
**closed form** for the exponential kernel (no discretization needed).

> **Code.** `interval_censored.sse_loss` is (36); `exp6` compares IC-LL vs SSE and
> reproduces the finding that SSE degrades in the near-critical regime
> ($\kappa=0.95$) while IC-LL stays accurate.

---

## 5. Observed exogenous stimuli: separable scenarios and exogenous functions

Often the exogenous (immigrant) events are themselves observed and *separable*
from the offspring — e.g. tweets (observed) drive YouTube views (observed). The
exogenous and endogenous streams may each be event-time (ET) or interval-censored
(IC), giving **six scenarios** (Table 2):

| Scenario | Immigrants | Offspring | Exogenous function | Loss | Endogenous? |
|---|---|---|---|---|---|
| A | — (ET, non-separable) | ET | known $s(t)$ | PP-LL | no |
| B | ET | ET | multi-impulse | PP-LL | yes |
| C | IC | ET | LHPP | PP-LL | yes |
| D | — (IC, non-separable) | IC | known $s(t)$ | IC-LL / SSE | no |
| E | ET | IC | multi-impulse | IC-LL / SSE | yes |
| F | IC | IC | LHPP | IC-LL / SSE | yes |

To handle the separable cases the paper introduces two exogenous functions.

**Multi-impulse (Definition 12)** — exogenous events observed as *event times*
$\{\mathsf s_z\}$:
$$
s_{m\delta}(t)=\sum_{z}\delta(t-\mathsf s_z).
$$
It is parameter-free and deterministic; being a Dirac comb it is not directly
evaluable, which is why it is used with the *endogenous* loss (below).

**Latent Homogeneous Poisson Process, LHPP (Definition 14, Proposition 13)** —
exogenous events observed *interval-censored*. On each interval $(q_{i-1},q_i]$
with observed immigrant volume $\mathsf S(q_{i-1},q_i]$, assume a homogeneous
Poisson rate $\lambda_i$. Its MLE is the empirical rate:

> **Proposition 13.** $\displaystyle \lambda_i=\frac{\mathsf S(q_{i-1},q_i]}{q_i-q_{i-1}}.$
>
> **Proof.** For a homogeneous Poisson process on $(x,y]$ with $\mathsf C$
> observed events, the log-likelihood in $\lambda$ is
> $\mathsf C\log(\lambda(y-x))-\lambda(y-x)-\log \mathsf C!$; it is concave in
> $\lambda$, and setting the derivative $\frac{\mathsf C}{\lambda}-(y-x)=0$ gives
> $\lambda^\star=\mathsf C/(y-x)$. Apply per interval. $\;\blacksquare$

This makes the exogenous function the piecewise-constant
$s_{\,\square}(t)=\sum_i\frac{\mathsf S(q_{i-1},q_i]}{q_i-q_{i-1}}\mathbb 1[q_{i-1}<t\le q_i]$.

> **Code.** `exogenous.MultiImpulse` and `exogenous.LHPP` implement Definitions 12
> and 14; the LHPP rate is Prop. 13 exactly. `Constant`, `Rectangle`,
> `PiecewiseConstant`, `Sine`, `Dassios` are the auxiliary exogenous functions
> used in Figs 2–3 and the synthetic datasets.

**Endogenous loss functions (Section 6.3).** When the data is separable, decompose
the intensity into exogenous + endogenous parts,
$\xi(t)=\xi^{\text{exo}}(t)+\xi^{\text{endo}}(t)$ with
$\xi^{\text{endo}}=\xi-s=h*s$ (Eq. 48), and fit only the endogenous piece against
the *offspring* events $\Upsilon$:

$$
\mathcal L^{\text{endo}}(\theta)=\mathcal L(\theta;\lambda^{\text{endo}},\Upsilon).
$$

Conceptually we needn't re-explain the already-observed immigrants;
computationally this sidesteps the incomputable Dirac comb of the multi-impulse
function. For interval-censored offspring (scenarios E, F) the endogenous IC-LL
uses $\Xi^{\text{endo}}(t)=\int_0^t\xi^{\text{endo}}$ and the offspring counts
$\mathsf F(o_{i-1},o_i]$.

> **Code.** `MBPP.endogenous_intensity` / `endogenous_compensator` return
> $\xi^{\text{endo}},\Xi^{\text{endo}}$; `fit_mbpp_ic(..., endogenous=True)` fits
> the endogenous loss. For the exponential kernel the endogenous compensator of a
> single impulse is $\frac{\kappa}{1-\kappa}\big(1-e^{(\kappa-1)\theta(t-a)}\big)$,
> whose limit $\frac{\kappa}{1-\kappa}$ is verified to equal the mean total
> progeny.

---

## 6. Forecasting (the ACTIVE prediction scheme)

Future counts are predicted with the Prop. 7 lower bound, replacing past
*expected* counts by *observed* ones (Eq. 55). With an augmented exogenous that
absorbs unobserved drivers,

$$
\hat s[i]=\gamma\,\mathbb 1[t=0]+\nu\,\mathbb 1[t>0]+\mu\cdot[\text{\#exogenous on day }i],
\tag{54}
$$

the forecast on a future interval is exogenous mass + excitation from all past and
already-predicted counts:

$$
\text{Predicted}[i]=\hat s[i]+\sum_{\text{past}}\mathsf C\!\int\!\phi+\sum_{\text{predicted}}\Xi\!\int\!\phi .
\tag{55}
$$

The recursion feeds each prediction back as a new excitation source.

> **Code.** `interval_censored.forecast_counts` implements (55). Anchoring each
> source interval's mass at its **right edge** makes the excitation telescope to
> exactly $\kappa$ per source, so the recursion is **mass-exact at stationarity**
> (verified: forecasting with the true parameters gives $<1\%$ cumulative bias;
> centre-anchoring would lose the near-field mass and bias the forecast low by
> $\sim\!28\%$). `exp7_forecast.py` shows the MBPP forecast roughly halving the
> sMAPE of an exogenous-only baseline — self-excitation matters for popularity.

---

## 7. Experiments

**Synthetic (Section 7).** From separable Hawkes realizations (two exogenous
functions HP-pc and HP-sin; a sub-critical regime $n^*=0.6$ and a near-critical
$n^*=0.95$) the authors construct all six scenarios and refit. Findings:
(i) MBPP recovers Hawkes parameters about as well as a Hawkes fit on event-time
data — MBPP is a drop-in; (ii) accuracy degrades gracefully as data is coarsened
from scenario A to F; (iii) the endogenous loss matches the non-endogenous one;
(iv) more discretization/observation intervals $\to$ better recovery;
(v) recovery is stable across a wide parameter grid.

**Real-world (Section 8).** On the **ACTIVE** dataset (≈14k YouTube videos with
their tweets), fitting in the IC-IC scenario F and forecasting views on days
91–120, the MBPP variants — and especially the **closed-form** exponential MBPP —
outperform HIP on APE and sMAPE. The closed-form MBPP beats the numerical
approximations (HIP and approximate MBPP are both approximations of it).

> **Code mapping.**
> `exp5_mbpp_impulse.py` → Figs 2–3 (MBPP = mean Hawkes intensity), validating the
> closed-form solver against Monte Carlo.
> `exp6_ic_scenarios.py` → Section 7: recovers $(\kappa,\theta)$ from
> interval-censored counts in scenarios E and F, in both regimes, IC-LL vs SSE,
> across observation granularities. Observed here: $\hat\kappa$ recovered to
> $\pm0.03$ (sub-critical) and to $0.95$ (near-critical) under IC-LL, while SSE
> runs to the boundary near criticality.
> `exp7_forecast.py` → Section 8: ACTIVE-style interval-censored forecasting,
> MBPP vs an exogenous-only baseline.

---

## 8. Extension implemented here: time-varying covariates in the baseline

The paper fixes the exogenous intensity to a chosen functional form $s(t)$. A
natural and useful extension — and the one this package adds beyond the paper —
is to drive the **baseline by observed covariates** through a log-linear link,
exactly as the event-time half of this package already does:

$$
s(t)=\exp\!\big(\gamma_0+\gamma^{\top}X(t)\big).
$$

**Why it requires no new MBPP machinery.** The MBPP solution $\xi=E*s=s+h*s$ is
**linear in $s$** (Corollary 3), and holds for *any* deterministic input $s(t)$.
The only question is tractability of the convolution. The key observation:

> If $X(t)$ is **piecewise-constant**, then $s(t)=\exp(\gamma_0+\gamma^\top X(t))$
> is piecewise-constant too — with rate $\exp(\gamma_0+\gamma^\top X_k)$ on the
> $k$-th covariate interval. Hence it is exactly the `PiecewiseConstant`
> exogenous of Section 5, and **the closed-form $\xi$ and $\Xi$ apply unchanged.**

So the entire IC-LL apparatus carries over verbatim; only the *parameterisation*
of the interval rates changes, from free rates to $\exp(\gamma_0+\gamma^\top X_k)$.
The interval-censored fitter then recovers $(\gamma_0,\gamma)$ jointly with the
kernel $(\kappa,\theta)$ by maximizing (19). The baseline compensator
$\mathsf S(t)=\int_0^t\exp(\gamma_0+\gamma^\top X)$ is the same closed-form sum
already used for the event-time covariate model. Continuous covariates are
handled identically, either by a fine piecewise-constant approximation or by the
numerical MBPP solver.

**A bonus in the interval-censored immigrant setting.** When immigrants are
interval-censored, the plain LHPP estimates each interval's rate as a free
parameter $\lambda_i=\mathsf S_i/\Delta$ (Prop. 13). Replacing this by the
log-linear regression $\log\lambda_i=\gamma_0+\gamma^\top X_i$ *shares strength*
across intervals through the covariates — a built-in regularizer that helps when
counts are sparse.

**Covariates in the *excitation* are harder.** Making the kernel itself depend on
covariates, $\alpha(t)=\alpha^{(0)}\exp(\delta^\top Z(t))$, destroys
time-invariance: the kernel becomes $\phi(t,\tau)$ (non-convolution), so the
impulse-response shortcut $h=\sum_n\phi^{*n}$ no longer applies and one must solve
a general (non-convolution) Volterra equation numerically, with a non-convex
likelihood. This is a genuine step up and is not implemented here.

**Identifiability caveat.** In interval-censored data, $\gamma$ is identified from
how counts vary with $X$ *across* observation intervals; if $X$ is constant within
every interval the contrast is clean, but covariate variation faster than the
observation grid is averaged out. Empirically (see `exp8` below), the *covariate
effect* $\gamma_1$ is recovered tightly, while the absolute split between baseline
level $\gamma_0$ and self-excitation $\kappa$ is more weakly identified in the
non-separable joint fit — a manifestation of the same $\alpha$–$\beta$ weak
identifiability noted for event-time Hawkes (NOTES §3).

> **Code.** `exogenous.CovariateExogenous(covariate, gamma0, gamma)` builds
> $s(t)=\exp(\gamma_0+\gamma^\top X(t))$ from a `PiecewiseConstantCovariate`;
> `interval_censored.fit_mbpp_ic_covariates` recovers $(\kappa,\theta,\gamma_0,\gamma)$
> from counts. `exp8_ic_covariates.py` is the interval-censored analogue of the
> event-time covariate experiment `exp3`: with a regime-switching baseline it
> recovers $\gamma_1=0.90$ as $0.90\pm0.06$ from counts alone.

---

## 9. Extension implemented here: the functional / operator view

The MBPP solution map $G:s\mapsto\xi$ defined by $\xi=s+\xi*\phi$ (Eq. 9) is a
**linear, translation-invariant operator**. Making that operator explicit gives a
flexible framework in which the same MBPP can be solved or *learned* in several
interchangeable ways — useful when the forcing has an arbitrary functional form,
when the kernel is unknown, or when the dynamics are nonlinear.

**The ODE reduction (rational Laplace $\Rightarrow$ finite linear ODE).** Split
off the endogenous part $y=\xi-s$ in Eq. (9) with the exponential kernel and
differentiate:

$$
y'(t)=(\kappa-1)\theta\,y(t)+\kappa\theta\,s(t),\qquad y(0)=0,\qquad \xi=s+y.
$$

So $\xi$ solves a **first-order linear ODE whose forcing term is the (covariate)
baseline $s(t)$**, with a stable pole at $(\kappa-1)\theta<0$. More generally a
convolution Volterra equation reduces to a *finite* linear ODE **iff the kernel
has a rational Laplace transform**, i.e. iff it is a sum of exponentials
$\phi(t)=\sum_{q}a_q e^{-b_q t}$. Then with states $u_q=(a_q e^{-b_q\cdot})*\xi$,

$$
u'(t)=A\,u(t)+a\,s(t),\quad \xi=s+\mathbf 1^{\top}u,\quad A=a\mathbf 1^{\top}-\operatorname{diag}(b),
$$

an LTI state-space system forced by $s$. Integrating it (RK4) solves the MBPP for
*any* functional form of the forcing and any sum-of-exponentials kernel (and sums
of exponentials are dense, so this approximates power-law/Rayleigh too).

**The spectral operator (a Fourier neural operator).** In frequency space the
operator is a pointwise multiplier,

$$
\hat\xi(\omega)=R(\omega)\,\hat s(\omega),\qquad R(\omega)=\frac{1}{1-\hat\phi(\omega)},
$$

which is Theorem 1 in Fourier form (using $\hat h=\sum_{n\ge1}\hat\phi^{\,n}=\hat\phi/(1-\hat\phi)$,
so $\hat E=1+\hat h=1/(1-\hat\phi)$). A single **linear Fourier-neural-operator
layer** computes exactly $\xi=\mathrm{IFFT}(R\cdot\mathrm{FFT}(s))$, so with
$R=1/(1-\hat\phi)$ it *is* the exact MBPP operator. Conversely, **learning**
$R(\omega)$ from $(s,\xi)$ pairs by per-frequency least squares is nonparametric
operator identification, after which the kernel is recovered as
$\hat\phi=1-1/R$ — the Bacry–Muzy Wiener–Hopf idea in neural-operator form. For
the exponential kernel $\hat\phi(\omega)=\kappa\theta/(\theta+i\omega)$ gives the
one-pole filter $R=(\theta+i\omega)/((1-\kappa)\theta+i\omega)$, whose pole is the
ODE rate $(\kappa-1)\theta$ — time-domain ODE, frequency-domain $R$, and the
Green's function $E$ are three views of one operator.

**Learned operators for the hard regimes.** A **DeepONet** (branch encodes the
forcing at sensor points, trunk encodes the query time, output is their inner
product) approximates $s\mapsto\xi$ *without* assuming linearity — the route when
there is no closed form (inhibition / non-linear Hawkes, marked/spatial kernels).
An **amortized inference** network learns the *inverse* map
$\text{counts}\mapsto(\kappa,\theta)$, turning a fit into a single forward pass.

**When to use which.** Exploit known structure when you have it: the ODE and
exact-spectral backends are essentially exact and free. The learned operators
earn their keep when the kernel is unknown/nonparametric (learned spectral), the
dynamics are nonlinear/no-closed-form (DeepONet), or inference must be amortized
across many series.

> **Code & numbers (`exp9_functional_operators.py`, all numpy).**
> `operators.solve_mbpp_ode` / `FunctionalMBPP(method="ode")` — matches the closed
> form to $\sim\!10^{-11}$ for smooth forcing. `operators.SpectralOperator` —
> exact from a kernel (err $\sim\!10^{-2}$) and **learned** from data (generalises
> to new forcings at $\sim\!5\%$ L2, recovers the branching ratio $0.64$ vs $0.6$).
> `operators_nn.DeepONetOperator` — learns $s\mapsto\xi$ to $\sim\!12\%$ test L2.
> `operators_nn.AmortizedInference` — predicts $\kappa$ in one pass (corr $0.93$),
> and, faithfully, struggles with the weakly-identified $\theta$ (corr $\sim\!0.1$).

---

## 10. Extension implemented here: high-dimensional MBPP and TensorFlow operators

Everything above is univariate. The same operator picture scales to a
**multivariate** point process of dimension $M$, and to **large-data /
high-dimensional** learning in TensorFlow.

**Multivariate MBPP.** The intensity is now a vector $\xi(t)\in\mathbb R^{M}$
driven by an $M\times M$ matrix of kernels:
$\xi(t)=s(t)+\int_0^t\Phi(t-\tau)\xi(\tau)\,d\tau$, with
$\Phi_{m,j}(u)=A_{m,j}e^{-B_{m,j}u}$. The state-space reduction generalises: with
states $y_{m,j}=(A_{m,j}e^{-B_{m,j}\cdot})*\xi_j$,

$$
y'_{m,j}=A_{m,j}\,\xi_j-B_{m,j}\,y_{m,j},\qquad \xi_m=s_m+\sum_j y_{m,j},
$$

a linear ODE in $M^2$ states forced by the multivariate baseline $s(t)$. Its
stationary solution under a constant baseline is the familiar
$\bar\xi=(I-G)^{-1}\bar\mu$ with branching matrix $G=A/B$ (stable iff the spectral
radius of $G$ is $<1$). This exact solver supplies ground-truth $(s,\xi)$ pairs
for the learned operators.

**TensorFlow neural operators (high-dimensional, lots of data).** Treating a
batch of series as a tensor of shape $(\text{batch},T,M)$, four Keras models learn
the evolution at scale (batched via `tf.data`, GPU-ready):

* a **multivariate Fourier Neural Operator** — stacked spectral-convolution layers
  that FFT over time and mix the $M$ channels with learnable complex weights on
  the low Fourier modes; the natural high-dimensional generalisation of the linear
  transfer-function operator $R(\omega)$ of §9 (one channel, one mode-band) to
  many channels, depth and non-linearity;
* a **multivariate DeepONet** (branch over the $M$-channel forcing, trunk over
  query time);
* a **learned state-space / Neural-ODE** (`StateSpaceMBPP`) that integrates a
  *stable* forced recurrence $z_{t+1}=z_t+\Delta t(-d\odot z_t+\tanh(\cdots)+\text{forcing})$
  and reads out $\xi$ — the trainable generalisation of the exact ODE drift, $O(T)$
  per series so it scales to long sequences and large $M$;
* an **amortised inference** CNN mapping interval counts $(\text{batch},T,M)$ to the
  $M\times M$ branching matrix in a single forward pass.

> **Code.** `operators.solve_mbpp_ode_multivariate` (numpy, exact ground truth);
> `operators_tf.py` (`FourierNeuralOperator`, `MBPPDeepONet`, `StateSpaceMBPP`,
> `AmortizedKernelInference`, plus `generate_operator_dataset` / `make_dataset` /
> `train_operator`). `operators_tf` is an optional extra — it requires TensorFlow
> and is not imported by the package `__init__`, so the rest of
> `hawkes_calibration` stays numpy-only.

---

## 11. Contributions, in the authors' numbering

* **C1.** Construct the MBPP approximating a Hawkes process's mean behavior
  (Section 3); its likelihood admits a Bregman family of losses (Section 5).
* **C2.** Prove the MBPP is a causal LTI system with impulse response an infinite
  convolution sum (Theorem 2), giving closed-form and numerical solutions.
* **C3.** A second, numerically efficient compensator approximation usable for
  forecasting (Section 4.3).
* **C4.** Tools — exogenous and loss functions — that fit Hawkes parameters when
  exogenous and/or endogenous events are interval-censored (Section 6).
* **C5.** On real data (ACTIVE), beat the prior state of the art HIP in popularity
  prediction (Section 8).

---

## 12. Paper → code reference

| Paper construct | Equation/Result | Code |
|---|---|---|
| Exponential / power-law kernel | Eq. 13 | `mbpp.ExponentialKernel`, `mbpp.PowerLawKernel` |
| MBPP intensity $\xi=s+\xi*\phi$ | Eq. 9 | `mbpp.MBPP.intensity` |
| MBPP compensator $\Xi=\mathsf S+\Xi*\phi$ | Eq. 10 (proof §2.2) | `mbpp.MBPP.compensator` |
| Laplace solution | Thm 1, Eq. 11 | (documented; closed forms used instead) |
| Impulse response $E=\delta+\sum\phi^{*n}$ | Thm 2 (proof §2.3) | `ExponentialKernel.impulse_response_h` |
| Solution $\xi=E*s$ | Cor. 3 | `MBPP` closed-form assembly |
| Exponential $h=\kappa\theta e^{(\kappa-1)\theta t}$ | Eq. 14 | `ExponentialKernel.impulse_response_h` |
| Numerical $h\approx\sum_{n\le N}\phi^{*n}$ | §3.3 | `MBPP._build_numeric` |
| Lower-bound compensator | Prop. 6–7 | `MBPP.compensator_lower_bound` |
| IC-LL | Eq. 19 | `interval_censored.ic_ll` |
| KL ↔ IC-LL | Prop. 8 (proof §4) | (objective identity; tested) |
| SSE / Bregman | Eq. 36 | `interval_censored.sse_loss` |
| HIP recovery | Lem. 9, Thm 10–11 | discussed in §4 |
| Multi-impulse $s_{m\delta}$ | Def. 12 | `exogenous.MultiImpulse` |
| LHPP $\lambda_i=\mathsf S_i/\Delta$ | Def. 14, Prop. 13 (proof §5) | `exogenous.LHPP` |
| Endogenous loss | §6.3, Eq. 48–51 | `MBPP.endogenous_*`, `fit_mbpp_ic(endogenous=True)` |
| Cluster simulation | §5, §7.1 | `ic_simulate.simulate_separable_hawkes` |
| Forecasting | Eq. 54–55 | `interval_censored.forecast_counts` |
| Six scenarios A–F | Table 2 | `exp6_ic_scenarios.py` |
| Figures 2–3 | §3 | `exp5_mbpp_impulse.py` |
| ACTIVE prediction | §8 | `exp7_forecast.py` |
| Covariate baseline $s=e^{\gamma_0+\gamma^\top X}$ | extension (§8 above) | `exogenous.CovariateExogenous` |
| Fit $(\kappa,\theta,\gamma_0,\gamma)$ from counts | extension | `interval_censored.fit_mbpp_ic_covariates` |
| Covariate recovery demo | extension | `exp8_ic_covariates.py` |
| ODE / state-space operator | extension (§9 above) | `operators.solve_mbpp_ode`, `FunctionalMBPP` |
| Spectral operator $R=1/(1-\hat\phi)$ + kernel ID | extension | `operators.SpectralOperator` |
| DeepONet / amortized inference | extension | `operators_nn.py` |
| Operator backends demo | extension | `exp9_functional_operators.py` |
| Multivariate MBPP $\xi\in\mathbb R^M$, matrix kernel | extension (§10) | `operators.solve_mbpp_ode_multivariate` |
| High-dim TF operators (FNO/DeepONet/SSM) | extension | `operators_tf.py` |
| Amortized $M\times M$ kernel inference | extension | `operators_tf.AmortizedKernelInference` |

---

# Mathematical Appendices

The body above is a working summary. These appendices give the supporting theory
in full: the measure-theoretic and martingale foundations (A), the Hawkes cluster
/ branching theory (B), existence–uniqueness and the complete solution proofs for
the MBPP equation (C), every closed-form exogenous response and compensator (D),
the interval-censored likelihood and its Bregman geometry (E), the operator /
ODE / spectral / fractional theory (F), and estimation & identifiability (G).
Equation numbers $(n)$ refer to the paper; appendix results are labelled by
letter.

---

## Standing assumptions and conventions

Unless explicitly relaxed, the following hold throughout the appendices. They are
mild and hold for every model in this package (exponential kernels; power-law
kernels with positive offset $c>0$; non-negative, locally integrable baselines).

- **(A1) Simplicity & non-explosion.** $N$ is a simple temporal point process on
  $[0,\infty)$: no two events coincide ($\mathbb P$-a.s.), and $N(t)<\infty$ a.s.
  for every finite $t$.
- **(A2) Stochastic intensity.** $N$ admits an $(\mathcal H_t)$-predictable
  conditional intensity $\lambda\ge0$ with respect to its internal history
  $\mathcal H_t=\sigma\{N(s):s\le t\}$; equivalently the compensator
  $\Lambda(t)=\int_0^t\lambda$ is absolutely continuous, with $\int_0^T\lambda<\infty$ a.s.
- **(A3) Kernel.** $\phi:\mathbb R\to[0,\infty)$ is measurable, causal
  ($\phi(u)=0$ for $u<0$), locally bounded, with branching ratio
  $n^*:=\int_0^\infty\phi=\kappa<\infty$. *Subcritical* means $\kappa<1$.
- **(A4) Baseline.** $s:[0,\infty)\to[0,\infty)$ is measurable and locally
  integrable; where the multi-impulse (Dirac comb) baseline appears it is a
  non-negative measure, and convolutions with it are read distributionally.
- **(A5) Finite mean intensity.** Wherever $\mathbb E[\lambda]$ is used,
  $\sup_{t\le T}\mathbb E[\lambda(t)]<\infty$ — automatic under (A3)-subcritical
  with locally integrable $s$.
- **(A6) Smooth identifiable parameterisation.** For estimation, $\theta$ ranges
  over an open $\Theta\subseteq\mathbb R^p$; $\theta\mapsto\lambda(t;\theta)$ is
  twice continuously differentiable with $\lambda(t;\theta)>0$ on the data, and the
  model is identifiable on $\Theta$.
- **(A7) Multivariate.** In dimension $M$: $N=(N_1,\dots,N_M)$, intensity
  $\lambda\in\mathbb R_+^M$, kernel matrix $\Phi=(\phi_{mj})$, branching matrix
  $G=(\int\phi_{mj})$; *stable* means spectral radius $\rho(G)<1$.

Each result below states any **extra** hypotheses beyond (A1)–(A7) explicitly.

---

## Appendix A. Foundations: random measures, compensators, and why Poisson ⇒ independent increments

### A.1 Point processes as random counting measures

Let $(\mathbb X,\mathcal B)$ be a Polish state space (here $\mathbb X=[0,\infty)$,
time) and $(\Omega,\mathcal F,\mathbb P)$ a probability space. A **point process**
is a measurable map $N:\Omega\to\mathsf N$ into the space of locally finite
counting measures on $\mathcal B$; for $A\in\mathcal B$, $N(A)$ counts events in
$A$. The temporal counting process is $N(t):=N((0,t])$, right-continuous,
integer-valued, with unit jumps for a **simple** process (no two events
coincide). The internal history (filtration) is
$\mathcal H_t=\sigma\{N(s):s\le t\}$.

### A.2 Compensator and conditional intensity (Doob–Meyer)

$N(t)$ is a submartingale; by the **Doob–Meyer decomposition** there is a unique
predictable, right-continuous, increasing process $\Lambda$ with $\Lambda(0)=0$
such that

$$
M(t):=N(t)-\Lambda(t)\quad\text{is an }(\mathcal H_t)\text{-local martingale.}
$$

$\Lambda$ is the **compensator**. When $\Lambda$ is absolutely continuous,
$\Lambda(t)=\int_0^t\lambda(s)\,ds$ and the density $\lambda$ is the **conditional
intensity**

$$
\lambda(t)=\lim_{h\downarrow0}\tfrac1h\,\mathbb E\!\big[N(t{+}h)-N(t)\mid\mathcal H_{t^-}\big].
$$

Heuristically $\mathbb E[dN(t)\mid\mathcal H_{t^-}]=\lambda(t)\,dt$, the identity
used throughout.

**Assumptions & intuition.** Needs only (A1)–(A2) (Doob–Meyer applies to any
integrable adapted submartingale). *Intuition:* $\Lambda$ is the predictable
"running forecast" of how many events should have accumulated; $\lambda$ is that
forecast's instantaneous rate. Subtracting the forecast from the realised count
leaves pure surprise — the martingale $M=N-\Lambda$ — so $\lambda$ is exactly the
part of the future the past lets you anticipate.

### A.3 The likelihood of a point process (Jacod formula)

**Theorem A.1 (Jacod).** *A simple point process with $(\mathbb P,\mathcal
H_t)$-intensity $\lambda$ admits, on $[0,T]$, a likelihood with respect to the
unit-rate Poisson measure*
$$
\frac{d\mathbb P}{d\mathbb P_0}\Big|_{\mathcal H_T}
=\Big(\prod_{t_i\le T}\lambda(t_i)\Big)\exp\!\Big(-\!\int_0^T(\lambda(s)-1)\,ds\Big).
$$
*Hence the log-likelihood, up to the constant $T$, is*
$\ell(\theta)=\sum_{t_i\le T}\log\lambda(t_i;\theta)-\int_0^T\lambda(s;\theta)\,ds$,
*which is exactly the PP-LL of Eq. (4).*

This is the change-of-measure (Girsanov) formula for point processes; the product
term rewards intensity at observed times, the integral penalises total intensity.

**Assumptions & intuition.** Beyond (A1)–(A2): $\lambda$ predictable,
$\int_0^T\lambda<\infty$ a.s., and $\lambda(t_i)>0$ at the observed events (so the
log is finite). *Intuition:* the likelihood rewards having forecast high intensity
exactly where events actually landed ($\prod_i\lambda(t_i)$) and penalises total
forecast mass ($\int\lambda$) — "put probability on the bumps, don't waste it
elsewhere." It is the point-process analogue of $\prod p(x_i)$.

### A.4 Laplace functional and the exponential formula

For a Poisson process with mean measure $\Lambda$, the **Laplace functional** is
$$
\mathbb E\Big[\exp\!\big(-\!\textstyle\int f\,dN\big)\Big]
=\exp\!\Big(-\!\int(1-e^{-f(x)})\,\Lambda(dx)\Big),\qquad f\ge0 .
$$
Setting $f=\theta\mathbb 1_A$ recovers $N(A)\sim\mathrm{Poisson}(\Lambda(A))$, and
products over disjoint sets factorise — the structural fact behind the IC-LL.

**Assumptions & intuition.** $f\ge0$ measurable, $\Lambda$ $\sigma$-finite.
*Intuition:* because a Poisson process scatters points independently across
disjoint cells, the transform of any additive functional factorises into a product
over cells — the analytic fingerprint of independent increments, and the engine
behind the IC-LL product.

### A.5 Watanabe's theorem: deterministic compensator ⇒ Poisson

**Theorem A.2 (Watanabe characterisation).** *Let $N$ be a simple point process
with continuous compensator $\Lambda$. Then $N$ is an inhomogeneous Poisson
process with mean measure $\Lambda$ if and only if $\Lambda$ is **deterministic**.
Equivalently, $N(t)-\Lambda(t)$ and $(N(t)-\Lambda(t))^2-\Lambda(t)$ are both
martingales.*

**Consequence (the crux of the whole construction).** The MBPP is *defined* to
have the deterministic intensity $\xi(t)=\mathbb E[\lambda(t)]$, hence a
deterministic compensator $\Xi(t)=\int_0^t\xi$. By Theorem A.2 it is therefore a
genuine Poisson process and **has independent increments**, so

$$
\mathbb P\big(N(o_{i-1},o_i]=c_i,\ i=1,\dots,m\big)=\prod_{i=1}^m\mathbb P\big(N(o_{i-1},o_i]=c_i\big),
\qquad N(o_{i-1},o_i]\sim\mathrm{Poisson}\big(\Xi(o_{i-1},o_i]\big),
$$

which is precisely the factorisation that makes IC-LL (Eq. 19) a tractable
product of Poisson terms. The Hawkes process fails Theorem A.2 because its
compensator $\Lambda$ is *random* (history-dependent), so its increments are
dependent — the obstruction the MBPP removes.

---

**Assumptions & intuition.** $N$ simple (A1) with a *continuous, deterministic*
compensator. *Intuition:* dependence between increments can only arise from the
intensity *reacting to history*; if the intensity — hence the compensator — is
fixed in advance, there is no channel through which the past can influence the
future, so the process must be Poisson. This is the whole reason the MBPP works:
replacing the random Hawkes intensity $\lambda$ by its deterministic mean $\xi$
severs the feedback and *manufactures* independent increments.

---

## Appendix B. The Hawkes process: cluster representation, branching, and stability

### B.1 Cluster (immigrant–offspring) representation

**Theorem B.1 (Hawkes–Oakes, 1974).** *The Hawkes process of Eq. (3) is
distributionally equal to the following branching construction:*

1. *immigrants arrive as an inhomogeneous Poisson process of intensity $s(t)$;*
2. *each event at time $u$ (immigrant or offspring) independently produces direct
   offspring as an inhomogeneous Poisson process of intensity $\phi(\cdot-u)$ on
   $(u,\infty)$;*
3. *the Hawkes process is the superposition of all immigrants and their
   descendants.*

Each immigrant together with its descendants is a **cluster** (cascade). The
construction makes the "self-excitation" literal: events beget events.

**Assumptions & intuition.** (A3)–(A4); the immigrant intensity is $s$ and each
event's offspring kernel is $\phi$. *Intuition:* self-excitation made literal —
every event founds a "family", and the whole process is a *forest of independent
family trees* whose roots (immigrants) are sprinkled by the baseline. This picture
is what makes the mean, the variance, and the simulation all tractable.

### B.2 Branching ratio and the mean-intensity equation

The expected number of **direct** offspring of one event is
$n^*:=\int_0^\infty\phi(t)\,dt=\kappa$ (the **branching ratio**). Taking
expectations in Eq. (3) and using $\mathbb E[dN(u)\mid\mathcal
H_{u^-}]=\lambda(u)\,du$ with the tower property:

**Proposition B.2 (rigorous Eq. 9).** *Writing $\xi(t)=\mathbb E[\lambda(t)]$ and
assuming $\sup_{t\le T}\xi(t)<\infty$,*
$$
\xi(t)=s(t)+\int_0^t\phi(t-u)\,\mathbb E[dN(u)]=s(t)+\int_0^t\phi(t-u)\,\xi(u)\,du .
$$
**Proof.** Take $\mathbb E[\cdot]$ of $\lambda(t)=s(t)+\int_0^{t^-}\phi(t-u)\,dN(u)$.
By Tonelli (all integrands $\ge0$) the expectation passes inside the integral, and
$\mathbb E[dN(u)]=\mathbb E[\mathbb E[dN(u)\mid\mathcal H_{u^-}]]=\mathbb E[\lambda(u)]\,du=\xi(u)\,du$.
$\;\blacksquare$

**Assumptions & intuition.** (A3)–(A5); the interchange uses Tonelli (non-negative
integrands). *Intuition:* averaging the *stochastic* feedback over all realisations
turns it into *deterministic* feedback — "on average, today's rate equals the
baseline plus the kernel-weighted echo of the average past." That is precisely the
MBPP equation (9), now derived rather than posited.

### B.3 Stability / subcriticality

**Proposition B.3.** *A stationary, finite-intensity version of the Hawkes process
exists iff $n^*=\kappa<1$ (spectral radius $<1$ in the multivariate case,
Appendix F.4). Then the stationary mean intensity is $\bar\xi=\bar s/(1-\kappa)$.*

**Proof sketch.** Each event spawns on average $\kappa$ direct offspring, so a
cluster is a Galton–Watson tree with mean offspring $\kappa$; it is a.s. finite
iff $\kappa\le1$ and has finite expected size iff $\kappa<1$. Stationarity of the
superposition then follows from stationarity of the immigrant stream. In the
stationary regime $\bar\xi=\bar s+\kappa\bar\xi$ from Prop. B.2, giving
$\bar\xi=\bar s/(1-\kappa)$. $\;\blacksquare$

**Assumptions & intuition.** Stationary version needs subcriticality $\kappa<1$
(A3) and a stationary baseline (e.g. constant $\bar s$). *Intuition:* one event
echoes into $\kappa$ direct offspring, each of which echoes again, so the total
gain is the geometric series $1+\kappa+\kappa^2+\dots=1/(1-\kappa)$. Subcritical
means the echoes fade; as $\kappa\uparrow1$ they never fully decay and the mean
blows up.

### B.4 Cluster size: Borel and Borel–Tanner laws

When immigrants arrive and offspring counts are Poisson, the **total progeny** $S$
of a single cluster (one immigrant + all descendants) follows the **Borel
distribution**:

$$
\mathbb P(S=n)=\frac{e^{-\kappa n}(\kappa n)^{n-1}}{n!},\quad n\ge1,
\qquad \mathbb E[S]=\frac1{1-\kappa},\qquad \mathrm{Var}(S)=\frac{\kappa}{(1-\kappa)^3}.
$$

A cluster seeded by $k$ immigrants has total size $\sim$ **Borel–Tanner**$(\kappa,k)$.
The expected number of *offspring* per immigrant is $\mathbb
E[S]-1=\kappa/(1-\kappa)$ — exactly the total endogenous compensator mass
$\Xi^{\mathrm{endo}}(\infty)$ of a single impulse computed in Appendix D.1, and the
divergence of $\mathrm{Var}(S)$ as $\kappa\uparrow1$ explains the heavy right-tail
and the slow Monte-Carlo convergence observed near criticality.

**Assumptions & intuition.** Offspring counts are $\mathrm{Poisson}(\kappa)$ per
event (the cluster representation of Thm B.1) and $\kappa<1$. *Intuition:* a
cluster's size is the total descendants of a branching tree; its mean is the
geometric sum $1/(1-\kappa)$, and near criticality a handful of *giant* cascades
dominate — the heavy Borel tail (variance $\kappa/(1-\kappa)^3$) that makes naïve
Monte-Carlo estimates of near-critical means converge painfully slowly.

### B.5 Why the interval-censored Hawkes likelihood is intractable

Decomposing $N(o_{i-1},o_i]$ over offspring generations yields nested **Borel
convolutions**; Kong et al. (2020) give the $t\to\infty$ closed form, but at finite
$T$ there is no closed form for the joint law of generation counts, and Monte
Carlo over the conditional generation factorisation scales poorly. This is the
formal statement of "the Hawkes process cannot be fit interval-censored," and the
reason the MBPP detour (Appendix A.5) is needed.

---

**Intuition.** A bin's count is a sum of contributions from an *unknown* number of
*unobserved* offspring generations; without the timestamps that reveal who
triggered whom, those generations cannot be disentangled, so the joint law has no
finite closed form. The MBPP sidesteps this by modelling the *mean* directly.

---

## Appendix C. The MBPP equation: existence, uniqueness, and complete solutions

### C.1 The Volterra operator

Equation (9), $\xi=s+\phi*\xi$, is a **linear Volterra equation of the second
kind**. Define the Volterra (causal convolution) operator

$$
(\mathcal V\xi)(t):=\int_0^t\phi(t-\tau)\,\xi(\tau)\,d\tau,\qquad t\in[0,T],
$$

so the equation reads $(\mathcal I-\mathcal V)\xi=s$.

**Intuition.** $\mathcal V$ feeds the output back through the kernel, so solving
$(\mathcal I-\mathcal V)\xi=s$ means *inverting a feedback loop*. Every result in
this appendix answers one question — when, and how, can $\mathcal I-\mathcal V$ be
inverted?

### C.2 Existence and uniqueness (always, on finite horizons)

**Theorem C.1.** *Let $\phi$ be bounded on $[0,T]$, $\bar\phi:=\sup_{[0,T]}\phi<\infty$,
and $s\in C[0,T]$. Then (9) has a unique solution $\xi\in C[0,T]$, given by the
Neumann series $\xi=\sum_{n\ge0}\mathcal V^n s$.*

**Proof.** $\mathcal V^n$ has the iterated kernel $\phi^{*n}$. By induction
$|(\mathcal V^n\xi)(t)|\le \bar\phi^{\,n}\,\|\xi\|_\infty\,t^n/n!$: the base case is
$|\mathcal V\xi(t)|\le\bar\phi\|\xi\|_\infty t$, and
$|\mathcal V^{n+1}\xi(t)|\le\int_0^t\bar\phi\,|\mathcal V^n\xi(\tau)|\,d\tau
\le\bar\phi^{\,n+1}\|\xi\|_\infty\int_0^t\tau^n/n!\,d\tau=\bar\phi^{\,n+1}\|\xi\|_\infty t^{n+1}/(n+1)!$.
Hence $\|\mathcal V^n\|_{C[0,T]}\le(\bar\phi T)^n/n!\to0$, so $\sum_n\mathcal V^n$
converges in operator norm, $(\mathcal I-\mathcal V)$ is invertible, and the unique
fixed point is $\xi=(\mathcal I-\mathcal V)^{-1}s=\sum_{n\ge0}\mathcal V^n s$.
$\;\blacksquare$

Note this needs **no** subcriticality assumption: on any finite horizon the MBPP
intensity exists and is unique for any bounded kernel. Subcriticality enters only
for the *global* ($t\to\infty$) resolvent.

**Assumptions & intuition.** Extra hypotheses: $\phi$ bounded on $[0,T]$ (true for
the exponential kernel and for power-law with offset $c>0$) and $s\in C[0,T]$; the
result extends to $\phi\in L^1_{\mathrm loc}$ via Volterra-resolvent theory
(Gripenberg–Londen–Staffans). *Intuition:* on a finite horizon, feedback can only
compound so fast — the $n$-th echo $\mathcal V^n$ carries a $t^n/n!$ factor that
crushes it — so the loop is **always** invertible there, regardless of the gain
$\kappa$. "No runaway in finite time." Subcriticality is needed only to push this
to $t=\infty$.

### C.3 The resolvent kernel and global convergence

Writing $\mathcal V^n s=\phi^{*n}*s$ and separating the $n=0$ term,
$\xi=s+h*s$ with the **resolvent kernel** $h=\sum_{n\ge1}\phi^{*n}$.

**Proposition C.2.** *If $\int_0^\infty\phi=\kappa<1$ then $h\in L^1(0,\infty)$ with
$\|h\|_1=\kappa/(1-\kappa)$, and the resolvent representation $\xi=s+h*s$ holds on
$[0,\infty)$.*

**Proof.** Young's convolution inequality gives
$\|\phi^{*n}\|_1\le\|\phi\|_1^{\,n}=\kappa^n$, so
$\|h\|_1\le\sum_{n\ge1}\kappa^n=\kappa/(1-\kappa)<\infty$. Dominated convergence
justifies summation. $\;\blacksquare$

(The total mass $\|h\|_1=\kappa/(1-\kappa)$ is the expected offspring-per-event of
Appendix B.4 — the resolvent *is* the renewal density of the branching process.)

**Assumptions & intuition.** Extra: $\kappa=\|\phi\|_1<1$ and $\phi\in L^1$.
*Intuition:* the resolvent $h=\sum_{n\ge1}\phi^{*n}$ sums the echo over *all*
generations; it is globally integrable exactly when each successive generation
shrinks ($\kappa<1$), and its total mass $\kappa/(1-\kappa)$ is the expected
offspring count — i.e. $h$ *is* the renewal density of the branching process
(Appendix B.4).

### C.4 Theorem 1 (Laplace solution), in full

**Theorem C.3 (= paper Thm 1).** *If the Laplace transforms $\mathcal L\{s\}$,
$\mathcal L\{\phi\}$ exist on a common right half-plane and $\mathcal
L\{\phi\}\neq1$ there, then*
$$
\xi(t)=\mathcal L^{-1}\!\Big\{\tfrac{\mathcal L\{s\}(z)}{1-\mathcal L\{\phi\}(z)}\Big\}(t).
$$
**Proof.** The Laplace transform turns causal convolution into a product:
$\mathcal L\{\phi*\xi\}=\mathcal L\{\phi\}\,\mathcal L\{\xi\}$. Transforming (9),
$\mathcal L\{\xi\}=\mathcal L\{s\}+\mathcal L\{\phi\}\mathcal L\{\xi\}$, so
$\mathcal L\{\xi\}=\mathcal L\{s\}/(1-\mathcal L\{\phi\})$; invert. Existence of the
inverse is guaranteed when the right side is, e.g., a proper rational function
(Appendix F.2). $\;\blacksquare$

**Assumptions & intuition.** Extra: $\mathcal L\{s\},\mathcal L\{\phi\}$ exist on a
common half-plane $\operatorname{Re}z>\sigma$, $\mathcal L\{\phi\}\neq1$ there, and
the inverse transform exists (guaranteed if the right side is proper rational,
F.2). *Intuition:* the Laplace transform turns convolution into multiplication, so
the feedback loop collapses to the scalar geometric series
$1/(1-\text{loop gain})$ with loop gain $\mathcal L\{\phi\}$ — the same
$1/(1-\kappa)$ amplification seen everywhere, now frequency-by-frequency.

### C.5 Theorem 2 (impulse response / LTI), in full

**Lemma C.4 (the map is causal LTI).** *The solution operator $G:s\mapsto\xi$ is
linear, time-invariant and causal.*

**Proof.** *Linearity* is immediate from linearity of (9). *Time-invariance:* let
$\xi$ solve (9) for input $s$ and put $s_{t_0}(t):=s(t-t_0)$. With the change of
variable $\tau\mapsto\tau-t_0$ and causality $\phi(u)=0$ for $u<0$,
$$
s_{t_0}(t)+\int_0^t\phi(t-\tau)\,\xi(\tau-t_0)\,d\tau
=s(t-t_0)+\int_0^{t-t_0}\phi((t-t_0)-\tau')\,\xi(\tau')\,d\tau'=\xi(t-t_0),
$$
so $G(s_{t_0})=(G s)_{t_0}$. *Causality* of $\xi$ follows from that of $\phi$.
$\;\blacksquare$

**Theorem C.5 (= paper Thm 2).** *If $\int_0^\infty\phi<1$, the impulse response of
$G$ is $E=\delta+h$ with $h=\sum_{n\ge1}\phi^{*n}$.*

**Proof.** Put $s=\delta$ in (9): $E=\delta+\mathcal V E=\delta+(\delta+h)*\phi
=\delta+\phi+h*\phi$. Matching non-impulsive parts gives the renewal identity
$h=\phi+h*\phi$, whose iteration yields $h=\sum_{n\ge1}\phi^{*n}$ (Appendix C.3 for
convergence). $\;\blacksquare$

**Corollary C.6 (= Cor. 3 & 5).** *For any input, $\xi=E*s=s+h*s$ solves (9); and
the compensator satisfies $\Xi=E*\mathsf S=\mathsf S+h*\mathsf S$, with
$\Xi(x,y]=(E*\mathsf S)(y)-(E*\mathsf S)(x)$.* (Proof: substitute and use
$h=E*\phi$; the compensator statement integrates (9), cf. body §2.2 for the Fubini
derivation of $\Xi=\mathsf S+\phi*\Xi$.)

**Assumptions & intuition.** Extra: $\kappa<1$ (for the global resolvent);
$E=\delta+h$ is a distribution and convolutions with the Dirac comb are
distributional. *Intuition:* the impulse response $E$ is the system's complete
"echo" to a single unit spike; because the system is linear and time-invariant,
the response to *any* input is just a superposition of shifted, scaled echoes,
$\xi=E*s$ — solve once for the spike, reuse forever.

### C.6 The exponential kernel closes the series

For $\phi(t)=\kappa\theta e^{-\theta t}$, $\phi^{*n}(t)=\kappa^n\theta^n
e^{-\theta t}t^{n-1}/(n-1)!$ (induction), and summing the exponential series gives
$h(t)=\kappa\theta\,e^{(\kappa-1)\theta t}$ (Eq. 14). The single exponential is why
the resolvent is one exponential and the ODE (Appendix F.1) is first order.

---

**Intuition.** A single memory timescale ($\phi$ one exponential) gives a single
exponential echo ($h$), hence first-order dynamics: one number $\kappa$ sets the
loop gain and one number $\theta$ sets its speed. Multiple timescales (sums of
exponentials) simply stack more first-order modes (Appendix F.2).

---

## Appendix D. All closed forms (Table 10) derived

Throughout, the exponential kernel gives resolvent $h(t)=\kappa\theta\,e^{ct}$ with
$$
c:=(\kappa-1)\theta<0,\qquad K:=\frac{\kappa}{1-\kappa},\qquad \xi=s+(h*s),\quad \Xi=\mathsf S+(h*\mathsf S).
$$
We repeatedly use $\displaystyle\int_0^{u}\!e^{cr}\,dr=\frac{e^{cu}-1}{c}$ and
$\dfrac{\kappa\theta}{c}=\dfrac{\kappa}{\kappa-1}=-K$.

**Assumptions & intuition (all of Appendix D).** Exponential kernel with
$\kappa<1$, so $c=(\kappa-1)\theta<0$ and every integral converges. *Intuition:*
each formula is the deterministic *wake* a given exogenous shape leaves behind —
the input $s$ plus its self-excited echo $h*s$. Read $\xi$ as "input $+$ decaying
memory of the input" and $\Xi$ as that wake's running total.

### D.1 Single impulse $s(t)=\delta(t-a)$

$(h*s)(t)=h(t-a)\,\mathbb 1[t>a]=\kappa\theta\,e^{c(t-a)}\mathbb 1[t>a]$, hence
$$
\xi(t)=\delta(t-a)+\kappa\theta\,e^{c(t-a)}\mathbb 1[t>a],\qquad
\Xi^{\mathrm{endo}}(t)=\int_a^t\!\kappa\theta e^{c(\tau-a)}d\tau=K\big(1-e^{c(t-a)}\big)\mathbb 1[t>a].
$$
As $t\to\infty$, $\Xi^{\mathrm{endo}}\to K=\kappa/(1-\kappa)$ — the mean offspring
count per immigrant (Appendix B.4). $\checkmark$

### D.2 Multi-impulse $s(t)=\sum_{z}\delta(t-\mathsf s_z)$

By linearity (LTI), sum D.1 over the impulses:
$$
\xi(t)=\sum_z\Big[\delta(t-\mathsf s_z)+\kappa\theta e^{c(t-\mathsf s_z)}\mathbb 1[t>\mathsf s_z]\Big],\quad
\Xi^{\mathrm{endo}}(t)=\sum_z K\big(1-e^{c(t-\mathsf s_z)}\big)\mathbb 1[t>\mathsf s_z].
$$

### D.3 Rectangle $s(t)=\mathbb 1[a<t\le b]$

With $u=t-\tau$, for $a<t\le b$:
$(h*s)(t)=\int_a^t\kappa\theta e^{c(t-\tau)}d\tau=\kappa\theta\frac{e^{c(t-a)}-1}{c}=K\big(1-e^{c(t-a)}\big)$;
for $t>b$:
$(h*s)(t)=\int_a^b\kappa\theta e^{c(t-\tau)}d\tau=\kappa\theta\frac{e^{c(t-a)}-e^{c(t-b)}}{c}=K\big(e^{c(t-b)}-e^{c(t-a)}\big)$.
Therefore
$$
\xi(t)=\mathbb 1[a<t\le b]+K\big(1-e^{c(t-a)}\big)\mathbb 1[a<t\le b]+K\big(e^{c(t-b)}-e^{c(t-a)}\big)\mathbb 1[t>b].
$$
Integrating each piece gives the compensator used in code
(`PiecewiseConstant._rect_compensator`); e.g. on $a<t\le b$,
$\Xi^{\mathrm{endo}}(t)=K\big[(t-a)-\tfrac{1}{c}(e^{c(t-a)}-1)\big]$.

### D.4 Piecewise-constant / LHPP $s(t)=\sum_i\lambda_i\mathbb 1[q_{i-1}<t\le q_i]$

By linearity, superpose D.3 with weights $\lambda_i$ (for LHPP,
$\lambda_i=\mathsf S(q_{i-1},q_i]/(q_i-q_{i-1})$, Prop. 13):
$\xi=\sum_i\lambda_i\,\xi_{\mathrm{rect}(q_{i-1},q_i)}$, likewise for $\Xi$. This is
the workhorse used for the interval-censored experiments and the ACTIVE data.

### D.5 Dassios–Zhao $s(t)=\kappa\theta+(u_0-\kappa\theta)e^{-\theta t}$ (Eq. 16)

Split $h*s=h*(\kappa\theta)+h*\big((u_0-\kappa\theta)e^{-\theta\cdot}\big)$.

*Constant part.* $h*(\kappa\theta)(t)=\kappa\theta\!\int_0^t\!\kappa\theta e^{c\tau}d\tau
=\dfrac{\kappa^2\theta^2}{c}(e^{ct}-1)=\dfrac{\kappa^2\theta}{1-\kappa}(1-e^{ct})$.

*Exponential part.* Using $c+\theta=\kappa\theta$,
$$
h*\big((u_0{-}\kappa\theta)e^{-\theta\cdot}\big)(t)
=(u_0{-}\kappa\theta)\kappa\theta e^{ct}\!\int_0^t\! e^{-(c+\theta)\tau}d\tau
=(u_0{-}\kappa\theta)e^{ct}\big(1-e^{-\kappa\theta t}\big)=(u_0{-}\kappa\theta)\big(e^{ct}-e^{-\theta t}\big).
$$
Adding $s$ and simplifying (the $e^{-\theta t}$ terms cancel,
$\kappa\theta+\kappa^2\theta/(1-\kappa)=\kappa\theta/(1-\kappa)$):
$$
\boxed{\;\xi(t)=\frac{\kappa\theta}{1-\kappa}\big(1-e^{-(1-\kappa)\theta t}\big)+u_0\,e^{-(1-\kappa)\theta t}\;}
$$
exactly Eq. (16). $\checkmark$

### D.6 Sinusoidal $s(t)=\sin t+\alpha$ (Table 10, Row VI)

Using $\displaystyle\int_0^t e^{c(t-\tau)}\sin\tau\,d\tau=\frac{e^{ct}-(c\sin t+\cos t)}{c^2+1}$
and $\int_0^t e^{c(t-\tau)}\alpha\,d\tau=K\alpha(1-e^{ct})$, with $D:=c^2+1=1+(1-\kappa)^2\theta^2$,
$$
\xi(t)=\sin t+\alpha+K\alpha\big(1-e^{ct}\big)+\kappa\theta\,\frac{e^{ct}-\big((\kappa-1)\theta\sin t+\cos t\big)}{D}.
$$
(Expanding and collecting the trigonometric, exponential and constant groups
reproduces the longer form printed in the paper's Appendix B.1.) The compensator is
obtained termwise; the package evaluates it by high-accuracy quadrature of this
closed-form $\xi$, validated against the analytic pieces to $\sim10^{-8}$.

---

## Appendix E. Interval-censored likelihood and Bregman geometry

### E.1 Derivation of IC-LL

By Appendix A.5 the MBPP gives independent Poisson interval counts with means
$\Xi_i:=\Xi(o_{i-1},o_i]$. The negative log-likelihood of the observed counts
$\mathsf C_i$ is
$$
\mathcal L(\theta)=-\log\prod_{i=1}^m\frac{\Xi_i^{\mathsf C_i}}{\mathsf C_i!}e^{-\Xi_i}
=\sum_{i=1}^m\Xi_i-\sum_{i=1}^m\mathsf C_i\log\Xi_i+\underbrace{\sum_i\log\mathsf C_i!}_{\text{const in }\theta},
$$
and dropping the constant gives IC-LL, Eq. (19). $\checkmark$

**Assumptions & intuition.** The MBPP is Poisson with independent increments
(Thm A.2), and the counts $\mathsf C_i$ are observed. *Intuition:* independent
Poisson bins ⇒ the joint probability is a product ⇒ the negative log-likelihood is
a sum of per-bin Poisson terms in the interval means $\Xi_i$. Nothing more than
"$\log$ of a product of Poissons."

### E.2 Bregman divergences

For strictly convex differentiable $\varphi$ on a convex set,
$B_\varphi(x,y)=\varphi(x)-\varphi(y)-\langle x-y,\nabla\varphi(y)\rangle\ge0$, with
equality iff $x=y$ (first-order convexity); $B_\varphi$ is convex in $x$. Two
generators:
$\varphi_{\mathrm{KL}}(x)=\sum_i(x_i\log x_i-x_i)$ gives the generalized KL
$\sum_i\big(x_i\log\tfrac{x_i}{y_i}-(x_i-y_i)\big)$;
$\varphi_{\mathrm{SE}}(x)=\tfrac12\|x\|^2$ gives $\tfrac12\|x-y\|^2$.

**Intuition.** $B_\varphi(x,y)$ is the vertical gap at $x$ between the convex
graph of $\varphi$ and its tangent drawn at $y$ — non-negative, and zero only when
$x=y$. It is a "distance" whose shape is dictated by the curvature of $\varphi$,
which is what lets one generator encode Poisson noise and another Gaussian.

### E.3 The Bregman ↔ exponential-family bijection

**Theorem E.1 (Banerjee et al. 2005).** *Let $p_\psi(x\mid\theta)=\exp(\langle
x,\theta\rangle-\psi(\theta))p_0(x)$ be a regular exponential family with cumulant
$\psi$, mean parameter $\mu=\nabla\psi(\theta)$ and Legendre conjugate
$\psi^\star$. Then*
$$
-\log p_\psi(x\mid\theta)=B_{\psi^\star}(x,\mu)+\text{const}(x).
$$
*Maximum likelihood is thus Bregman projection onto the model mean, and the choice
of Bregman generator fixes the assumed noise family.*

**The two instances we use.**
*Poisson:* $\psi(\theta)=e^\theta$, $\mu=e^\theta$, $\psi^\star(\mu)=\mu\log\mu-\mu$,
so $B_{\psi^\star}(x,\mu)=x\log(x/\mu)-(x-\mu)=\mathrm{KL}(x,\mu)$ — **IC-LL assumes
each interval is Poisson** (Prop. 8 below).
*Gaussian (unit variance):* $\psi(\theta)=\theta^2/2$,
$\psi^\star(\mu)=\mu^2/2$, $B=\tfrac12(x-\mu)^2$ — **SSE assumes each interval is
Gaussian with constant variance.**

**Proposition 8 (restated).** $\mathcal L_{\text{IC-LL}}(\theta)=\sum_i
B_{\varphi_{\mathrm{KL}}}(\mathsf C_i,\Xi_i)+\Gamma(\mathsf C)=\mathrm{KL}(\mathsf
C,\Xi(\theta))+\Gamma(\mathsf C)$, with $\Gamma$ free of $\theta$ (proof in body
§4); hence minimising IC-LL $=$ minimising the KL projection of the counts onto
the compensator manifold.

**Assumptions & intuition.** Theorem E.1 needs a *regular* (minimal, steep)
exponential family with cumulant $\psi$ of Legendre type. *Intuition:* maximum
likelihood in any exponential family is "move the model mean as close as possible
to the data, measured in the geometry set by the family's variance function."
Choosing the Bregman generator therefore *is* choosing the noise model:
$x\log x$ ⟶ Poisson ⟶ KL ⟶ IC-LL; $\tfrac12x^2$ ⟶ Gaussian ⟶ SSE.

### E.4 Why KL (IC-LL) is more robust than SSE

If an interval's count is the sum/average of many independent contributions, the
CLT gives $\hat{\mathsf C}_i\approx\mathcal N(\Xi_i,\Xi_i/N)$ (a Poisson rate has
**variance equal to its mean**). KL is the correctly *heteroscedastic* loss
(weighting each interval by $1/\Xi_i$ through $\partial_\Xi\mathrm{KL}=1-\mathsf
C/\Xi$), whereas SSE imposes a single constant variance. The two agree as
$\Xi_i\to\infty$ (Poisson $\to$ Gaussian), but when counts span orders of magnitude
— bursty regimes, near criticality — SSE over-weights the large-count intervals and
destabilises; this is exactly the near-critical failure of SSE seen in `exp6`.

**Assumptions & intuition.** The CLT step assumes many independent, finite-variance
contributions per interval. *Intuition:* a Poisson bin has variance equal to its
mean, so the statistically correct loss must down-weight high-count bins by
$1/\Xi_i$ — which KL does and SSE does not. When counts span orders of magnitude
(bursts, near criticality), SSE is dominated by the loudest bins and destabilises;
they agree only when all bins are large (Poisson $\to$ Gaussian).

### E.5 Recovering HIP

**Lemma 9.** *If $\xi$ is constant on the unit interval $(i-1,i]$, then
$\Xi(i-1,i]=\xi(i)$.* (Proof: $\int_{i-1}^i\xi=\xi(i)\cdot1$.)

**Theorem 10.** *With unit intervals and per-interval-constant intensity, the SSE
loss (36) equals the HIP objective $\sum_i(\mathsf C(i-1,i]-\xi(i))^2$.* (Proof:
substitute Lemma 9 into (36).)

**Theorem 11.** *Additionally taking a power-law kernel and the discrete-convolution
approximation of $\xi$ recovers the full HIP model of Rizoiu et al. (2017b).*

Thus HIP is a triply-restricted special case of the MBPP+SSE framework — which is
how the present work repairs HIP's conceptual gaps (it now *is* the likelihood of a
defined process, IC-LL/KL replaces the heuristic SSE, and the exponential kernel
admits the exact compensator instead of a discretisation).

---

**Assumptions & intuition.** Lemma 9: unit intervals and a per-interval-constant
intensity. Theorem 11 adds a power-law kernel and the discrete-convolution
approximation of $\xi$. *Intuition:* HIP is the MBPP with three corners cut
(compensator $\to$ intensity, exact integral $\to$ discrete sum, arbitrary bins
$\to$ unit bins); seeing exactly which corners are cut is what tells you when HIP
is safe and what each simplification costs.

---

## Appendix F. Operator theory: ODE reduction, multivariate stationarity, fractional kernels, neural operators

### F.1 The exponential kernel gives a first-order ODE

$\mathcal L\{\phi\}(z)=\kappa\theta/(z+\theta)$, so
$1-\mathcal L\{\phi\}=\dfrac{z+(1-\kappa)\theta}{z+\theta}$ and (Thm C.3)
$\mathcal L\{\xi\}=\mathcal L\{s\}\,\dfrac{z+\theta}{z+(1-\kappa)\theta}$. Cross-multiplying
and inverting (with $z\leftrightarrow d/dt$),
$$
\xi'(t)+(1-\kappa)\theta\,\xi(t)=s'(t)+\theta s(t)
\;\Longleftrightarrow\;
y'(t)=(\kappa-1)\theta\,y(t)+\kappa\theta\,s(t),\ \ \xi=s+y,
$$
the stable ODE used in §9 (forced by the baseline $s$; pole at $(\kappa-1)\theta<0$).

**Assumptions & intuition.** Exponential kernel, $\kappa<1$. *Intuition:* a single
memory timescale is a single pole, i.e. a first-order filter — hence a first-order
ODE. The baseline $s$ is the forcing and $(1-\kappa)\theta$ is the relaxation rate:
push the system and it relaxes back at that rate.

### F.2 Rational Laplace transform ⇔ finite linear ODE

**Theorem F.1.** *The convolution Volterra equation $\xi=s+\phi*\xi$ is equivalent
to a finite-order linear constant-coefficient ODE (forced by $s$ and its
derivatives) if and only if $\phi$ has a strictly proper rational Laplace
transform — equivalently, iff $\phi$ is a finite exponential-polynomial
$\sum_q p_q(t)e^{-b_q t}$. Sums of exponentials (distinct $b_q$, $p_q$ constant)
are the generic case, realised by the diagonal state space of §9/§10.*

**Proof.** ($\Leftarrow$) Let $\mathcal L\{\phi\}=P/Q$ with $\deg P<\deg Q=:d$.
Then $\mathcal L\{\xi\}=\dfrac{\mathcal L\{s\}}{1-P/Q}=\dfrac{Q}{\,Q-P\,}\mathcal
L\{s\}$, so $(Q-P)(z)\,\mathcal L\{\xi\}=Q(z)\,\mathcal L\{s\}$. Since $Q-P$ has
degree $d$, reading $z^k\leftrightarrow d^k/dt^k$ yields the order-$d$ ODE
$(Q-P)(\tfrac{d}{dt})\xi=Q(\tfrac{d}{dt})s$. ($\Rightarrow$) If such an ODE holds for
all inputs, the transfer function $\mathcal L\{\xi\}/\mathcal L\{s\}=Q/(Q-P)$ is
rational, hence $\mathcal L\{\phi\}=1-(Q-P)/Q=P/Q$ is rational; and a function has
rational Laplace transform iff it is an exponential-polynomial (partial fractions
+ inverse Laplace). $\;\blacksquare$

The exponential kernel ($d=1$) is the simplest instance; a sum of $Q$ exponentials
gives the order-$Q$ realisation
$u'=Au+a\,s,\ \xi=s+\mathbf 1^\top u$ with $A=a\mathbf 1^\top-\operatorname{diag}(b)$.
A power-law kernel has a **non-rational** transform, so **no finite ODE exists**
(Appendix F.5).

**Assumptions & intuition.** $\phi$ has a strictly proper rational Laplace
transform (equivalently $\phi=\sum_q p_q(t)e^{-b_qt}$); the ODE equivalence is for
inputs smooth enough to differentiate $\deg Q$ times. *Intuition:* a memory built
from finitely many decaying modes can be tracked by finitely many state variables —
"rational transfer function $\Leftrightarrow$ finite-dimensional machine"
(realisation theory). A power-law memory is *not* rational: it has infinitely many
timescales, so no finite ODE exists (F.5).

### F.3 Transfer function and the spectral (Fourier) operator

On the Fourier axis $z=i\omega$, $\hat\xi(\omega)=R(\omega)\hat s(\omega)$ with
$R=1/(1-\hat\phi)$ (Thm C.3). Because $R$ multiplies pointwise in frequency, the
solution operator is *diagonalised by the Fourier transform* — which is exactly
what a Fourier-neural-operator layer implements, $\xi=\mathrm{IFFT}(R\cdot\mathrm{FFT}(s))$.
For the exponential kernel $R(\omega)=\dfrac{\theta+i\omega}{(1-\kappa)\theta+i\omega}$,
a one-pole all-pass-like filter; its single pole $-(1-\kappa)\theta$ is the ODE rate
of F.1, unifying the time-, frequency-, and state-space pictures.

**Assumptions & intuition.** Fourier transforms of $s,\phi$ exist and
$1-\hat\phi(\omega)\neq0$. *Intuition:* shift-invariant systems are *diagonal in
frequency* — each frequency is amplified independently by $R(\omega)=1/(1-\text{loop
gain})$. A Fourier-neural-operator layer is exactly this "FFT, multiply per
frequency, inverse FFT," which is why one linear FNO layer reproduces the operator
with no error.

### F.4 Multivariate resolvent, stationarity and spectral radius

For $\xi\in\mathbb R^M$ with kernel matrix $\Phi$, $\hat\xi(\omega)=(I-\hat\Phi(\omega))^{-1}\hat s(\omega)$.
At $\omega=0$, $\hat\Phi(0)=G$ the **branching matrix** $G_{mj}=\int\phi_{mj}=A_{mj}/B_{mj}$.

**Proposition F.2.** *Under a constant baseline $\bar s=\bar\mu$, the stationary
mean intensity is $\bar\xi=(I-G)^{-1}\bar\mu=\sum_{n\ge0}G^n\bar\mu$, convergent iff
the spectral radius $\rho(G)<1$.* (Proof: stationarity in Prop. B.2 gives
$\bar\xi=\bar\mu+G\bar\xi$; the Neumann series converges iff $\rho(G)<1$.) The
time-domain solver realises this as the $M^2$-state linear ODE
$y'_{mj}=A_{mj}\xi_j-B_{mj}y_{mj}$, $\xi_m=s_m+\sum_jy_{mj}$
(`solve_mbpp_ode_multivariate`); the numerical check $\bar\xi=(I-G)^{-1}\bar\mu$ to
$10^{-15}$ is the high-dimensional analogue of Appendix D.1's mass check.

**Assumptions & intuition.** (A7) with $\rho(G)<1$ and a constant baseline.
*Intuition:* the stationary intensity is the baseline amplified by the network's
*total* feedback, $(I-G)^{-1}=I+G+G^2+\dots$ — direct excitation, plus two-hop, plus
three-hop, … The spectral-radius condition $\rho(G)<1$ is exactly what makes these
multi-hop echoes sum to a finite total.

### F.5 Power-law kernels: Mittag–Leffler resolvent / fractional dynamics

For the canonical fractional kernel $\phi(t)=\rho\,t^{\alpha-1}/\Gamma(\alpha)$
($0<\alpha\le1$), the resolvent of $\xi=s+\phi*\xi$ is the **Mittag–Leffler**
function
$$
h(t)=\rho\,t^{\alpha-1}E_{\alpha,\alpha}(\rho\,t^{\alpha}),\qquad
E_{\alpha,\beta}(z)=\sum_{k\ge0}\frac{z^k}{\Gamma(\alpha k+\beta)},
$$
and the equation is equivalent to a **fractional-order** integro-differential
equation (apply the Riemann–Liouville derivative $D^\alpha$), not a finite ODE.
As $\alpha\to1$, $E_{1,1}(z)=e^z$ and one recovers the exponential resolvent of
F.1 — the fractional case is the genuine infinite-dimensional generalisation. In
practice one approximates the power-law by a sum of exponentials (Prony / the
`PowerLawKernel`→`kernel_exponentials` fit), restoring a finite state-space ODE to
any desired accuracy.

**Assumptions & intuition.** $0<\alpha\le1$, Riemann–Liouville power-law kernel with
$\rho>0$ (the Mittag–Leffler series converges for all $t$). *Intuition:* a
power-law memory has no single timescale — it is a continuum of exponential modes —
so the dynamics are genuinely *fractional* rather than a finite ODE. The
Mittag–Leffler function is the "stretched exponential" that interpolates between an
ordinary exponential ($\alpha=1$) and a power law, and a sum-of-exponentials fit
just approximates that continuum by finitely many modes.

### F.6 Neural-operator approximation theory

**Theorem F.3 (operator universal approximation; Chen & Chen 1995).** *A
single-hidden-layer operator network of the branch/trunk (DeepONet) form
approximates any continuous nonlinear operator between compact sets of continuous
functions to arbitrary uniform accuracy.* This is the theoretical license for
`MBPPDeepONet`/`DeepONetOperator` in the regimes with no closed form (inhibition,
marks, learned dynamics).

**Exactness of the linear spectral operator.** Conversely, the MBPP operator is
*linear and translation-invariant*, hence a convolution; a single linear FNO layer
with spectral multiplier $R(\omega)=1/(1-\hat\phi(\omega))$ represents it
**exactly** (no approximation). Learning $R(\omega)$ from $(s,\xi)$ pairs is
therefore *nonparametric operator/kernel identification*, with the kernel recovered
as $\hat\phi=1-1/R$ — the Wiener–Hopf / Bacry–Muzy programme in neural-operator
form. The practical lesson (borne out by `exp9`): exploit the linear structure when
it is present (spectral/ODE are exact and cheap); reserve the heavier nonlinear
learners for when it is not.

---

**Assumptions & intuition.** Theorem F.3 needs a continuous operator between
compact sets of continuous functions and a non-polynomial (Tauber–Wiener)
activation. *Intuition:* operator networks are universal — they can approximate
*any* nonlinear input-function-to-output-function map. But when the operator is
linear and shift-invariant, the exact answer is a single spectral multiply: don't
spend a deep network learning what one line of algebra already gives you.

---

## Appendix G. Estimation and identifiability

### G.1 Score and Fisher information

For the PP-LL (Thm A.1), the score is
$$
\partial_\theta\ell(\theta)=\sum_{t_i\le T}\frac{\partial_\theta\lambda(t_i)}{\lambda(t_i)}-\int_0^T\partial_\theta\lambda(s)\,ds,
$$
a martingale in $T$ (its predictable mean is zero). The **expected Fisher
information** is
$$
\mathcal I(\theta)=\mathbb E\!\Big[\int_0^T\frac{\partial_\theta\lambda(s)\,\partial_\theta\lambda(s)^\top}{\lambda(s)}\,ds\Big],
$$
and the **observed information** is the Hessian $-\partial_\theta^2\ell$ of the
negative log-likelihood, evaluated at $\hat\theta$ — what `estimate.py` inverts for
standard errors. For interval-censored data the analogue is the Poisson
information of the compensators,
$\mathcal I_{jk}^{\mathrm{IC}}=\sum_i\dfrac{\partial_{\theta_j}\Xi_i\,\partial_{\theta_k}\Xi_i}{\Xi_i}$.

**Assumptions & intuition.** (A6): $\lambda(\cdot;\theta)>0$, predictable, twice
continuously differentiable in $\theta$. *Intuition:* the score balances "intensity
where events actually occurred" against "predicted intensity everywhere," and is a
martingale because, on average, the model can't anticipate its own residuals. Its
variance — the Fisher information — is the curvature of the log-likelihood: how
sharply the data pin down $\theta$.

### G.2 Consistency and asymptotic normality

**Theorem G.1 (Ogata 1978; Bowsher 2007, multivariate).** *For a stationary,
ergodic, subcritical Hawkes process with true parameter $\theta_0$ in the interior
of a compact identifiable set, and the usual smoothness/moment regularity, the MLE
is consistent and asymptotically normal:*
$$
\hat\theta_T\xrightarrow{a.s.}\theta_0,\qquad
\sqrt{T}\,(\hat\theta_T-\theta_0)\xrightarrow{d}\mathcal N\!\big(0,\ \mathcal i(\theta_0)^{-1}\big),
$$
*where $\mathcal i=\lim_T\mathcal I/T$ is the per-unit-time information.* Standard
errors are read from $\mathcal I(\hat\theta)^{-1}$ (Cramér–Rao: any unbiased
estimator has $\mathrm{Var}\ge\mathcal I^{-1}$).

**Assumptions & intuition.** Precisely: a stationary, ergodic, *subcritical*
($\kappa<1$) process; true $\theta_0$ in the interior of a compact, *identifiable*
$\Theta$; log-intensity twice continuously differentiable with integrable
dominating envelopes (for a uniform LLN/CLT); and a finite, *nonsingular*
per-time information $\mathfrak i(\theta_0)$. *Intuition:* with enough data the MLE
concentrates Gaussianly on the truth, with covariance equal to the inverse
information — more curvature means tighter estimates, and Cramér–Rao says no
unbiased estimator does better. (Nonsingularity is exactly what fails, in the
limit, for the $\kappa$–$\theta$ pair under coarse censoring, G.4.)

### G.3 Time-rescaling theorem (goodness of fit)

**Theorem G.2 (random time change; Meyer/Papangelou).** *If $t_1<t_2<\dots$ are the
events of a simple process with compensator $\Lambda$, then the rescaled times
$\tau_k:=\Lambda(t_k)$ form a unit-rate Poisson process; equivalently the rescaled
inter-event gaps $\Lambda(t_k)-\Lambda(t_{k-1})$ are i.i.d. $\mathrm{Exp}(1)$.*
**Proof idea.** $M=N-\Lambda$ is a martingale; the time change $t\mapsto\Lambda(t)$
turns $N$ into a process whose compensator is the identity, which by Watanabe
(Thm A.2) is the unit-rate Poisson process. $\;\blacksquare$
This yields the diagnostic: fit, compute $\{\Lambda(t_k)-\Lambda(t_{k-1})\}$, and
KS-test / QQ-plot against $\mathrm{Exp}(1)$ to detect mis-specification.

**Assumptions & intuition.** $\Lambda$ continuous, strictly increasing, with
$\Lambda(\infty)=\infty$ (non-explosive, infinitely many events). *Intuition:*
measure time not in seconds but in *accumulated compensator* — "operational time"
that ticks fast when the intensity is high. In that clock the events are a plain
unit-rate Poisson process, which gives a completely model-free residual check:
rescaled inter-event gaps should look $\mathrm{Exp}(1)$.

### G.4 The $\kappa$–$\theta$ weak identifiability (the central caveat)

**Assumptions & intuition.** $\Xi_i(\theta)$ twice differentiable in
$(\kappa,\theta)$; the relevant object is the interval Poisson information
$\mathcal I^{\mathrm{IC}}$ at the truth. *Intuition:* the counts pin down "how much"
(the gain $\kappa$) but not "how fast" (the timescale $\theta$), because a bin
integrates away the within-bin timing that carries $\theta$. The two sensitivities
$\partial_\kappa\Xi$ and $\partial_\theta\Xi$ become nearly collinear, so the
information matrix is ill-conditioned along the $\alpha/\beta$-preserving ridge.

Write $\phi(t)=\alpha e^{-\beta t}$ with $\alpha=\kappa\theta$, $\beta=\theta$. The
data inform two nearly-orthogonal aspects: the **branching ratio**
$n^*=\int\phi=\alpha/\beta=\kappa$ (the offspring fraction, which sets the
stationary multiplier $(1-\kappa)^{-1}$ and is visible in *counts* and
cross-correlations), and the **timescale** $1/\beta=1/\theta$ (visible only in
*timing*). 

**Proposition G.3 (information geometry of the ridge).** *The $(\alpha,\beta)$
Fisher information matrix is typically ill-conditioned, with its small-eigenvalue
direction aligned with curves of constant $\alpha/\beta$. Consequently $\kappa$ is
well-identified while $\theta$ is weakly identified, and the more so the coarser
the observation.* **Why interval censoring sharpens this:** the interval
information depends on $\partial_\theta\Xi_i$; since $\Xi_i=\int_{o_{i-1}}^{o_i}\xi$
integrates over the bin, the *timing-borne* sensitivity $\partial_\theta\Xi_i$ is
small and sign-oscillating, whereas $\partial_\kappa\Xi_i$ is monotone and
order-one. As the bin width $\to\infty$ (counts only), $\partial_\theta\Xi_i\to0$:
$\theta$ becomes non-identified while $\kappa$ persists.

This is not a defect of the method but a property of the data, and it is exactly
what every experiment shows: `exp6` recovers $\kappa$ to $\pm0.03$ with $\theta$
far noisier; the granularity sweep tightens $\theta$ as bins shrink; `exp8`
recovers the covariate effect $\gamma_1$ tightly while $\theta,\gamma_0,\kappa$
scatter; and the amortised net (`exp9`) predicts $\kappa$ at correlation $0.93$ but
$\theta$ at $\approx0.1$. **Practical guidance:** treat $\kappa$ (branching ratio)
and the covariate effects $\gamma$ as the trustworthy estimands from
interval-censored data; fix or strongly prior-constrain $\theta$ from domain
knowledge or finer-grained data, and report $\theta$ with wide intervals.

---

### References (selected)

Hawkes (1971), *Spectra of some self-exciting and mutually exciting point
processes*. Hawkes & Oakes (1974), cluster representation. Dassios & Zhao (2013),
exponential-kernel closed form. Rizoiu et al. (2017b), HIP. Banerjee et al.
(2005), Bregman divergences and exponential families. Rizoiu, Soen, Li, Calderon,
Dong, Menon, Xie (2022), *Interval-censored Hawkes processes*, JMLR 23(1):1–84
— the paper summarized here.
