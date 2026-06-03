# Calibration of Multivariate Hawkes Processes with Exponential Kernel and Covariates

## 1. Model

We consider an $M$-dimensional point process $N(t) = (N_1(t), \dots, N_M(t))$ whose component $m$ has stochastic intensity (conditional on the history $\mathcal{F}_{t^-}$):

$$
\lambda_m(t \mid \mathcal{F}_{t^-}) \;=\; \mu_m(t)
\;+\; \sum_{j=1}^{M} \alpha_{m,j}\!\!\sum_{t_{j,k}<t} e^{-\beta_{m,j}\,(t - t_{j,k})}.
$$

The interpretation of the symbols is the following: $\mu_m(t)\ge 0$ is the baseline (exogenous) intensity for component $m$; $\alpha_{m,j}\ge 0$ measures the instantaneous *jump* of $\lambda_m$ produced by an event in component $j$ — this is what we will call the **elicitation coefficient** (also known in the literature as the cross-excitation coefficient or, after normalization by $\beta_{m,j}$, the branching coefficient); $\beta_{m,j}>0$ is the exponential decay rate of the influence of past events of $j$ over $m$; $\{t_{j,k}\}_k$ is the ordered set of jump times of component $j$ in $[0,T]$.

The matrix $A=(\alpha_{m,j})$ encodes the *elicitation network*: a non-zero entry $\alpha_{m,j}$ means events of $j$ elicit events of $m$. The diagonal entries represent self-excitation. Stationarity (and the existence of a stationary version of the process) holds whenever the spectral radius of the branching-ratio matrix $G = (\alpha_{m,j}/\beta_{m,j})$ is strictly less than 1.

### 1.1 Adding covariates

We allow time-varying covariates $X_m(t)\in\mathbb{R}^p$ to enter the baseline through a log-linear link:

$$
\mu_m(t) \;=\; \exp\!\big(\gamma_{m,0} + \gamma_m^{\top} X_m(t)\big),
$$

so that intensities remain non-negative without parameter constraints on $\gamma_m$. In the experiments below we use piecewise-constant covariates (typical for low-frequency information like macroeconomic regime, time-of-day bucket, treatment indicator).

If one also wants the elicitation strength itself to depend on covariates $Z(t)$, a natural specification is

$$
\alpha_{m,j}(t) \;=\; \alpha^{(0)}_{m,j}\,\exp\!\big(\delta_{m,j}^{\top} Z(t)\big),
$$

which keeps positivity. We implement the simpler version with covariates only in the baseline, but the code is structured so the second form can be added.

## 2. Likelihood

For a multivariate point process with intensities $\lambda_m$, the log-likelihood on $[0,T]$ is

$$
\ell(\theta) \;=\; \sum_{m=1}^{M}\!\left[\;\sum_{k:\,t_{m,k}\le T}\!\!\log \lambda_m(t_{m,k}) \;-\; \int_{0}^{T}\!\lambda_m(s)\,ds\;\right].
$$

The compensator integral splits into a baseline piece and an excitation piece:

$$
\int_0^T\!\lambda_m(s)\,ds = \int_0^T\!\mu_m(s)\,ds \;+\; \sum_{j=1}^{M}\frac{\alpha_{m,j}}{\beta_{m,j}}\sum_{t_{j,k}\le T}\!\!\Big(1 - e^{-\beta_{m,j}(T-t_{j,k})}\Big).
$$

For piecewise-constant baseline $\mu_m(t)=\bar\mu_m^{(\ell)}$ on interval $[s_{\ell-1},s_\ell]$, the first integral is just $\sum_\ell \bar\mu_m^{(\ell)} (s_\ell - s_{\ell-1})$.

### 2.1 Recursive computation

For the exponential kernel the sums inside $\log\lambda_m(t_{m,k})$ admit a recursion that brings the cost from $O(N^2)$ to $O(N)$. Define

$$
R^{(m,j)}_k \;=\; \sum_{t_{j,l}<\,t_{m,k}} e^{-\beta_{m,j}(t_{m,k}-t_{j,l})}.
$$

If $t_{m,k}$ and $t_{m,k-1}$ are consecutive events of component $m$, and $\mathcal{E}_{k}^{(j)} := \{l: t_{m,k-1}\le t_{j,l} < t_{m,k}\}$ is the set of events of component $j$ that occurred between them, then

$$
R^{(m,j)}_k \;=\; e^{-\beta_{m,j}(t_{m,k}-t_{m,k-1})}\;R^{(m,j)}_{k-1} \;+\; \sum_{l\in \mathcal{E}_k^{(j)}} e^{-\beta_{m,j}(t_{m,k}-t_{j,l})}.
$$

With this recursion the log-likelihood becomes a $O(\sum_m N_m \cdot M)$ computation per evaluation.

## 3. Estimation

We treat $\beta_{m,j}$ as fixed (as is standard in financial and biological applications, because $\alpha$ and $\beta$ are weakly jointly identified) and estimate $\theta=(\gamma, \alpha)$ by maximum likelihood. We parametrize $\alpha_{m,j}=\exp(a_{m,j})$ internally to guarantee positivity, then transform back. The optimizer is L-BFGS-B with analytic-free numerical gradients; for moderate $M$ and $N$ this is sufficient and very robust. We also report standard errors from the inverse Hessian (observed Fisher information) at the optimum.

The identifiability story:

1. Mean intensity matrix $\bar\Lambda$ satisfies, in the stationary case, $\bar\Lambda = (I - G)^{-1}\bar\mu$ with $G=(\alpha/\beta)$. This makes $\bar\mu$ and $G$ jointly identifiable from event counts and cross-correlations *provided* the spectral radius is strictly below 1.

2. For finite $T$ and a single sample path, identification is driven by the timing of events — the second-order properties of $N$ — not just counts. The MLE consistency proof (Ogata 1978, and Bowsher 2007 for the multivariate case) requires that the realized process visits a sufficiently rich set of intensity values.

## 4. Synthetic experiments

The package contains three experiments under `experiments/`:

`exp1_1d.py` calibrates a univariate Hawkes process and inspects bias and standard error of $(\mu,\alpha,\beta)$ as a function of horizon $T$.

`exp2_multidim.py` simulates a 3-dimensional Hawkes process where the elicitation matrix has a known sparsity pattern, and recovers $A$ by MLE — the central experiment for this work.

`exp3_covariates.py` adds a piecewise-constant covariate that switches the baseline intensity between two regimes, and we recover both the regime coefficients $\gamma$ and the elicitation matrix jointly.

For each experiment we report, alongside point estimates, (a) the absolute bias and (b) coverage of asymptotic 95% confidence intervals constructed from the observed information matrix.

## 5. Simulation algorithm

We simulate via Ogata's thinning. At each step we keep a running per-component intensity using the same recursion as in §2.1 (so that adding the next event updates intensities in $O(M)$), upper-bound the next intensity by the value just after the last event (intensities decay between events), and accept/reject candidate times. This procedure is exact.

## 6. Scaling to higher dimensions

Three things have to work for $M$ to grow beyond, say, ten:

1. **Likelihood evaluation cost.** A naive event-by-event implementation of $\sum_k \log \lambda_m(t_{m,k})$ is $O(N^2 M)$ — for each event we recompute kernel sums against every past event. The recursion in §2.1 reduces this to $O(N M)$ per likelihood evaluation. We further vectorize it by computing, per pair $(m,j)$, the "self kernel sum" $K^{(j)}_l = \sum_{l'\le l}e^{-\beta_{m,j}(t_{j,l}-t_{j,l'})}$ over the events of $j$ (one cheap recursion per pair), and then reading off $R^{(m,j)}(t_{m,k})$ at every event of $m$ via $\texttt{searchsorted}$ plus a single decay term: $R^{(m,j)}(t_{m,k}) = K^{(j)}_{L_k}\,e^{-\beta_{m,j}(t_{m,k}-t_{j,L_k})}$, where $L_k$ is the index of the most recent $j$-event before $t_{m,k}$. The cumulative-sum step inside $K$ is numerically stabilized by chunking time into blocks of length $\le 400/\beta$ so that $e^{\beta\Delta t}$ stays in range. In practice this is roughly 15–20× faster than the per-event Python loop, with identical output.

2. **Optimizer cost per step.** A finite-difference gradient costs $2n$ likelihood evaluations, where $n = M + Mp + M^2$. At $M = 15$ with $p = 0$ that is already $n = 240$, and the Hessian for asymptotic standard errors costs $2n^2 \approx 10^5$ evaluations — prohibitive. The package supplies an analytical gradient (§7 below) accumulated in the same forward pass as the log-likelihood, so each BFGS iteration is one likelihood-plus-gradient pass instead of $\sim 2n$ likelihood passes. Standard errors then come from differentiating that gradient finite-difference style — $2n$ gradient evals — which is feasible up to a few hundred parameters.

3. **Statistical conditioning.** The number of parameters $M + Mp + M^2$ grows quadratically in $M$, while events per component typically grow only linearly in $T$. The likelihood becomes flat in the direction of small $\alpha_{m,j}$ for many $(m,j)$ pairs, and unregularized MLE produces noisy non-zero estimates for entries that are truly zero. We address this with an $\ell_1$ penalty on $A$ (§8), which is a standard remedy when the elicitation network is sparse — as it generally is in applied work.

## 7. Analytical gradient

The log-likelihood gradient has a closed form. Let $\lambda_{m,k}\equiv \lambda_m(t_{m,k}^-)$ and $\mu_{m,k}\equiv \mu_m(t_{m,k})$. Then

$$
\frac{\partial \ell}{\partial \gamma_{m,0}}
= \sum_k \frac{\mu_{m,k}}{\lambda_{m,k}}
\;-\; \int_0^T \mu_m(s)\,ds,
\qquad
\frac{\partial \ell}{\partial \gamma_{m,p}}
= \sum_k \frac{\mu_{m,k}\,X_p(t_{m,k})}{\lambda_{m,k}}
\;-\; \int_0^T \mu_m(s)\,X_p(s)\,ds,
$$

$$
\frac{\partial \ell}{\partial \alpha_{m,j}}
= \sum_k \frac{R^{(m,j)}(t_{m,k})}{\lambda_{m,k}}
\;-\; \frac{1}{\beta_{m,j}}\sum_{t_{j,l}\le T}\!\!\Big(1 - e^{-\beta_{m,j}(T-t_{j,l})}\Big).
$$

For the piecewise-constant covariate, $\int_0^T \mu_m(s) X_p(s)\,ds = \sum_\ell X_p^{(\ell)} \mu_m^{(\ell)}(s_\ell - s_{\ell-1})$ in closed form. All of these can be accumulated in the same forward pass that produces $\ell$ itself, at no asymptotic extra cost. The implementation contains a finite-difference verification test (max abs error $\sim 10^{-7}$) — see `tests/` notes in the README.

## 8. Sparse elicitation via $\ell_1$ regularization

For larger $M$ we estimate the elicitation matrix under an $\ell_1$ penalty:

$$
\hat\theta_\lambda \;=\; \arg\min_{\gamma,\alpha\ge 0}\; -\ell(\theta) \;+\; \lambda \sum_{m,j} \alpha_{m,j}.
$$

The non-negativity constraint on $\alpha$ is enforced through the proximal operator, so we parametrize $\alpha$ directly (not as $e^{a}$). The smooth part has the analytical gradient above; we minimize with FISTA (accelerated proximal gradient) where the proximal step is $\operatorname{prox}_{\tau,\lambda}(z) = \max(z - \tau\lambda,\,0)$ — a one-sided soft-threshold that produces exact zeros. We pick $\lambda$ from a grid by BIC,

$$
\mathrm{BIC}(\lambda) \;=\; -2\,\hat\ell(\lambda) \;+\; k(\lambda)\,\log\!\bigg(\sum_m N_m\bigg),
$$

where $k(\lambda)$ counts the unpenalized parameters plus the number of non-zero entries of $\hat A$.

In `experiments/exp4_highdim.py`, with $M = 12$ and a true elicitation density of about a third, the unpenalized MLE selects edges with precision $0.61$ and recall $0.79$ at a $2\sigma$ threshold, while the BIC-tuned lasso selects with precision $0.74$ and recall $0.94$, and reduces the mean absolute error on true-zero entries from $0.016$ to $0.002$.

Group lasso would be a natural variant when one expects entire rows or columns of $A$ to be zero (e.g.\ pure recipients vs. pure broadcasters of excitation).

## 9. Generalizations of the model

Beyond what is implemented above, the same architecture extends naturally in several directions.

**Sum-of-exponentials kernel.** Replace $\alpha_{m,j}e^{-\beta_{m,j}t}$ by $\sum_{q=1}^{Q}\alpha^{(q)}_{m,j} e^{-\beta^{(q)}_{m,j} t}$ to capture multiple time-scales (e.g.\ fast and slow excitation). All the recursions in §2.1 still apply per $q$, and the parameter count multiplies by $Q$. Identifiability of $\beta^{(q)}$ requires distinct decay rates.

**Power-law / Mittag–Leffler kernel.** For heavy-tailed excitation (typical in seismology and social media), $\phi(t)\propto (t+c)^{-(1+\eta)}$. The compensator integral is still closed-form, but the $O(N)$ recursion is lost — one falls back to $O(N^2)$ or to a sum-of-exponentials approximation that retains the recursion.

**Marked / spatial Hawkes.** Each event $t_{m,k}$ carries a mark $\xi_{m,k}\in\mathbb{R}^q$ (a magnitude, location, transaction size). The intensity becomes
$\lambda_m(t,\xi) = \mu_m(t,\xi) + \sum_j\sum_{t_{j,k}<t} g_{m,j}(\xi_{m,k};\xi_{j,k})\,e^{-\beta_{m,j}(t-t_{j,k})}$,
where $g$ scales excitation by the mark of the parent event (e.g.\ ETAS uses $g(\xi)\propto 10^{a\,\xi}$ for earthquake magnitude). The likelihood adds a mark density term and the optimizer otherwise looks identical.

**Covariate-dependent elicitation.** Make $A$ a function of $Z(t)$,
$\alpha_{m,j}(t) = \alpha^{(0)}_{m,j}\exp(\delta_{m,j}^{\top}Z(t))$,
useful when triggering depends on regime (e.g.\ market stress, time of day). The closed-form integral in §2.1 generalizes to a piecewise integral, and the gradient closed form extends with one extra term per $\delta_{m,j}$.

**Non-parametric kernels.** When the parametric form is in doubt, one either estimates $\phi_{m,j}(\cdot)$ on a spline basis or uses the EM algorithm of Lewis–Mohler / Veen–Schoenberg with a histogram representation. The Bacry–Muzy Wiener–Hopf approach recovers the kernels directly from the empirical second-order moments without solving an MLE — extremely useful at very high $M$ but harder to combine with covariates.

**Inhibition.** Negative $\alpha_{m,j}$ means events of $j$ suppress events of $m$. Removing the non-negativity constraint on $\alpha$ and replacing $\lambda_m$ by $\max(0, \lambda_m)$ or by a softplus link gives a "non-linear Hawkes" model (Brémaud–Massoulié), at the cost of a non-convex likelihood.

**Bayesian inference.** Same likelihood, replace the BFGS/FISTA estimator by HMC or variational inference. Particularly useful for the small-$T$ regime and for full posterior over $A$.

**Goodness-of-fit.** By the time-rescaling theorem, the compensated process $\tilde t_{m,k}=\int_0^{t_{m,k}}\lambda_m(s)\,ds$ should be a unit-rate Poisson process. KS tests or QQ plots against $\mathrm{Exp}(1)$ on the rescaled inter-arrival times of each component diagnose mis-specification.

**Branching ratio and elicitation summary statistics.** Beyond entry-wise inspection, one can summarize the elicitation structure by (i) the spectral radius of $G=A/\beta$, which is the asymptotic fraction of events that are "offspring" rather than "exogenous"; (ii) the right principal eigenvector of $G$, which gives a Pagerank-style importance ranking of components in the network; and (iii) the row sums $\sum_j \alpha_{m,j}/\beta_{m,j}$, which measure how susceptible $m$ is to being excited overall.
