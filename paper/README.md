# LaTeX source — *Interval-censored Hawkes Processes via the MBPP*

A self-contained mathematical treatment, reorganised from `../PAPER_SUMMARY.md`
into a structured LaTeX project, and extended with the **covariate-modulated
excitation** case and a from-scratch **Volterra equations** primer.

## Build

```bash
cd paper
latexmk -pdf main.tex      # runs pdflatex the right number of times
# or, without latexmk:
pdflatex main && pdflatex main && pdflatex main
```

Output: `main.pdf`. Requires a standard TeX Live with `amsmath, amsthm,
mathtools, hyperref, geometry, booktabs, enumitem, lmodern, microtype`.

## Structure

```
main.tex                         preamble, theorem environments, title, \input order, bibliography
sections/
  01-intro.tex                   why interval-censored Hawkes is hard; how the MBPP solves it; roadmap
  02-poisson-hawkes.tex          Poisson, compensators, Watanabe; Hawkes, cluster/branching; the MBPP equation
  03-assumptions.tex             standing assumptions (A1)–(A7)
  04-no-covariates.tex           univariate & multivariate MBPP (convolution Volterra, closed forms, IC-LL)
  05-baseline-covariates.tex     s(t)=exp(gamma0+gamma^T X(t)) — stays a convolution; fitting gamma
  06-excitation-covariates.tex   alpha(t)=alpha0 exp(delta^T Z(t)) — general (non-convolution) Volterra; LTV ODE
appendices/
  A-volterra.tex                 Volterra integral equations from scratch (kinds, resolvent, existence, methods)
  B-transforms.tex               Laplace & Fourier; transfer function; why they fail off the convolution case
  C-ode.tex                      exponential->ODE; rational<=>finite ODE; state space; LTV integration
  D-estimation.tex               IC-LL/Bregman; MLE asymptotics; time-rescaling; kappa-theta identifiability
```

## Reading order

The body is organised by increasing model complexity: **no covariates →
baseline covariates → excitation covariates**, with the general theory (Volterra,
transforms, ODE, estimation) in the appendices. A reader new to integral equations
should read Appendix A before §6.

## Relation to the code

The theory is implemented in the `hawkes_calibration` package: the convolution
case in `mbpp.py` / `operators.solve_mbpp_ode_multivariate`, and the
excitation-covariate (LTV) case of §6 in `operators.solve_mbpp_ltv` (validated:
with no modulation it reproduces the LTI solver to ~1e-16).
