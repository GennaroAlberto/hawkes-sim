"""
Interval-censored Hawkes calibration via the Mean Behavior Poisson Process.

When only aggregate counts per interval are observed, the Hawkes likelihood is
unavailable, so we fit through the MBPP -- a Poisson process whose deterministic
intensity equals the *expected* Hawkes intensity (Rizoiu et al., 2022).

Modules
-------
core               MBPP intensity & compensator; exponential / power-law kernels;
                   (kappa, theta) <-> (alpha, beta) conversions
exogenous          baseline functions (constant/rectangle/sine/Dassios/
                   multi-impulse/LHPP) + log-linear CovariateExogenous
ic_simulate        separable + covariate-modulated-excitation simulators; censoring
interval_censored  IC-LL & SSE losses; all MBPP fitters; SEs; forecasting
gof                goodness-of-fit: time-rescaling KS, Pearson dispersion, residuals
bayes              Bayesian posterior over (kappa, theta): adaptive Metropolis,
                   Laplace, hierarchical pooling, priors, R-hat
"""

from .bayes import (
    BayesResult,
    GaussianPrior,
    adaptive_metropolis,
    default_prior,
    fit_mbpp_bayes,
    fit_mbpp_bayes_hierarchical,
    laplace_posterior,
    rhat,
)
from .core import (
    MBPP,
    ExponentialKernel,
    PowerLawKernel,
    alpha_beta_to_kappa_theta,
    kappa_theta_to_alpha_beta,
    make_mbpp,
)
from .exogenous import (
    LHPP,
    Constant,
    CovariateExogenous,
    Dassios,
    MultiImpulse,
    PiecewiseConstant,
    Rectangle,
    Sine,
)
from .gof import (
    dispersion,
    ks_test_exp1,
    poisson_pearson_residuals,
    qq_exp1,
    time_rescaling_residuals,
)
from .ic_simulate import (
    interval_censor,
    simulate_hawkes_excitation,
    simulate_separable_hawkes,
    uniform_obs_times,
)
from .interval_censored import (
    ICFitResult,
    fit_mbpp_ic,
    fit_mbpp_ic_covariates,
    fit_mbpp_ic_excitation,
    fit_mbpp_ic_excitation_multi,
    fit_mbpp_ic_multi,
    fit_mbpp_ic_sumexp,
    forecast_counts,
    ic_ll,
    sse_loss,
)

__all__ = [
    # core
    "MBPP",
    "make_mbpp",
    "ExponentialKernel",
    "PowerLawKernel",
    "kappa_theta_to_alpha_beta",
    "alpha_beta_to_kappa_theta",
    # exogenous
    "Constant",
    "Rectangle",
    "PiecewiseConstant",
    "LHPP",
    "MultiImpulse",
    "Sine",
    "Dassios",
    "CovariateExogenous",
    # simulation / censoring
    "simulate_separable_hawkes",
    "simulate_hawkes_excitation",
    "interval_censor",
    "uniform_obs_times",
    # losses / fitting / forecasting
    "ic_ll",
    "sse_loss",
    "fit_mbpp_ic",
    "fit_mbpp_ic_multi",
    "fit_mbpp_ic_covariates",
    "fit_mbpp_ic_excitation",
    "fit_mbpp_ic_excitation_multi",
    "fit_mbpp_ic_sumexp",
    "forecast_counts",
    "ICFitResult",
    # goodness of fit
    "time_rescaling_residuals",
    "ks_test_exp1",
    "qq_exp1",
    "poisson_pearson_residuals",
    "dispersion",
    # bayesian
    "fit_mbpp_bayes",
    "fit_mbpp_bayes_hierarchical",
    "GaussianPrior",
    "default_prior",
    "BayesResult",
    "adaptive_metropolis",
    "laplace_posterior",
    "rhat",
]
