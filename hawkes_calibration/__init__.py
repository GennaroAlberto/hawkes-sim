"""
hawkes_calibration: Calibration of Hawkes processes.

Two regimes are supported:

* **Event-time calibration** (exact timestamps observed): maximum-likelihood
  estimation of multivariate Hawkes processes with exponential kernels and
  optional covariates in the baseline -- see ``simulate``, ``likelihood``,
  ``estimate``, ``lasso``, ``covariates`` and ``NOTES.md``.

* **Interval-censored calibration** (only aggregate counts per interval
  observed): fitting Hawkes processes through the Mean Behavior Poisson process
  (MBPP), following Rizoiu et al. (2022), "Interval-censored Hawkes processes",
  JMLR 23(1).  See ``mbpp``, ``exogenous``, ``ic_simulate``,
  ``interval_censored`` and ``PAPER_SUMMARY.md``.
"""

# --- Event-time Hawkes (original package) ----------------------------------
from .simulate import simulate_multivariate_hawkes
from .likelihood import log_likelihood, neg_log_likelihood
from .estimate import (
    fit_univariate,
    fit_multivariate,
    fit_multivariate_with_covariates,
)
from .lasso import fit_multivariate_lasso
from .covariates import PiecewiseConstantCovariate

# --- Interval-censored Hawkes via the MBPP (new) ---------------------------
from .mbpp import (
    MBPP,
    make_mbpp,
    ExponentialKernel,
    PowerLawKernel,
    kappa_theta_to_alpha_beta,
    alpha_beta_to_kappa_theta,
)
from .exogenous import (
    Constant,
    Rectangle,
    PiecewiseConstant,
    LHPP,
    MultiImpulse,
    Sine,
    Dassios,
    CovariateExogenous,
)
from .ic_simulate import (
    simulate_separable_hawkes,
    interval_censor,
    uniform_obs_times,
)
from .interval_censored import (
    ic_ll,
    sse_loss,
    fit_mbpp_ic,
    fit_mbpp_ic_multi,
    fit_mbpp_ic_covariates,
    fit_mbpp_ic_excitation,
    forecast_counts,
    ICFitResult,
)
from .operators import (
    FunctionalMBPP,
    SpectralOperator,
    solve_mbpp_ode,
    solve_mbpp_ode_multivariate,
    solve_mbpp_ltv,
    kernel_exponentials,
    make_exp_sum_kernel,
)
from .operators_nn import (
    MLP,
    DeepONetOperator,
    AmortizedInference,
)

__all__ = [
    # event-time
    "simulate_multivariate_hawkes",
    "log_likelihood",
    "neg_log_likelihood",
    "fit_univariate",
    "fit_multivariate",
    "fit_multivariate_with_covariates",
    "fit_multivariate_lasso",
    "PiecewiseConstantCovariate",
    # MBPP / interval-censored
    "MBPP",
    "make_mbpp",
    "ExponentialKernel",
    "PowerLawKernel",
    "kappa_theta_to_alpha_beta",
    "alpha_beta_to_kappa_theta",
    "Constant",
    "Rectangle",
    "PiecewiseConstant",
    "LHPP",
    "MultiImpulse",
    "Sine",
    "Dassios",
    "CovariateExogenous",
    "simulate_separable_hawkes",
    "interval_censor",
    "uniform_obs_times",
    "ic_ll",
    "sse_loss",
    "fit_mbpp_ic",
    "fit_mbpp_ic_multi",
    "fit_mbpp_ic_covariates",
    "fit_mbpp_ic_excitation",
    "forecast_counts",
    "ICFitResult",
    # functional / operator views
    "FunctionalMBPP",
    "SpectralOperator",
    "solve_mbpp_ode",
    "solve_mbpp_ode_multivariate",
    "solve_mbpp_ltv",
    "kernel_exponentials",
    "make_exp_sum_kernel",
    "MLP",
    "DeepONetOperator",
    "AmortizedInference",
]
