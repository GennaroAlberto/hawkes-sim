"""
Event-time Hawkes calibration -- exact timestamps observed.

Maximum-likelihood estimation of multivariate Hawkes processes with exponential
kernels and optional time-varying covariates in the baseline, including an
L1-regularised fitter for sparse high-dimensional excitation networks.

Modules
-------
simulate     exact simulation via Ogata thinning
likelihood   log-likelihood + analytical gradient (vectorised per-pair)
estimate     MLE wrappers + asymptotic standard errors
lasso        L1-penalised MLE via FISTA (sparse high-M elicitation)
covariates   piecewise-constant covariates with closed-form baseline integral
"""

from .simulate import simulate_multivariate_hawkes
from .likelihood import log_likelihood, neg_log_likelihood
from .estimate import (
    fit_univariate,
    fit_multivariate,
    fit_multivariate_with_covariates,
)
from .lasso import fit_multivariate_lasso
from .covariates import PiecewiseConstantCovariate

__all__ = [
    "simulate_multivariate_hawkes",
    "log_likelihood",
    "neg_log_likelihood",
    "fit_univariate",
    "fit_multivariate",
    "fit_multivariate_with_covariates",
    "fit_multivariate_lasso",
    "PiecewiseConstantCovariate",
]
