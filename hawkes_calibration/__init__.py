"""
hawkes_calibration: Calibration of Hawkes processes.

Two data regimes are supported, organised into subpackages:

* ``eventtime``  -- **event-time calibration** (exact timestamps observed):
  maximum-likelihood estimation of multivariate Hawkes processes with exponential
  kernels and optional covariates in the baseline.

* ``mbpp``       -- **interval-censored calibration** (only aggregate counts per
  interval observed): fitting Hawkes processes through the Mean Behavior Poisson
  Process, following Rizoiu et al. (2022), "Interval-censored Hawkes processes",
  JMLR 23(1).  Includes goodness-of-fit and Bayesian inference.

* ``operators``  -- **functional / operator views**: exact linear-operator solvers
  (ODE / spectral / Volterra) plus learned surrogates (DeepONet, FNO, PINN, PINO).

* ``sector_ranker`` -- **investment-market prototype**: a weekly sector-level
  Hawkes/count model plus a dynamic startup risk-set ranker for marked-event
  simulation and synthetic backtests.

* ``optim``      -- a shared numpy BFGS used by the estimators.

The whole public API is re-exported here, so ``from hawkes_calibration import X``
continues to work for every X.  See the ``README`` for usage and
``paper/textbook.pdf`` for the full mathematical treatment.
"""

from .eventtime import *  # noqa: F401,F403
from .mbpp import *  # noqa: F401,F403
from .operators import *  # noqa: F401,F403
from .sector_ranker import *  # noqa: F401,F403

from .eventtime import __all__ as _eventtime_all
from .mbpp import __all__ as _mbpp_all
from .operators import __all__ as _operators_all
from .sector_ranker import __all__ as _sector_ranker_all

__all__ = [*_eventtime_all, *_mbpp_all, *_operators_all, *_sector_ranker_all]
