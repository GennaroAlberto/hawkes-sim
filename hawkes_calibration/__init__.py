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

* ``sector_stability`` -- stability-constrained sector count fitting; the public
  ``fit_sector_count_model`` is the noncritical version by default.

* ``sector_survival`` -- **startup survival stage**: Cox/McFadden partial likelihood
  over live sector risk sets, with an outside option for firms not tracked in the
  dataset/watch-list.

* ``sector_hazard`` -- **discrete-time hazard baseline**: supervised firm-week hazard
  model with negative sampling for weekly PitchBook-style data.

* ``optim``      -- a shared numpy BFGS used by the estimators.

The whole public API is re-exported here, so ``from hawkes_calibration import X``
continues to work for every X.  See the ``README`` for usage and
``paper/textbook.pdf`` for the full mathematical treatment.
"""

from . import sector_backtest as _sector_backtest_mod
from . import sector_stability as _sector_stability_mod
from .eventtime import *  # noqa: F401,F403
from .eventtime import __all__ as _eventtime_all
from .mbpp import *  # noqa: F401,F403
from .mbpp import __all__ as _mbpp_all
from .models.event_block_hawkes import *  # noqa: F401,F403  # event-time block Hawkes
from .models.event_block_hawkes import __all__ as _models_block_all
from .models.hawkes_ranker import *  # noqa: F401,F403  # 1-D Hawkes within-sector rankers
from .models.hawkes_ranker import __all__ as _models_ranker_all
from .operators import *  # noqa: F401,F403
from .operators import __all__ as _operators_all
from .sector_backtest import *  # noqa: F401,F403
from .sector_backtest import __all__ as _sector_backtest_all
from .sector_hazard import *  # noqa: F401,F403
from .sector_hazard import __all__ as _sector_hazard_all
from .sector_ranker import *  # noqa: F401,F403
from .sector_ranker import __all__ as _sector_ranker_all
from .sector_stability import *  # noqa: F401,F403
from .sector_stability import __all__ as _sector_stability_all
from .sector_survival import *  # noqa: F401,F403
from .sector_survival import __all__ as _sector_survival_all

# ---------------------------------------------------------------------------
# Canonical implementations for names that exist in more than one module.
# Plain assignments (NOT imports) so import-sorters cannot reorder them above
# the star imports -- these are the single source of truth for the collisions:
#   * the stability-constrained sector fitter supersedes the legacy one in
#     sector_ranker (noncritical excitation; see sector_stability.py);
#   * the backtest wrapper in sector_backtest supersedes the prototype in
#     sector_ranker (zeroed post-train histories, stable fitter, survival stage).
# ---------------------------------------------------------------------------
backtest_synthetic_pipeline = _sector_backtest_mod.backtest_synthetic_pipeline
fit_sector_count_model = _sector_stability_mod.fit_sector_count_model
sector_rate_at = _sector_stability_mod.sector_rate_at

# Public API: union of the submodule __all__ lists, deduplicated (collisions
# above appear once, resolved to the canonical binding).
__all__ = list(
    dict.fromkeys(
        [
            *_eventtime_all,
            *_mbpp_all,
            *_operators_all,
            *_sector_ranker_all,
            *_sector_stability_all,
            *_sector_backtest_all,
            *_sector_survival_all,
            *_models_block_all,
            *_models_ranker_all,
            *_sector_hazard_all,
        ]
    )
)
