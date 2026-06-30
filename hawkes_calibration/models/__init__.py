"""
Application models for investment-market funding events.

This subpackage is the single logical home for the *application* models, kept as
distinct, self-contained modules:

* :mod:`event_block_hawkes` -- a **one-layer, marked, block-structured log-linear
  (exp-link) Hawkes** for **exact event times** in an **open population**.  The
  exponential link allows genuine firm-level self-inhibition; the sector-block
  excitation matrix keeps the parameter count at ``O(M^2)`` rather than ``O(N^2)``;
  at-risk indicators handle firm entry/exit.  Concave MLE.

* the **two-stage counts prototype** -- a sector-level Hawkes/Poisson count model
  (``fit_sector_count_model``) and a within-sector survival selector with an outside
  option (``fit_startup_survival``).  These live in ``hawkes_calibration.sector_ranker``
  and ``hawkes_calibration.sector_survival`` and are re-exported here so all three
  application models can be reached from one place::

      from hawkes_calibration.models import fit_block_hawkes, fit_startup_survival
"""

from .event_block_hawkes import *  # noqa: F401,F403
from .event_block_hawkes import __all__ as _block_all

# Re-export the existing application models so this is one logical home
# (the implementations stay in their original modules; nothing is moved).
from .. import sector_ranker as _sr
from .. import sector_survival as _ss

_reexport = list(getattr(_sr, "__all__", [])) + list(getattr(_ss, "__all__", []))
for _name in _reexport:
    globals()[_name] = getattr(_sr, _name, None) or getattr(_ss, _name)

__all__ = [*_block_all, *_reexport]
