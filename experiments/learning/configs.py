"""
Shared configuration for the pure-learning experiments: the parameter prior box,
the data-generating kernels, the covariate levels, and the collocation grid.  Keep
these in one place so the forward-operator, PINN/PINO, and amortised-inference
experiments all train and evaluate on the same investment family.
"""

import numpy as np

# Prior box over the scalar parameters (used for PINN/PINO sampling and for the
# amortised-inference training distribution).
PARAM_BOX = {
    "kappa": (0.10, 0.95),  # branching ratio (well identified)
    "theta": (0.40, 2.50),  # decay (weakly identified)
    "mu": (0.50, 3.00),  # constant baseline
    "delta": (-1.00, 1.50),  # covariate effect on excitation
}
PARAM_ORDER = ("kappa", "theta", "mu", "delta")

KERNELS = ("exp", "sumexp3", "powerlaw")  # data-generating kernels to sweep
COV_LEVELS = {"none": 0.0, "small": 0.35, "large": 0.8}

T_DEFAULT = 60.0
N_GRID = 96


def collocation_grid(T=T_DEFAULT, n=N_GRID):
    return np.linspace(0.0, T, n)


def sample_params(n, seed=0):
    """Sample n parameter vectors [kappa, theta, mu, delta] uniformly from the box."""
    rng = np.random.default_rng(seed)
    cols = [rng.uniform(*PARAM_BOX[k], size=n) for k in PARAM_ORDER]
    return np.column_stack(cols)


def effective_branching(P, zmax):
    """Effective branching ratio kappa * exp(|delta| * max_t|Z(t)|) for rows of P.

    Excitation is modulated multiplicatively by exp(delta . Z), so stability of the
    MBPP (a bounded intensity) requires this to stay below 1 -- otherwise the target
    diverges and no solver, learned or exact, can represent it.
    """
    P = np.atleast_2d(np.asarray(P, float))
    return P[:, 0] * np.exp(np.abs(P[:, 3]) * float(zmax))


def sample_params_stable(n, seed=0, zmax=0.5, cap=0.9):
    """Sample n parameter vectors restricted to the stable region

        kappa * exp(|delta| * zmax) <= cap < 1,

    by clipping kappa down to the stability ceiling.  This is the prior the
    learned solvers are actually trained and evaluated on (see docs/package_guide.md):
    the raw box bleeds ~18% into the supercritical region where xi is unbounded.
    """
    P = sample_params(n, seed=seed)
    ceil = cap / np.exp(np.abs(P[:, 3]) * float(zmax))
    P[:, 0] = np.minimum(P[:, 0], ceil)
    return P


def results_dir():
    import os

    d = os.path.join("results", "learning")
    os.makedirs(d, exist_ok=True)
    return d
