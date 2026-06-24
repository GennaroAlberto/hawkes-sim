"""
Functional / operator views of the MBPP, and learned solvers.

The MBPP solution map ``s -> xi`` is a linear, translation-invariant operator, so
it can be solved exactly or *learned*.  The exact (numpy) solvers and the numpy
DeepONet / amortized surrogates are exported here.

The optional learned backends are **not** imported here -- they pull in heavy
dependencies (TensorFlow / JAX), so import them explicitly:

    from hawkes_calibration.operators.tf import FourierNeuralOperator        # needs tensorflow
    from hawkes_calibration.operators.neural_solver import make_neural_solver  # needs jax / tensorflow

Modules
-------
linear            exact solvers: state-space ODE (incl. multivariate), spectral
                  (Fourier transfer function), Volterra-Nystrom, LTV; kernel helpers
nn                numpy DeepONet surrogate + amortized inference net
tf                TensorFlow (optional): high-dim FNO / DeepONet / state-space /
                  amortized kernel inference
neural_solver     physics-informed neural PDE solver (PINN / PINO) dispatcher +
                  numpy reference; jax / tf backends in neural_solver_{jax,tf}
"""

from .linear import (
    FunctionalMBPP,
    SpectralOperator,
    solve_mbpp_ode,
    solve_mbpp_ode_multivariate,
    solve_mbpp_ltv,
    solve_mbpp_volterra,
    kernel_exponentials,
    make_exp_sum_kernel,
)
from .nn import (
    MLP,
    DeepONetOperator,
    AmortizedInference,
)
from .pino import (
    MultivariateMBPPOperator,
    sample_instances,
    exact_solution,
    conv_matrix,
)

__all__ = [
    "FunctionalMBPP",
    "SpectralOperator",
    "solve_mbpp_ode",
    "solve_mbpp_ode_multivariate",
    "solve_mbpp_ltv",
    "solve_mbpp_volterra",
    "kernel_exponentials",
    "make_exp_sum_kernel",
    "MLP",
    "DeepONetOperator",
    "AmortizedInference",
    "MultivariateMBPPOperator",
    "sample_instances",
    "exact_solution",
    "conv_matrix",
]
