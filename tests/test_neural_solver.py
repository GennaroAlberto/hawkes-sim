r"""
Tests for the neural MBPP solver -- the numpy-testable core (the residual and the
PINN objective formulation).  The JAX/TensorFlow PINN backends require those
frameworks (not installed in CI); their training target is exactly the residual
validated here.

Run:  PYTHONPATH=. python tests/test_neural_solver.py
"""

import numpy as np

from hawkes_calibration import PiecewiseConstantCovariate
from hawkes_calibration.operators import solve_mbpp_volterra
from hawkes_calibration.operators.neural_solver import (
    excitation_kernel_matrix,
    family_residual,
    make_anchor_data,
    make_neural_solver,
    mbpp_volterra_residual,
    sample_covariate_paths,
    solve_mbpp_residual_linear,
    solver_accuracy_report,
    trapezoid_weight_matrix,
)


def _excitation_problem(m=30, N=121):
    kappa, theta, mu, delta = 0.5, 1.0, 2.0, 1.2
    obs = np.arange(m + 1, dtype=float)
    Z = PiecewiseConstantCovariate(obs, np.array([[((i // 6) % 2) - 0.5] for i in range(m)], float))
    t = np.linspace(0, m, N)

    def forcing(tt):
        return mu

    def kernel(tt, uu):
        z = np.atleast_1d(np.asarray(Z(tt), float)).reshape(-1)
        return kappa * theta * np.exp(delta * z[0]) * np.exp(-theta * (tt - uu))

    return t, forcing, kernel, dict(mu=mu, kappa=kappa, theta=theta, delta=[delta], Z=Z)


def test_residual_zero_at_solution_large_elsewhere():
    t, forcing, kernel, _ = _excitation_problem()
    xi = solve_mbpp_volterra(lambda tt: np.array([forcing(tt)]), kernel, t, M=1)[:, 0]
    R = mbpp_volterra_residual(xi, t, forcing, kernel)
    assert np.max(np.abs(R)) < 1e-9  # exact solution -> residual 0
    assert (
        np.max(np.abs(mbpp_volterra_residual(1.5 * xi, t, forcing, kernel))) > 0.1
    )  # wrong -> nonzero


def test_pinn_objective_minimizer_is_the_solution():
    # the residual is linear in xi, so its zero (the PINN global minimum) equals
    # the time-stepping Volterra solve.
    t, forcing, kernel, _ = _excitation_problem()
    xi_lin = solve_mbpp_residual_linear(t, forcing, kernel)
    xi_step = solve_mbpp_volterra(lambda tt: np.array([forcing(tt)]), kernel, t, M=1)[:, 0]
    assert np.max(np.abs(xi_lin - xi_step)) < 1e-9


def test_numpy_backend_dispatch():
    t, _, _, params = _excitation_problem()
    solver = make_neural_solver(backend="numpy")
    xi = solver.solve(params, t)
    xi_ref = solve_mbpp_volterra(
        lambda tt: np.array([params["mu"]]),
        lambda tt, uu: (
            params["kappa"]
            * params["theta"]
            * np.exp(params["delta"][0] * np.atleast_1d(params["Z"](tt)).reshape(-1)[0])
            * np.exp(-params["theta"] * (tt - uu))
        ),
        t,
        M=1,
    )[:, 0]
    assert np.allclose(xi, xi_ref)
    # compensator is the cumulative integral
    Xi = solver.compensator(params, t)
    assert Xi[0] == 0.0 and np.all(np.diff(Xi) >= -1e-9)


def test_pino_family_residual_well_posed():
    # the PINO objective: a SINGLE operator that drives the residual to zero over a
    # whole family of covariate paths AND parameters solves every instance.
    rng = np.random.default_rng(0)
    t = np.linspace(0, 30, 121)
    B = 10
    Z = sample_covariate_paths(t, B, n_steps=5, seed=3)
    P = np.column_stack(
        [
            rng.uniform(0.2, 0.7, B),
            rng.uniform(0.5, 2.0, B),
            rng.uniform(1.0, 3.0, B),
            rng.uniform(-1.5, 1.5, B),
        ]
    )
    W = trapezoid_weight_matrix(t)
    xi = np.stack(
        [
            np.linalg.solve(
                np.eye(t.size) - W * excitation_kernel_matrix(t, Z[b], P[b, 0], P[b, 1], P[b, 3]),
                np.full(t.size, P[b, 2]),
            )
            for b in range(B)
        ]
    )
    R = family_residual(xi, t, Z, P)
    assert np.max(np.abs(R)) < 1e-9  # zero across the whole family


def test_anchor_data_satisfies_equation():
    # exact supervised anchors (for hybrid training) solve the MBPP residual
    rng = np.random.default_rng(0)
    t = np.linspace(0, 30, 121)
    Z = sample_covariate_paths(t, 6, seed=4)
    P = np.column_stack(
        [
            rng.uniform(0.2, 0.8, 6),
            rng.uniform(0.5, 2, 6),
            rng.uniform(1, 3, 6),
            rng.uniform(-1.5, 1.5, 6),
        ]
    )
    XI = make_anchor_data(t, P, Z)
    assert np.max(np.abs(family_residual(XI, t, Z, P))) < 1e-9


def test_accuracy_report_on_exact_and_perturbed():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 30, 121)
    P = np.column_stack(
        [
            rng.uniform(0.2, 0.8, 12),
            rng.uniform(0.5, 2, 12),
            rng.uniform(1, 3, 12),
            rng.uniform(-1, 1, 12),
        ]
    )
    Z = sample_covariate_paths(t, 12, seed=5)
    make_anchor_data(t, P, Z)

    # a perfect predictor -> ~0 error; a perturbed one -> larger, with by-kappa bins
    def exact_predict(p, z):
        return make_anchor_data(t, p[None, :], z[None, :])[0]

    rep_ok = solver_accuracy_report(exact_predict, t, P, Z)
    assert rep_ok["mean"] < 1e-9

    def perturbed(p, z):
        return 1.2 * exact_predict(p, z)

    rep_bad = solver_accuracy_report(perturbed, t, P, Z)
    assert rep_bad["mean"] > 0.05 and len(rep_bad["by_kappa"]) >= 1


def test_pino_dispatch_optional():
    for backend in ("jax", "tensorflow"):
        try:
            make_neural_solver(backend=backend, mode="pino", coll_grid=np.linspace(0, 10, 16))
        except ImportError as e:
            assert "jax" in str(e).lower() or "tensor" in str(e).lower()


def test_neural_backends_optional():
    # jax/tensorflow backends raise a clear ImportError if the framework is absent;
    # if present, the constructor should succeed.
    for backend in ("jax", "tensorflow"):
        try:
            make_neural_solver(backend=backend, coll_grid=np.linspace(0, 10, 16))
        except ImportError as e:
            assert backend.split("flow")[0] in str(e).lower() or "jax" in str(e).lower()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} neural-solver tests passed.")
