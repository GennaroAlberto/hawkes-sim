r"""
Runner for the pure-learning experiments (see ../../docs/package_guide.md).

* ``--selfcheck``  runs the numpy-only well-posedness checks (works anywhere) and is
  the gate every learned solver depends on.
* ``train_pinn / train_pino``  require JAX; ``train_fno / train_amortized`` require
  TensorFlow.  They are thin, correct wrappers around the package API, meant to be
  executed by a coding agent in an environment with those backends installed.

Run:  PYTHONPATH=. python experiments/learning/run_learning.py --selfcheck
"""

import sys
import numpy as np

from hawkes_calibration.operators.neural_solver import (
    make_neural_solver, sample_covariate_paths, make_anchor_data, family_residual,
    solve_mbpp_residual_linear, solver_accuracy_report,
)
from hawkes_calibration.operators.linear import solve_mbpp_volterra
from experiments.learning import configs


# ---------------------------------------------------------------------------
# 0. numpy well-posedness self-check (runs anywhere).
# ---------------------------------------------------------------------------
def numpy_selfcheck():
    t = configs.collocation_grid()
    P = configs.sample_params(6, seed=0)
    Z = sample_covariate_paths(t, len(P), seed=0)
    XI = make_anchor_data(t, P, Z)

    # (i) the residual the PINN/PINO minimises is ~0 at the exact solutions
    res = float(np.max(np.abs(family_residual(XI, t, Z, P))))

    # (ii) the linear residual's zero reproduces the Volterra solve
    kappa, theta, mu, delta = 0.5, 1.0, 2.0, 0.8
    z1 = sample_covariate_paths(t, 1, seed=3)[0]
    Zfn = _grid_callable(t, z1)
    forcing = lambda tt: mu
    kernel = lambda tt, uu: kappa * theta * np.exp(np.clip(delta * float(Zfn(tt)[0]), -30, 30)) * np.exp(-theta * (tt - uu))
    xi_lin = solve_mbpp_residual_linear(t, forcing, kernel)
    xi_vol = solve_mbpp_volterra(lambda tt: np.array([forcing(tt)]), lambda tt, uu: kernel(tt, uu), t, M=1)[:, 0]
    lin_err = float(np.max(np.abs(xi_lin - xi_vol)))

    # (iii) the exact numpy reference solver is the oracle; accuracy report ~0 on it
    ref = make_neural_solver(backend="numpy", mode="pino", coll_grid=t, T=float(t[-1]))
    predict = lambda p, z: ref.solve(dict(kappa=p[0], theta=p[1], mu=p[2], delta=[p[3]], Z=_grid_callable(t, z)), t)
    rep = solver_accuracy_report(predict, t, P, Z)
    ref_err = float(rep.get("mean", np.nan))

    print("=== pure-learning numpy self-check ===")
    print("  family residual at exact solutions : %.2e   (PINN/PINO objective is well-posed)" % res)
    print("  linear-residual vs Volterra solve  : %.2e   (the residual's zero = exact solution)" % lin_err)
    print("  numpy reference accuracy report     : %.2e   (oracle self-consistency)" % ref_err)
    ok = res < 1e-9 and lin_err < 1e-6
    print("  --> %s" % ("OK" if ok else "CHECK FAILED"))
    return ok


def _grid_callable(t, z):
    """Wrap a covariate sampled on grid t (shape (n,) or (n,p)) as a callable Z(s)."""
    z = np.atleast_2d(np.asarray(z, float))
    if z.shape[0] != t.size:
        z = z.T
    p = z.shape[1]
    return lambda s: np.array([np.interp(float(s), t, z[:, k]) for k in range(p)])


# ---------------------------------------------------------------------------
# 1. supervised dataset from the exact solver (targets for operator learning).
# ---------------------------------------------------------------------------
def build_supervised_dataset(n=4000, seed=0):
    t = configs.collocation_grid()
    P = configs.sample_params(n, seed=seed)
    Z = sample_covariate_paths(t, n, seed=seed + 1)
    XI = make_anchor_data(t, P, Z)          # exact intensities (n, len(t))
    return t, P, Z, XI


# ---------------------------------------------------------------------------
# 2. PINN / PINO  (JAX).      3. FNO / amortised (TensorFlow).
#    Thin wrappers around the real API; raise a clear message if the backend
#    is missing so the agent knows to `pip install` it.
# ---------------------------------------------------------------------------
def _require(backend):
    import importlib
    try:
        importlib.import_module(backend)
    except Exception as e:  # pragma: no cover - environment dependent
        raise SystemExit("This experiment needs '%s' installed (pip install %s). [%s]"
                         % (backend, "jax jaxlib" if backend == "jax" else backend, e))


def train_pinn(n_steps=20000, T=configs.T_DEFAULT):
    _require("jax")
    t = configs.collocation_grid(T)
    Z = ((np.floor(t / 10) % 2) - 0.5)[:, None]
    solver = make_neural_solver(backend="jax", mode="pinn", coll_grid=t, Z_on_grid=Z, T=T, n_delta=1)
    solver.train(n_steps=n_steps, batch=64)
    rep = solver_accuracy_report(
        lambda p, z: solver.solve(dict(kappa=p[0], theta=p[1], mu=p[2], delta=[p[3]]), t),
        t, configs.sample_params(64, seed=99), Z_array=None,
        plot_path=configs.results_dir() + "/pinn_accuracy.png")
    return rep


def train_pino(n_steps=30000, T=configs.T_DEFAULT, hybrid=True):
    _require("jax")
    t = configs.collocation_grid(T)
    op = make_neural_solver(backend="jax", mode="pino", coll_grid=t, T=T)
    kw = {}
    if hybrid:
        P = configs.sample_params(64, seed=7)
        P[:, 0] = np.clip(P[:, 0], 0.75, 0.95)               # stiff anchors near criticality
        Z = sample_covariate_paths(t, len(P), seed=7)
        kw = dict(anchors=(Z, P, make_anchor_data(t, P, Z)), data_weight=1.0,
                  curriculum=True, reweight=True)
    op.train(n_steps=n_steps, batch=32, **kw)
    P_te = configs.sample_params(64, seed=123)
    Z_te = sample_covariate_paths(t, len(P_te), seed=123)
    return solver_accuracy_report(
        lambda p, z: op.solve(dict(kappa=p[0], theta=p[1], mu=p[2], delta=[p[3]]), t, Z_on_grid=z),
        t, P_te, Z_te, plot_path=configs.results_dir() + "/pino_accuracy.png")


def train_fno(epochs=100, M=20, seq_len=128, n_samples=20000):
    _require("tensorflow")
    from hawkes_calibration.operators.tf import (
        FourierNeuralOperator, generate_operator_dataset, train_operator)
    S, XI = generate_operator_dataset(n_samples=n_samples, M=M, seq_len=seq_len)
    fno = FourierNeuralOperator(seq_len=seq_len, n_channels=M, width=64, modes=32)
    train_operator(fno, S, XI, epochs=epochs, batch_size=128)
    return fno


def train_amortized(epochs=60, M=10):
    _require("tensorflow")
    from hawkes_calibration.operators.tf import AmortizedKernelInference, generate_inference_dataset
    C, K = generate_inference_dataset(n_samples=20000, M=M)
    net = AmortizedKernelInference(M=M)
    net.compile(optimizer="adam", loss="mse"); net.fit(C, K, epochs=epochs, batch_size=128, verbose=0)
    return net


# ---------------------------------------------------------------------------
# 4. Surrogate-in-the-loop IC-LL fit (template; needs JAX for autodiff).
# ---------------------------------------------------------------------------
def ic_fit_with_surrogate(solver, obs_times, counts):
    """Template: drop the jitted, differentiable surrogate into an IC-LL objective."""
    _require("jax")
    import jax, jax.numpy as jnp
    f = solver.intensity_fn()                                   # jitted f(p_vec, t) -> xi

    def ic_nll(p_vec):
        xi = f(p_vec, jnp.asarray(obs_times))
        Xi = jnp.concatenate([jnp.array([0.0]),
                              jnp.cumsum(0.5 * (xi[1:] + xi[:-1]) * jnp.diff(jnp.asarray(obs_times)))])
        dXi = jnp.maximum(jnp.diff(Xi), 1e-9)
        return jnp.sum(dXi) - jnp.sum(jnp.asarray(counts) * jnp.log(dXi))

    grad = jax.jit(jax.grad(ic_nll))
    return ic_nll, grad


def main(argv):
    if "--selfcheck" in argv or len(argv) == 0:
        ok = numpy_selfcheck()
        print("\nNext: install jax / tensorflow and run train_pinn / train_pino / train_fno /"
              " train_amortized (see docs/package_guide.md).")
        sys.exit(0 if ok else 1)
    print("Usage: run_learning.py --selfcheck   (then call the train_* entry points from Python)")


if __name__ == "__main__":
    main(sys.argv[1:])
