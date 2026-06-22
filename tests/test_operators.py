r"""
Verification tests for the functional / operator backends.

These assert that the interchangeable representations of the MBPP solution
operator agree with the exact closed form, that the learned spectral operator
recovers the kernel, and that the small numpy neural nets actually learn.

Run with:  python -m pytest tests/test_operators.py -q
       or:  PYTHONPATH=. python tests/test_operators.py
"""

import numpy as np

from hawkes_calibration import (
    MBPP, ExponentialKernel, Sine, Constant, PiecewiseConstant,
    FunctionalMBPP, SpectralOperator, solve_mbpp_ode, solve_mbpp_ode_multivariate,
    solve_mbpp_ltv, kernel_exponentials, make_exp_sum_kernel, MLP, DeepONetOperator,
    AmortizedInference, simulate_separable_hawkes, interval_censor, uniform_obs_times,
    PiecewiseConstantCovariate, fit_mbpp_ic_excitation,
)
from hawkes_calibration.mbpp.interval_censored import _excitation_modulation


def test_kernel_exponentials_exponential():
    a, b = kernel_exponentials(ExponentialKernel(0.6, 0.8))
    assert np.allclose(a, [0.48]) and np.allclose(b, [0.8])
    # branching ratio = sum a_q / b_q = kappa
    assert abs(float((a / b).sum()) - 0.6) < 1e-12


def test_ode_backend_matches_closed_form():
    # smooth forcing -> the state-space ODE is exact to integrator tolerance
    ker = ExponentialKernel(0.6, 0.8)
    t = np.linspace(0, 30, 1501)
    exo = Sine(2.0)
    a, b = kernel_exponentials(ker)
    xi_ode = solve_mbpp_ode(exo.intensity, a, b, t)
    xi_cf = MBPP(ker, exo, method="closed").intensity(t)
    assert np.max(np.abs(xi_ode - xi_cf)) < 1e-6


def test_spectral_exact_matches_closed_form():
    ker = ExponentialKernel(0.6, 0.8)
    t = np.linspace(0, 40, 2001)
    exo = Sine(2.0)
    xi_spec = SpectralOperator(t, pad=4).from_kernel(ker)(exo.intensity(t))
    xi_cf = MBPP(ker, exo, method="closed").intensity(t)
    m = (t > 1) & (t < 35)
    assert np.max(np.abs(xi_spec - xi_cf)[m]) < 0.02


def test_functional_dispatch_agrees():
    ker = ExponentialKernel(0.5, 1.0)
    t = np.linspace(0, 30, 1501)
    exo = Constant(2.0, 60)
    a = FunctionalMBPP(ker, method="ode").solve(exo.intensity, t)
    b = FunctionalMBPP(ker, method="spectral").solve(exo.intensity, t)
    m = (t > 1) & (t < 25)
    assert np.max(np.abs(a - b)[m]) < 0.05


def test_spectral_learned_recovers_operator_and_branching():
    rng = np.random.default_rng(0)
    ker = ExponentialKernel(0.6, 0.8)
    T = 40.0
    t = np.linspace(0, T, 2001)
    exos = []
    for _ in range(40):
        nb = rng.integers(4, 9)
        br = np.sort(np.concatenate([[0], rng.uniform(0, T, nb - 1), [T]]))
        exos.append(PiecewiseConstant(br, rng.uniform(0.2, 3.0, br.size - 1)))
    S = np.array([e.intensity(t) for e in exos])
    XI = np.array([MBPP(ker, e, method="closed").intensity(t) for e in exos])
    sp = SpectralOperator(t, pad=4).fit(S, XI)
    # generalize to a new forcing
    e_new = PiecewiseConstant([0, 7, 18, 30, 40], [1.5, 0.5, 2.5, 1.0])
    xi_pred = sp(e_new.intensity(t))
    xi_true = MBPP(ker, e_new, method="closed").intensity(t)
    m = (t > 1) & (t < 35)
    rel = np.linalg.norm((xi_pred - xi_true)[m]) / np.linalg.norm(xi_true[m])
    assert rel < 0.12
    # recovered branching ratio (integral of recovered kernel) ~ kappa
    tt, phi_rec = sp.recover_kernel()
    br = np.trapezoid(np.maximum(phi_rec, 0)[tt < 20], tt[tt < 20])
    assert abs(br - 0.6) < 0.15


def test_multivariate_mbpp_solver():
    # M=1 must match the univariate solver
    ker = ExponentialKernel(0.6, 0.8)
    a, b = kernel_exponentials(ker)
    t = np.linspace(0, 30, 1501)
    xi1 = solve_mbpp_ode(Constant(2.0, 60).intensity, a, b, t)
    xiM = solve_mbpp_ode_multivariate(lambda tt: np.array([2.0 if tt > 0 else 0.0]),
                                      np.array([[0.48]]), np.array([[0.8]]), t)[:, 0]
    assert np.max(np.abs(xi1 - xiM)) < 1e-9
    # M=3 constant baseline: stationary intensity -> (I-G)^{-1} mu
    A = np.array([[0.3, 0.1, 0.0], [0.0, 0.25, 0.15], [0.1, 0.0, 0.2]])
    B = np.ones((3, 3))
    mu = np.array([0.5, 0.3, 0.4])
    G = A / B
    xi_star = np.linalg.solve(np.eye(3) - G, mu)
    t = np.linspace(0, 80, 8001)
    xi = solve_mbpp_ode_multivariate(lambda tt: mu, A, B, t)
    assert np.max(np.abs(xi[-1] - xi_star)) < 1e-6


def test_ltv_solver_reduces_to_lti():
    # covariate-modulated excitation with NO modulation must match the LTI solver
    A = np.array([[0.3, 0.1], [0.0, 0.25]])
    B = np.ones((2, 2))
    mu = np.array([0.5, 0.3])
    t = np.linspace(0, 40, 4001)
    xi_lti = solve_mbpp_ode_multivariate(lambda tt: mu, A, B, t)
    xi_ltv = solve_mbpp_ltv(lambda tt: mu, A, B, t, modulation=None)
    assert np.max(np.abs(xi_lti - xi_ltv)) < 1e-9
    # doubling excitation over a window elevates the intensity, then it relaxes back
    def mod(tt):
        return np.full((2, 2), 2.0 if 15 <= tt < 25 else 1.0)
    xi_mod = solve_mbpp_ltv(lambda tt: mu, A, B, t, modulation=mod)
    i_mid = np.searchsorted(t, 22)
    i_end = np.searchsorted(t, 38)
    assert np.all(xi_mod[i_mid] > xi_lti[-1] + 1e-3)        # elevated during window
    assert np.allclose(xi_mod[i_end], xi_lti[-1], atol=1e-2)  # relaxed back after


def _excitation_dXi(obs, Z, kappa, theta, mu, delta, n_sub=6):
    fine = np.unique(np.concatenate(
        [np.linspace(obs[i], obs[i + 1], n_sub + 1) for i in range(obs.size - 1)]))
    _, Xi = solve_mbpp_ltv(lambda t: np.array([mu]), np.array([[kappa * theta]]),
                           np.array([[theta]]), fine,
                           modulation=_excitation_modulation(Z, [delta]), return_compensator=True)
    return np.diff(np.interp(obs, fine, Xi[:, 0]))


def test_excitation_likelihood_detects_modulation():
    # the IC-LL built on the LTV solver correctly prefers the true excitation
    # modulation over no modulation (delta=0): the machinery is sensitive to delta.
    from hawkes_calibration.mbpp.interval_censored import ic_ll
    rng = np.random.default_rng(0)
    m = 60
    obs = np.arange(m + 1, dtype=float)
    zvals = np.array([[((i // 10) % 2) - 0.5] for i in range(m)], dtype=float)
    Z = PiecewiseConstantCovariate(obs, zvals)
    kappa, theta, mu, delta = 0.4, 1.0, 2.0, 1.2
    dXi_true = _excitation_dXi(obs, Z, kappa, theta, mu, delta)
    counts = [rng.poisson(np.maximum(dXi_true, 0)).astype(float) for _ in range(20)]
    # loss at the true delta vs at delta=0 (no excitation modulation)
    dXi_0 = _excitation_dXi(obs, Z, kappa, theta, mu, 0.0)
    loss_true = sum(ic_ll(c, dXi_true) for c in counts)
    loss_zero = sum(ic_ll(c, dXi_0) for c in counts)
    assert loss_true < loss_zero          # the modulated model fits the data better


def test_excitation_fit_runs_and_is_sane():
    # smoke test: the fitter runs and returns finite, in-range parameters.
    rng = np.random.default_rng(1)
    m = 30
    obs = np.arange(m + 1, dtype=float)
    zvals = np.array([[((i // 10) % 2) - 0.5] for i in range(m)], dtype=float)
    Z = PiecewiseConstantCovariate(obs, zvals)
    dXi = _excitation_dXi(obs, Z, 0.4, 1.0, 2.0, 1.0)
    counts = [rng.poisson(np.maximum(dXi, 0)).astype(float) for _ in range(6)]
    res = fit_mbpp_ic_excitation(obs, counts, Z, n_sub=5, n_restarts=1)
    assert 0.0 < res.kappa < 1.0 and res.theta > 0 and res.baseline > 0
    assert np.isfinite(res.delta[0])


def test_mlp_learns_simple_function():
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, size=(800, 1))
    Y = np.sin(2 * X) + 0.3 * X
    net = MLP([1, 32, 32, 1], seed=0)
    for _ in range(2000):
        p = net.forward(X)
        d = (2.0 / len(X)) * (p - Y)
        net.backward(d); net.adam_step(3e-3)
    assert np.mean((net.forward(X) - Y) ** 2) < 1e-2


def test_amortized_inference_recovers_kappa():
    rng = np.random.default_rng(0)
    Ta = 40.0
    obs = uniform_obs_times(Ta, 40)
    exo = Constant(3.0, Ta)

    def make(K):
        X, Y = [], []
        for _ in range(K):
            ka, th = rng.uniform(0.2, 0.85), rng.uniform(0.4, 2.0)
            imm, off = simulate_separable_hawkes(exo, ka, th, Ta, seed=int(rng.integers(1e9)))
            X.append(interval_censor(np.sort(np.concatenate([imm, off])), obs))
            Y.append([ka, th])
        return np.array(X, float), np.array(Y)

    Xtr, Ytr = make(1000)
    Xte, Yte = make(200)
    net = AmortizedInference(40, hidden=64).fit(Xtr, Ytr, epochs=600, lr=2e-3)
    P = net.predict(Xte)
    # branching ratio kappa is identifiable -> strong correlation in one pass
    assert np.corrcoef(P[:, 0], Yte[:, 0])[0, 1] > 0.8


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} operator tests passed.")
