r"""
Tests for the multivariate MBPP physics-informed neural operator (numpy).

Run:  PYTHONPATH=. python tests/test_pino.py
"""

import numpy as np

from hawkes_calibration.operators.pino import (
    MultivariateMBPPOperator,
    conv_matrix,
    exact_solution,
    sample_instances,
)


def test_residual_of_exact_solution_is_small():
    # The discrete MBPP residual evaluated at the EXACT solution is at the
    # discretization floor (-> the physics objective is well posed).
    M = 3
    t = np.linspace(0, 12, 96)
    S, A = sample_instances(8, M, seed=0)
    XI = exact_solution(S, A, t, 1.0)
    op = MultivariateMBPPOperator(M, t, theta=1.0, seed=0)
    R = op.residual(XI, S, A)
    assert np.sqrt(np.mean(R**2)) < 1e-2


def test_pino_learns_the_operator():
    # Train briefly on the physics residual (no exact solver in the loss) and
    # check it generalises to unseen instances vs the exact multivariate solver.
    M = 2
    t = np.linspace(0, 12, 48)
    Str, Atr = sample_instances(256, M, seed=1)
    Sval, Aval = sample_instances(24, M, seed=2)
    op = MultivariateMBPPOperator(M, t, theta=1.0, p=24, hidden=48, seed=0)
    op.train(Str, Atr, epochs=900, lr=3e-3, batch=64, val=(Sval, Aval), log_every=0)
    # residual must have dropped a lot, and held-out accuracy must be good
    rel = op._rel_l2(Sval, Aval)
    assert rel < 0.12  # < 12% relative L2 on unseen instances


def test_delta_zero_conv_matrix_shapes():
    t = np.linspace(0, 5, 20)
    G = conv_matrix(t, 1.0)
    assert G.shape == (20, 20)
    assert np.allclose(np.triu(G, 1), 0.0)  # causal: strictly upper triangle is zero


def test_jax_pino_learns_and_roundtrips(tmp_path=None):
    # The fast JAX operator (hybrid physics+anchor loss) learns the M-sector
    # operator well and its saved weights reload bit-exactly.  Skipped if JAX
    # is not installed (it is optional).
    try:
        from hawkes_calibration.operators.pino_jax import JAXMultivariateMBPPOperator
    except ImportError:
        import pytest  # type: ignore

        pytest.skip("jax/optax not installed")
        return
    import os
    import tempfile

    M = 3
    t = np.linspace(0, 12, 80)
    Str, Atr = sample_instances(400, M, seed=1)
    Sval, Aval = sample_instances(48, M, seed=2)
    Sa, Aa = sample_instances(300, M, seed=4)
    XIa = exact_solution(Sa, Aa, t, 1.0)
    op = JAXMultivariateMBPPOperator(M, t, p=48, hidden=128, depth=3, seed=0)
    op.train(Str, Atr, epochs=1500, lr=2e-3, batch=128, anchors=(Sa, Aa, XIa), log_every=0)
    assert op._rel_l2(Sval, Aval) < 0.05  # < 5% held-out with short training
    # save / load roundtrip is exact
    path = os.path.join(tempfile.mkdtemp(), "pino.npz")
    op.save(path)
    op2 = JAXMultivariateMBPPOperator.load(path)
    St, At = sample_instances(20, M, seed=9)
    assert np.max(np.abs(op.predict(St, At) - op2.predict(St, At))) < 1e-10


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} PINO tests passed.")
