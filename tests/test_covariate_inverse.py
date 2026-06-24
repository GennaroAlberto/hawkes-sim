r"""
Tests for the covariate-modulated multivariate MBPP inverse (shared + private
covariates).  Skipped if JAX is not installed.

Run:  PYTHONPATH=. python tests/test_covariate_inverse.py
"""

import numpy as np


def _imports():
    from hawkes_calibration.operators.covariate_inverse import (
        CovariateDesign, sample_dataset, batched_forward_numpy, jax_batched_forward,
        recover_params,
    )
    return (CovariateDesign, sample_dataset, batched_forward_numpy,
            jax_batched_forward, recover_params)


def test_design_shared_and_private_mask():
    try:
        CovariateDesign = _imports()[0]
    except ImportError:
        import pytest; pytest.skip("jax not installed"); return
    d = CovariateDesign(M=4, K_shared=2)
    assert d.p == 2 + 4                                  # shared + one private per group
    assert d.n_delta == 4 * 2 + 4                        # M*K_shared + M live coeffs
    # every group sees the shared covariates...
    assert d.mask[:, :2].all()
    # ...and exactly its own private one
    for m in range(4):
        assert d.mask[m, 2 + m] and d.mask[m, 2:].sum() == 1


def test_jax_forward_matches_numpy_oracle():
    try:
        CovariateDesign, sample_dataset, fwd_np, fwd_jx, _ = _imports()
    except ImportError:
        import pytest; pytest.skip("jax not installed"); return
    design = CovariateDesign(M=3, K_shared=2)
    t = np.linspace(0, 12, 80)
    d = sample_dataset(20, design, t, seed=0)
    xi_jax = np.asarray(fwd_jx(d["s"], d["A"], d["delta"], d["Z"], t))
    xi_np = fwd_np(d["s"], d["A"], d["delta"], d["Z"], t)
    assert np.max(np.abs(xi_jax - xi_np)) < 1e-5         # RK4 vs the trusted solve_mbpp_ltv


def test_clean_recovery_under_5_percent():
    try:
        CovariateDesign, sample_dataset, _, _, recover = _imports()
    except ImportError:
        import pytest; pytest.skip("jax not installed"); return
    design = CovariateDesign(M=3, K_shared=2)
    t = np.linspace(0, 16, 100)
    d = sample_dataset(24, design, t, n_switch=7, diag=(0.2, 0.55), off=(0.05, 0.4),
                       rho_max=0.85, seed=3)
    r = recover(design, d, sigma=0.0, n_obs=1, steps=6000, early_stop=False, seed=1)
    # on clean observations both the excitation matrix and the covariate
    # coefficients are recovered essentially exactly
    assert r["A_rel_mean"] < 0.05
    assert r["delta_rel_mean"] < 0.02


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} covariate-inverse tests passed.")
