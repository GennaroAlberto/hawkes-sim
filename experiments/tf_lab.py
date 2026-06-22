r"""
tf_lab.py -- a synthetic-data laboratory for stress-testing the MBPP neural
operators (``hawkes_calibration/operators_tf.py``).

What it gives you
-----------------
1. **Forcing generators** -- several functional forms for the exogenous baseline
   s(t) in R^M (piecewise-constant, sinusoidal, impulse trains, smooth/GP-like,
   bursty), so you can probe how operator learning depends on the input class.
2. **A batched multivariate MBPP solver** -- exact ground-truth intensities
   xi(t) in R^M for a whole batch of forcings sharing one kernel matrix
   (vectorised RK4; fast enough for medium/large datasets).
3. **Noise models** -- additive Gaussian, multiplicative, Poisson-count and
   missing-data corruption of the targets, with a single scalar ``level`` so you
   can sweep noise from clean to broken.
4. **Instance builder** -- ``make_instance(scale=...)`` with ``small`` / ``medium``
   / ``large`` presets (dimension M, sequence length T, #samples), returning
   train/test splits ready for Keras.
5. **A numpy baseline learner** -- ``MultivariateSpectralOperator`` fits the exact
   *linear* operator R(omega) (an M x M complex matrix per frequency).  Because
   the MBPP operator really is linear, this is a strong, instant baseline and a
   reference "breaking point" to compare the TF models against -- and it needs no
   TensorFlow.
6. **A sweep harness** -- ``noise_sweep`` trains on noisy targets, evaluates
   against the *clean* targets, and reports where each forcing class breaks.

This module is numpy + matplotlib only, so it runs today.  To stress-test the
TensorFlow models instead of the linear baseline, pass a ``fit_eval`` callback to
``noise_sweep`` (see the README, "High-dimensional learning + noise-stress lab",
for a ready-made one).
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, field, asdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ===========================================================================
# Forcing generators: each returns a non-negative (T, M) array on grid ``t``.
# ===========================================================================
def forcing_piecewise_constant(M, t, rng, n_steps=5):
    T = t.size
    s = np.zeros((T, M))
    for m in range(M):
        edges = np.sort(rng.uniform(t[0], t[-1], n_steps - 1))
        levels = rng.uniform(0.2, 3.0, n_steps)
        s[:, m] = levels[np.searchsorted(edges, t)]
    return s


def forcing_sine(M, t, rng):
    s = np.zeros((t.size, M))
    for m in range(M):
        offset = rng.uniform(1.0, 3.0)
        comp = sum(rng.uniform(0.2, 1.2) * np.sin(rng.uniform(0.1, 1.0) * t + rng.uniform(0, 2 * np.pi))
                   for _ in range(rng.integers(1, 4)))
        s[:, m] = np.maximum(offset + comp, 0.05)
    return s


def forcing_impulse(M, t, rng, n_impulses=6, width=None):
    """Sparse tall narrow rectangles -- a smoothed impulse train."""
    T = t.size
    dt = t[1] - t[0]
    width = width or 2 * dt
    s = np.full((T, M), 0.05)
    for m in range(M):
        for _ in range(n_impulses):
            t0 = rng.uniform(t[0], t[-1])
            s[:, m] += rng.uniform(2.0, 6.0) * (np.abs(t - t0) < width)
    return s


def forcing_smooth(M, t, rng, n_freq=6):
    """Random-Fourier smooth positive functions (GP-like)."""
    s = np.zeros((t.size, M))
    span = t[-1] - t[0]
    for m in range(M):
        v = np.zeros_like(t)
        for _ in range(n_freq):
            w = rng.uniform(0.5, 6.0) * (2 * np.pi / span)
            v += rng.normal() * np.sin(w * t) + rng.normal() * np.cos(w * t)
        s[:, m] = np.logaddexp(0.0, v / np.sqrt(n_freq) + 0.5)   # softplus -> positive
    return s


def forcing_bursty(M, t, rng, n_bursts=4):
    """Sum of exponentially-decaying bursts at random onsets."""
    s = np.full((t.size, M), 0.05)
    for m in range(M):
        for _ in range(n_bursts):
            t0 = rng.uniform(t[0], t[-1])
            rate = rng.uniform(0.5, 2.0)
            s[:, m] += rng.uniform(2.0, 5.0) * np.exp(-rate * np.maximum(t - t0, 0)) * (t >= t0)
    return s


FORCINGS = {
    "pc": forcing_piecewise_constant,
    "sine": forcing_sine,
    "impulse": forcing_impulse,
    "smooth": forcing_smooth,
    "bursty": forcing_bursty,
}


# ===========================================================================
# Batched, exact multivariate MBPP solver (vectorised RK4 over the batch).
# ===========================================================================
def solve_mbpp_batch(S_grid, A, B, t):
    r"""
    xi for a batch of forcings ``S_grid`` (K, T, M) sharing kernel matrices
    A, B (M, M).  States Y_{m,j}, xi_m = s_m + sum_j Y_{m,j}.  Returns (K, T, M).
    """
    A = np.asarray(A, float); B = np.asarray(B, float)
    K, T, M = S_grid.shape
    Y = np.zeros((K, M, M))
    XI = np.empty((K, T, M))
    XI[:, 0, :] = S_grid[:, 0, :] + Y.sum(2)

    def deriv(Y, s):                       # Y:(K,M,M)  s:(K,M)
        xi = s + Y.sum(axis=2)             # (K,M)
        return A[None] * xi[:, None, :] - B[None] * Y

    for n in range(1, T):
        dt = t[n] - t[n - 1]
        s0, s1 = S_grid[:, n - 1, :], S_grid[:, n, :]
        smid = 0.5 * (s0 + s1)
        k1 = deriv(Y, s0)
        k2 = deriv(Y + dt / 2 * k1, smid)
        k3 = deriv(Y + dt / 2 * k2, smid)
        k4 = deriv(Y + dt * k3, s1)
        Y = Y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        XI[:, n, :] = s1 + Y.sum(2)
    return XI


def random_branching_matrix(M, density, max_radius, rng):
    """Non-negative branching matrix G with spectral radius < max_radius."""
    mask = rng.random((M, M)) < density
    G = rng.uniform(0.05, 1.0, size=(M, M)) * mask
    radius = max(np.max(np.abs(np.linalg.eigvals(G))), 1e-9)
    return G * (max_radius * rng.uniform(0.6, 0.95) / radius)


# ===========================================================================
# Noise models on the targets (a single scalar ``level`` controls severity).
# ===========================================================================
def add_noise(XI, kind, level, rng, dt=1.0):
    if level <= 0:
        return XI.copy(), np.ones_like(XI)
    mask = np.ones_like(XI)
    if kind == "gauss":                    # additive observation noise
        noisy = XI + level * XI.std() * rng.standard_normal(XI.shape)
    elif kind == "mult":                   # multiplicative (heteroscedastic)
        noisy = XI * (1.0 + level * rng.standard_normal(XI.shape))
    elif kind == "poisson":                # count noise via finite exposure
        exposure = 1.0 / max(level, 1e-6)
        counts = rng.poisson(np.maximum(XI, 0) * exposure)
        noisy = counts / exposure
    elif kind == "missing":                # randomly drop a fraction of time steps
        noisy = XI.copy()
        drop = rng.random(XI.shape[:2]) < level
        noisy[drop] = 0.0
        mask[drop] = 0.0
    else:
        raise ValueError(f"unknown noise kind {kind!r}")
    return np.maximum(noisy, 0.0), mask


# ===========================================================================
# Instances at small / medium / large scale.
# ===========================================================================
SCALES = {
    "small": dict(M=3, T=64, n_samples=512, density=0.5, max_radius=0.7),
    "medium": dict(M=10, T=128, n_samples=4000, density=0.4, max_radius=0.8),
    "large": dict(M=40, T=256, n_samples=20000, density=0.25, max_radius=0.85),
}


@dataclass
class Instance:
    S_train: np.ndarray
    XI_train: np.ndarray          # clean targets (reference)
    XI_train_noisy: np.ndarray    # corrupted targets (what you fit on)
    mask_train: np.ndarray
    S_test: np.ndarray
    XI_test: np.ndarray           # clean test targets (what you evaluate against)
    t: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def shape(self):
        return dict(M=self.meta["M"], T=self.meta["T"],
                    n_train=len(self.S_train), n_test=len(self.S_test))


def make_instance(scale="small", forcing="pc", noise_kind="gauss", noise_level=0.0,
                  theta=1.0, vary_system=False, horizon=30.0, test_frac=0.2,
                  seed=0, **overrides):
    r"""
    Build a synthetic operator-learning instance.

    Parameters
    ----------
    scale : {"small","medium","large"}   preset (override M/T/n_samples via kwargs).
    forcing : key of FORCINGS              the input functional form.
    noise_kind : {"gauss","mult","poisson","missing"}.
    noise_level : float                    severity (0 = clean).
    theta : float                          common kernel decay (B matrix).
    vary_system : bool                     if True each sample has its own kernel
        matrix (harder; learn a family) instead of one shared system.
    horizon : float                        physical time covered by the T-grid.

    Returns
    -------
    Instance
    """
    cfg = dict(SCALES[scale]); cfg.update(overrides)
    M, T, n = cfg["M"], cfg["T"], cfg["n_samples"]
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, horizon, T)
    B = np.full((M, M), float(theta))
    gen = FORCINGS[forcing]

    S = np.stack([gen(M, t, rng) for _ in range(n)]).astype(float)   # (n,T,M)
    if vary_system:
        XI = np.empty_like(S)
        for i in range(n):
            G = random_branching_matrix(M, cfg["density"], cfg["max_radius"], rng)
            XI[i] = solve_mbpp_batch(S[i:i + 1], G * B, B, t)[0]
        G_used = None
    else:
        G = random_branching_matrix(M, cfg["density"], cfg["max_radius"], rng)
        XI = solve_mbpp_batch(S, G * B, B, t)
        G_used = G

    XI_noisy, mask = add_noise(XI, noise_kind, noise_level, rng, dt=t[1] - t[0])

    n_test = int(n * test_frac)
    sl_tr, sl_te = slice(n_test, None), slice(0, n_test)
    meta = dict(M=M, T=T, n_samples=n, scale=scale, forcing=forcing,
                noise_kind=noise_kind, noise_level=noise_level, theta=theta,
                vary_system=vary_system, horizon=horizon,
                spectral_radius=None if G_used is None else float(np.max(np.abs(np.linalg.eigvals(G_used)))))
    return Instance(
        S_train=S[sl_tr], XI_train=XI[sl_tr], XI_train_noisy=XI_noisy[sl_tr], mask_train=mask[sl_tr],
        S_test=S[sl_te], XI_test=XI[sl_te], t=t, meta=meta,
    )


# ===========================================================================
# Numpy baseline learner: the exact LINEAR operator R(omega) (M x M per freq).
# ===========================================================================
class MultivariateSpectralOperator:
    r"""
    Fits the multivariate transfer operator: for each frequency a complex
    M x M matrix W_f with xi_hat(f) = s_hat(f) W_f, by ridge least squares.  The
    MBPP operator is exactly of this (linear, per-frequency) form, so on clean
    data this baseline is near-exact; under noise its degradation is the natural
    yardstick for the TF models.
    """

    def __init__(self, ridge=1e-2):
        self.ridge = float(ridge)
        self.W = None

    def fit(self, S, XI):
        Sf = np.fft.rfft(S, axis=1)            # (K,F,M)
        Xf = np.fft.rfft(XI, axis=1)
        K, Fr, M = Sf.shape
        W = np.empty((Fr, M, M), complex)
        eye = self.ridge * np.eye(M)
        for f in range(Fr):
            Sff = Sf[:, f, :]                  # (K,M)
            G = Sff.conj().T @ Sff + eye       # (M,M)
            P = Sff.conj().T @ Xf[:, f, :]     # (M,M)
            W[f] = np.linalg.solve(G, P)       # R_f^T
        self.W = W
        return self

    def predict(self, S):
        Sf = np.fft.rfft(S, axis=1)
        Xf = np.einsum("kfi,fio->kfo", Sf, self.W)
        return np.fft.irfft(Xf, n=S.shape[1], axis=1)


# ===========================================================================
# Metrics and the noise sweep.
# ===========================================================================
def rel_l2(pred, true, trim=0.1):
    """Relative L2 error, trimming a fraction ``trim`` off each time boundary."""
    T = true.shape[1]
    lo, hi = int(T * trim), T - int(T * trim)
    a, b = pred[:, lo:hi], true[:, lo:hi]
    return float(np.linalg.norm((a - b).ravel()) / (np.linalg.norm(b.ravel()) + 1e-12))


def find_breaking_point(levels, errors, factor=2.0, min_abs=0.08):
    r"""
    The first noise level at which the (clean-target) test error blows up: it
    exceeds ``factor`` x the noiseless error *and* is at least ``min_abs`` in
    absolute terms (so tiny base errors don't trigger on rounding).  Returns None
    if it never breaks over the swept range.
    """
    base = errors[0]
    thr = factor * base
    for lvl, err in zip(levels, errors):
        if err > thr and err >= min_abs:
            return float(lvl)
    return None


def noise_sweep(scale="small", forcings=("pc", "sine", "smooth", "bursty"),
                noise_kind="gauss", levels=(0.0, 0.05, 0.1, 0.2, 0.4, 0.8),
                fit_eval=None, seed=0, ridge=1e-2):
    r"""
    For each forcing class, build instances across ``levels`` (train on noisy,
    test on clean) and record the relative L2 error and the breaking point.

    Parameters
    ----------
    fit_eval : callable(Instance) -> float, optional
        Trains a model on ``inst.S_train, inst.XI_train_noisy`` and returns the
        relative L2 error against ``inst.XI_test``.  If None, the numpy
        ``MultivariateSpectralOperator`` baseline is used (no TensorFlow needed).

    Returns
    -------
    dict : { forcing: {"levels":[...], "errors":[...], "breaking": lvl_or_None} }
    """
    levels = list(levels)

    def default_fit_eval(inst):
        op = MultivariateSpectralOperator(ridge=ridge).fit(inst.S_train, inst.XI_train_noisy)
        return rel_l2(op.predict(inst.S_test), inst.XI_test)

    fe = fit_eval or default_fit_eval
    out = {}
    for fc in forcings:
        errs = []
        for lvl in levels:
            inst = make_instance(scale=scale, forcing=fc, noise_kind=noise_kind,
                                 noise_level=lvl, seed=seed)
            errs.append(fe(inst))
        out[fc] = dict(levels=levels, errors=errs,
                       breaking=find_breaking_point(levels, errs))
    return out


def plot_sweep(sweep, title, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for fc, r in sweep.items():
        line, = ax.plot(r["levels"], r["errors"], "o-", label=fc)
        if r["breaking"] is not None:
            ax.axvline(r["breaking"], color=line.get_color(), ls=":", alpha=0.5)
    ax.set_xlabel("noise level"); ax.set_ylabel("test rel. L2 vs clean targets")
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def run_demo(out_dir="results", scale="small", noise_kind="gauss"):
    """Numpy-baseline noise sweep across forcing classes -> figure + json."""
    os.makedirs(out_dir, exist_ok=True)
    sweep = noise_sweep(scale=scale, noise_kind=noise_kind)
    print(f"=== tf_lab noise sweep ({scale}, {noise_kind}, linear spectral baseline) ===")
    for fc, r in sweep.items():
        bp = r["breaking"]
        print(f"  {fc:8s}: errors={[round(e,3) for e in r['errors']]}  breaks@{bp}")
    path = os.path.join(out_dir, "tf_lab_noise_sweep.png")
    plot_sweep(sweep, f"MBPP operator: noise tolerance by forcing ({scale}, {noise_kind})", path)
    with open(os.path.join(out_dir, "tf_lab_noise_sweep.json"), "w") as f:
        json.dump(sweep, f, indent=2)
    print(f"\nWrote {path}")
    return sweep


if __name__ == "__main__":
    run_demo()
