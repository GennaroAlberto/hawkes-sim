r"""
Neural operator backends for the MBPP, in pure numpy (no torch/JAX dependency).

Two learned representations of the MBPP operator are provided:

* :class:`DeepONetOperator` -- a branch/trunk operator network approximating the
  *forward* solution map  s |-> xi  (forcing -> intensity).  Unlike the spectral
  operator it does not assume linearity or translation-invariance, so the same
  architecture extends to the no-closed-form regimes (inhibition, marks).

* :class:`AmortizedInference` -- a network approximating the *inverse* map
  counts |-> (kappa, theta), so fitting a new interval-censored series is a
  single forward pass instead of an optimization.

Both are built on a small fully-connected :class:`MLP` with Adam.  These are
deliberately compact reference implementations meant to demonstrate the operator
structure and be validated against the exact solvers -- not to compete with a
GPU framework.
"""

from __future__ import annotations

import numpy as np


# ===========================================================================
# Minimal MLP with Adam (tanh hidden activations, linear output).
# ===========================================================================
class MLP:
    def __init__(self, sizes, seed=0, act="tanh"):
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            scale = np.sqrt(2.0 / (sizes[i] + sizes[i + 1]))
            self.W.append(rng.normal(0, scale, size=(sizes[i], sizes[i + 1])))
            self.b.append(np.zeros(sizes[i + 1]))
        self.act = act
        self._init_adam()

    def _init_adam(self):
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.t = 0

    def _phi(self, x):
        return np.tanh(x) if self.act == "tanh" else np.maximum(x, 0)

    def _dphi(self, x):
        return 1 - np.tanh(x) ** 2 if self.act == "tanh" else (x > 0).astype(float)

    def forward(self, X):
        self.zs, self.as_ = [], [X]
        a = X
        for i in range(len(self.W)):
            z = a @ self.W[i] + self.b[i]
            self.zs.append(z)
            a = z if i == len(self.W) - 1 else self._phi(z)
            self.as_.append(a)
        return a

    def backward(self, dout):
        """Given dL/d(output), return dL/d(input) and stash weight grads."""
        self.gW = [None] * len(self.W)
        self.gb = [None] * len(self.b)
        d = dout
        for i in reversed(range(len(self.W))):
            if i != len(self.W) - 1:
                d = d * self._dphi(self.zs[i])
            self.gW[i] = self.as_[i].T @ d
            self.gb[i] = d.sum(0)
            d = d @ self.W[i].T
        return d

    def adam_step(self, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for i in range(len(self.W)):
            for g, p, m, v in ((self.gW[i], self.W[i], self.mW[i], self.vW[i]),
                               (self.gb[i], self.b[i], self.mb[i], self.vb[i])):
                m[...] = b1 * m + (1 - b1) * g
                v[...] = b2 * v + (1 - b2) * g * g
                mhat = m / (1 - b1 ** self.t)
                vhat = v / (1 - b2 ** self.t)
                p -= lr * mhat / (np.sqrt(vhat) + eps)


# ===========================================================================
# DeepONet: forward operator s |-> xi.
# ===========================================================================
class DeepONetOperator:
    r"""
    Branch/trunk operator network.  The branch encodes a forcing sampled at
    ``n_sensors`` fixed times into ``p`` coefficients; the trunk encodes a query
    time into ``p`` basis values; the prediction is their inner product:

        xi(s)(t) ~ sum_k branch_k(s_sampled) * trunk_k(t) + b0.
    """

    def __init__(self, n_sensors, p=24, hidden=48, seed=0):
        self.p = p
        self.branch = MLP([n_sensors, hidden, hidden, p], seed=seed)
        self.trunk = MLP([1, hidden, hidden, p], seed=seed + 1)
        self.b0 = 0.0
        self._mb0 = self._vb0 = 0.0
        self._t = 0
        self.x_mean = self.x_std = self.y_mean = self.y_std = None

    def fit(self, F, t_query, Y, epochs=400, lr=2e-3, verbose=False):
        r"""
        F : (K, n_sensors) forcings at the sensor times.
        t_query : (N,) query times.
        Y : (K, N) target intensities xi.
        """
        F = np.asarray(F, float); Y = np.asarray(Y, float)
        tq = np.asarray(t_query, float).reshape(-1, 1)
        # standardize
        self.x_mean, self.x_std = F.mean(), F.std() + 1e-8
        self.y_mean, self.y_std = Y.mean(), Y.std() + 1e-8
        self.t_mean, self.t_std = tq.mean(), tq.std() + 1e-8
        Fs = (F - self.x_mean) / self.x_std
        Ys = (Y - self.y_mean) / self.y_std
        tqs = (tq - self.t_mean) / self.t_std

        K, N = Y.shape
        for ep in range(epochs):
            B = self.branch.forward(Fs)               # (K, p)
            Tr = self.trunk.forward(tqs)              # (N, p)
            pred = B @ Tr.T + self.b0                  # (K, N)
            err = pred - Ys
            loss = np.mean(err ** 2)
            dpred = (2.0 / (K * N)) * err
            # grads to branch and trunk via the bilinear form
            dB = dpred @ Tr                            # (K, p)
            dTr = dpred.T @ B                          # (N, p)
            gb0 = dpred.sum()
            self.branch.backward(dB); self.branch.adam_step(lr)
            self.trunk.backward(dTr); self.trunk.adam_step(lr)
            # adam for scalar bias
            self._t += 1
            self._mb0 = 0.9 * self._mb0 + 0.1 * gb0
            self._vb0 = 0.999 * self._vb0 + 0.001 * gb0 * gb0
            self.b0 -= lr * (self._mb0 / (1 - 0.9 ** self._t)) / (np.sqrt(self._vb0 / (1 - 0.999 ** self._t)) + 1e-8)
            if verbose and ep % 100 == 0:
                print(f"  epoch {ep:4d}  mse={loss:.5f}")
        return self

    def predict(self, F, t_query):
        F = np.atleast_2d(np.asarray(F, float))
        tq = np.asarray(t_query, float).reshape(-1, 1)
        Fs = (F - self.x_mean) / self.x_std
        tqs = (tq - self.t_mean) / self.t_std
        B = self.branch.forward(Fs)
        Tr = self.trunk.forward(tqs)
        return (B @ Tr.T + self.b0) * self.y_std + self.y_mean


# ===========================================================================
# Amortized inference: counts |-> (kappa, theta).
# ===========================================================================
class AmortizedInference:
    r"""
    Network approximating the inverse map: a feature vector summarising the
    observed interval counts -> the kernel parameters (kappa, theta).  Trained on
    simulated (counts -> true params) pairs; inference on a new series is then a
    single forward pass.  kappa is squashed to (0,1) by a logistic output and
    theta is forced positive by a softplus, so predictions are always valid.
    """

    def __init__(self, n_features, hidden=64, seed=0):
        self.net = MLP([n_features, hidden, hidden, 2], seed=seed)
        self.x_mean = self.x_std = None

    @staticmethod
    def _squash(raw):
        kappa = 1.0 / (1.0 + np.exp(-raw[:, 0:1]))
        theta = np.log1p(np.exp(raw[:, 1:2]))         # softplus > 0
        return np.concatenate([kappa, theta], axis=1)

    def fit(self, X, Y, epochs=600, lr=2e-3, verbose=False):
        """X : (K, n_features) count features.  Y : (K, 2) true (kappa, theta)."""
        X = np.asarray(X, float); Y = np.asarray(Y, float)
        self.x_mean, self.x_std = X.mean(0), X.std(0) + 1e-8
        Xs = (X - self.x_mean) / self.x_std
        for ep in range(epochs):
            raw = self.net.forward(Xs)
            pred = self._squash(raw)
            err = pred - Y
            loss = np.mean(err ** 2)
            # d(pred)/d(raw): kappa'=k(1-k); theta'=sigmoid(raw2)
            k = pred[:, 0:1]
            dk = k * (1 - k)
            dth = 1.0 / (1.0 + np.exp(-raw[:, 1:2]))
            draw = np.concatenate([(2.0 / len(Y)) * err[:, 0:1] * dk,
                                   (2.0 / len(Y)) * err[:, 1:2] * dth], axis=1)
            self.net.backward(draw); self.net.adam_step(lr)
            if verbose and ep % 150 == 0:
                print(f"  epoch {ep:4d}  mse={loss:.5f}")
        return self

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        Xs = (X - self.x_mean) / self.x_std
        return self._squash(self.net.forward(Xs))
