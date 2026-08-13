r"""
A **multivariate** Physics-Informed Neural Operator (PINO) for the Mean Behavior
Poisson Process -- pure numpy, no GPU framework.

It learns the solution operator of an *M-dimensional* Hawkes/MBPP,

    (s, A) |-> xi,      xi(t) in R^M  solving   xi = s + Phi * xi,
                        Phi_{m,j}(tau) = A_{m,j} e^{-theta tau},

i.e.\ given a constant baseline vector ``s`` and an ``M x M`` excitation matrix
``A`` (the cross-/self-excitation structure), it produces the whole mean-intensity
path of every sector.  A single trained operator solves *any* instance in the
family in one forward pass.

It is trained **physics-informed**: the loss is the collocation residual of the
multivariate MBPP equation,

    R[xi] = xi - s - A (G xi),   (G = discrete  int_0^t e^{-theta(t-u)}(.) du),

with no exact solver in the loss.  Because R is *linear* in xi, its global minimum
is the exact solution and its gradient w.r.t. the network output is
``2 (R - G^T (R A))`` -- which we backpropagate through a DeepONet built on the
package's numpy MLP.  Optional exact ``anchors`` add a supervised term (hybrid PINO).

This is the runnable, multivariate counterpart of the JAX/TF PINO reference
implementations; it executes anywhere numpy does and is validated against
:func:`hawkes_calibration.solve_mbpp_ode_multivariate`.
"""

from __future__ import annotations

import numpy as np

from .linear import solve_mbpp_ode_multivariate
from .nn import MLP


# ---------------------------------------------------------------------------
# Discrete causal convolution operator  (G xi)(t_n) = int_0^{t_n} e^{-theta(t_n-u)} xi(u) du.
# ---------------------------------------------------------------------------
def conv_matrix(t_grid, theta):
    t = np.asarray(t_grid, float)
    N = t.size
    G = np.zeros((N, N))
    for n in range(1, N):
        tl = t[: n + 1]
        dt = np.diff(tl)
        w = np.zeros(n + 1)
        w[0] = dt[0] / 2.0
        w[-1] += dt[-1] / 2.0
        if n > 1:
            w[1:-1] = (dt[:-1] + dt[1:]) / 2.0
        G[n, : n + 1] = w * np.exp(-theta * (t[n] - tl))
    return G


# ---------------------------------------------------------------------------
# Instance sampler: subcritical excitation matrices + baseline vectors.
# ---------------------------------------------------------------------------
def sample_instances(
    n, M, seed=0, diag=(0.05, 0.45), off=(0.0, 0.25), s_range=(0.4, 2.2), rho_max=0.75
):
    rng = np.random.default_rng(seed)
    S = rng.uniform(*s_range, size=(n, M))
    A = np.empty((n, M, M))
    for k in range(n):
        a = rng.uniform(*off, size=(M, M))
        a[np.diag_indices(M)] = rng.uniform(*diag, size=M)
        rho = float(np.max(np.abs(np.linalg.eigvals(a))))
        if rho > rho_max:
            a *= rho_max / rho  # keep subcritical (bounded xi)
        A[k] = a
    return S, A


def exact_solution(S, A, t_grid, theta=1.0):
    """Exact multivariate MBPP intensities (the oracle), shape (n, N, M)."""
    t = np.asarray(t_grid, float)
    M = A.shape[1]
    B = theta * np.ones((M, M))
    out = np.empty((len(A), t.size, M))
    for k in range(len(A)):
        out[k] = solve_mbpp_ode_multivariate(lambda tt, s=S[k]: s, A[k], B, t)
    return out


# ===========================================================================
# The operator: a DeepONet branch(params) (x) trunk(time) -> xi in R^M.
# ===========================================================================
class MultivariateMBPPOperator:
    def __init__(self, M, t_grid, theta=1.0, p=32, hidden=64, seed=0):
        self.M = M
        self.t = np.asarray(t_grid, float)
        self.N = self.t.size
        self.theta = float(theta)
        self.p = p
        self.G = conv_matrix(self.t, theta)
        self.branch = MLP([M + M * M, hidden, hidden, p * M], seed=seed)
        self.trunk = MLP([1, hidden, hidden, p], seed=seed + 1)
        self.b0 = np.zeros(M)
        self._mb0 = np.zeros(M)
        self._vb0 = np.zeros(M)
        self._tb0 = 0
        self.pin_mean = None
        self.pin_std = None
        self.yscale = 1.0
        self.tq = None

    # -- helpers ------------------------------------------------------------
    def _params(self, S, A):
        return np.concatenate(
            [np.asarray(S, float), np.asarray(A, float).reshape(len(A), -1)], axis=1
        )

    def _forward(self, S, A):
        P = (self._params(S, A) - self.pin_mean) / self.pin_std
        C = self.branch.forward(P).reshape(-1, self.p, self.M)  # (b, p, M)
        Tr = self.trunk.forward(self.tq)  # (N, p)
        xi = np.einsum("bkm,nk->bnm", C, Tr) * self.yscale + self.b0  # (b, N, M)
        return xi, C, Tr

    def residual(self, xi, S, A):
        Gxi = np.einsum("nl,blm->bnm", self.G, xi)  # (b, N, M)
        exc = np.einsum("bnj,bmj->bnm", Gxi, np.asarray(A, float))  # sum_j A_{mj}(Gxi)_{n,j}
        return xi - np.asarray(S, float)[:, None, :] - exc

    # -- training -----------------------------------------------------------
    def train(
        self,
        S,
        A,
        epochs=4000,
        lr=2e-3,
        batch=64,
        anchors=None,
        data_weight=0.0,
        seed=0,
        val=None,
        log_every=0,
    ):
        """Physics-informed training on the MBPP residual (optionally + anchors).

        anchors : (S_a, A_a, XI_a) exact solutions to add a supervised MSE term.
        val     : (S_v, A_v) held-out instances; if given, prints/records rel-L2 vs exact.
        Returns a history dict.
        """
        S = np.asarray(S, float)
        A = np.asarray(A, float)
        self.pin_mean = self._params(S, A).mean(0)
        self.pin_std = self._params(S, A).std(0) + 1e-8
        self.yscale = float(S.mean() / (1.0 - 0.3))  # rough output scale
        self.tq = (self.t.reshape(-1, 1) - self.t.mean()) / (self.t.std() + 1e-8)
        rng = np.random.default_rng(seed)
        n = len(S)
        hist = {"epoch": [], "residual": [], "val_rel_l2": []}
        XIv = exact_solution(*val, self.t, self.theta) if val is not None else None

        for ep in range(epochs):
            cur_lr = lr * (0.3 ** (ep / max(1, epochs)))  # gentle exp decay to 0.3x
            idx = rng.choice(n, size=min(batch, n), replace=False)
            Sb, Ab = S[idx], A[idx]
            xi, C, Tr = self._forward(Sb, Ab)
            R = self.residual(xi, Sb, Ab)
            cnt = R.size
            # physics gradient w.r.t. xi:  (2/cnt) (R - G^T (R A))
            RA = np.einsum("bnm,bmj->bnj", R, Ab)
            GtRA = np.einsum("nl,bnj->blj", self.G, RA)
            dxi = (2.0 / cnt) * (R - GtRA)
            res_loss = float(np.mean(R**2))

            # optional supervised anchor term
            if anchors is not None and data_weight > 0:
                Sa, Aa, XIa = anchors
                xia, Ca, Tra = self._forward(Sa, Aa)
                Ra = xia - XIa
                dxi_a = (2.0 * data_weight / Ra.size) * Ra
                # accumulate by a second backward pass below (kept separate for clarity)
            else:
                dxi_a = None

            self._backward(dxi, C, Tr, cur_lr)
            if dxi_a is not None:
                xia, Ca, Tra = self._forward(Sa, Aa)
                self._backward(dxi_a, Ca, Tra, cur_lr)

            if val is not None and (log_every and ep % log_every == 0 or ep == epochs - 1):
                rel = self._rel_l2(val[0], val[1], XIv)
                hist["epoch"].append(ep)
                hist["residual"].append(res_loss)
                hist["val_rel_l2"].append(rel)
                if log_every:
                    print(f"  epoch {ep:5d}  residual={res_loss:.3e}  val rel-L2={rel:.4f}")
        return hist

    def _backward(self, dxi, C, Tr, lr):
        dyi = dxi * self.yscale
        dC = np.einsum("bnm,nk->bkm", dyi, Tr)  # (b, p, M)
        dTr = np.einsum("bnm,bkm->nk", dyi, C)  # (N, p)
        db0 = dxi.sum((0, 1))  # (M,)
        self.branch.backward(dC.reshape(dC.shape[0], -1))
        self.branch.adam_step(lr)
        self.trunk.backward(dTr)
        self.trunk.adam_step(lr)
        # adam on b0
        self._tb0 += 1
        self._mb0 = 0.9 * self._mb0 + 0.1 * db0
        self._vb0 = 0.999 * self._vb0 + 0.001 * db0 * db0
        self.b0 -= (
            lr
            * (self._mb0 / (1 - 0.9**self._tb0))
            / (np.sqrt(self._vb0 / (1 - 0.999**self._tb0)) + 1e-8)
        )

    # -- inference / evaluation --------------------------------------------
    def predict(self, S, A):
        xi, _, _ = self._forward(np.atleast_2d(S), np.atleast_3d(A).reshape(-1, self.M, self.M))
        return xi

    def _rel_l2(self, S, A, XI_exact=None):
        if XI_exact is None:
            XI_exact = exact_solution(S, A, self.t, self.theta)
        xi = self.predict(S, A)
        num = np.linalg.norm((xi - XI_exact).reshape(len(S), -1), axis=1)
        den = np.linalg.norm(XI_exact.reshape(len(S), -1), axis=1) + 1e-12
        return float(np.mean(num / den))

    # -- persistence --------------------------------------------------------
    def save(self, path):
        np.savez(
            path,
            bW=np.array(self.branch.W, dtype=object),
            bb=np.array(self.branch.b, dtype=object),
            tW=np.array(self.trunk.W, dtype=object),
            tb=np.array(self.trunk.b, dtype=object),
            b0=self.b0,
            pin_mean=self.pin_mean,
            pin_std=self.pin_std,
            yscale=self.yscale,
            t=self.t,
            theta=self.theta,
            p=self.p,
            M=self.M,
        )
