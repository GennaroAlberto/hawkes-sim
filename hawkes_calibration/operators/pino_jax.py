r"""
JAX physics-informed neural operator (PINO) for the **multivariate** MBPP.

This is the fast, jit/autodiff counterpart of the numpy
:class:`hawkes_calibration.operators.pino.MultivariateMBPPOperator`.  It learns the
same solution operator

    (s, A) |-> xi,    xi(t) in R^M   solving   xi = s + A (G xi),
    (G xi)(t) = int_0^t e^{-theta (t-u)} xi(u) du,

as a DeepONet (branch on the instance ``(s, vec A)``, trunk on the query time).
Training minimises a **hybrid, scale-invariant** objective:

* a *relative* collocation residual  ||R||^2 / ||xi||^2   (physics, no solver),
* an optional *relative* supervised term ||xi - xi*||^2 / ||xi*||^2 on a modest set
  of exact "anchor" solutions (cheap to make with the numpy ODE solver),
* an initial-condition penalty  ||xi(0) - s||^2.

Optimising the *relative* error directly (i) targets the held-out rel-L2 metric and
(ii) stops the stiff, large-intensity instances from dominating the loss -- the two
things that capped the plain numpy PINO.  Autodiff supplies the residual gradient,
so no hand-derived backprop is needed.

Requires JAX + optax; not imported by ``hawkes_calibration.__init__``.
"""

from __future__ import annotations

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    import optax
except Exception as exc:  # pragma: no cover
    raise ImportError("pino_jax requires `jax` and `optax`.") from exc

from .pino import conv_matrix, exact_solution  # reuse the problem def


# ---------------------------------------------------------------------------
# small MLP as an explicit pytree of (W, b)
# ---------------------------------------------------------------------------
def _init_mlp(sizes, key, scale=None):
    params = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        key, k = jax.random.split(key)
        s = scale if scale is not None else np.sqrt(2.0 / (a + b))
        params.append((jax.random.normal(k, (a, b)) * s, jnp.zeros(b)))
    return params


def _mlp(params, x, act=jnp.tanh):
    for W, b in params[:-1]:
        x = act(x @ W + b)
    W, b = params[-1]
    return x @ W + b


class JAXMultivariateMBPPOperator:
    def __init__(self, M, t_grid, theta=1.0, p=64, hidden=160, depth=3, n_fourier=16, seed=0):
        self.M = int(M)
        self._seed = int(seed)
        self.t = np.asarray(t_grid, float)
        self.N = self.t.size
        self.theta = float(theta)
        self.p = int(p)
        self.G = jnp.asarray(conv_matrix(self.t, theta))
        # trunk input: scaled time + random Fourier features (helps represent the
        # exponential-relaxation shapes with a small MLP)
        tt = (self.t - self.t.mean()) / (self.t.std() + 1e-8)
        rng = np.random.default_rng(seed)
        freqs = rng.normal(0, 1.5, size=n_fourier)
        feats = [tt[:, None]]
        feats += [np.sin(tt[:, None] * f) for f in freqs]
        feats += [np.cos(tt[:, None] * f) for f in freqs]
        self.tq = jnp.asarray(np.concatenate(feats, axis=1))  # (N, 1+2F)
        trunk_in = self.tq.shape[1]

        key = jax.random.PRNGKey(seed)
        kb, ktr = jax.random.split(key)
        self.branch = _init_mlp([M + M * M] + [hidden] * depth + [p * M], kb)
        self.trunk = _init_mlp([trunk_in] + [hidden] * depth + [p], ktr)
        self.b0 = jnp.zeros(M)
        self.pin_mean = None
        self.pin_std = None
        self.yscale = 1.0

    # -- core forward (single instance) -------------------------------------
    def _xi_one(self, branch, trunk, b0, p_vec):
        C = _mlp(branch, p_vec[None, :])[0].reshape(self.p, self.M)  # (p, M)
        Tr = _mlp(trunk, self.tq)  # (N, p)
        return (Tr @ C) * self.yscale + b0  # (N, M)

    def _params_vec(self, S, A):
        P = np.concatenate([np.asarray(S, float), np.asarray(A, float).reshape(len(A), -1)], axis=1)
        return (P - self.pin_mean) / self.pin_std

    # -- training -----------------------------------------------------------
    def train(
        self,
        S,
        A,
        epochs=8000,
        lr=2e-3,
        batch=128,
        anchors=None,
        w_res=1.0,
        w_anchor=1.0,
        w_ic=1.0,
        seed=0,
        val=None,
        log_every=0,
    ):
        S = np.asarray(S, float)
        A = np.asarray(A, float)
        Praw = np.concatenate([S, A.reshape(len(A), -1)], axis=1)
        self.pin_mean = Praw.mean(0)
        self.pin_std = Praw.std(0) + 1e-8
        self.yscale = float(S.mean() / 0.7)

        P = jnp.asarray(self._params_vec(S, A))
        Sj = jnp.asarray(S)
        Aj = jnp.asarray(A)
        n = len(S)

        has_anchor = anchors is not None and w_anchor > 0
        if has_anchor:
            Sa, Aa, XIa = anchors
            Pa = jnp.asarray(self._params_vec(Sa, Aa))
            Saj = jnp.asarray(Sa)
            Aaj = jnp.asarray(Aa)
            XIaj = jnp.asarray(XIa)
            na = len(Sa)

        G = self.G

        def residual(xi, s, a):
            Gxi = G @ xi  # (N, M)
            exc = Gxi @ a.T  # sum_j a_{mj}(Gxi)_j
            return xi - s[None, :] - exc

        def loss_fn(params, pb, sb, ab, pa, sa, aa, xia):
            branch, trunk, b0 = params
            xi = jax.vmap(lambda pv: self._xi_one(branch, trunk, b0, pv))(pb)  # (B,N,M)
            R = jax.vmap(residual)(xi, sb, ab)
            rel_res = jnp.mean(jnp.sum(R**2, (1, 2)) / (jnp.sum(xi**2, (1, 2)) + 1e-6))
            ic = jnp.mean((xi[:, 0, :] - sb) ** 2)
            tot = w_res * rel_res + w_ic * ic
            if has_anchor:
                xa = jax.vmap(lambda pv: self._xi_one(branch, trunk, b0, pv))(pa)
                rel_a = jnp.mean(
                    jnp.sum((xa - xia) ** 2, (1, 2)) / (jnp.sum(xia**2, (1, 2)) + 1e-6)
                )
                tot = tot + w_anchor * rel_a
            return tot

        sched = optax.cosine_decay_schedule(lr, epochs, alpha=0.05)
        # Gradient clipping tames the divergence spikes that otherwise hit the
        # higher-M (stiff, near-critical) instances mid-training.
        opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched))
        params = (self.branch, self.trunk, self.b0)
        opt_state = opt.init(params)
        loss_grad = jax.jit(jax.value_and_grad(loss_fn))

        key = jax.random.PRNGKey(seed)
        hist = {"epoch": [], "loss": [], "val_rel_l2": []}
        XIv = exact_solution(*val, self.t, self.theta) if val is not None else None
        # keep-best-on-validation: training can still spike; we return the best checkpoint
        best_val = np.inf
        best_params = params
        val_every = max(1, epochs // 40)  # ~40 validation checks for keep-best

        for ep in range(epochs):
            key, k = jax.random.split(key)
            idx = jax.random.choice(k, n, (min(batch, n),), replace=False)
            if has_anchor:
                key, k2 = jax.random.split(key)
                ia = jax.random.choice(k2, na, (min(batch, na),), replace=False)
                args = (P[idx], Sj[idx], Aj[idx], Pa[ia], Saj[ia], Aaj[ia], XIaj[ia])
            else:
                args = (P[idx], Sj[idx], Aj[idx], P[idx], Sj[idx], Aj[idx], Sj[idx][:, None, :])
            loss, grads = loss_grad(params, *args)
            updates, opt_state = opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            if val is not None and (ep % val_every == 0 or ep == epochs - 1):
                self.branch, self.trunk, self.b0 = params
                rel = self._rel_l2(val[0], val[1], XIv)
                hist["epoch"].append(ep)
                hist["loss"].append(float(loss))
                hist["val_rel_l2"].append(rel)
                if rel < best_val:
                    best_val = rel
                    best_params = jax.tree_util.tree_map(lambda x: x, params)
                if log_every:
                    print(
                        f"  epoch {ep:5d}  loss={float(loss):.3e}  val rel-L2={rel:.4f}"
                        f"  (best {best_val:.4f})",
                        flush=True,
                    )
        # keep-best only when validation was tracked; otherwise keep the final params
        self.branch, self.trunk, self.b0 = best_params if val is not None else params
        return hist

    # -- inference ----------------------------------------------------------
    def predict(self, S, A):
        S = np.atleast_2d(np.asarray(S, float))
        A = np.asarray(A, float).reshape(-1, self.M, self.M)
        P = jnp.asarray(self._params_vec(S, A))
        xi = jax.vmap(lambda pv: self._xi_one(self.branch, self.trunk, self.b0, pv))(P)
        return np.asarray(xi)

    def _rel_l2(self, S, A, XI_exact=None):
        if XI_exact is None:
            XI_exact = exact_solution(S, A, self.t, self.theta)
        xi = self.predict(S, A)
        num = np.linalg.norm((xi - XI_exact).reshape(len(S), -1), axis=1)
        den = np.linalg.norm(XI_exact.reshape(len(S), -1), axis=1) + 1e-12
        return float(np.mean(num / den))

    def save(self, path):
        flat = {}
        for i, (W, b) in enumerate(self.branch):
            flat[f"bW{i}"] = np.asarray(W)
            flat[f"bb{i}"] = np.asarray(b)
        for i, (W, b) in enumerate(self.trunk):
            flat[f"tW{i}"] = np.asarray(W)
            flat[f"tb{i}"] = np.asarray(b)
        np.savez(
            path,
            b0=np.asarray(self.b0),
            pin_mean=self.pin_mean,
            pin_std=self.pin_std,
            yscale=self.yscale,
            t=self.t,
            theta=self.theta,
            p=self.p,
            M=self.M,
            n_fourier=(self.tq.shape[1] - 1) // 2,
            seed_used=self._seed,
            n_branch=len(self.branch),
            n_trunk=len(self.trunk),
            **flat,
        )

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=False)
        op = cls(
            int(d["M"]),
            d["t"],
            theta=float(d["theta"]),
            p=int(d["p"]),
            n_fourier=int(d["n_fourier"]),
            seed=int(d["seed_used"]),
        )
        op.branch = [
            (jnp.asarray(d[f"bW{i}"]), jnp.asarray(d[f"bb{i}"])) for i in range(int(d["n_branch"]))
        ]
        op.trunk = [
            (jnp.asarray(d[f"tW{i}"]), jnp.asarray(d[f"tb{i}"])) for i in range(int(d["n_trunk"]))
        ]
        op.b0 = jnp.asarray(d["b0"])
        op.pin_mean = d["pin_mean"]
        op.pin_std = d["pin_std"]
        op.yscale = float(d["yscale"])
        return op
