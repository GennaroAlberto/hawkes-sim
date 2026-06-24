r"""
JAX physics-informed neural solver for the MBPP equation (optional backend).

A *parametric* network ``N(t, p) ~ xi(t; p)`` is trained so that, for any model
parameter vector ``p`` in a prior range, it satisfies the MBPP Volterra equation
in residual form (no ground-truth solver needed).  After training it is a fast,
JIT-compiled, autodifferentiable forward map: ``solve(p)`` is a single network
evaluation and ``d xi / d p`` comes from autodiff -- so it can replace the
pure-Python ODE solve inside the fitter and remove both the time-stepping loop
and the finite-difference gradient.

Model family (the user's target, the non-closed-form case): univariate MBPP with
covariate-modulated exponential excitation,

    s(t) = mu,   K(t,u) = kappa*theta * exp(delta . Z(t)) * exp(-theta (t-u)),

with the covariate ``Z`` supplied as samples on a fixed collocation grid (so the
whole residual is a static, vectorised, JIT-friendly computation).  The free
parameter vector is ``p = [kappa, theta, mu, delta_1, ..., delta_q]``.

Requires ``jax``; this module is not imported by ``hawkes_calibration.__init__``.
It is a reference implementation (JAX is not available in the build environment,
so it is not executed here); the residual it minimises is validated in numpy in
:func:`hawkes_calibration.neural_solver.mbpp_volterra_residual`.
"""

from __future__ import annotations

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "neural_solver_jax requires JAX (`pip install jax jaxlib`). The rest of "
        "hawkes_calibration is numpy-only and does not need it."
    ) from exc


def _trapezoid_weight_matrix(t):
    """Lower-triangular trapezoid weights W with (W g)_i ~ int_0^{t_i} g."""
    t = np.asarray(t, float)
    M = t.size
    W = np.zeros((M, M))
    for i in range(1, M):
        for k in range(1, i + 1):
            h = t[k] - t[k - 1]
            W[i, k - 1] += 0.5 * h
            W[i, k] += 0.5 * h
    return W


class JAXNeuralMBPP:
    def __init__(self, mode="pinn", coll_grid=None, Z_on_grid=None, T=30.0,
                 width=64, depth=3, n_delta=1, seed=0,
                 kappa_range=(0.05, 0.9), theta_range=(0.2, 3.0),
                 mu_range=(0.5, 5.0), delta_range=(-2.0, 2.0), ic_weight=10.0,
                 stability_cap=0.95):
        self.mode = mode
        self.T = float(T)
        self.n_delta = int(n_delta)
        self.ranges = dict(kappa=kappa_range, theta=theta_range, mu=mu_range, delta=delta_range)
        self.ic_weight = float(ic_weight)
        # Excitation is modulated multiplicatively by exp(delta . Z), so the
        # *effective* branching ratio is kappa * exp(delta . Z).  If this reaches 1
        # the MBPP has no bounded solution (the target intensity diverges), which
        # poisons both the residual loss and any supervised anchor.  We clip the
        # sampled kappa so kappa * exp(max_t delta . Z(t)) <= stability_cap < 1.
        # Set stability_cap=None to disable (legacy behaviour).
        self.stability_cap = stability_cap

        if coll_grid is None:
            coll_grid = np.linspace(0.0, T, 128)
        self.t = jnp.asarray(coll_grid, float)
        self.W = jnp.asarray(_trapezoid_weight_matrix(np.asarray(coll_grid)), float)
        if Z_on_grid is None:
            Z_on_grid = np.zeros((coll_grid.size, n_delta))
        self.Z = jnp.asarray(np.atleast_2d(np.asarray(Z_on_grid, float)).reshape(coll_grid.size, -1), float)

        # parameter dimension fed to the network: [t, kappa, theta, mu, delta...]
        self.p_dim = 3 + self.n_delta
        sizes = [1 + self.p_dim] + [width] * depth + [1]
        key = jax.random.PRNGKey(seed)
        self.params = self._init_mlp(sizes, key)

    # -- MLP -------------------------------------------------------------
    def _init_mlp(self, sizes, key):
        params = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            key, k = jax.random.split(key)
            W = jax.random.normal(k, (a, b)) * np.sqrt(2.0 / a)
            params.append((W, jnp.zeros(b)))
        return params

    @staticmethod
    def _mlp(params, x):
        for (W, b) in params[:-1]:
            x = jnp.tanh(x @ W + b)
        W, b = params[-1]
        return x @ W + b                                   # (B, 1)

    # -- network intensity on the collocation grid for one p -------------
    def _xi_grid(self, params, p):
        # p = [kappa, theta, mu, delta...]; build network inputs (M, 1+p_dim)
        M = self.t.shape[0]
        tt = (self.t / self.T)[:, None]                    # scaled time
        pmat = jnp.broadcast_to(p[None, :], (M, p.shape[0]))
        x = jnp.concatenate([tt, pmat], axis=1)
        return self._mlp(params, x)[:, 0]                  # (M,)

    def _kernel_matrix(self, p):
        kappa, theta = p[0], p[1]
        delta = jax.lax.dynamic_slice(p, (3,), (self.n_delta,))
        mod = jnp.exp(jnp.clip(self.Z @ delta, -30.0, 30.0))         # (M,)
        dt = self.t[:, None] - self.t[None, :]                       # (M, M)
        K = kappa * theta * mod[:, None] * jnp.exp(-theta * dt)
        return jnp.tril(K)                                           # causal/lower-tri

    def _residual_loss_one(self, params, p, reweight=False):
        xi = self._xi_grid(params, p)
        mu = p[2]
        K = self._kernel_matrix(p)
        integ = (self.W * K) @ xi
        R = xi - mu - integ
        if reweight:                       # normalise by the solution scale so
            R = R / (jnp.sqrt(jnp.mean(xi ** 2)) + 1e-3)   # stiff (large-xi) cases don't dominate
        ic = (xi[0] - mu) ** 2
        return jnp.mean(R ** 2) + self.ic_weight * ic

    def _sample_params(self, key, batch, kappa_max=None):
        kmax = self.ranges["kappa"][1] if kappa_max is None else kappa_max
        ks = jax.random.uniform(key, (batch,), minval=self.ranges["kappa"][0], maxval=kmax)
        key, k = jax.random.split(key)
        th = jax.random.uniform(k, (batch,), minval=self.ranges["theta"][0], maxval=self.ranges["theta"][1])
        key, k = jax.random.split(key)
        mu = jax.random.uniform(k, (batch,), minval=self.ranges["mu"][0], maxval=self.ranges["mu"][1])
        key, k = jax.random.split(key)
        dl = jax.random.uniform(k, (batch, self.n_delta),
                                minval=self.ranges["delta"][0], maxval=self.ranges["delta"][1])
        if self.stability_cap is not None:
            # max_t (delta . Z(t)) <= sum_q |delta_q| * max_t |Z_q(t)|  (conservative)
            zmax = jnp.max(jnp.abs(self.Z), axis=0)                      # (n_delta,)
            expo = jnp.sum(jnp.abs(dl) * zmax[None, :], axis=1)          # (batch,)
            ks = jnp.minimum(ks, self.stability_cap / jnp.exp(expo))
        return jnp.concatenate([ks[:, None], th[:, None], mu[:, None], dl], axis=1)

    def train(self, n_steps=20000, batch=64, lr=1e-3, seed=0,
              anchors=None, data_weight=1.0, curriculum=False, reweight=False):
        r"""
        Train the PINN.  Options to stabilise the hard (stiff, near-critical) regime:

        * ``anchors=(P_anchor, XI_anchor)`` -- a few exact solutions (from
          :func:`hawkes_calibration.neural_solver.make_anchor_data`) added as a
          supervised MSE term weighted by ``data_weight`` (the *hybrid* loss);
        * ``curriculum=True`` -- sample kappa from an easy range first and widen
          towards criticality over training;
        * ``reweight=True`` -- normalise the residual by the per-example solution
          scale so large-intensity (stiff) cases do not dominate the loss.
        """
        opt_state = [(jnp.zeros_like(W), jnp.zeros_like(b),
                      jnp.zeros_like(W), jnp.zeros_like(b)) for (W, b) in self.params]
        has_data = anchors is not None
        if has_data:
            Pa = jnp.asarray(np.asarray(anchors[0], float))
            XIa = jnp.asarray(np.asarray(anchors[1], float))

        def data_loss(params):
            if not has_data:
                return 0.0
            preds = jax.vmap(lambda p: self._xi_grid(params, p))(Pa)    # (Ka, M)
            return jnp.mean((preds - XIa) ** 2)

        def batched_loss(params, P):
            res = jnp.mean(jax.vmap(lambda p: self._residual_loss_one(params, p, reweight))(P))
            return res + data_weight * data_loss(params)

        loss_grad = jax.jit(jax.value_and_grad(batched_loss))

        @jax.jit
        def adam_step(params, state, grads, t, lr):
            new_p, new_s = [], []
            for (W, b), (mW, mb, vW, vb), (gW, gb) in zip(params, state, grads):
                mW = 0.9 * mW + 0.1 * gW; vW = 0.999 * vW + 0.001 * gW ** 2
                mb = 0.9 * mb + 0.1 * gb; vb = 0.999 * vb + 0.001 * gb ** 2
                bc1 = 1 - 0.9 ** t; bc2 = 1 - 0.999 ** t
                W = W - lr * (mW / bc1) / (jnp.sqrt(vW / bc2) + 1e-8)
                b = b - lr * (mb / bc1) / (jnp.sqrt(vb / bc2) + 1e-8)
                new_p.append((W, b)); new_s.append((mW, mb, vW, vb))
            return new_p, new_s

        key = jax.random.PRNGKey(seed)
        k_lo, k_hi = self.ranges["kappa"]
        for step in range(1, n_steps + 1):
            key, k = jax.random.split(key)
            if curriculum:                                   # widen kappa easy -> stiff
                frac = min(1.0, step / (0.7 * n_steps))
                kmax = min(k_hi, max(k_lo + 0.1, 0.5) + (k_hi - 0.5) * frac)
            else:
                kmax = k_hi
            P = self._sample_params(k, batch, kappa_max=kmax)
            loss, grads = loss_grad(self.params, P)
            self.params, opt_state = adam_step(self.params, opt_state, grads, step, lr)
        return self

    # -- inference: fast differentiable forward solve --------------------
    def solve(self, params, t_grid):
        """xi(t) for the parameter dict, as a vectorised network evaluation."""
        p = self._params_vec(params)
        t = jnp.asarray(t_grid, float)
        x = jnp.concatenate([(t / self.T)[:, None],
                             jnp.broadcast_to(p[None, :], (t.shape[0], p.shape[0]))], axis=1)
        return np.asarray(self._mlp(self.params, x)[:, 0])

    def compensator(self, params, t_grid):
        t = np.asarray(t_grid, float)
        xi = self.solve(params, t)
        return np.concatenate([[0.0], np.cumsum(0.5 * (xi[1:] + xi[:-1]) * np.diff(t))])

    def _params_vec(self, params):
        delta = np.atleast_1d(np.asarray(params.get("delta", [0.0]), float)).reshape(-1)
        return jnp.asarray(np.concatenate([[params["kappa"], params["theta"], params.get("mu", 1.0)], delta]))

    # -- the payoff: an autodiff intensity usable inside a JAX optimiser --
    def intensity_fn(self):
        """
        Return a pure jitted function ``f(p_vec, t) -> xi`` differentiable in
        ``p_vec`` -- drop this into a JAX/optax interval-censored fit to get exact
        gradients with no finite differences and no Python ODE loop.
        """
        net = self.params
        T = self.T

        @jax.jit
        def f(p_vec, t):
            x = jnp.concatenate([(t / T)[:, None],
                                 jnp.broadcast_to(p_vec[None, :], (t.shape[0], p_vec.shape[0]))], axis=1)
            return JAXNeuralMBPP._mlp(net, x)[:, 0]

        return f


class JAXDeepONetPINO:
    r"""
    Physics-Informed **Neural Operator** for the MBPP family (JAX).

    Unlike the parametric PINN (which fixes the covariate path and varies only the
    scalar parameters), this DeepONet trains over the whole *family*: the branch
    ingests a sampled covariate path ``Z(.)`` (at the collocation grid) together
    with the scalar parameters, the trunk encodes the query time, and the operator
    is trained to drive the MBPP Volterra residual to zero over a *distribution*
    of covariate paths and parameters.  One trained operator then solves any new
    instance (any ``Z``, any ``p``) in a single forward pass.

        xi(t) = sum_k branch_k([Z_sensors, kappa, theta, mu, delta]) * trunk_k(t).
    """

    def __init__(self, coll_grid=None, T=30.0, p_latent=64, width=96, depth=3, seed=0,
                 kappa_range=(0.05, 0.9), theta_range=(0.2, 3.0), mu_range=(0.5, 5.0),
                 delta_range=(-2.0, 2.0), z_steps=5, ic_weight=10.0, stability_cap=0.95):
        self.T = float(T)
        self.stability_cap = stability_cap   # clip kappa to keep kappa*exp(delta.Z) < 1
        if coll_grid is None:
            coll_grid = np.linspace(0.0, T, 96)
        self.t = jnp.asarray(coll_grid, float)
        self.M = int(coll_grid.size)
        self.W = jnp.asarray(_trapezoid_weight_matrix(np.asarray(coll_grid)), float)
        self.ranges = dict(kappa=kappa_range, theta=theta_range, mu=mu_range, delta=delta_range)
        self.z_steps = int(z_steps); self.ic_weight = float(ic_weight)
        self.p_latent = int(p_latent)
        key = jax.random.PRNGKey(seed)
        k1, k2 = jax.random.split(key)
        self.branch = JAXNeuralMBPP._init_mlp(self, [self.M + 4] + [width] * depth + [p_latent], k1)
        self.trunk = JAXNeuralMBPP._init_mlp(self, [1] + [width] * depth + [p_latent], k2)
        self.b0 = jnp.array(0.0)

    def _xi_grid(self, branch_w, trunk_w, b0, Z, p):
        binp = jnp.concatenate([Z, p])[None, :]                 # (1, M+4)
        bcoef = JAXNeuralMBPP._mlp(branch_w, binp)[0]           # (p_latent,)
        Tr = JAXNeuralMBPP._mlp(trunk_w, (self.t / self.T)[:, None])   # (M, p_latent)
        return Tr @ bcoef + b0                                  # (M,)

    def _loss_one(self, branch_w, trunk_w, b0, Z, p, reweight=False):
        xi = self._xi_grid(branch_w, trunk_w, b0, Z, p)
        kappa, theta, mu, delta = p[0], p[1], p[2], p[3]
        mod = jnp.exp(jnp.clip(delta * Z, -30.0, 30.0))
        dt = self.t[:, None] - self.t[None, :]
        K = jnp.tril(kappa * theta * mod[:, None] * jnp.exp(-theta * dt))
        integ = (self.W * K) @ xi
        R = xi - mu - integ
        if reweight:
            R = R / (jnp.sqrt(jnp.mean(xi ** 2)) + 1e-3)
        return jnp.mean(R ** 2) + self.ic_weight * (xi[0] - mu) ** 2

    def _sample_batch(self, key, batch, kappa_max=None):
        kmax = self.ranges["kappa"][1] if kappa_max is None else kappa_max
        # random params
        keys = jax.random.split(key, 6)
        ka = jax.random.uniform(keys[0], (batch,), minval=self.ranges["kappa"][0], maxval=kmax)
        th = jax.random.uniform(keys[1], (batch,), minval=self.ranges["theta"][0], maxval=self.ranges["theta"][1])
        mu = jax.random.uniform(keys[2], (batch,), minval=self.ranges["mu"][0], maxval=self.ranges["mu"][1])
        dl = jax.random.uniform(keys[3], (batch,), minval=self.ranges["delta"][0], maxval=self.ranges["delta"][1])
        # random piecewise-constant covariate paths (via numpy; static per step)
        Zb = jnp.asarray(np.stack([
            _random_step_path(np.asarray(self.t), self.z_steps, int(s))
            for s in np.asarray(jax.random.randint(keys[4], (batch,), 0, 2 ** 31 - 1))
        ]))
        if self.stability_cap is not None:
            # clip kappa per sample so kappa * exp(delta * max_t|Z(t)|) <= cap < 1
            zmax = jnp.max(jnp.abs(Zb), axis=1)                          # (batch,)
            ka = jnp.minimum(ka, self.stability_cap / jnp.exp(jnp.abs(dl) * zmax))
        P = jnp.stack([ka, th, mu, dl], axis=1)
        return Zb, P

    def train(self, n_steps=20000, batch=32, lr=1e-3, seed=0,
              anchors=None, data_weight=1.0, curriculum=False, reweight=False):
        r"""
        Train the operator on the family.  As for the PINN, ``anchors=(Z_anchor,
        P_anchor, XI_anchor)`` adds a supervised MSE term (the *hybrid* loss --
        recommended for a wide family / near criticality), ``curriculum=True``
        widens kappa from easy to stiff, and ``reweight=True`` normalises the
        residual by the solution scale.
        """
        state = self._adam_init()
        has_data = anchors is not None
        if has_data:
            Za = jnp.asarray(np.asarray(anchors[0], float))
            Pa = jnp.asarray(np.asarray(anchors[1], float))
            XIa = jnp.asarray(np.asarray(anchors[2], float))

        def data_loss(bw, tw, b0):
            if not has_data:
                return 0.0
            preds = jax.vmap(lambda Z, p: self._xi_grid(bw, tw, b0, Z, p))(Za, Pa)
            return jnp.mean((preds - XIa) ** 2)

        def batched_loss(bw, tw, b0, Zb, P):
            res = jnp.mean(jax.vmap(lambda Z, p: self._loss_one(bw, tw, b0, Z, p, reweight))(Zb, P))
            return res + data_weight * data_loss(bw, tw, b0)

        vg = jax.jit(jax.value_and_grad(batched_loss, argnums=(0, 1, 2)))
        key = jax.random.PRNGKey(seed)
        k_lo, k_hi = self.ranges["kappa"]
        for step in range(1, n_steps + 1):
            key, k = jax.random.split(key)
            if curriculum:
                frac = min(1.0, step / (0.7 * n_steps))
                kmax = min(k_hi, max(k_lo + 0.1, 0.5) + (k_hi - 0.5) * frac)
            else:
                kmax = k_hi
            Zb, P = self._sample_batch(k, batch, kappa_max=kmax)
            loss, grads = vg(self.branch, self.trunk, self.b0, Zb, P)
            (self.branch, self.trunk, self.b0), state = self._adam_apply(
                (self.branch, self.trunk, self.b0), state, grads, step, lr)
        return self

    def solve(self, params, t_grid, Z_on_grid):
        """xi(t) for a covariate path ``Z_on_grid`` (sampled on the collocation grid)
        and scalar params -- one forward pass, any instance in the family."""
        p = jnp.asarray([params["kappa"], params["theta"], params.get("mu", 1.0),
                         np.atleast_1d(params.get("delta", [0.0]))[0]], float)
        Z = jnp.asarray(Z_on_grid, float)
        bcoef = JAXNeuralMBPP._mlp(self.branch, jnp.concatenate([Z, p])[None, :])[0]
        t = jnp.asarray(t_grid, float)
        Tr = JAXNeuralMBPP._mlp(self.trunk, (t / self.T)[:, None])
        return np.asarray(Tr @ bcoef + self.b0)

    # -- tiny Adam over the (branch, trunk, b0) pytree -------------------
    def _adam_init(self):
        def z(p):
            return [(jnp.zeros_like(W), jnp.zeros_like(b)) for (W, b) in p]
        return dict(mb=z(self.branch), vb=z(self.branch), mt=z(self.trunk), vt=z(self.trunk),
                    mb0=jnp.array(0.0), vb0=jnp.array(0.0))

    def _adam_apply(self, params, state, grads, t, lr):
        (bw, tw, b0) = params
        (gb, gt, gb0) = grads
        def upd(ws, gs, ms, vs):
            nw, nm, nv = [], [], []
            for (W, b), (gW, gb_), (mW, mb), (vW, vb) in zip(ws, gs, ms, vs):
                mW = 0.9 * mW + 0.1 * gW; vW = 0.999 * vW + 0.001 * gW ** 2
                mb = 0.9 * mb + 0.1 * gb_; vb = 0.999 * vb + 0.001 * gb_ ** 2
                c1 = 1 - 0.9 ** t; c2 = 1 - 0.999 ** t
                W = W - lr * (mW / c1) / (jnp.sqrt(vW / c2) + 1e-8)
                b = b - lr * (mb / c1) / (jnp.sqrt(vb / c2) + 1e-8)
                nw.append((W, b)); nm.append((mW, mb)); nv.append((vW, vb))
            return nw, nm, nv
        bw, state["mb"], state["vb"] = upd(bw, gb, state["mb"], state["vb"])
        tw, state["mt"], state["vt"] = upd(tw, gt, state["mt"], state["vt"])
        state["mb0"] = 0.9 * state["mb0"] + 0.1 * gb0; state["vb0"] = 0.999 * state["vb0"] + 0.001 * gb0 ** 2
        b0 = b0 - lr * (state["mb0"] / (1 - 0.9 ** t)) / (jnp.sqrt(state["vb0"] / (1 - 0.999 ** t)) + 1e-8)
        return (bw, tw, b0), state


def _random_step_path(t, n_steps, seed):
    rng = np.random.default_rng(seed)
    edges = np.sort(rng.uniform(0, t[-1], n_steps - 1))
    levels = rng.uniform(-0.5, 0.5, n_steps)
    return levels[np.searchsorted(edges, t)]
