r"""
TensorFlow physics-informed neural solver for the MBPP equation (optional backend).

The TensorFlow twin of :mod:`hawkes_calibration.neural_solver_jax`: the same
parametric network ``N(t, p) ~ xi(t; p)`` trained on the MBPP Volterra residual,
behind the same interface (``make_neural_solver(backend="tensorflow")``).  Once
trained it is a fast, ``tf.function``-compiled, autodifferentiable forward map,
so it replaces the pure-Python ODE solve in the optimiser and removes both the
time-stepping loop and the finite-difference gradient.

Model family: univariate MBPP with covariate-modulated exponential excitation,
``K(t,u)=kappa*theta*exp(delta . Z(t))*exp(-theta(t-u))``, baseline ``mu``;
free parameter vector ``p=[kappa, theta, mu, delta...]``, covariate ``Z`` sampled
on a fixed collocation grid.

Requires ``tensorflow``; not imported by ``hawkes_calibration.__init__``.  Reference
implementation (TensorFlow is not available in the build environment, so it is not
executed here); the residual it minimises is validated in numpy in
:func:`hawkes_calibration.neural_solver.mbpp_volterra_residual`.
"""

from __future__ import annotations

import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras import layers
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "neural_solver_tf requires TensorFlow (`pip install tensorflow`). The rest "
        "of hawkes_calibration is numpy-only and does not need it."
    ) from exc


def _trapezoid_weight_matrix(t):
    t = np.asarray(t, float)
    M = t.size
    W = np.zeros((M, M))
    for i in range(1, M):
        for k in range(1, i + 1):
            h = t[k] - t[k - 1]
            W[i, k - 1] += 0.5 * h
            W[i, k] += 0.5 * h
    return W


class TFNeuralMBPP:
    def __init__(
        self,
        mode="pinn",
        coll_grid=None,
        Z_on_grid=None,
        T=30.0,
        width=64,
        depth=3,
        n_delta=1,
        seed=0,
        kappa_range=(0.05, 0.9),
        theta_range=(0.2, 3.0),
        mu_range=(0.5, 5.0),
        delta_range=(-2.0, 2.0),
        ic_weight=10.0,
    ):
        tf.random.set_seed(seed)
        self.mode = mode
        self.T = float(T)
        self.n_delta = int(n_delta)
        self.ranges = dict(kappa=kappa_range, theta=theta_range, mu=mu_range, delta=delta_range)
        self.ic_weight = float(ic_weight)

        if coll_grid is None:
            coll_grid = np.linspace(0.0, T, 128)
        self.t = tf.constant(coll_grid, tf.float32)
        self.M = int(coll_grid.size)
        self.W = tf.constant(_trapezoid_weight_matrix(np.asarray(coll_grid)), tf.float32)
        if Z_on_grid is None:
            Z_on_grid = np.zeros((self.M, n_delta))
        self.Z = tf.constant(
            np.atleast_2d(np.asarray(Z_on_grid, float)).reshape(self.M, -1), tf.float32
        )
        # pairwise time differences t_i - t_j
        self.dt = self.t[:, None] - self.t[None, :]
        self.tril = tf.constant(np.tril(np.ones((self.M, self.M))), tf.float32)

        self.p_dim = 3 + self.n_delta
        self.net = tf.keras.Sequential(
            [layers.Input(shape=(1 + self.p_dim,))]
            + [layers.Dense(width, activation="tanh") for _ in range(depth)]
            + [layers.Dense(1)]
        )
        self.opt = tf.keras.optimizers.Adam(1e-3)

    # -- batched PINN residual loss --------------------------------------
    def _xi_on_grid(self, P):
        # P: (B, p_dim).  Returns xi on the collocation grid: (B, M)
        B = tf.shape(P)[0]
        tt = tf.tile((self.t / self.T)[None, :, None], [B, 1, 1])  # (B, M, 1)
        pp = tf.tile(P[:, None, :], [1, self.M, 1])  # (B, M, p_dim)
        x = tf.reshape(tf.concat([tt, pp], axis=2), (-1, 1 + self.p_dim))
        return tf.reshape(self.net(x), (B, self.M))

    def _loss(self, P, reweight=False):
        xi = self._xi_on_grid(P)  # (B, M)
        kappa = P[:, 0:1]
        theta = P[:, 1:2]
        mu = P[:, 2:3]
        delta = P[:, 3 : 3 + self.n_delta]  # (B, q)
        mod = tf.exp(
            tf.clip_by_value(tf.matmul(delta, self.Z, transpose_b=True), -30.0, 30.0)
        )  # (B, M)
        K = (
            kappa[:, :, None]
            * theta[:, :, None]
            * mod[:, :, None]
            * tf.exp(-theta[:, :, None] * self.dt[None])
        )  # (B, M, M)
        K = K * self.tril[None]
        integ = tf.matmul(self.W[None] * K, xi[:, :, None])[:, :, 0]  # (B, M)
        R = xi - mu - integ
        if reweight:
            R = R / (tf.sqrt(tf.reduce_mean(xi**2, axis=1, keepdims=True)) + 1e-3)
        ic = tf.reduce_mean((xi[:, 0:1] - mu) ** 2)
        return tf.reduce_mean(R**2) + self.ic_weight * ic

    def _sample(self, batch, kappa_max=None):
        kmax = self.ranges["kappa"][1] if kappa_max is None else kappa_max

        def u(lo, hi, shape):
            return tf.random.uniform(shape, lo, hi)

        r = self.ranges
        return tf.concat(
            [
                u(r["kappa"][0], kmax, (batch, 1)),
                u(*r["theta"], (batch, 1)),
                u(*r["mu"], (batch, 1)),
                u(*r["delta"], (batch, self.n_delta)),
            ],
            axis=1,
        )

    def train(
        self,
        n_steps=20000,
        batch=64,
        lr=1e-3,
        anchors=None,
        data_weight=1.0,
        curriculum=False,
        reweight=False,
    ):
        r"""
        Train the PINN.  ``anchors=(P_anchor, XI_anchor)`` adds a supervised MSE
        term (hybrid loss) weighted by ``data_weight``; ``curriculum=True`` widens
        kappa from easy to stiff; ``reweight=True`` normalises the residual by the
        solution scale.
        """
        self.opt.learning_rate.assign(lr)
        has_data = anchors is not None
        if has_data:
            Pa = tf.constant(np.asarray(anchors[0], np.float32))
            XIa = tf.constant(np.asarray(anchors[1], np.float32))

        @tf.function
        def step(P):
            with tf.GradientTape() as tape:
                loss = self._loss(P, reweight)
                if has_data:
                    loss = loss + data_weight * tf.reduce_mean((self._xi_on_grid(Pa) - XIa) ** 2)
            grads = tape.gradient(loss, self.net.trainable_variables)
            self.opt.apply_gradients(zip(grads, self.net.trainable_variables))
            return loss

        k_lo, k_hi = self.ranges["kappa"]
        for s in range(n_steps):
            if curriculum:
                frac = min(1.0, s / (0.7 * n_steps))
                kmax = min(k_hi, max(k_lo + 0.1, 0.5) + (k_hi - 0.5) * frac)
            else:
                kmax = k_hi
            step(self._sample(batch, kmax))
        return self

    # -- inference -------------------------------------------------------
    def _p_vec(self, params):
        delta = np.atleast_1d(np.asarray(params.get("delta", [0.0]), float)).reshape(-1)
        return np.concatenate(
            [[params["kappa"], params["theta"], params.get("mu", 1.0)], delta]
        ).astype(np.float32)

    def solve(self, params, t_grid):
        t = np.asarray(t_grid, np.float32)
        p = self._p_vec(params)
        x = np.concatenate([(t / self.T)[:, None], np.tile(p[None, :], (t.size, 1))], axis=1)
        return self.net(tf.constant(x)).numpy()[:, 0]

    def compensator(self, params, t_grid):
        t = np.asarray(t_grid, float)
        xi = self.solve(params, t)
        return np.concatenate([[0.0], np.cumsum(0.5 * (xi[1:] + xi[:-1]) * np.diff(t))])

    def intensity_fn(self):
        """A tf.function f(p_vec, t) -> xi, differentiable in p_vec (autodiff) for
        use inside a TF interval-censored fit (no finite differences, no ODE loop)."""
        net, T = self.net, self.T

        @tf.function
        def f(p_vec, t):
            x = tf.concat([(t / T)[:, None], tf.tile(p_vec[None, :], [tf.shape(t)[0], 1])], axis=1)
            return net(x)[:, 0]

        return f


class TFDeepONetPINO:
    r"""
    Physics-Informed **Neural Operator** for the MBPP family (TensorFlow).

    The TF twin of :class:`hawkes_calibration.neural_solver_jax.JAXDeepONetPINO`:
    a DeepONet whose branch ingests a sampled covariate path ``Z(.)`` and the
    scalar parameters, trunk over the query time, trained on the MBPP Volterra
    residual over a *distribution* of covariate paths and parameters.  One trained
    operator solves any instance in the family in a single forward pass.
    """

    def __init__(
        self,
        coll_grid=None,
        T=30.0,
        p_latent=64,
        width=96,
        depth=3,
        seed=0,
        kappa_range=(0.05, 0.9),
        theta_range=(0.2, 3.0),
        mu_range=(0.5, 5.0),
        delta_range=(-2.0, 2.0),
        z_steps=5,
        ic_weight=10.0,
    ):
        tf.random.set_seed(seed)
        self.T = float(T)
        if coll_grid is None:
            coll_grid = np.linspace(0.0, T, 96)
        self.M = int(coll_grid.size)
        self.t = tf.constant(coll_grid, tf.float32)
        self.W = tf.constant(_trapezoid_weight_matrix(np.asarray(coll_grid)), tf.float32)
        self.dt = self.t[:, None] - self.t[None, :]
        self.tril = tf.constant(np.tril(np.ones((self.M, self.M))), tf.float32)
        self.ranges = dict(kappa=kappa_range, theta=theta_range, mu=mu_range, delta=delta_range)
        self.z_steps = int(z_steps)
        self.ic_weight = float(ic_weight)

        def mk(nin):
            return tf.keras.Sequential(
                [layers.Input(shape=(nin,))]
                + [layers.Dense(width, activation="tanh") for _ in range(depth)]
                + [layers.Dense(p_latent)]
            )

        self.branch = mk(self.M + 4)
        self.trunk = mk(1)
        self.opt = tf.keras.optimizers.Adam(1e-3)

    def _xi_grid(self, Zb, P):
        # Zb: (B,M), P: (B,4) -> xi on grid (B, M)
        bcoef = self.branch(tf.concat([Zb, P], axis=1))  # (B, p_latent)
        Tr = self.trunk((self.t / self.T)[:, None])  # (M, p_latent)
        return tf.matmul(bcoef, Tr, transpose_b=True)  # (B, M)

    def _loss(self, Zb, P, reweight=False):
        xi = self._xi_grid(Zb, P)
        kappa = P[:, 0:1]
        theta = P[:, 1:2]
        mu = P[:, 2:3]
        delta = P[:, 3:4]
        mod = tf.exp(tf.clip_by_value(delta * Zb, -30.0, 30.0))  # (B, M)
        K = (
            kappa[:, :, None]
            * theta[:, :, None]
            * mod[:, :, None]
            * tf.exp(-theta[:, :, None] * self.dt[None])
        ) * self.tril[None]
        integ = tf.matmul(self.W[None] * K, xi[:, :, None])[:, :, 0]
        R = xi - mu - integ
        if reweight:
            R = R / (tf.sqrt(tf.reduce_mean(xi**2, axis=1, keepdims=True)) + 1e-3)
        return tf.reduce_mean(R**2) + self.ic_weight * tf.reduce_mean((xi[:, 0:1] - mu) ** 2)

    def _sample(self, batch, kappa_max=None):
        kmax = self.ranges["kappa"][1] if kappa_max is None else kappa_max
        r = self.ranges
        P = tf.concat(
            [
                tf.random.uniform((batch, 1), r["kappa"][0], kmax),
                tf.random.uniform((batch, 1), *r["theta"]),
                tf.random.uniform((batch, 1), *r["mu"]),
                tf.random.uniform((batch, 1), *r["delta"]),
            ],
            axis=1,
        )
        t = np.asarray(self.t)
        Zb = np.stack(
            [
                _random_step_path(t, self.z_steps, int(np.random.randint(2**31)))
                for _ in range(batch)
            ]
        )
        return tf.constant(Zb, tf.float32), P

    def train(
        self,
        n_steps=20000,
        batch=32,
        lr=1e-3,
        anchors=None,
        data_weight=1.0,
        curriculum=False,
        reweight=False,
    ):
        r"""
        Train the operator on the family.  ``anchors=(Z_anchor, P_anchor,
        XI_anchor)`` adds a supervised MSE term (hybrid loss; recommended for a
        wide family / near criticality); ``curriculum=True`` widens kappa easy ->
        stiff; ``reweight=True`` normalises the residual by the solution scale.
        """
        self.opt.learning_rate.assign(lr)
        has_data = anchors is not None
        if has_data:
            Za = tf.constant(np.asarray(anchors[0], np.float32))
            Pa = tf.constant(np.asarray(anchors[1], np.float32))
            XIa = tf.constant(np.asarray(anchors[2], np.float32))

        @tf.function
        def step(Zb, P):
            with tf.GradientTape() as tape:
                loss = self._loss(Zb, P, reweight)
                if has_data:
                    loss = loss + data_weight * tf.reduce_mean((self._xi_grid(Za, Pa) - XIa) ** 2)
            vars_ = self.branch.trainable_variables + self.trunk.trainable_variables
            self.opt.apply_gradients(zip(tape.gradient(loss, vars_), vars_))
            return loss

        k_lo, k_hi = self.ranges["kappa"]
        for s in range(n_steps):
            if curriculum:
                frac = min(1.0, s / (0.7 * n_steps))
                kmax = min(k_hi, max(k_lo + 0.1, 0.5) + (k_hi - 0.5) * frac)
            else:
                kmax = k_hi
            step(*self._sample(batch, kmax))
        return self

    def solve(self, params, t_grid, Z_on_grid):
        p = np.array(
            [
                [
                    params["kappa"],
                    params["theta"],
                    params.get("mu", 1.0),
                    float(np.atleast_1d(params.get("delta", [0.0]))[0]),
                ]
            ],
            np.float32,
        )
        bcoef = self.branch(
            tf.concat([tf.constant(np.asarray(Z_on_grid, np.float32)[None, :]), p], axis=1)
        )
        Tr = self.trunk((tf.constant(np.asarray(t_grid, np.float32)) / self.T)[:, None])
        return tf.matmul(bcoef, Tr, transpose_b=True).numpy()[0]


def _random_step_path(t, n_steps, seed):
    rng = np.random.default_rng(seed)
    edges = np.sort(rng.uniform(0, t[-1], n_steps - 1))
    levels = rng.uniform(-0.5, 0.5, n_steps)
    return levels[np.searchsorted(edges, t)]
