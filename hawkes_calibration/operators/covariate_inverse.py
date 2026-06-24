r"""
Covariate-modulated multivariate MBPP + an amortised **inverse** operator that
recovers the excitation matrix *and* the per-group covariate coefficients from the
(noisy) observed intensity path.

Model.  ``M`` groups.  The excitation of group ``m`` is modulated in time by a
log-linear covariate response,

    alpha_{m,j}(t) = A_{m,j} * exp(eta_m(t)),
    eta_m(t)       = sum_k delta^s_{m,k} Z^s_k(t)  +  delta^p_m Z^p_m(t),

so every group reacts to a pool of ``K_shared`` **shared** covariates ``Z^s`` (the
*overlap* -- each group with its own coefficients, "similar or different values")
plus **one private** covariate ``Z^p_m`` only it sees (the *different* one).  The
mean intensity solves the covariate-modulated MBPP (``solve_mbpp_ltv``).

Inverse task.  Given the observed covariates ``Z`` and a noisy intensity path
``xi``, recover the excitation matrix ``A`` (M*M numbers) and the covariate
coefficients ``delta`` (the M*(K_shared+1) live entries).  This module provides:

* the data model (design / samplers / a fast JAX batched forward validated against
  the numpy ``solve_mbpp_ltv`` oracle), and
* a JAX amortised inverse network trained on it, evaluated with observation noise.

Requires jax + optax (optional; not imported by ``hawkes_calibration.__init__``).
"""

from __future__ import annotations

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    import optax
except Exception as exc:  # pragma: no cover
    raise ImportError("covariate_inverse requires `jax` and `optax`.") from exc

from .linear import solve_mbpp_ltv


# ===========================================================================
# Design + samplers
# ===========================================================================
class CovariateDesign:
    """Channel layout: columns 0..K_shared-1 are shared; column K_shared+m is the
    private covariate of group m.  ``mask`` (M, p) marks the live delta entries."""

    def __init__(self, M, K_shared):
        self.M = int(M)
        self.K_shared = int(K_shared)
        self.p = self.K_shared + self.M
        mask = np.zeros((self.M, self.p), bool)
        mask[:, :self.K_shared] = True                  # every group sees the shared ones
        for m in range(self.M):
            mask[m, self.K_shared + m] = True           # plus its own private one
        self.mask = mask
        self.n_delta = int(mask.sum())                  # M*K_shared + M

    def pack_delta(self, delta):
        """(.., M, p) -> (.., n_delta) keeping only the live entries (row-major)."""
        d = np.asarray(delta)
        return d[..., self.mask]

    def unpack_delta(self, vec):
        """(.., n_delta) -> (.., M, p) zero-filled off the mask."""
        vec = np.asarray(vec)
        out = np.zeros(vec.shape[:-1] + (self.M, self.p))
        out[..., self.mask] = vec
        return out


def _pc_paths(n, N, p, n_switch, rng, lo=-1.0, hi=1.0):
    """Random zero-mean piecewise-constant covariate paths, shape (n, N, p)."""
    Z = np.empty((n, N, p))
    for b in range(n):
        for c in range(p):
            edges = np.sort(rng.integers(1, N, size=n_switch - 1)) if n_switch > 1 else np.array([], int)
            levels = rng.uniform(lo, hi, size=n_switch)
            seg = np.searchsorted(edges, np.arange(N))
            z = levels[seg]
            Z[b, :, c] = z - z.mean()                    # center -> identifies A vs delta
    return Z


def sample_dataset(n, design, t_grid, theta=1.0, n_switch=4,
                   diag=(0.05, 0.45), off=(0.0, 0.25), s_range=(0.4, 2.2),
                   delta_range=(-0.9, 0.9), rho_max=0.8, seed=0):
    """Sample ``n`` covariate-modulated instances and solve them exactly.

    Returns dict with s (n,M), A (n,M,M), delta (n,M,p), delta_vec (n,n_delta),
    Z (n,N,p), xi (n,N,M)."""
    rng = np.random.default_rng(seed)
    M, p = design.M, design.p
    t = np.asarray(t_grid, float); N = t.size
    Z = _pc_paths(n, N, p, n_switch, rng)
    s = rng.uniform(*s_range, size=(n, M))
    A = np.empty((n, M, M)); delta = np.zeros((n, M, p))
    for b in range(n):
        a = rng.uniform(*off, size=(M, M)); a[np.diag_indices(M)] = rng.uniform(*diag, size=M)
        d = rng.uniform(*delta_range, size=(M, p)) * design.mask
        # worst-case (over time) effective excitation -> keep subcritical for stability
        eta = Z[b] @ d.T                                   # (N, M)
        max_mod = np.exp(eta.max(0))                       # (M,)
        W = a * max_mod[:, None]
        rho = float(np.max(np.abs(np.linalg.eigvals(W))))
        if rho > rho_max:
            a *= rho_max / rho
        A[b] = a; delta[b] = d
    # the JAX forward matches the numpy solve_mbpp_ltv oracle to ~1e-6 (see tests),
    # so we use it to generate data fast.
    xi = np.asarray(jax_batched_forward(s, A, delta, Z, t, theta))
    return dict(s=s, A=A, delta=delta, delta_vec=design.pack_delta(delta),
                Z=Z, xi=xi, t=t, theta=theta)


# ===========================================================================
# Forward solvers
# ===========================================================================
def _modulation_callable(Z, delta, t_grid):
    """t -> (M,M) modulation exp(eta_m(t)) (row-constant), Z piecewise-constant on t_grid."""
    t = np.asarray(t_grid, float)
    M = delta.shape[0]

    def mod(tt):
        i = min(int(np.searchsorted(t, tt, "right") - 1), t.size - 1)
        eta = delta @ Z[max(i, 0)]                          # (M,)
        return np.exp(eta)[:, None] * np.ones((M, M))
    return mod


def batched_forward_numpy(s, A, delta, Z, t_grid, theta=1.0):
    """Exact oracle: per-instance covariate-modulated MBPP via solve_mbpp_ltv."""
    s = np.asarray(s, float); A = np.asarray(A, float); delta = np.asarray(delta, float)
    Z = np.asarray(Z, float); t = np.asarray(t_grid, float)
    n, M = s.shape[0], s.shape[1]
    B = theta * np.ones((M, M))
    out = np.empty((n, t.size, M))
    for b in range(n):
        out[b] = solve_mbpp_ltv(lambda tt, ss=s[b]: ss, A[b], B, t,
                                modulation=_modulation_callable(Z[b], delta[b], t))
    return out


def _fine_grid_mod(Z, delta, t_grid):
    """exp(eta) on the union of the grid and its midpoints, shape (n, 2N-1, M).

    Z is piecewise-constant on ``t_grid``; a midpoint takes the left cell's value."""
    Z = np.asarray(Z, float); delta = np.asarray(delta, float)
    n, N, _ = Z.shape
    eta = np.einsum("bnc,bmc->bnm", Z, delta)               # (n, N, M) on the grid
    eta_mid = eta[:, :-1, :]                                # midpoint n uses left cell n
    fine = np.empty((n, 2 * N - 1, eta.shape[2]))
    fine[:, 0::2, :] = eta
    fine[:, 1::2, :] = eta_mid
    return np.exp(fine)


def jax_batched_forward(s, A, delta, Z, t_grid, theta=1.0):
    """Fast, vectorised JAX RK4 forward (matches the numpy oracle)."""
    s = jnp.asarray(s); A = jnp.asarray(A); t = np.asarray(t_grid, float)
    Modf = jnp.asarray(_fine_grid_mod(Z, np.asarray(delta), t))   # (n, 2N-1, M)
    return _jax_forward_from_mod(s, A, Modf, jnp.asarray(np.diff(t)), float(theta))


def fine_covariates(Z, t_grid):
    """Z on the union of the grid and its midpoints (n, 2N-1, p), piecewise-constant."""
    Z = np.asarray(Z, float); n, N, p = Z.shape
    out = np.empty((n, 2 * N - 1, p))
    out[:, 0::2, :] = Z
    out[:, 1::2, :] = Z[:, :-1, :]
    return out


@jax.jit
def jax_forward_diff(s, A, delta, Zfine, dts, theta):
    """Differentiable forward in (A, delta): Mod = exp(Z.delta) built inside JAX."""
    Modf = jnp.exp(jnp.einsum("bnc,bmc->bnm", Zfine, delta))      # (B, 2N-1, M)
    return _jax_forward_from_mod(s, A, Modf, dts, theta)


# ===========================================================================
# Differentiable parameter recovery (analysis-by-synthesis).
#
# The intensity path identifies (A, delta) essentially exactly on clean data;
# observation noise makes a naive fit overfit, so we regularise two physical ways:
#   * average ``n_obs`` repeated noisy observations  (sigma_eff = sigma/sqrt(n_obs));
#   * EARLY STOP at the noise floor -- stop refining once the relative recon loss
#     reaches the irreducible noise level (so we don't fit the noise).
# ===========================================================================
def recover_params(design, data, theta=1.0, sigma=0.0, n_obs=1, steps=6000,
                   lr=3e-2, early_stop=True, seed=0):
    s = jnp.asarray(data["s"]); Zf = jnp.asarray(fine_covariates(data["Z"], data["t"]))
    dts = jnp.asarray(np.diff(data["t"])); mask = jnp.asarray(design.mask.astype(float))
    n, M = data["s"].shape

    rng = np.random.default_rng(seed)
    xi = data["xi"]
    if sigma > 0:
        obs = np.mean([np.maximum(xi * (1.0 + sigma * rng.standard_normal(xi.shape)), 0.0)
                       for _ in range(n_obs)], axis=0)
    else:
        obs = xi
    xo = jnp.asarray(obs)
    floor = (sigma / np.sqrt(n_obs)) ** 2 if sigma > 0 else 0.0   # noise floor of the rel recon loss

    def loss(par):
        A, dl = par
        xih = jax_forward_diff(s, A, dl * mask[None], Zf, dts, theta)
        return jnp.mean(jnp.sum((xih - xo) ** 2, (1, 2)) / (jnp.sum(xo ** 2, (1, 2)) + 1e-9))

    par = (jnp.tile(0.2 * jnp.eye(M)[None], (n, 1, 1)), jnp.zeros((n, M, design.p)))
    opt = optax.adam(optax.cosine_decay_schedule(lr, steps, alpha=0.02))
    st = opt.init(par); g = jax.jit(jax.value_and_grad(loss))
    for i in range(steps):
        l, gr = g(par); up, st = opt.update(gr, st); par = optax.apply_updates(par, up)
        if early_stop and floor > 0 and float(l) <= 1.05 * floor:
            break
    A, dl = par
    Ah = np.asarray(A); dfull = np.asarray(dl) * design.mask[None]
    dvech = dfull[:, design.mask]
    A_rel = np.linalg.norm((Ah - data["A"]).reshape(n, -1), axis=1) / (np.linalg.norm(data["A"].reshape(n, -1), axis=1) + 1e-12)
    d_rel = np.linalg.norm(dvech - data["delta_vec"], axis=1) / (np.linalg.norm(data["delta_vec"], axis=1) + 1e-12)
    return dict(A=Ah, delta_vec=dvech, steps_used=i + 1, final_loss=float(l),
                A_rel_mean=float(A_rel.mean()), A_rel_median=float(np.median(A_rel)),
                delta_rel_mean=float(d_rel.mean()), delta_rel_median=float(np.median(d_rel)),
                A_rel=A_rel, d_rel=d_rel)


@jax.jit
def _jax_forward_from_mod(s, A, Modf, dts, theta):
    # s (B,M), A (B,M,M), Modf (B,2N-1,M), dts (N-1,)
    def xi_of(Y, mod):                                      # mod (B,M)
        AY = jnp.einsum("bmj,bmj->bm", A, Y)
        return s + mod * AY
    def deriv(Y, mod):
        xi = xi_of(Y, mod)
        return xi[:, None, :] - theta * Y                  # y'_{mj}=xi_j-theta y_{mj}
    Y0 = jnp.zeros_like(A)
    xi0 = xi_of(Y0, Modf[:, 0, :])

    def step(Y, inp):
        dt, m0, mm, m1 = inp
        k1 = deriv(Y, m0)
        k2 = deriv(Y + dt / 2 * k1, mm)
        k3 = deriv(Y + dt / 2 * k2, mm)
        k4 = deriv(Y + dt * k3, m1)
        Y = Y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return Y, xi_of(Y, m1)

    inps = (dts, Modf[:, 0:-1:2, :].transpose(1, 0, 2),
            Modf[:, 1::2, :].transpose(1, 0, 2),
            Modf[:, 2::2, :].transpose(1, 0, 2))
    _, xis = jax.lax.scan(step, Y0, inps)
    return jnp.concatenate([xi0[None], xis], axis=0).transpose(1, 0, 2)   # (B,N,M)


# ===========================================================================
# Amortised inverse network:  (noisy xi, Z)  ->  (A, delta)
# ===========================================================================
def _add_noise(xi, sigma, key):
    """Relative Gaussian observation noise on the intensity path."""
    if sigma <= 0:
        return xi
    eps = jax.random.normal(key, xi.shape)
    return jnp.maximum(xi * (1.0 + sigma * eps), 0.0)


def _conv1d(x, W, b):
    y = jax.lax.conv_general_dilated(x, W, (1,), "SAME",
                                     dimension_numbers=("NWC", "WIO", "NWC"))
    return y + b


class AmortizedCovariateInverse:
    r"""Temporal-CNN regressor mapping the observed ``(xi, Z)`` channels to the
    excitation matrix ``A`` and the live covariate coefficients ``delta``."""

    def __init__(self, design, theta=1.0, width=128, kernel=5, depth=3, hidden=256, seed=0):
        self.design = design
        self.M = design.M
        self.theta = float(theta)
        self.out_A = self.M * self.M
        self.out_d = design.n_delta
        self.width = width; self.kernel = kernel; self.depth = depth; self.hidden = hidden
        self._seed = int(seed)
        self.in_ch = self.M + design.p
        self.params = None
        self.x_mean = self.x_std = None
        self.y_mean = self.y_std = None

    # -- param init ---------------------------------------------------------
    def _init(self, key):
        ps = {"conv": [], "dense": []}
        cin = self.in_ch
        for _ in range(self.depth):
            key, k = jax.random.split(key)
            W = jax.random.normal(k, (self.kernel, cin, self.width)) * np.sqrt(2.0 / (self.kernel * cin))
            ps["conv"].append((W, jnp.zeros(self.width)))
            cin = self.width
        sizes = [2 * self.width, self.hidden, self.hidden, self.out_A + self.out_d]
        for a, b in zip(sizes[:-1], sizes[1:]):
            key, k = jax.random.split(key)
            ps["dense"].append((jax.random.normal(k, (a, b)) * np.sqrt(2.0 / (a + b)), jnp.zeros(b)))
        return ps

    def _net(self, params, X):
        h = X
        for (W, b) in params["conv"]:
            h = jax.nn.gelu(_conv1d(h, W, b))
        h = jnp.concatenate([h.mean(1), h.max(1)], axis=-1)          # global mean+max pool
        for (W, b) in params["dense"][:-1]:
            h = jax.nn.gelu(h @ W + b)
        W, b = params["dense"][-1]
        return h @ W + b                                             # (B, out)

    # -- training -----------------------------------------------------------
    def train(self, data, epochs=4000, lr=2e-3, batch=128, sigma_train=0.1,
              val=None, log_every=0, seed=0):
        Z = np.asarray(data["Z"]); xi = np.asarray(data["xi"])
        Y = np.concatenate([data["A"].reshape(len(Z), -1), data["delta_vec"]], axis=1)
        self.x_mean = np.concatenate([xi.reshape(-1, self.M).mean(0), Z.reshape(-1, self.design.p).mean(0)])
        self.x_std = np.concatenate([xi.reshape(-1, self.M).std(0), Z.reshape(-1, self.design.p).std(0)]) + 1e-6
        self.y_mean = Y.mean(0); self.y_std = Y.std(0) + 1e-8
        Zj = jnp.asarray(Z); xij = jnp.asarray(xi)
        Yj = jnp.asarray((Y - self.y_mean) / self.y_std)
        xm = jnp.asarray(self.x_mean); xs = jnp.asarray(self.x_std)
        n = len(Z)

        def make_X(xi_b, Z_b):
            chan = jnp.concatenate([xi_b, Z_b], axis=-1)             # (B,N,in_ch)
            return (chan - xm) / xs

        def loss_fn(params, xi_b, Z_b, y_b, key):
            xin = _add_noise(xi_b, sigma_train, key)
            pred = self._net(params, make_X(xin, Z_b))
            return jnp.mean((pred - y_b) ** 2)

        sched = optax.cosine_decay_schedule(lr, epochs, alpha=0.05)
        opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched))
        key = jax.random.PRNGKey(seed)
        params = self._init(key)
        opt_state = opt.init(params)
        step = jax.jit(jax.value_and_grad(loss_fn))

        best = np.inf; best_params = params
        hist = {"epoch": [], "loss": [], "val": []}
        for ep in range(epochs):
            key, ks, kn = jax.random.split(key, 3)
            idx = jax.random.choice(ks, n, (min(batch, n),), replace=False)
            loss, grads = step(params, xij[idx], Zj[idx], Yj[idx], kn)
            updates, opt_state = opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            if val is not None and (ep % max(1, epochs // 40) == 0 or ep == epochs - 1):
                self.params = params
                v = self.eval_param_error(val, sigma=sigma_train)["A_rel_mean"]
                hist["epoch"].append(ep); hist["loss"].append(float(loss)); hist["val"].append(v)
                if v < best:
                    best = v; best_params = jax.tree_util.tree_map(lambda x: x, params)
                if log_every:
                    print(f"  epoch {ep:5d}  loss={float(loss):.4e}  val A-rel={v:.4f} (best {best:.4f})", flush=True)
        self.params = best_params if val is not None else params
        return hist

    # -- inference / evaluation --------------------------------------------
    def predict(self, Z, xi):
        Z = np.atleast_3d(np.asarray(Z, float)); xi = np.asarray(xi, float)
        chan = (np.concatenate([xi, Z], axis=-1) - self.x_mean) / self.x_std
        out = np.asarray(self._net(self.params, jnp.asarray(chan))) * self.y_std + self.y_mean
        A = out[:, :self.out_A].reshape(-1, self.M, self.M)
        dvec = out[:, self.out_A:]
        return A, dvec

    def eval_param_error(self, data, sigma=0.0, seed=123):
        Z = np.asarray(data["Z"]); xi = np.asarray(data["xi"]).copy()
        if sigma > 0:
            rng = np.random.default_rng(seed)
            xi = np.maximum(xi * (1.0 + sigma * rng.standard_normal(xi.shape)), 0.0)
        Ah, dvech = self.predict(Z, xi)
        A, dvec = data["A"], data["delta_vec"]
        A_rel = np.linalg.norm((Ah - A).reshape(len(A), -1), axis=1) / (np.linalg.norm(A.reshape(len(A), -1), axis=1) + 1e-12)
        d_rel = np.linalg.norm(dvech - dvec, axis=1) / (np.linalg.norm(dvec, axis=1) + 1e-12)
        # end-to-end: reconstruct the intensity from recovered params
        dfull = self.design.unpack_delta(dvech)
        xi_rec = np.asarray(jax_batched_forward(data["s"], Ah, dfull, Z, data["t"], self.theta))
        xi_true = data["xi"]
        rec = np.linalg.norm((xi_rec - xi_true).reshape(len(A), -1), axis=1) / (np.linalg.norm(xi_true.reshape(len(A), -1), axis=1) + 1e-12)
        return dict(A_rel_mean=float(A_rel.mean()), A_rel_median=float(np.median(A_rel)),
                    delta_rel_mean=float(d_rel.mean()), delta_rel_median=float(np.median(d_rel)),
                    recon_rel_mean=float(rec.mean()), recon_rel_median=float(np.median(rec)))
