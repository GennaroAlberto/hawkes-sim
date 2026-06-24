r"""
TensorFlow / Keras neural operators for the MULTIVARIATE Mean Behavior Poisson
process -- the high-dimensional, large-data counterpart of ``operators_nn.py``.

Everything in ``operators_nn.py`` is univariate and numpy; this module scales the
same ideas up:

* the forcing ``s(t)`` and intensity ``xi(t)`` are now vector-valued in R^M,
  sampled on a length-``T`` grid, so a batch of series is a tensor of shape
  ``(batch, T, M)``;
* the operators are Keras models trained on *large* simulated datasets, fed by a
  ``tf.data`` pipeline (shuffended/batched/prefetched, GPU-ready).

We learn either the forward solution operator ``G : s -> xi`` (how the system
evolves) or the inverse map ``counts -> kernel matrix`` (amortised inference).

Models
------
- :class:`SpectralConv1D` / :class:`FourierNeuralOperator` -- a multivariate FNO,
  the natural architecture for high-dimensional function-to-function operator
  learning.  Generalises ``operators.SpectralOperator`` from one linear spectral
  multiply to ``M``-channel mixing with depth and non-linearity.
- :class:`MBPPDeepONet` -- branch/trunk operator for the no-closed-form regime.
- :class:`StateSpaceMBPP` -- a learned (leaky neural-ODE) state-space model that
  integrates the *evolution* ``dz/dt`` forced by ``s(t)`` and reads out ``xi`` --
  the trainable generalisation of the exact ODE reduction
  ``operators.solve_mbpp_ode_multivariate``.
- :class:`AmortizedKernelInference` -- a temporal CNN mapping interval counts to
  the M x M branching matrix in a single forward pass.

Data
----
- :func:`generate_operator_dataset` / :func:`generate_inference_dataset` build
  large training sets from the multivariate MBPP solver / Hawkes simulator.
- :func:`make_dataset` wraps numpy arrays into a batched ``tf.data.Dataset``.
- :func:`train_operator` is a ready-to-run Keras training loop.

This module is an *optional* extra: importing it requires ``tensorflow``.  It is
deliberately NOT imported by ``hawkes_calibration.__init__``, so the rest of the
package stays numpy-only.  (Provided as a reference implementation; not unit
tested in this environment.)
"""

from __future__ import annotations

import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras import layers
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "operators_tf requires TensorFlow.  Install it with `pip install tensorflow` "
        "(the rest of hawkes_calibration is numpy-only and does not need it)."
    ) from exc

from .linear import solve_mbpp_ode_multivariate


# ===========================================================================
# Fourier Neural Operator (multivariate, channels = M point-process dimensions).
# ===========================================================================
class SpectralConv1D(layers.Layer):
    r"""
    Spectral convolution over the time axis with channel mixing (FNO core).

    Input/output: ``(batch, T, C)``.  Acts by FFT over time, a learnable complex
    matrix multiply across channels on the lowest ``modes`` frequencies, and an
    inverse FFT.  With a single channel and ``modes`` covering the band this is
    exactly the linear transfer-function operator ``R(omega)`` of
    ``operators.SpectralOperator``; here it is multi-channel, learnable and stacked.

    ``seq_len`` (=T) is fixed at construction so all frequency shapes are static.
    """

    def __init__(self, out_channels, modes, seq_len, **kw):
        super().__init__(**kw)
        self.out_channels = int(out_channels)
        self.seq_len = int(seq_len)
        self.full = self.seq_len // 2 + 1
        self.modes = min(int(modes), self.full)   # cannot keep more modes than exist

    def build(self, input_shape):
        in_ch = int(input_shape[-1])
        scale = 1.0 / (in_ch * self.out_channels)
        init = tf.random_normal_initializer(stddev=scale)
        self.w_real = self.add_weight(
            name="w_real", shape=(in_ch, self.out_channels, self.modes),
            initializer=init, trainable=True)
        self.w_imag = self.add_weight(
            name="w_imag", shape=(in_ch, self.out_channels, self.modes),
            initializer=init, trainable=True)

    def call(self, x):
        x_t = tf.transpose(x, [0, 2, 1])                  # (B, C_in, T)
        x_ft = tf.signal.rfft(x_t)                        # (B, C_in, full) complex64
        x_ft_modes = x_ft[..., : self.modes]              # (B, C_in, modes)
        w = tf.complex(self.w_real, self.w_imag)          # (C_in, C_out, modes)
        out_modes = tf.einsum("bim,iom->bom", x_ft_modes, w)   # (B, C_out, modes)
        pad = self.full - self.modes
        out_ft = tf.pad(out_modes, [[0, 0], [0, 0], [0, pad]])  # (B, C_out, full)
        out = tf.signal.irfft(out_ft, fft_length=[self.seq_len])  # (B, C_out, T)
        return tf.transpose(out, [0, 2, 1])               # (B, T, C_out)


class FourierNeuralOperator(tf.keras.Model):
    r"""
    Multivariate FNO learning the MBPP solution operator s(t) -> xi(t),
    s, xi in R^M sampled on a length-T grid.  Input/output: (batch, T, M).

    Parameters
    ----------
    seq_len : int          length T of the time grid.
    n_channels : int       M, the point-process dimension.
    width : int            lifted hidden channel width.
    modes : int            number of Fourier modes kept per spectral layer.
    n_layers : int         number of Fourier blocks.
    """

    def __init__(self, seq_len, n_channels, width=48, modes=24, n_layers=4, **kw):
        super().__init__(**kw)
        self.lift = layers.Dense(width)
        self.spectral = [SpectralConv1D(width, modes, seq_len) for _ in range(n_layers)]
        self.pointwise = [layers.Dense(width) for _ in range(n_layers)]   # 1x1 conv
        self.proj1 = layers.Dense(128, activation="gelu")
        self.proj2 = layers.Dense(n_channels)

    def call(self, x, training=False):
        v = self.lift(x)
        for sp, pw in zip(self.spectral, self.pointwise):
            v = tf.nn.gelu(sp(v) + pw(v))
        return self.proj2(self.proj1(v))


# ===========================================================================
# DeepONet (multivariate branch/trunk).
# ===========================================================================
class MBPPDeepONet(tf.keras.Model):
    r"""
    Multivariate DeepONet:  xi(s)(t)_m = sum_k branch_{m,k}(s) * trunk_k(t) + b0_m.

    Inputs to ``call`` are ``(forcing, t_query)`` with
    ``forcing : (batch, n_sensors, M)`` and ``t_query : (T, 1)``;
    output is ``(batch, T, M)``.
    """

    def __init__(self, n_sensors, n_channels, p=64, width=128, **kw):
        super().__init__(**kw)
        self.M = int(n_channels)
        self.p = int(p)
        self.branch = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(width, activation="gelu"),
            layers.Dense(width, activation="gelu"),
            layers.Dense(self.M * self.p),
        ])
        self.trunk = tf.keras.Sequential([
            layers.Dense(width, activation="gelu"),
            layers.Dense(width, activation="gelu"),
            layers.Dense(self.p, activation="gelu"),
        ])
        self.b0 = self.add_weight(name="b0", shape=(self.M,), initializer="zeros")

    def call(self, inputs, training=False):
        forcing, t_query = inputs
        b = self.branch(forcing)                                   # (B, M*p)
        b = tf.reshape(b, (-1, self.M, self.p))                    # (B, M, p)
        tr = self.trunk(t_query)                                   # (T, p)
        return tf.einsum("bmp,tp->btm", b, tr) + self.b0           # (B, T, M)


# ===========================================================================
# Learned state-space / Neural-ODE: integrate the evolution forced by s(t).
# ===========================================================================
class MBPPCell(layers.Layer):
    r"""
    One step of a learned, *stable* continuous-time evolution:

        z_{t+1} = z_t + dt * ( -d (.) z_t + tanh(z_t W_rec + s_t W_in + b) ),
        xi_t    = z_{t+1} C + s_t D,

    where d = softplus(raw_decay) > 0 guarantees a contracting (stable) drift --
    the learned generalisation of the exact linear drift -(1-kappa)theta in
    ``solve_mbpp_ode_multivariate``.  ``s_t`` (the multivariate baseline) is the
    forcing, exactly as in the ODE reduction.
    """

    def __init__(self, state_dim, n_channels, dt=1.0, **kw):
        super().__init__(**kw)
        self.state_dim = int(state_dim)
        self.M = int(n_channels)
        self.dt = float(dt)
        self.state_size = self.state_dim
        self.output_size = self.M

    def build(self, input_shape):
        H, M = self.state_dim, self.M
        self.W_in = self.add_weight(name="W_in", shape=(M, H), initializer="glorot_uniform")
        self.W_rec = self.add_weight(name="W_rec", shape=(H, H), initializer="orthogonal")
        self.b = self.add_weight(name="b", shape=(H,), initializer="zeros")
        self.raw_decay = self.add_weight(name="raw_decay", shape=(H,), initializer=tf.constant_initializer(0.5))
        self.C = self.add_weight(name="C", shape=(H, M), initializer="glorot_uniform")
        self.D = self.add_weight(name="D", shape=(M, M), initializer="zeros")

    def call(self, inputs, states):
        s_t = inputs                                    # (B, M)
        z = states[0]                                   # (B, H)
        d = tf.nn.softplus(self.raw_decay)              # (H,) > 0  -> stable drift
        drift = -d * z + tf.tanh(tf.matmul(z, self.W_rec) + tf.matmul(s_t, self.W_in) + self.b)
        z_new = z + self.dt * drift
        xi = tf.matmul(z_new, self.C) + tf.matmul(s_t, self.D)
        return xi, [z_new]


class StateSpaceMBPP(tf.keras.Model):
    r"""
    Learns the *evolution* xi(t) by integrating a forced latent state-space model
    over the forcing sequence s(t).  Input/output: ``(batch, T, M)``.  This is the
    trainable cousin of the exact state-space MBPP solver and scales to long
    sequences and high dimension (the recurrence is O(T) per series).
    """

    def __init__(self, n_channels, state_dim=64, dt=1.0, **kw):
        super().__init__(**kw)
        self.rnn = layers.RNN(MBPPCell(state_dim, n_channels, dt=dt), return_sequences=True)

    def call(self, s_seq, training=False):
        return self.rnn(s_seq)


# ===========================================================================
# Amortized inference: interval counts -> M x M branching matrix.
# ===========================================================================
class AmortizedKernelInference(tf.keras.Model):
    r"""
    A temporal CNN mapping interval-censored counts ``(batch, T, M)`` to the
    estimated branching matrix ``G = A/B`` of shape ``(batch, M, M)`` in one
    forward pass.  Entries pass through a sigmoid so they lie in (0, 1); the
    caller may rescale rows to control the spectral radius.
    """

    def __init__(self, n_channels, hidden=128, **kw):
        super().__init__(**kw)
        self.M = int(n_channels)
        self.body = tf.keras.Sequential([
            layers.Conv1D(hidden, 5, padding="same", activation="gelu"),
            layers.Conv1D(hidden, 5, padding="same", activation="gelu"),
            layers.GlobalAveragePooling1D(),
            layers.Dense(hidden, activation="gelu"),
        ])
        self.head = layers.Dense(self.M * self.M, activation="sigmoid")

    def call(self, counts, training=False):
        h = self.body(counts)
        return tf.reshape(self.head(h), (-1, self.M, self.M))


# ===========================================================================
# Data generation (numpy) + tf.data pipeline.
# ===========================================================================
def _random_branching_matrix(M, density, max_radius, rng):
    """Random non-negative branching matrix G with spectral radius < max_radius."""
    mask = rng.random((M, M)) < density
    G = rng.uniform(0.05, 1.0, size=(M, M)) * mask
    radius = max(np.max(np.abs(np.linalg.eigvals(G))), 1e-9)
    return G * (max_radius * rng.uniform(0.5, 0.95) / radius)


def _random_forcing(M, seq_len, T, rng, n_steps=5):
    """A random piecewise-constant, non-negative multivariate forcing on the grid."""
    t = np.linspace(0, T, seq_len)
    s = np.zeros((seq_len, M))
    for m in range(M):
        edges = np.sort(rng.uniform(0, T, n_steps - 1))
        levels = rng.uniform(0.2, 3.0, n_steps)
        idx = np.searchsorted(edges, t)
        s[:, m] = levels[idx]
    return t, s


def generate_operator_dataset(n_samples, M, seq_len=128, T=30.0,
                              theta=1.0, density=0.4, max_radius=0.8, seed=0):
    r"""
    Build (forcing, intensity) pairs for operator learning, using the exact
    multivariate MBPP solver as ground truth.

    Returns
    -------
    S, XI : (n_samples, seq_len, M) float32 arrays.
    """
    rng = np.random.default_rng(seed)
    B = np.full((M, M), float(theta))
    S = np.empty((n_samples, seq_len, M), np.float32)
    XI = np.empty((n_samples, seq_len, M), np.float32)
    for i in range(n_samples):
        G = _random_branching_matrix(M, density, max_radius, rng)
        A = G * B                                  # a_{m,j} = G_{m,j} * b_{m,j}
        t, s = _random_forcing(M, seq_len, T, rng)
        forcing = lambda tt, t=t, s=s: s[min(int(np.searchsorted(t, tt, "right") - 1), seq_len - 1)]
        xi = solve_mbpp_ode_multivariate(forcing, A, B, t)
        S[i] = s
        XI[i] = xi
    return S, XI


def generate_inference_dataset(n_samples, M, seq_len=128, T=30.0,
                               theta=1.0, density=0.4, max_radius=0.8,
                               baseline=0.5, seed=0):
    r"""
    Build (counts, branching-matrix) pairs for amortised inference.  Counts are
    the MBPP expected counts per interval (Poisson means); for genuinely
    stochastic training data, sample ``np.random.poisson`` on these means.

    Returns
    -------
    C : (n_samples, seq_len, M) float32 counts.
    G : (n_samples, M, M) float32 branching matrices.
    """
    rng = np.random.default_rng(seed)
    Bm = np.full((M, M), float(theta))
    C = np.empty((n_samples, seq_len, M), np.float32)
    Gs = np.empty((n_samples, M, M), np.float32)
    t = np.linspace(0, T, seq_len + 1)
    for i in range(n_samples):
        G = _random_branching_matrix(M, density, max_radius, rng)
        A = G * Bm
        mu = rng.uniform(0.2, 1.5, size=M)
        xi, Xi = solve_mbpp_ode_multivariate(lambda tt: mu, A, Bm, t, return_compensator=True)
        counts = np.diff(Xi, axis=0)               # expected counts per interval
        C[i] = rng.poisson(np.maximum(counts, 0)).astype(np.float32)
        Gs[i] = G
    return C, Gs


def make_dataset(X, Y, batch_size=64, shuffle=True, buffer=4096):
    """Wrap numpy arrays (or a tuple of inputs) into a batched tf.data.Dataset."""
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    if shuffle:
        ds = ds.shuffle(buffer)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def train_operator(model, S, XI, epochs=50, batch_size=64, lr=1e-3, val_split=0.1):
    r"""
    Compile and fit a forward operator (FNO or StateSpaceMBPP) on (forcing,
    intensity) pairs.  Returns the Keras ``History``.
    """
    n_val = int(len(S) * val_split)
    train = make_dataset(S[n_val:], XI[n_val:], batch_size, shuffle=True)
    val = make_dataset(S[:n_val], XI[:n_val], batch_size, shuffle=False)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")
    return model.fit(train, validation_data=val, epochs=epochs)


# ---------------------------------------------------------------------------
# Example end-to-end usage (high-dimensional, lots of data):
#
#   M, T = 20, 128
#   S, XI = generate_operator_dataset(n_samples=20000, M=M, seq_len=T)
#   fno = FourierNeuralOperator(seq_len=T, n_channels=M, width=64, modes=32)
#   train_operator(fno, S, XI, epochs=100, batch_size=128)
#   xi_pred = fno(S[:4])                      # operator applied to new forcings
#
#   # learn the evolution with the state-space model instead:
#   ssm = StateSpaceMBPP(n_channels=M, state_dim=128)
#   train_operator(ssm, S, XI, epochs=100)
#
#   # amortised inference of the branching matrix from counts:
#   C, G = generate_inference_dataset(n_samples=20000, M=M, seq_len=T)
#   inf = AmortizedKernelInference(n_channels=M)
#   inf.compile(optimizer="adam", loss="mse"); inf.fit(make_dataset(C, G), epochs=100)
#   G_hat = inf(C[:4])
# ---------------------------------------------------------------------------
