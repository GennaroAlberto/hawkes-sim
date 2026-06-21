# TF Playbook — experimenting with the MBPP neural operators

**Audience:** an agent picking up this repo to run TensorFlow experiments.
**Goal of the experiments:** learn the MBPP solution operator `s(t) ↦ ξ(t)` with the
TF models, then **add noise to a smart synthetic dataset and find where learning
breaks**, for different *forcing* classes and at small/medium/large scale.

Read this end-to-end once (5 min), then jump to [Quickstart](#quickstart).

---

## 1. What this repo is (30-second version)

`hawkes_calibration` calibrates Hawkes processes. A Hawkes process has a
**stochastic** intensity, so its expected behaviour is captured by the **Mean
Behavior Poisson process (MBPP)**: a deterministic intensity `ξ(t)` solving the
Volterra equation

```
ξ(t) = s(t) + (Φ * ξ)(t)        # Φ = triggering kernel, s = exogenous "forcing"
```

Key fact we exploit everywhere: **the map `G : s ↦ ξ` is a linear,
translation-invariant operator.** So "learning the evolution of the system" =
learning this operator. For the exponential kernel it also equals a linear ODE
(`ξ' = (κ−1)θ·ξ + θ·s`, forcing `s`) and, in Fourier space, a transfer function
`ξ̂ = ŝ/(1−φ̂)`. The full math is in **`PAPER_SUMMARY.md`** (§9 functional/operator
view, §10 high-dim + TF). You don't need to re-derive anything — the ground-truth
solver is already implemented.

**Univariate vs multivariate.** Most of the package is 1-D. For these experiments
we work **multivariate**: `ξ(t) ∈ ℝᴹ`, driven by an `M×M` kernel matrix. A batch
of series is a tensor `(batch, T, M)`.

---

## 2. The pieces you'll use

| File | What it gives you |
|---|---|
| `hawkes_calibration/operators.py` | `solve_mbpp_ode_multivariate` — **exact** ground-truth ξ for one system. |
| `experiments/tf_lab.py` | the **synthetic-data lab**: forcings, noise, `make_instance`, a numpy baseline learner, and the `noise_sweep` harness. **Numpy-only — runs without TF.** |
| `hawkes_calibration/operators_tf.py` | the **TF/Keras models** to test: `FourierNeuralOperator`, `MBPPDeepONet`, `StateSpaceMBPP`, `AmortizedKernelInference`, plus `generate_operator_dataset` / `make_dataset` / `train_operator`. **Requires `pip install tensorflow`.** |
| `hawkes_calibration/operators_nn.py` | the 1-D numpy reference versions (DeepONet, amortized) — handy for sanity intuition. |

> The TF module is **not** imported by `hawkes_calibration/__init__` on purpose, so
> the core stays numpy-only. Import it explicitly:
> `from hawkes_calibration.operators_tf import FourierNeuralOperator`.

---

## 3. The synthetic-data lab (`experiments/tf_lab.py`)

Everything you need to build noisy datasets and measure breaking points.

### Forcing classes (the input functional forms)
`FORCINGS = {"pc", "sine", "impulse", "smooth", "bursty"}`
- `pc` — piecewise-constant (step) baselines.
- `sine` — sums of sinusoids (smooth, periodic).
- `impulse` — sparse tall narrow rectangles (≈ impulse train, high-frequency).
- `smooth` — random-Fourier GP-like positive functions.
- `bursty` — exponentially-decaying bursts at random onsets.

These deliberately span the frequency content spectrum — that's what determines
noise tolerance.

### Scales
`SCALES = {"small": M=3,T=64,n=512; "medium": M=10,T=128,n=4000; "large": M=40,T=256,n=20000}`
Override any field via kwargs to `make_instance`.

### Noise models (one scalar `level` controls severity, 0 = clean)
- `"gauss"` — additive observation noise (`level` × signal std).
- `"mult"` — multiplicative / heteroscedastic.
- `"poisson"` — count noise via finite exposure (`level` ↑ ⇒ lower exposure ⇒ noisier). This is the realistic interval-censored regime.
- `"missing"` — randomly drop a fraction `level` of time steps (harsh).

### Build one instance
```python
from experiments.tf_lab import make_instance
inst = make_instance(scale="medium", forcing="smooth",
                     noise_kind="gauss", noise_level=0.2, seed=0)
inst.S_train          # (n_train, T, M) forcings
inst.XI_train         # (n_train, T, M) CLEAN targets (reference)
inst.XI_train_noisy   # (n_train, T, M) NOISY targets  <- train on these
inst.S_test, inst.XI_test   # clean test set  <- evaluate against these
inst.t, inst.meta     # time grid + metadata (M, T, spectral_radius, ...)
```
Train on `XI_train_noisy`, **evaluate against the clean `XI_test`** — that's how
we measure whether noise corrupted the learned operator (not just whether it
memorized noise). Set `vary_system=True` to make each sample its own kernel (the
harder "learn a family of systems" regime).

### The breaking-point sweep
```python
from experiments.tf_lab import noise_sweep, plot_sweep, run_demo
# numpy linear baseline (no TF), across forcings:
run_demo(scale="small", noise_kind="gauss")   # writes results/tf_lab_noise_sweep.png
```
`noise_sweep(...)` returns `{forcing: {"levels", "errors", "breaking"}}` where
`breaking` is the first noise level at which the clean-target test error blows up
(> 2× the noiseless error). The numpy `MultivariateSpectralOperator` baseline —
the *exact linear operator* fitted by per-frequency least squares — is your
**reference**: because it's the true operator class, it's strongly noise-robust
(LSQ averages noise over samples). The scientific question for the TF models:

> **Does the TF model match the linear operator's noise robustness, or break
> earlier?** Breaking earlier ⇒ it's overfitting / under-regularized / mis-sized.

Baseline behaviour you'll reproduce: under `gauss`, `pc`/`sine` barely move while
`smooth`/`bursty` break around level ≈ 0.8; `missing` is catastrophic (breaks by
≈ 0.1). Use these as sanity anchors.

---

## 4. Wiring in the TF models

`noise_sweep` takes a `fit_eval(instance) -> rel_L2_error` callback. Drop in any
TF model. Ready-made for the FNO:

```python
import tensorflow as tf
from hawkes_calibration.operators_tf import FourierNeuralOperator
from experiments.tf_lab import noise_sweep, rel_l2

def make_fno_fit_eval(epochs=60, width=48, modes=24, lr=1e-3, batch=64):
    def fit_eval(inst):
        T, M = inst.meta["T"], inst.meta["M"]
        model = FourierNeuralOperator(seq_len=T, n_channels=M,
                                      width=width, modes=min(modes, T // 2 + 1))
        model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")
        model.fit(inst.S_train.astype("float32"),
                  inst.XI_train_noisy.astype("float32"),
                  epochs=epochs, batch_size=batch, verbose=0)
        pred = model.predict(inst.S_test.astype("float32"), verbose=0)
        return rel_l2(pred, inst.XI_test)
    return fit_eval

sweep = noise_sweep(scale="medium", forcings=("pc", "smooth", "bursty"),
                    noise_kind="gauss", levels=[0, 0.1, 0.2, 0.4, 0.8],
                    fit_eval=make_fno_fit_eval())
```

`StateSpaceMBPP` (the "learn the evolution" recurrent model) is a drop-in
replacement for `FourierNeuralOperator` above — same `(B,T,M) → (B,T,M)` signature:
```python
from hawkes_calibration.operators_tf import StateSpaceMBPP
model = StateSpaceMBPP(n_channels=M, state_dim=128)
```
`MBPPDeepONet` is different: it takes `(forcing_at_sensors, t_query)` as a tuple
and returns `(B, T, M)`. Sample the forcing at `n_sensors` fixed times for the
branch and pass the trunk a `(T,1)` time grid. (Use it for the no-closed-form /
nonlinear regime; for these linear-MBPP experiments FNO/SSM are the natural fit.)

`AmortizedKernelInference` solves the *inverse* problem — feed it interval counts
`(B,T,M)`, it predicts the `M×M` branching matrix. Build data with
`generate_inference_dataset`.

---

## 5. Experiment protocol (suggested)

1. **Reproduce the baseline.** `run_demo(scale="small")` and `("medium")` for
   `gauss`, `poisson`, `missing`. Record breaking points per forcing. (No TF.)
2. **FNO vs baseline, clean data.** Confirm the FNO reaches the baseline's
   noiseless error (≈ a few %). If not, increase `width`/`modes`/`epochs`.
3. **Noise sweep with the FNO.** Same forcings/levels as the baseline; overlay
   the curves (`plot_sweep`). Where does the FNO break vs the linear operator?
4. **Forcing dependence.** Which input classes are hardest? (Expect high-frequency
   `impulse`/`bursty`/`smooth` to break first — they stress the kept Fourier modes.)
5. **Scale.** Repeat at `medium` then `large`. Track: does more data push the
   breaking point higher? Does the gap to the linear baseline shrink?
6. **Architecture ablations.** FNO `modes` (spectral capacity) and `width`;
   `StateSpaceMBPP` `state_dim`; regularization / early stopping. Plot breaking
   point vs each knob.
7. **(Optional) `vary_system=True`.** Learning a family of operators is much
   harder — a good stress test of generalization.

**Metrics to log per run:** noiseless rel-L2, rel-L2 at each noise level, breaking
point, train time, model size. Save figures to `results/` and a JSON of the sweep.

---

## 6. Known TF gotchas (the module is a careful reference impl, untested here)

- **TensorFlow isn't installed in the base env.** `pip install tensorflow` first.
- **Fixed sequence length.** The FNO's `SpectralConv1D` bakes in `seq_len=T`; keep
  `T` constant across a sweep (the lab already does). `modes ≤ T//2 + 1` (auto-clamped).
- **rfft/irfft plumbing.** If a shape error appears in `SpectralConv1D`, check the
  `tf.signal.rfft` (input float32 → complex64) / `irfft(..., fft_length=[T])` path
  and the `einsum("bim,iom->bom", ...)` channel mixing. This is the single most
  likely spot to need a tweak.
- **DeepONet tuple inputs.** `model.fit` on `MBPPDeepONet` needs a dataset that
  yields `((forcing, t_query), target)`; simplest is to call it directly in a
  custom train step, or just use FNO/SSM for the sweep.
- **dtype.** Cast inputs/targets to `float32` before `fit/predict`.
- **Determinism.** `tf.keras.utils.set_random_seed(0)` for reproducible sweeps.

---

## 7. Quickstart

```bash
# 0. (optional) inspect the math: PAPER_SUMMARY.md §9–§10
# 1. numpy baseline breaking-point sweep — runs immediately, no TF:
python -m experiments.tf_lab                      # -> results/tf_lab_noise_sweep.png
# 2. install TF, then run a TF sweep:
pip install tensorflow
python - <<'PY'
from experiments.tf_lab import noise_sweep, plot_sweep
# paste make_fno_fit_eval from §4 here
sweep = noise_sweep(scale="medium", forcings=("pc","smooth","bursty"),
                    noise_kind="gauss", levels=[0,0.1,0.2,0.4,0.8],
                    fit_eval=make_fno_fit_eval())
plot_sweep(sweep, "FNO noise tolerance (medium, gauss)", "results/fno_sweep.png")
print({k: v["breaking"] for k, v in sweep.items()})
PY
```

That's it. The lab gives you exact ground truth, configurable forcings and noise,
a strong linear baseline, and a one-line harness to find where each TF model
breaks. Start at `small`, confirm against the baseline, then scale up.
