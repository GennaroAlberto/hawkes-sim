# Pure-learning experiments for the covariate-augmented MBPP

**Audience.** A coding agent (with JAX and/or TensorFlow available) who will run the
machine-learning experiments that the numpy-only environment used for the paper
cannot execute. Everything here is wired to the real package API; the numpy-checkable
pieces are validated in `experiments/learning/run_learning.py` (run it first).

**Goal.** Replace, or accelerate, the classical interval-censored MBPP fitting of the
investment model with *learned* components, and characterise where learning helps and
where it breaks. Three tasks:

1. **Forward operator learning** — learn the MBPP solution map so a forward solve is a
   single network call (no ODE/Volterra integration in the optimiser loop).
2. **Physics-informed solving (PINN / PINO)** — train that map from the MBPP *residual*
   (no ground-truth solver needed), parametrically (PINN) or over the whole functional
   family (PINO), and use it as a fast differentiable surrogate inside the IC-LL fit.
3. **Amortised inverse inference** — learn `counts -> (kappa, theta, delta, ...)` directly,
   so calibrating a new sector is a forward pass.

All three are evaluated on the same investment data-generating family as the paper
(`experiments/exp12_investments.py`, `experiments/exp13_grid.py`).

---

## 0. Setup and ground truth

```bash
pip install jax jaxlib          # for the PINN/PINO solver backend
pip install tensorflow          # for the FNO / DeepONet / state-space / amortised models
PYTHONPATH=. python experiments/learning/run_learning.py --selfcheck   # numpy, runs anywhere
```

The self-check validates the **well-posedness** that every learned solver relies on:
the MBPP residual is *linear in xi*, so its global minimum is the exact solution. In
numpy we confirm `family_residual(exact_xi, ...) ~= 1.8e-15` and that
`solve_mbpp_residual_linear` reproduces the Volterra solve to `1e-9`. The exact
(non-neural) `NumpyReferenceSolver` is the oracle you benchmark every learned model
against.

The MBPP equation being learned (multivariate, covariate-modulated excitation):
```
xi(t) = s(t) + \int_0^t K(t,u) xi(u) du,   K_{m,j}(t,u) = kappa_{m,j} theta exp(delta^T Z(t)) e^{-theta (t-u)}.
```

---

## 1. Forward operator learning  (TensorFlow: `hawkes_calibration.operators.tf`)

Learn `G: s(.) -> xi(.)` (or `(params, Z) -> xi`) as a neural operator. The exact
ground truth is `solve_mbpp_ode_multivariate` (constant excitation) or
`solve_mbpp_ltv` (covariate-modulated); both are numpy and fast enough to generate
large training sets.

**Models** (all in `operators/tf.py`):
- `FourierNeuralOperator(seq_len=T, n_channels=M, width, modes)` — the natural choice for
  the *linear, translation-invariant* convolution case; for the exponential kernel the
  true operator is exactly `1/(1 - phi_hat(omega))`, so the FNO should reach a few-percent
  relative error and be strongly noise-robust.
- `StateSpaceMBPP(n_channels=M, state_dim)` — a learned stable state-space model that
  integrates the evolution forced by `s(t)`; drop-in `(B,T,M)->(B,T,M)` like the FNO.
- `MBPPDeepONet(...)` — branch ingests the sampled forcing/covariate, trunk the query
  time; use for the covariate-modulated (non-convolution) regime.

**Data**: `generate_operator_dataset(n_samples, M, seq_len)` returns `(S, XI)`; or build
your own from the exp13 family (vary kernel in {exp, sumexp3, powerlaw}, covariate level).

**Run**:
```python
from hawkes_calibration.operators.tf import FourierNeuralOperator, generate_operator_dataset, train_operator
S, XI = generate_operator_dataset(n_samples=20000, M=20, seq_len=128)
fno = FourierNeuralOperator(seq_len=128, n_channels=20, width=64, modes=32)
train_operator(fno, S, XI, epochs=100, batch_size=128)
```

**Experiments to run** (record rel-L2 vs the exact solver):
| id | question | sweep |
|----|----------|-------|
| F1 | FNO reaches the exact-operator error on clean data? | width in {32,64,128}, modes in {16,32,64} |
| F2 | FNO vs StateSpaceMBPP vs DeepONet on the *covariate-modulated* (non-convolution) regime | covariate level none/small/large |
| F3 | does learning a *family of operators* (varying kernel) degrade accuracy? | `vary_system=True` (see tf_lab) |
| F4 | **noise breaking point** vs the exact linear baseline | use `experiments/tf_lab.py` `noise_sweep` |

**Noise stress (the headline ML result).** `experiments/tf_lab.py` is a numpy-only lab that
builds noisy datasets across forcing classes (`pc/sine/impulse/smooth/bursty`), scales,
and noise models (`gauss/mult/poisson/missing`), and reports the *breaking point* (first
noise level where the clean-target test error exceeds 2x the noiseless error). Plug any TF
model in via a `fit_eval(instance)->rel_L2` callback. The exact
`MultivariateSpectralOperator` baseline is the reference: **does the TF model match its
noise robustness or break earlier?** A ready-made FNO `fit_eval` and the full 7-step
protocol are in the `tf_lab.py` docstrings.

---

## 2. Physics-informed solving — PINN & PINO  (`operators.neural_solver`, JAX or TF)

Train the solver from the **collocation residual** of the MBPP equation, so no
ground-truth solver is needed in the loss. This is the fast, differentiable surrogate
that removes the `(1 + 2*n_params) x Python-ODE` inner cost from the IC-LL fit.

```python
import numpy as np
from hawkes_calibration.operators.neural_solver import make_neural_solver, sample_covariate_paths, make_anchor_data

t = np.linspace(0, 60, 96)
Z = ((np.floor(t/10) % 2) - 0.5)[:, None]                       # a regime covariate

# (a) parametric PINN: fixed covariate path, generalise over (kappa,theta,mu,delta)
pinn = make_neural_solver(backend="jax", mode="pinn", coll_grid=t, Z_on_grid=Z, T=60., n_delta=1)
pinn.train(n_steps=20000, batch=64)

# (b) PINO: generalise over the covariate *paths* too (one operator for the whole family)
pino = make_neural_solver(backend="jax", mode="pino", coll_grid=t, T=60.)
pino.train(n_steps=30000, batch=32)

xi = pino.solve(dict(kappa=0.4, theta=1.0, mu=2.0, delta=[1.2]), t,
                Z_on_grid=sample_covariate_paths(t, 1, seed=1)[0])
```

**Hybrid stabilisers** (all four trainers `JAXNeuralMBPP / JAXDeepONetPINO /
TFNeuralMBPP / TFDeepONetPINO` accept them) — essential near criticality `kappa->1`:
```python
P  = np.array([[0.85,1.0,2.0,1.0],[0.9,0.6,2.0,-1.0]])          # stiff rows
Z  = sample_covariate_paths(t, len(P), seed=0)
XI = make_anchor_data(t, P, Z)                                  # a few EXACT anchors
pino.train(n_steps=30000, batch=32, anchors=(Z, P, XI), data_weight=1.0,
           curriculum=True,      # widen kappa easy->stiff over training
           reweight=True)        # normalise residual by solution scale
```

**Use it inside the fit (the payoff).** `solver.intensity_fn()` returns a jitted
`f(p_vec, t)->xi` differentiable in `p_vec`; drop it into a JAX/optax IC-LL objective so
each fit step is one network pass with exact autodiff gradients (template in
`run_learning.py: ic_fit_with_surrogate`).

**Experiments to run**:
| id | question | metric |
|----|----------|--------|
| P1 | PINN matches the exact solver across the prior box | `solver_accuracy_report` rel-L2, binned by kappa |
| P2 | PINO solves a *new* covariate path in one pass (no retrain) | rel-L2 on held-out Z |
| P3 | hybrid anchors + curriculum + reweight fix the stiff `kappa->1` regime | rel-L2 in the top-kappa bin |
| P4 | surrogate-in-the-loop fit vs classical fit: speed and the recovered `(kappa,theta,delta)` | wall-clock, parameter error |

`solver_accuracy_report(predict, t, P_test, Z_test, plot_path=...)` bins error by kappa and
writes a figure — use it for P1/P3. (Pass `predict(p, z)` that wraps `z` as a covariate
callable; see `run_learning.py`.)

---

## 3. Amortised inverse inference  (`operators.tf` / `operators.nn`)

Learn the inverse map directly: feed interval counts, predict the kernel parameters.

- `AmortizedKernelInference` (TF) — `counts (B,T,M) -> MxM branching matrix`. Build training
  data with `generate_inference_dataset`.
- `AmortizedInference` (numpy, `operators/nn.py`) — the 1-D reference (`counts -> (kappa,theta)`),
  useful as a sanity baseline and runnable without TF.

**Experiments to run**:
| id | question | metric |
|----|----------|--------|
| A1 | amortised `(kappa,theta,delta)` vs the classical IC-LL fit on the exp13 grid | recovery error, inference time |
| A2 | does the amortised net degrade gracefully under kernel misspecification (sumexp3, powerlaw)? | error vs DGP kernel |
| A3 | calibrate on the multivariate investment family (M sectors) | per-sector and cross error |

---

## 4. Datasets — the investment family

Reuse the paper's DGP so the learned models are tested on the same distribution:
- `experiments/exp13_grid.py: clean_sim(...)` — multivariate nonlinear Hawkes with
  kernel in {exp, sumexp3, powerlaw}, covariate-modulated excitation.
- `make_anchor_data(t, params, Z)` — exact intensities for supervised targets.
- `sample_covariate_paths(t, n, seed)` — random piecewise covariate paths for PINO.
- Parameter prior box (suggested): `kappa in [0.1, 0.95]`, `theta in [0.4, 2.5]`,
  `mu in [0.5, 3]`, `delta in [-1, 1.5]`. See `experiments/learning/configs.py`.

---

## 5. Success criteria & deliverables

For each experiment, log: clean rel-L2 (vs exact), rel-L2 at each noise level + breaking
point (F4), held-out IC-LL when used in fitting (P4), parameter recovery error (A1–A3),
train time, and model size. Save figures + a JSON per run under `results/learning/`.

Headline questions to answer:
- Does a learned forward operator match the exact linear operator's **accuracy and noise
  robustness**, or break earlier (over-fitting / under-regularised)?
- Does the PINO remove the per-instance solve cost **without** losing accuracy in the
  stiff regime, once hybrid training is on?
- Is amortised inference competitive with classical IC-LL fitting on recovery, and how
  much faster?

Everything is scaffolded in `experiments/learning/`; start with `--selfcheck`, then work
through F, P, A in order.
