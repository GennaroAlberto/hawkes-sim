# GENERATE.md — quick start for the experiments agent

You are working in the `hawkes_calibration` repository. This file tells you how to
generate the synthetic data, run the experiment protocols of the design document
(`design-doc-unified/main.pdf`, chapter "The Code and Test Plan"), and where to write
the results. The two results chapters of the document ("Results on Synthetic Data",
"Results on the Spain VC Ecosystem") are empty on purpose — your output fills the
first one; real Spanish data fills the second.

## 0. Environment

```bash
cd <repo root>            # the folder containing hawkes_calibration/ and synthetic/
pip install scipy pandas --break-system-packages          # numpy assumed present
pip install xgboost scikit-learn --break-system-packages  # only for benchmarks B1/B3
export PYTHONPATH=.
```

Everything below runs from the repo root. If your shell kills long commands
(~45 s caps), stage regime A: call `simulate_world()`, pickle the result, then
`emit_dataset()` in a second command.

## 1. Generate the synthetic worlds

```bash
python -m synthetic.generate --regime A --out data/synthetic_A --seed 0
python -m synthetic.generate --regime B --out data/synthetic_B --seed 0
# repeat with --seed 1..4 for error bars
```

- Regime A: well-specified weekly two-stage world (sector count GLM with lagged
  excitation + risk-set choice model with a 26-week cooldown). ~50k deals,
  520 weeks, 12 sectors, 8,000 firms. Runs ~40-60 s.
- Regime B: misspecified daily world (firm-level intensity with 30-day-half-life
  self-inhibition, 7-day-half-life sector contagion, latent quality). Day-resolution
  deal dates. Runs ~15 s.
- Knobs: `simulate_regime_a(..., target_mean, effective_radius, latent_strength,
  covariate_scale)`, `simulate_regime_b(..., target_total_events, quality_strength,
  contagion_exponent, covariate_scale)`.

Each output folder contains the observed world (`deals.csv`, `companies.csv`,
`investors.csv`, `macro.csv`) and oracle-only files (`ground_truth.npz`,
`private_deal_map.csv`, `meta.json`). **No fitting code may read the oracle files;
they are for evaluation only.**

## 2. Load fitter-ready arrays (point-in-time safe)

```python
from synthetic.loaders import load_dataset, build_choice_sets

d  = load_dataset("data/synthetic_A")            # weekly arrays
cs = build_choice_sets("data/synthetic_A", max_candidates=64)  # event-level choice sets
```

- `d`: `sector_counts (T,M)`, `covariates_raw (T,3)` (as-of FFUND/CPI_YOY/RUNEMP),
  `features (T,N,5)` (LOCF firm features), `startup_counts`, `active`, `events (E,3)`.
  `active="true"` + `ground_truth=` path gives the oracle risk-set mask.
- `cs`: padded tensors for the funded-pool conditional logit — candidates are
  previously funded active firms of the event's sector, excluding the last-funded
  firm, plus one newcomer slot; features include `log1p(days_since_last_funding/30)`.

## 3. Reference fitters

```python
from synthetic.fast_fit import fit_sector_glm_fast, fit_choice_fast, fit_ranker_fast

a, b, exc = fit_sector_glm_fast(d.sector_counts, X_std, n_lags=4, train_end=416)
w0, u, tr_nll, te_nll, rnd_nll, top5, ok = fit_choice_fast(cs)
```

Standardize covariates with train-window moments. The package's loop fitters
(`hawkes_calibration.sector_ranker`) are the reference implementations; verify the
fast fitters match them on a subsample once (tolerance 1e-6), then use the fast ones.

## 4. The protocols to run

Defined in the design document's "Code and Test Plan" chapter; summary:

| ID | What | Pass criterion |
|----|------|----------------|
| E1 | Weekly GLM vs event-time fits; bucket width 1d/7d/28d; latent-confounding curve (`latent_strength` 0 / 0.15 / 0.3) | truth within 2 SE over 5 seeds; monotone radius-vs-latent curve |
| E2 | Choice model: gap functional form (log / binned / exp-decay), newcomer variants (ASC, context-ASC, representative entrant, immigrant split), sampled-softmax 32/64/128, exclusion rule | recovers true weights in regime A; newcomer share calibrated per quarter |
| E3 | Two-stage vs single-layer block Hawkes head-to-head (regime B) | joint NLL + inhibition ordering rank-corr >= 0.9 |
| E4 | Neural-operator accuracy vs exact solvers; wall-clock crossover | rel-L2 <= 2% inside training box |
| E5 | Walk-forward backtest: top-k, MRR, recall@K, PIT calibration, covariate ablation | beats historical-mean and random baselines |
| B1 | XGBoost ranking benchmark on the same choice sets (feature-parity + kitchen sink) | decision rule: material win -> missing structure |
| B2 | Covariate-only Poisson null; improvement decomposition covariates -> +excitation -> +timing | quantifies what excitation adds |
| B3 | Nonparametric intensity challenger (XGB / RF / small NN on binned likelihood with engineered history features, per the "Nonparametric Intensities" chapter) | compare held-out likelihood + calibration vs log-linear GLM and Hawkes |
| C  | Causal pipeline on known ground truth: naive vs backdoor-adjusted vs AIPW, cluster bootstrap, E-values | adjusted estimate covers truth; naive shows the expected bias |

Rules for every protocol: temporal splits only (train weeks < 416); 5 seeds;
quasi-Poisson dispersion reported next to any Poisson NLL; effective spectral
radius reported for any fitted excitation (never simulate above 0.95); every
number regenerable by one seeded script under `experiments/`.

## 5. B3 implementation sketch (the non-log-linear regime)

1. Build (unit, bin) rows: weekly bins per sector (or firm), exposure = bin length,
   features = as-of covariates + engineered history (time since last event, decayed
   event counts at half-lives {7, 30, 90} days, sector recent counts).
2. Fit XGBoost with `objective="count:poisson"`, `base_margin = log(exposure)`,
   monotone constraints where the economic sign is known, early stopping on a
   temporal validation tail. Same rows for a Poisson-deviance random forest and a
   small softplus-output NN.
3. Evaluate exactly like the GLM: held-out Poisson NLL, dispersion, PIT, and
   time-rescaling diagnostics on the fitted intensity. Uncertainty: firm-cluster
   bootstrap.

## 6. Where results go

Write one markdown + CSV per protocol under `results/` (`results/E1_seed{k}.csv`, ...)
and fill the corresponding empty tables in
`design-doc-unified/chapters/ch-results-synth.tex` (synthetic) — table shells and
`\todo` markers are already in place. Real-data results for Spain go in
`ch-results-spain.tex` only; never mix the two. Rebuild the document with
`pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`
from `design-doc-unified/`.
