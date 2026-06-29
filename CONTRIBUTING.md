# Contributing

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # core + pytest, ruff, matplotlib, scipy
```

Optional heavy backends (only needed for the learned-operator / inverse modules):

```bash
pip install -e ".[jax]"          # JAX PINO / PINN, differentiable inverse
pip install -e ".[tf]"           # TensorFlow operators
pip install -e ".[all]"          # everything
```

## Running the tests

```bash
pytest                            # all tests; JAX/TF tests skip if not installed
pytest tests/test_interval_censored.py
```

The core package imports with **numpy only**. SciPy is optional — when it is absent the
`sector_ranker` estimators fall back to a numpy optimiser, so `pytest` still runs (install
the `dev` extra to use the faster SciPy path). Tests that require JAX or TensorFlow skip
automatically when those packages are not installed.

## Linting and formatting

```bash
ruff check .                      # lint
ruff check --fix .                # autofix
ruff format .                     # format (line length 100)
```

Configuration lives in `pyproject.toml` (`[tool.ruff]`, `[tool.black]`, line length 100).

## Layout

```
hawkes_calibration/   the package (eventtime, mbpp, operators, sector_ranker, optim)
tests/                pytest suite
experiments/          runnable demos that reproduce the figures in results/
paper/                LaTeX sources and compiled PDFs
docs/                 design notes
```

## Conventions

- New estimators expose a small, typed result dataclass and a `fit_*` function.
- Optional dependencies are imported lazily or behind a `try/except ImportError`; the
  numpy core must always import.
- Every numerical claim in a report should be backed by a test or an experiment script.
