"""
Experiment 14: sector Hawkes/count model + startup second-stage models.

Run:
    PYTHONPATH=. python -m experiments.exp14_sector_ranker

The experiment creates a synthetic weekly startup-funding dataset, fits the
sector-level positive-lag count model and several within-sector second stages:

1. conditional risk-set ranker;
2. Cox-style startup survival model with an outside option;
3. discrete-time logistic hazard model.

It scores the held-out period and writes metrics to
``results/exp14_sector_ranker.json``.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass

import numpy as np

from hawkes_calibration import backtest_synthetic_pipeline


def _jsonify(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if is_dataclass(obj):
        return {k: _jsonify(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return obj


def _print_stage(name, metrics):
    print(f"  {name} held-out:")
    print(f"    NLL   = {metrics.get('nll', float('nan')):.4f}")
    print(f"    MRR   = {metrics.get('mrr', float('nan')):.4f}")
    print(f"    top@5 = {metrics.get('top5', float('nan')):.4f}")
    if "random_nll" in metrics:
        print(f"    random NLL/top@5 = {metrics.get('random_nll', float('nan')):.4f} / {metrics.get('random_top5', float('nan')):.4f}")
    if "outside_rate" in metrics:
        print(f"    outside rate/pred/AUC = {metrics.get('outside_rate', float('nan')):.4f} / {metrics.get('outside_pred_mean', float('nan')):.4f} / {metrics.get('outside_auc', float('nan')):.4f}")


def main():
    out = backtest_synthetic_pipeline(
        seed=7,
        T=180,
        train_end=120,
        n_sectors=11,
        startups_per_sector=35,
        n_lags=4,
        cooldown_weeks=26,
        n_paths=80,
    )
    metrics = out["metrics"]

    os.makedirs("results", exist_ok=True)
    path = "results/exp14_sector_ranker.json"
    payload = {
        "metrics": metrics,
        "sector_fit": out["sector_fit"],
        "ranker_fit": out["ranker_fit"],
        "survival_fit": out["survival_fit"],
        "hazard_fit": out["hazard_fit"],
        "tracked": out["tracked"],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonify(payload), f, indent=2)

    print("Experiment 14: sector count model + startup second-stage models")
    print(f"  total events: {metrics['n_events_total']}  train/test: {metrics['n_events_train']}/{metrics['n_events_test']}")
    print("  sector NLL per cell:")
    print(f"    model    = {metrics['sector_model_nll_per_cell']:.4f}")
    print(f"    baseline = {metrics['sector_baseline_nll_per_cell']:.4f}")
    print(f"    improvement = {metrics['sector_nll_improvement']:.4f}")
    print("  simulation MAE per sector-week:")
    print(f"    fitted simulator = {metrics['sim_sector_mae']:.4f}")
    print(f"    baseline         = {metrics['baseline_sector_mae']:.4f}")
    _print_stage("risk-set ranker", metrics["ranker"])
    _print_stage("startup survival + outside", metrics["survival"])
    _print_stage("discrete hazard", metrics["discrete_hazard"])
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
