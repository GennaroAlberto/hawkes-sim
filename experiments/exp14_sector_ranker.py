"""
Experiment 14: sector Hawkes/count model + startup risk-set ranker.

Run:
    PYTHONPATH=. python -m experiments.exp14_sector_ranker

The experiment creates a synthetic weekly startup-funding dataset, fits the
sector-level positive-lag count model and the within-sector dynamic risk-set
ranker, scores both on a held-out period, and writes metrics to
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
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonify(payload), f, indent=2)

    print("Experiment 14: sector count model + startup risk-set ranker")
    print(f"  total events: {metrics['n_events_total']}  train/test: {metrics['n_events_train']}/{metrics['n_events_test']}")
    print("  sector NLL per cell:")
    print(f"    model    = {metrics['sector_model_nll_per_cell']:.4f}")
    print(f"    baseline = {metrics['sector_baseline_nll_per_cell']:.4f}")
    print(f"    improvement = {metrics['sector_nll_improvement']:.4f}")
    print("  simulation MAE per sector-week:")
    print(f"    fitted simulator = {metrics['sim_sector_mae']:.4f}")
    print(f"    baseline         = {metrics['baseline_sector_mae']:.4f}")
    r = metrics["ranker"]
    print("  ranker held-out:")
    print(f"    NLL     = {r.get('nll', float('nan')):.4f}  random = {r.get('random_nll', float('nan')):.4f}")
    print(f"    MRR     = {r.get('mrr', float('nan')):.4f}  random = {r.get('random_mrr', float('nan')):.4f}")
    print(f"    top@5   = {r.get('top5', float('nan')):.4f}  random = {r.get('random_top5', float('nan')):.4f}")
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
