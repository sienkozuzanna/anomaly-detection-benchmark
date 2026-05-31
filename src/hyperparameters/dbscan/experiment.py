"""
src/hyperparameters/dbscan/experiment.py
-----------------------------------------
Hyperparameter sweep for DBSCAN.

Params explored:
  eps x min_samples : 2D grid heatmap
  eps : [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]  (per dataset, relative to auto-eps)
  min_samples : [2, 3, 5, 7, 10, 15, 20]

Output:
  results/hyperparameters/dbscan/
    eps_x_minsamples_grid.csv <- full 2D grid (eps x min_samples per dataset)
    eps_sweep.csv <- eps as fraction of auto-eps
    minsamples_sweep.csv
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hp_utils import load_all_datasets, preprocess, cv_score, save_results, _src
from sklearn.neighbors import NearestNeighbors
import numpy as np

sys.path.insert(0, _src)
from models import DBSCANDetector

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "results", "hyperparameters", "dbscan")

EPS_FRACTIONS  = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]  # multipliers of auto-eps
MIN_SAMPLES_GRID = [2, 3, 5, 7, 10, 15, 20]


def auto_eps(X, percentile=90.0, min_eps=0.05):
    k = min(5, len(X) - 1)
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    d, _ = nbrs.kneighbors(X)
    return max(float(np.percentile(d[:, -1], percentile)), min_eps)


def run():
    datasets = load_all_datasets()
    print(f"Loaded {len(datasets)} datasets\n")

    # ── 1. Full 2D grid: eps × min_samples ───────────────────────────────────
    print("=== eps × min_samples grid ===")
    grid_rows = []
    for ds_name, data in datasets.items():
        X, contam = preprocess(data["X"], data["y"])
        base_eps = auto_eps(X)
        print(f"  {ds_name}  auto_eps={base_eps:.3f}")
        for frac in EPS_FRACTIONS:
            eps_val = base_eps * frac
            for ms in MIN_SAMPLES_GRID:
                metrics = cv_score(
                    lambda c, _e=eps_val, _ms=ms: DBSCANDetector(eps=_e, min_samples=_ms),
                    X, data["y"], contam,
                )
                grid_rows.append({
                    "dataset": ds_name,
                    "eps_fraction": frac,
                    "eps_abs": round(eps_val, 4),
                    "min_samples": ms,
                    "auto_eps": round(base_eps, 4),
                    "true_contam": contam,
                    **metrics,
                })
    save_results(grid_rows, os.path.join(OUT, "eps_x_minsamples_grid.csv"))

    # ── 2. eps sweep (min_samples=5 fixed) ───────────────────────────────────
    print("\n=== eps sweep (min_samples=5) ===")
    eps_rows = []
    for ds_name, data in datasets.items():
        X, contam = preprocess(data["X"], data["y"])
        base_eps = auto_eps(X)
        for frac in EPS_FRACTIONS:
            eps_val = base_eps * frac
            metrics = cv_score(
                lambda c, _e=eps_val: DBSCANDetector(eps=_e, min_samples=5),
                X, data["y"], contam,
            )
            eps_rows.append({"dataset": ds_name, "eps_fraction": frac, "eps_abs": round(eps_val, 4), "auto_eps": round(base_eps, 4),
                             "true_contam": contam, **metrics})
    save_results(eps_rows, os.path.join(OUT, "eps_sweep.csv"))

    # ── 3. min_samples sweep (eps=auto) ──────────────────────────────────────
    print("\n=== min_samples sweep (eps=auto) ===")
    ms_rows = []
    for ds_name, data in datasets.items():
        X, contam = preprocess(data["X"], data["y"])
        base_eps = auto_eps(X)
        for ms in MIN_SAMPLES_GRID:
            metrics = cv_score(lambda c, _ms=ms: DBSCANDetector(eps=base_eps, min_samples=_ms), X, data["y"], contam)
            ms_rows.append({"dataset": ds_name, "min_samples": ms, "auto_eps": round(base_eps, 4),
                            "true_contam": contam, **metrics})
    save_results(ms_rows, os.path.join(OUT, "minsamples_sweep.csv"))

    print("\nDone.")

if __name__ == "__main__":
    run()