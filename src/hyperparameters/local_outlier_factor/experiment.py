"""
src/hyperparameters/local_outlier_factor/experiment.py
--------------------------------------
Hyperparameter sweep for Local Outlier Factor.

Params explored:
  n_neighbors : [5, 10, 15, 20, 30, 50, 100]
  metric : [euclidean, manhattan, cosine, minkowski]
  contamination : [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]

Output:
  results/hyperparameters/local_outlier_factor/
    n_neighbors_sweep.csv
    metric_sweep.csv
    contamination_sweep.csv
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hp_utils import load_all_datasets, preprocess, cv_score, save_results
from sklearn.neighbors import LocalOutlierFactor
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "results", "hyperparameters", "local_outlier_factor")

N_NEIGH_GRID = [5, 10, 15, 20, 30, 50, 100]
METRIC_GRID = ["euclidean", "manhattan", "cosine", "minkowski"]
CONTAM_GRID = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]


class _LOFWrapper:
    def __init__(self, n_neighbors=20, metric="euclidean", contamination="auto"):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.contamination = contamination
        self._model = None

    def fit(self, X, contamination=None):
        c = contamination if contamination is not None else self.contamination
        self._model = LocalOutlierFactor(n_neighbors=min(self.n_neighbors, len(X) - 1), metric=self.metric, contamination=c, novelty=True, n_jobs=-1)
        self._model.fit(X)
        return self

    def predict(self, X):
        return (self._model.predict(X) == -1).astype(int)

    def score_samples(self, X):
        return -self._model.score_samples(X)


def run():
    datasets = load_all_datasets()
    print(f"Loaded {len(datasets)} datasets\n")

    # ── 1. n_neighbors sweep ─────────────────────────────────────────────────
    print("=== n_neighbors sweep ===")
    rows = []
    for ds_name, data in datasets.items():
        X, contam = preprocess(data["X"], data["y"])
        print(f"  {ds_name} (n={len(X)})")
        for k in N_NEIGH_GRID:
            if k >= len(X):
                continue
            metrics = cv_score(lambda c, _k=k: _LOFWrapper(n_neighbors=_k, contamination=c), X, data["y"], contam)
            rows.append({"dataset": ds_name, "n_neighbors": k, "true_contam": contam, **metrics})
    save_results(rows, os.path.join(OUT, "n_neighbors_sweep.csv"))

    # ── 2. metric sweep ──────────────────────────────────────────────────────
    print("\n=== metric sweep ===")
    rows = []
    for ds_name, data in datasets.items():
        X, contam = preprocess(data["X"], data["y"])
        print(f"  {ds_name}")
        for metric in METRIC_GRID:
            metrics = cv_score(lambda c, _m=metric: _LOFWrapper(n_neighbors=20, metric=_m, contamination=c), X, data["y"], contam)
            rows.append({"dataset": ds_name, "metric": metric, "true_contam": contam, **metrics})
    save_results(rows, os.path.join(OUT, "metric_sweep.csv"))

    # ── 3. contamination sweep ───────────────────────────────────────────────
    print("\n=== contamination sweep ===")
    rows = []
    for ds_name, data in datasets.items():
        X, contam = preprocess(data["X"], data["y"])
        print(f"  {ds_name} (true={contam:.3f})")
        for c_val in CONTAM_GRID:
            metrics = cv_score(lambda c, _cv=c_val: _LOFWrapper(n_neighbors=20, contamination=_cv), X, data["y"], contam)
            rows.append({"dataset": ds_name, "contamination": c_val, "true_contam": contam, **metrics})
    save_results(rows, os.path.join(OUT, "contamination_sweep.csv"))

    print("\nDone.")

if __name__ == "__main__":
    run()