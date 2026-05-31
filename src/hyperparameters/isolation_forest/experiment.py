"""
src/hyperparameters/isolation_forest/experiment.py
------------------------------------------
Hyperparameter sweep for Isolation Forest.

Params explored:
  n_estimators : [10, 25, 50, 100, 200, 500]
  contamination : [0.01, 0.05, 0.1, 0.2, 0.3, 0.5] vs true contamination
  max_features: [0.25, 0.5, 0.75, 1.0]  (fraction of features)

Output:
  results/hyperparameters/isolation_forest/
    n_estimators_sweep.csv
    contamination_sweep.csv
    max_features_sweep.csv
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hp_utils import load_all_datasets, preprocess, cv_score, save_results
from sklearn.ensemble import IsolationForest
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "results", "hyperparameters", "isolation_forest")

N_EST_GRID   = [10, 25, 50, 100, 200, 500]
CONTAM_GRID  = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]
MAXFEAT_GRID = [0.25, 0.5, 0.75, 1.0]

class _IFWrapper:
    """Thin wrapper so IsolationForest fits the BaseOutlierDetector interface."""
    def __init__(self, n_estimators=200, contamination="auto", max_features=1.0):
        self.n_estimators  = n_estimators
        self.contamination = contamination
        self.max_features  = max_features
        self._model = None

    def fit(self, X, contamination=None):
        c = contamination if contamination is not None else self.contamination
        self._model = IsolationForest(n_estimators=self.n_estimators, contamination=c, max_features=self.max_features, random_state=42, n_jobs=-1)
        self._model.fit(X)
        return self

    def predict(self, X):
        return (self._model.predict(X) == -1).astype(int)

    def score_samples(self, X):
        return -self._model.score_samples(X)


def run():
    datasets = load_all_datasets()
    print(f"Loaded {len(datasets)} datasets\n")

    # ── 1. n_estimators sweep ────────────────────────────────────────────────
    print("=== n_estimators sweep ===")
    rows = []
    for ds_name, data in datasets.items():
        X, contam = preprocess(data["X"], data["y"])
        print(f"  {ds_name}")
        for n in N_EST_GRID:
            metrics = cv_score(lambda c, _n=n: _IFWrapper(n_estimators=_n, contamination=c), X, data["y"], contam)
            rows.append({"dataset": ds_name, "n_estimators": n, "true_contam": contam, **metrics})
    save_results(rows, os.path.join(OUT, "n_estimators_sweep.csv"))

    # ── 2. contamination sweep ───────────────────────────────────────────────
    print("\n=== contamination sweep ===")
    rows = []
    for ds_name, data in datasets.items():
        X, contam = preprocess(data["X"], data["y"])
        print(f"  {ds_name} (true={contam:.3f})")
        for c_val in CONTAM_GRID:
            metrics = cv_score(lambda c, _cv=c_val: _IFWrapper(n_estimators=200, contamination=_cv), X, data["y"], contam)
            rows.append({"dataset": ds_name, "contamination": c_val, "true_contam": contam, **metrics})
    save_results(rows, os.path.join(OUT, "contamination_sweep.csv"))

    # ── 3. max_features sweep ────────────────────────────────────────────────
    print("\n=== max_features sweep ===")
    rows = []
    for ds_name, data in datasets.items():
        X, contam = preprocess(data["X"], data["y"])
        print(f"  {ds_name}")
        for mf in MAXFEAT_GRID:
            metrics = cv_score(lambda c, _mf=mf: _IFWrapper(n_estimators=200, contamination=c, max_features=_mf), X, data["y"], contam)
            rows.append({"dataset": ds_name, "max_features": mf, "true_contam": contam, **metrics})
    save_results(rows, os.path.join(OUT, "max_features_sweep.csv"))

    print("\nDone.")

if __name__ == "__main__":
    run()