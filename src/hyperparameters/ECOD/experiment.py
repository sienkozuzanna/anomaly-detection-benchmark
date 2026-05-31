"""
src/hyperparameters/ECOD/experiment.py
--------------------------------------
Hyperparameter sweep for ECOD.

ECOD is parameter-free — the only tunable parameter is contamination,
which sets the decision threshold (does NOT affect the anomaly scores,
only the binary classification cutoff).

Params explored:
  contamination : [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]

Output:
  results/hyperparameters/ecod/
    contamination_sweep.csv
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hp_utils import load_all_datasets, preprocess, cv_score, save_results, _src
import numpy as np

sys.path.insert(0, _src)
from models import ECODDetector

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                   "results", "hyperparameters", "ecod")

CONTAM_GRID = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]


def run():
    datasets = load_all_datasets()
    print(f"Loaded {len(datasets)} datasets\n")

    # ── contamination sweep ──────────────────────────────────────────────────
    print("=== contamination sweep ===")
    rows = []
    for ds_name, data in datasets.items():
        X, contam = preprocess(data["X"], data["y"])
        print(f"  {ds_name} (true={contam:.3f})")
        for c_val in CONTAM_GRID:
            metrics = cv_score(lambda c, _cv=c_val: ECODDetector(contamination=_cv), X, data["y"], contam)
            rows.append({
                "dataset":      ds_name,
                "contamination": c_val,
                "true_contam":  contam,
                **metrics,
            })
    save_results(rows, os.path.join(OUT, "contamination_sweep.csv"))

    print("\nDone.")

if __name__ == "__main__":
    run()
