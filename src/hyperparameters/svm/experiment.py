"""
src/hyperparameters/svm/experiment.py
----------------------------------------
Hyperparameter sweep for One-Class SVM.

Params explored:
  nu : [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
  kernel : [rbf, linear, poly, sigmoid]
  gamma : [scale, auto, 0.001, 0.01, 0.1, 1.0]  (rbf only)

Output:
  results/hyperparameters/svm/
    nu_sweep.csv
    kernel_sweep.csv
    gamma_sweep.csv
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hp_utils import load_all_datasets, preprocess, cv_score, save_results
from models import OneClassSVMDetector
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "results", "hyperparameters", "svm")

NU_GRID = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
KERNEL_GRID = ["rbf", "linear", "poly", "sigmoid"]
GAMMA_GRID = ["scale", "auto", 0.001, 0.01, 0.1, 1.0]

MAX_N = 5_000


def subsample(X, y, max_n=MAX_N, seed=42):
    """Stratified subsample to max_n points, preserving outlier ratio."""
    if len(X) <= max_n:
        return X, y
    rng = np.random.default_rng(seed)
    # sample separately from each class to preserve contamination rate
    idx_out = np.where(y == 1)[0]
    idx_in  = np.where(y == 0)[0]
    ratio = len(idx_out) / len(y)
    n_out = max(1, int(max_n * ratio))
    n_in = max_n - n_out
    idx = np.concatenate([rng.choice(idx_out, min(n_out, len(idx_out)), replace=False), rng.choice(idx_in,  min(n_in,  len(idx_in)),  replace=False)])
    return X[idx], y[idx]


def run():
    datasets = load_all_datasets()
    print(f"Loaded {len(datasets)} datasets: {list(datasets.keys())}\n")

    # ── 1. nu sweep (rbf kernel, gamma=scale) ────────────────────────────────
    print("=== nu sweep ===")
    nu_rows = []
    for ds_name, data in datasets.items():
        X, contam = preprocess(data["X"], data["y"])
        X, y = subsample(X, data["y"])
        print(f"  {ds_name} (n={len(X)}, contam={contam:.3f})")
        for nu in NU_GRID:
            metrics = cv_score(lambda c, _nu=nu: OneClassSVMDetector(nu=_nu, kernel="rbf", gamma="scale"), X, y, contam)
            nu_rows.append({"dataset": ds_name, "nu": nu, "true_contam": contam, **metrics})
    save_results(nu_rows, os.path.join(OUT, "nu_sweep.csv"))

    # ── 2. kernel sweep (nu=true_contam, gamma=scale) ────────────────────────
    print("\n=== kernel sweep ===")
    kernel_rows = []
    for ds_name, data in datasets.items():
        X, contam = preprocess(data["X"], data["y"])
        X, y = subsample(X, data["y"])
        print(f"  {ds_name} (n={len(X)})")
        for kernel in KERNEL_GRID:
            metrics = cv_score(lambda c, _k=kernel: OneClassSVMDetector(nu=c, kernel=_k, gamma="scale"), X, y, contam)
            kernel_rows.append({"dataset": ds_name, "kernel": kernel, "true_contam": contam, **metrics})
    save_results(kernel_rows, os.path.join(OUT, "kernel_sweep.csv"))

    # ── 3. gamma sweep (rbf only, nu=true_contam) ────────────────────────────
    print("\n=== gamma sweep (rbf) ===")
    gamma_rows = []
    for ds_name, data in datasets.items():
        X, contam = preprocess(data["X"], data["y"])
        X, y = subsample(X, data["y"])
        print(f"  {ds_name} (n={len(X)})")
        for gamma in GAMMA_GRID:
            metrics = cv_score(lambda c, _g=gamma: OneClassSVMDetector(nu=c, kernel="rbf", gamma=_g), X, y, contam)
            gamma_rows.append({"dataset": ds_name, "gamma": str(gamma),  "true_contam": contam, **metrics})
    save_results(gamma_rows, os.path.join(OUT, "gamma_sweep.csv"))

    print("\nDone.")

if __name__ == "__main__":
    run()
