from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.model_selection import StratifiedKFold

_here = os.path.dirname(os.path.abspath(__file__))
_src  = os.path.dirname(_here)
for p in [_here, _src]:
    if p not in sys.path:
        sys.path.insert(0, p)

from preprocessing import handle_missing, scale


def load_odds(results_base: str = None) -> dict[str, dict]:
    """
    Load all ODDS datasets from data/odds/*.csv
    Returns {name: {"X": ndarray, "y": ndarray}}
    """
    root = os.path.dirname(_src)
    odds_dir = os.path.join(root, "data", "odds")
    datasets = {}
    if not os.path.isdir(odds_dir):
        print(f"[WARN] ODDS dir not found: {odds_dir}")
        return datasets
    for fname in sorted(os.listdir(odds_dir)):
        if not fname.endswith(".csv"):
            continue
        name = fname.replace(".csv", "")
        df = pd.read_csv(os.path.join(odds_dir, fname))
        y = df["y"].values.astype(int)
        X = df.drop(columns=["y"]).values.astype(float)
        datasets[name] = {"X": X, "y": y}
    return datasets


def load_clustering(results_base: str = None) -> dict[str, dict]:
    """
    Load clustering-benchmark datasets from data/clustering_benchmark/*.csv
    Returns {name: {"X": ndarray, "y": ndarray}}  (y = is_noise)
    """
    root = os.path.dirname(_src)
    clust_dir = os.path.join(root, "data", "clustering_benchmark")
    datasets = {}
    if not os.path.isdir(clust_dir):
        print(f"[WARN] Clustering dir not found: {clust_dir}")
        return datasets
    for fname in sorted(os.listdir(clust_dir)):
        if not fname.endswith(".csv"):
            continue
        name = fname.replace(".csv", "")
        df   = pd.read_csv(os.path.join(clust_dir, fname))
        if "is_noise" not in df.columns:
            continue
        y = df["is_noise"].values.astype(int)
        if y.sum() == 0:
            continue
        X = df.drop(columns=["cluster", "is_noise"]).values.astype(float)
        datasets[name] = {"X": X, "y": y}
    return datasets


def load_all_datasets() -> dict[str, dict]:
    d = load_odds()
    d.update(load_clustering())
    return d

def preprocess(X: np.ndarray, y: np.ndarray):
    X_clean = handle_missing(X)
    X_scaled, _ = scale(X_clean)
    contamination = float(y.mean())
    return X_scaled, contamination

def cv_score(detector_fn, X: np.ndarray, y: np.ndarray,
             contamination: float, n_splits: int = 5) -> dict[str, float]:
    """
    Stratified k-fold CV.  detector_fn(contamination) -> fitted detector instance.
    Returns mean +- std for auc_roc, auc_pr, f1.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs, aps, f1s = [], [], []

    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_te = y[test_idx]

        try:
            det = detector_fn(contamination)
            det.fit(X_tr, contamination=contamination)
            scores = det.score_samples(X_te)
            preds = det.predict(X_te)

            if len(np.unique(y_te)) < 2:
                continue

            aucs.append(roc_auc_score(y_te, scores))
            aps.append(average_precision_score(y_te, scores))
            f1s.append(f1_score(y_te, preds, zero_division=0))
        except Exception as e:
            print(f"    [SKIP] {type(e).__name__}: {e}")
            continue

    def _stats(vals):
        if not vals:
            return float("nan"), float("nan")
        return float(np.mean(vals)), float(np.std(vals))

    auc_mean, auc_std = _stats(aucs)
    ap_mean, ap_std = _stats(aps)
    f1_mean, f1_std = _stats(f1s)

    return {
        "auc_roc": auc_mean, "auc_roc_std": auc_std,
        "auc_pr": ap_mean, "auc_pr_std": ap_std,
        "f1": f1_mean, "f1_std": f1_std,
    }


def save_results(rows: list[dict], out_path: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}  ({len(df)} rows)")
    return df