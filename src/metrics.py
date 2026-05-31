from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, precision_recall_curve)

def compute_auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the ROC curve (AUC-ROC)."""
    return float(roc_auc_score(y_true, y_score))


def compute_auc_pr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the Precision-Recall curve (AUC-PR / Average Precision)."""
    return float(average_precision_score(y_true, y_score))


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


def compute_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(precision_score(y_true, y_pred, zero_division=0))


def compute_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(recall_score(y_true, y_pred, zero_division=0))


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, zero_division=0))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> dict[str, float]:
    """
    Compute a full suite of evaluation metrics.

    Parameters
    y_true  : ground-truth binary labels (1 = outlier)
    y_pred  : hard binary predictions (1 = outlier)
    y_score : continuous anomaly scores (higher = more anomalous). If None, AUC-ROC and AUC-PR are omitted.

    Returns
    dict with keys:
        accuracy, precision, recall, f1,
        auc_roc (if y_score provided),
        auc_pr (if y_score provided),
        tp, fp, tn, fn
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics: dict[str, float] = {
        "accuracy": compute_accuracy(y_true, y_pred),
        "precision": compute_precision(y_true, y_pred),
        "recall": compute_recall(y_true, y_pred),
        "f1": compute_f1(y_true, y_pred),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }

    if y_score is not None:
        try:
            metrics["auc_roc"] = compute_auc_roc(y_true, y_score)
        except ValueError:
            metrics["auc_roc"] = float("nan")
        try:
            metrics["auc_pr"] = compute_auc_pr(y_true, y_score)
        except ValueError:
            metrics["auc_pr"] = float("nan")

    return metrics


def evaluate_all_detectors(detectors, X: np.ndarray, y_true: np.ndarray, contamination: float | None = None) -> pd.DataFrame:
    """
    Fit every detector in `detectors` on X (labels NOT used during training), then evaluate against y_true.

    Parameters
    detectors : list of BaseOutlierDetector instances (from models.py)
    X : scaled feature matrix
    y_true  : ground-truth binary labels (1 = outlier)
    contamination: if provided, passed to each detector's fit() method

    Returns
    pd.DataFrame  - one row per detector, columns = metric names
    """
    records = []
    for det in detectors:
        fit_kwargs: dict = {}
        if contamination is not None:
            fit_kwargs["contamination"] = contamination

        det.fit(X, **fit_kwargs)

        y_pred = det.predict(X)
        y_score = det.score_samples(X)

        row = {"detector": det.name}
        row.update(evaluate(y_true, y_pred, y_score))
        records.append(row)

    df = pd.DataFrame(records).set_index("detector")
    cols_order = ["auc_roc", "auc_pr", "f1", "accuracy", "precision", "recall",
                  "tp", "fp", "tn", "fn"]
    df = df[[c for c in cols_order if c in df.columns]]
    return df


def get_roc_curve_data(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (fpr, tpr, thresholds) for an ROC curve plot."""
    return roc_curve(y_true, y_score)

def get_pr_curve_data(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (precision, recall, thresholds) for a PR curve plot."""
    return precision_recall_curve(y_true, y_score)


def format_results_table(df: pd.DataFrame, float_fmt: str = ".4f") -> pd.DataFrame:
    """
    Return a display-ready copy of the results DataFrame with rounded values and highlighted best values per metric column.

    Parameters
    df : output of evaluate_all_detectors()
    float_fmt : format string for rounding

    Returns
    pd.DataFrame with float columns rounded to float_fmt
    """
    float_cols = [c for c in df.columns if df[c].dtype == float and c not in ("tp", "fp", "tn", "fn")]
    display = df.copy()
    for c in float_cols:
        display[c] = display[c].map(lambda v: float(f"{v:{float_fmt}}"))
    return display


def rank_detectors(df: pd.DataFrame, primary: str = "auc_roc") -> pd.DataFrame:
    """
    Sort the results DataFrame by `primary` metric (descending) and add a rank column.

    Parameters
    df : output of evaluate_all_detectors()
    primary : column to rank by

    Returns
    pd.DataFrame sorted by primary metric with a 'rank' column prepended
    """
    if primary not in df.columns:
        raise ValueError(f"Column '{primary}' not in DataFrame. "
                         f"Available: {list(df.columns)}")
    ranked = df.sort_values(primary, ascending=False).copy()
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked
