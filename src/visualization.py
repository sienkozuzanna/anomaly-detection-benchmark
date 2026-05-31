from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from metrics import get_roc_curve_data, get_pr_curve_data

PALETTE = {
    "One-Class SVM": "#E63946", 
    "Isolation Forest": "#457B9D", 
    "Local Outlier Factor":"#2A9D8F", 
    "DBSCAN": "#E9C46A", 
    "ECOD": "#9B5DE5", 
}
_DEFAULT_COLORS = list(PALETTE.values())

INLIER_COLOR  = "#AECBF0" 
OUTLIER_COLOR = "#E63946"

FIG_DPI = 130

def _color_for(name: str, idx: int = 0) -> str:
    return PALETTE.get(name, _DEFAULT_COLORS[idx % len(_DEFAULT_COLORS)])


def plot_scatter_ground_truth(X_2d: np.ndarray, y_true: np.ndarray, title: str = "Ground Truth", ax: plt.Axes | None = None) -> plt.Figure:
    """
    Scatter plot of a 2-D projection coloured by ground-truth labels.

    Parameters
    X_2d : (n, 2) projected coordinates (e.g. from PCA / UMAP)
    y_true : binary labels, 1 = outlier
    title : plot title
    ax : optional existing Axes; creates new figure if None
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(5, 4), dpi=FIG_DPI)
    else:
        fig = ax.get_figure()

    inliers = y_true == 0
    outliers = y_true == 1

    ax.scatter(X_2d[inliers, 0], X_2d[inliers, 1], c=INLIER_COLOR,  s=18, alpha=0.7, linewidths=0, label="Inlier")
    ax.scatter(X_2d[outliers, 0], X_2d[outliers, 1], c=OUTLIER_COLOR, s=30, alpha=0.9, marker="x", linewidths=1.2, label="Outlier")

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Component 1", fontsize=8)
    ax.set_ylabel("Component 2", fontsize=8)
    ax.legend(fontsize=8, framealpha=0.6)
    ax.tick_params(labelsize=7)

    if standalone:
        fig.tight_layout()
    return fig

def plot_scatter_predictions(X_2d: np.ndarray, y_true: np.ndarray, predictions: dict[str, np.ndarray], dataset_name: str = "", ncols: int = 3) -> plt.Figure:
    """
    Grid of scatter plots: one per detector + one for ground truth.

    Parameters
    X_2d : (n, 2) 2-D projection
    y_true : ground-truth binary labels
    predictions : {detector_name: y_pred array}
    dataset_name : used in the figure suptitle
    ncols : number of subplot columns
    """
    items = [("Ground Truth", y_true)] + list(predictions.items())
    n_plots = len(items)
    nrows = int(np.ceil(n_plots / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.8), dpi=FIG_DPI)
    axes = np.array(axes).flatten()

    for idx, (name, labels) in enumerate(items):
        ax = axes[idx]
        inliers = labels == 0
        outliers = labels == 1

        ax.scatter(X_2d[inliers,  0], X_2d[inliers,  1], c=INLIER_COLOR,  s=14, alpha=0.65, linewidths=0)
        ax.scatter(X_2d[outliers, 0], X_2d[outliers, 1], c=OUTLIER_COLOR, s=24, alpha=0.90, marker="x", linewidths=1.2)

        n_out = int(labels.sum())
        ax.set_title(f"{name}\n(n_outlier={n_out})", fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=6)
        ax.set_xlabel("C1", fontsize=7)
        ax.set_ylabel("C2", fontsize=7)

    for ax in axes[n_plots:]:
        ax.set_visible(False)

    legend_elements = [mpatches.Patch(color=INLIER_COLOR,  label="Inlier"), mpatches.Patch(color=OUTLIER_COLOR, label="Outlier / Detected")]
    fig.legend(handles=legend_elements, loc="lower right", fontsize=9, framealpha=0.7)
    sup = f"Outlier Detection – {dataset_name}" if dataset_name else "Outlier Detection"
    fig.suptitle(sup, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig

def plot_roc_curves(y_true: np.ndarray, scores: dict[str, np.ndarray], title: str = "ROC Curves") -> plt.Figure:
    """
    Overlay ROC curves for multiple detectors on one axes.

    Parameters
    y_true : ground-truth binary labels
    scores : {detector_name: anomaly_score_array}
    title : plot title
    """
    fig, ax = plt.subplots(figsize=(6, 5), dpi=FIG_DPI)

    for idx, (name, y_score) in enumerate(scores.items()):
        try:
            fpr, tpr, _ = get_roc_curve_data(y_true, y_score)
            auc = np.trapz(tpr, fpr)
            color = _color_for(name, idx)
            ax.plot(fpr, tpr, color=color, lw=1.8, label=f"{name}  (AUC={auc:.3f})")
        except Exception:
            pass

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=9)
    ax.set_ylabel("True Positive Rate", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.7)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig

def plot_pr_curves(y_true: np.ndarray, scores: dict[str, np.ndarray], title: str = "Precision-Recall Curves") -> plt.Figure:
    """Overlay PR curves for multiple detectors."""
    baseline = float(y_true.mean())
    fig, ax = plt.subplots(figsize=(6, 5), dpi=FIG_DPI)

    for idx, (name, y_score) in enumerate(scores.items()):
        try:
            prec, rec, _ = get_pr_curve_data(y_true, y_score)
            ap = np.trapz(prec, rec) * -1 # precision_recall_curve returns decreasing recall
            ap_sk = float(__import__("sklearn.metrics", fromlist=["average_precision_score"]).average_precision_score(y_true, y_score))
            color = _color_for(name, idx)
            ax.plot(rec, prec, color=color, lw=1.8, label=f"{name}  (AP={ap_sk:.3f})")
        except Exception:
            pass

    ax.axhline(baseline, color="k", lw=0.8, linestyle="--", alpha=0.5, label=f"Random  ({baseline:.3f})")
    ax.set_xlabel("Recall", fontsize=9)
    ax.set_ylabel("Precision", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.7)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig

def plot_metric_bars(results_df: pd.DataFrame, metrics: list[str] | None = None, title: str = "Detector Comparison") -> plt.Figure:
    """
    Grouped bar chart comparing detectors across multiple metrics.

    Parameters
    results_df : DataFrame from metrics.evaluate_all_detectors() (index = detector name, columns = metric names)
    metrics : list of column names to plot; defaults to ['auc_roc', 'auc_pr', 'f1', 'precision', 'recall']
    title : figure title
    """
    if metrics is None:
        metrics = [m for m in ["auc_roc", "auc_pr", "f1", "precision", "recall"] if m in results_df.columns]

    detectors = list(results_df.index)
    n_det = len(detectors)
    n_met = len(metrics)
    x = np.arange(n_met)
    width = 0.8 / n_det

    fig, ax = plt.subplots(figsize=(max(8, n_met * 1.8), 5), dpi=FIG_DPI)

    for i, det in enumerate(detectors):
        vals   = [results_df.loc[det, m] if m in results_df.columns else 0.0 for m in metrics]
        offset = (i - n_det / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9, label=det, color=_color_for(det, i), alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics], fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.7, loc="upper right")
    ax.tick_params(axis="y", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig

def plot_metric_heatmap(results_df: pd.DataFrame, metrics: list[str] | None = None, title: str = "Performance Heatmap", cmap: str = "YlGn") -> plt.Figure:
    """
    Heatmap of metric values; rows = detectors, columns = metrics.

    Parameters
    results_df : DataFrame from metrics.evaluate_all_detectors()
    metrics : columns to include; defaults to main performance metrics
    title : plot title
    cmap : matplotlib colour map
    """
    if metrics is None:
        metrics = [m for m in ["auc_roc", "auc_pr", "f1", "accuracy", "precision", "recall"] if m in results_df.columns]

    data = results_df[metrics].values.astype(float)

    fig, ax = plt.subplots(figsize=(len(metrics) * 1.5 + 1, len(results_df) * 0.8 + 1.2), dpi=FIG_DPI)
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics], fontsize=9)
    ax.set_yticks(range(len(results_df)))
    ax.set_yticklabels(list(results_df.index), fontsize=9)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            text_color = "white" if v > 0.65 else "black"
            ax.text(j, i, f"{v:.3f}" if not np.isnan(v) else "—", ha="center", va="center", fontsize=8.5, color=text_color, fontweight="bold")

    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig

def plot_score_distributions(y_true: np.ndarray, scores: dict[str, np.ndarray], title: str = "Anomaly Score Distributions",) -> plt.Figure:
    """
    For each detector, plot overlapping histograms of anomaly scores split by ground-truth class (inlier vs outlier).

    Parameters
    y_true : ground-truth binary labels
    scores : {detector_name: anomaly_score_array}
    title : figure suptitle
    """
    n = len(scores)
    ncols = min(n, 3)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3), dpi=FIG_DPI)
    axes = np.array(axes).flatten()

    for idx, (name, y_score) in enumerate(scores.items()):
        ax = axes[idx]
        s_in = y_score[y_true == 0]
        s_out = y_score[y_true == 1]

        bins = min(50, max(10, len(y_score) // 20))
        ax.hist(s_in,  bins=bins, color=INLIER_COLOR,  alpha=0.65, label="Inlier",  density=True, edgecolor="white", linewidth=0.3)
        ax.hist(s_out, bins=bins, color=OUTLIER_COLOR, alpha=0.70, label="Outlier", density=True, edgecolor="white", linewidth=0.3)

        ax.set_title(name, fontsize=9, fontweight="bold")
        ax.set_xlabel("Anomaly score", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.legend(fontsize=7, framealpha=0.6)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_multi_dataset_summary(all_results: dict[str, pd.DataFrame], metric: str = "auc_roc", title: str | None = None) -> plt.Figure:
    """
    Compare detector performance across multiple datasets using one metric.

    Parameters
    all_results : {dataset_name: results_df from evaluate_all_detectors()}
    metric : metric column to compare
    title : figure title; defaults to metric name
    """
    if title is None:
        title = f"Cross-dataset comparison – {metric.upper()}"

    datasets = list(all_results.keys())
    detectors = list(next(iter(all_results.values())).index)

    n_ds = len(datasets)
    n_det = len(detectors)
    x  = np.arange(n_ds)
    width = 0.8 / n_det

    fig, ax = plt.subplots(figsize=(max(8, n_ds * 1.5), 5), dpi=FIG_DPI)

    for i, det in enumerate(detectors):
        vals = [all_results[ds].loc[det, metric] if det in all_results[ds].index else np.nan for ds in datasets]
        offset = (i - n_det / 2 + 0.5) * width
        ax.bar(x + offset, vals, width * 0.9, label=det, color=_color_for(det, i), alpha=0.85, edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=25, ha="right", fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel(metric.replace("_", " ").upper(), fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig
