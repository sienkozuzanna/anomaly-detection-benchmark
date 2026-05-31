from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from metrics import (
    evaluate as compute_metrics,
    evaluate_all_detectors,
    rank_detectors,
    format_results_table,
    get_roc_curve_data,
    get_pr_curve_data,
)

from visualization import PALETTE as DETECTOR_COLORS


def rank_by(df: pd.DataFrame, metric: str = "auc_roc") -> pd.DataFrame:
    """Sort results df by metric descending, prepend rank column."""
    ranked = df.sort_values(metric, ascending=False).copy()
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked


def load_results_csv(results_dir: str = "results/performance") -> dict[str, pd.DataFrame]:
    """Walk results/ and load every results.csv keyed by dataset name."""
    out = {}
    for root, dirs, files in os.walk(results_dir):
        if "results.csv" in files:
            ds = os.path.relpath(root, results_dir)
            df = pd.read_csv(os.path.join(root, "results.csv"), index_col=0)
            out[ds] = df
    return out


def mean_performance(all_results: dict[str, pd.DataFrame],
                     metrics: list[str] = None,
                     include_std: bool = False) -> pd.DataFrame:
    """Mean (and optionally std) per detector across all datasets."""
    metrics = metrics or ["auc_roc", "auc_pr", "f1"]
    rows = []
    for ds, df in all_results.items():
        for det in df.index:
            row = {"dataset": ds, "detector": det}
            row.update(df.loc[det].to_dict())
            rows.append(row)
    agg = pd.DataFrame(rows)
    grouped = agg.groupby("detector")[metrics]

    if not include_std:
        return grouped.mean().sort_values(metrics[0], ascending=False)

    mean = grouped.mean()
    std = grouped.std().rename(columns={m: f"{m}_std" for m in metrics})
    result = pd.concat([mean, std], axis=1)
    cols = [c for m in metrics for c in (m, f"{m}_std")]
    return result[cols].sort_values(metrics[0], ascending=False)


def tune_dbscan_eps(X: np.ndarray, percentile: float = 90.0,
                    min_eps: float = 0.05) -> float:
    """k-NN elbow heuristic for DBSCAN eps (from main.py)."""
    k = min(5, len(X) - 1)
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nbrs.kneighbors(X)
    return max(float(np.percentile(distances[:, -1], percentile)), min_eps)


def iqr_contamination(X: np.ndarray, factor: float = 1.5) -> float:
    """Estimate contamination via IQR rule (fraction of rows flagged in any feature)."""
    Q1, Q3 = np.percentile(X, 25, axis=0), np.percentile(X, 75, axis=0)
    IQR = Q3 - Q1
    mask = ((X < Q1 - factor * IQR) | (X > Q3 + factor * IQR)).any(axis=1)
    return float(np.clip(mask.mean(), 0.01, 0.49))


def scale_robust(X: np.ndarray):
    """RobustScaler — less sensitive to outliers than StandardScaler."""
    scaler = RobustScaler()
    return scaler.fit_transform(X), scaler


def plot_mean_performance(mean_df: pd.DataFrame, metrics: list[str] = None,
                          title: str = "Mean performance across datasets"):
    """Horizontal bar chart with optional std error bars."""

    metrics = metrics or ["auc_roc", "auc_pr", "f1"]
    metrics = [m for m in metrics if m in mean_df.columns]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]

    for ax, m in zip(axes, metrics):
        vals = mean_df[m].sort_values(ascending=False)
        colors = [DETECTOR_COLORS.get(d, "#888") for d in vals.index]
        std_col = f"{m}_std"
        xerr = mean_df.loc[vals.index, std_col].values if std_col in mean_df.columns else None

        bars = ax.barh(vals.index, vals.values, xerr=xerr, color=colors, edgecolor="white",
                       error_kw={"ecolor": "#555", "capsize": 4, "lw": 1.5})
        ax.set_xlim(0, 1)
        ax.set_title(m.upper())
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        ax.invert_yaxis()

    fig.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()
    return fig


def plot_radar(mean_df: pd.DataFrame, metrics: list[str] = None,
               title: str = "Detector Radar Chart"):
    """Radar / spider chart comparing detectors across metrics."""
    import matplotlib.pyplot as plt

    metrics = metrics or ["auc_roc", "auc_pr", "f1", "precision", "recall"]
    metrics = [m for m in metrics if m in mean_df.columns]
    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    for det in mean_df.index:
        vals = mean_df.loc[det, metrics].values.tolist() + [mean_df.loc[det, metrics[0]]]
        ax.plot(angles, vals, label=det, color=DETECTOR_COLORS.get(det), lw=2)
        ax.fill(angles, vals, color=DETECTOR_COLORS.get(det), alpha=0.07)

    ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
    ax.set_ylim(0, 1)
    ax.set_title(title, y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)
    plt.tight_layout()
    return fig

def plot_metric_heatmap(results: dict[str, pd.DataFrame], metric: str = "auc_roc", title: str | None = None, ax=None):
    """
    Heatmap: rows = detectors, columns = datasets.
    results : {dataset_name: metrics_df}
    """
    table = pd.DataFrame({ds: df[metric] for ds, df in results.items()})
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(max(6, len(table.columns) * 1.2), 4))
    sns.heatmap(table, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0.4, vmax=1.0, ax=ax, linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_title(title or f"{metric.upper()} across datasets")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    if standalone:
        plt.tight_layout()
        return fig

def show_dataset_summary(ds_name, plots):
    slug = ds_name.replace("/", "_")
    paths = [(p, f"results/performance/{slug}/{p}.png") for p in plots]
    paths = [(p, f) for p, f in paths if os.path.exists(f)]
    if not paths:
        print(f"Missing results for {ds_name}")
        return

    n = len(paths)
    ncols = 2
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows))
    axes = np.array(axes).flatten()

    for ax, (_, path) in zip(axes, paths):
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.axis("off")

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(ds_name, fontsize=13, fontweight="bold")
    plt.subplots_adjust(top=0.95, wspace=0.005, hspace=0.01)
    plt.show()

