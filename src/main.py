"""
main.py
-------
End-to-end experiment runner for the outlier detection project.

Usage:
    python main.py

Output structure (results/performance/ is wiped on every run):
    results/
      performance/
        <dataset_name>/
          results.csv
          roc.png
          pr.png
          metric_bars.png
          heatmap.png
          score_dist.png
          scatter.png  (clustering-benchmark only)
        cross_dataset_auc_roc.png
        cross_dataset_auc_pr.png
        cross_dataset_f1.png
        mean_performance.csv
"""

from __future__ import annotations
import os
import shutil
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

PERF_DIR = os.path.join("results", "performance")
if os.path.exists(PERF_DIR):
    shutil.rmtree(PERF_DIR)
os.makedirs(PERF_DIR)
print("results/performance/ folder cleared.\n")

from data_loader import convert_mat_to_csv, load_clustering_benchmark
from preprocessing import preprocess_odds, reduce_for_visualization
from metrics import evaluate_all_detectors, rank_detectors, format_results_table
from visualization import (plot_scatter_predictions, plot_roc_curves, plot_pr_curves, plot_metric_bars, plot_metric_heatmap,
    plot_score_distributions, plot_multi_dataset_summary)


def outdir(name: str) -> str:
    """Create and return results/performance/<name>/"""
    slug = name.replace("/", "_")
    path = os.path.join(PERF_DIR, slug)
    os.makedirs(path, exist_ok=True)
    return path

def savefig(fig: plt.Figure, path: str) -> None:
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {path}")


def tune_dbscan_eps(X: np.ndarray, percentile: float = 90.0, min_eps: float = 0.05) -> float:
    from sklearn.neighbors import NearestNeighbors
    k = min(5, len(X) - 1)
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nbrs.kneighbors(X)
    eps = float(np.percentile(distances[:, -1], percentile))
    return max(eps, min_eps)


def build_detectors(contamination: float, eps: float):
    from models import (OneClassSVMDetector, IsolationForestDetector, LOFDetector, DBSCANDetector, ECODDetector)
    return [OneClassSVMDetector(nu=contamination), IsolationForestDetector(contamination=contamination),
        LOFDetector(contamination=contamination), DBSCANDetector(eps=eps, min_samples=5),
        ECODDetector(contamination=contamination)]

def run_odds_dataset(name: str, data: dict, all_results: dict) -> None:
    print(f"\n{'='*60}")
    print(f"ODDS: {name}")
    print(f"{'='*60}")

    X, y = data["X"], data["y"]
    X_scaled, contam = preprocess_odds(X, y)
    print(f"n={len(X)}, d={X.shape[1]}, contamination={contam:.3f}")

    eps = tune_dbscan_eps(X_scaled)
    detectors = build_detectors(contam, eps)

    results_df = evaluate_all_detectors(detectors, X_scaled, y, contamination=contam)
    print(format_results_table(rank_detectors(results_df)).to_string())

    out = outdir(name)
    results_df.to_csv(os.path.join(out, "results.csv"))

    scores, preds = {}, {}
    for det in detectors:
        scores[det.name] = det.score_samples(X_scaled)
        preds[det.name]  = det.predict(X_scaled)

    savefig(plot_roc_curves(y, scores, title=f"ROC - {name}"), os.path.join(out, "roc.png"))
    savefig(plot_pr_curves(y, scores, title=f"PR - {name}"), os.path.join(out, "pr.png"))
    savefig(plot_metric_bars(results_df, title=f"Metric Comparison - {name}"), os.path.join(out, "metric_bars.png"))
    savefig(plot_metric_heatmap(results_df, title=f"Heatmap - {name}"), os.path.join(out, "heatmap.png"))
    savefig(plot_score_distributions(y, scores, title=f"Score Distributions - {name}"), os.path.join(out, "score_dist.png"))

    all_results[name] = results_df

def run_clustering_dataset(key: str, data: dict, all_results: dict) -> None:
    print(f"\n{'='*60}")
    print(f"Clustering-benchmark: {key}")
    print(f"{'='*60}")

    X = data["X"].astype(float)
    y_noise = data["is_noise"]

    if y_noise.sum() == 0:
        print(" No noise/outlier points – skipping.")
        return

    from preprocessing import scale, handle_missing
    X_scaled, _ = scale(handle_missing(X))
    contam = float(y_noise.mean())
    print(f"n={len(X)}, d={X.shape[1]}, contamination={contam:.3f}")

    X_2d = X_scaled if X.shape[1] == 2 else reduce_for_visualization(X_scaled)[0]

    eps = tune_dbscan_eps(X_scaled)
    detectors = build_detectors(contam, eps)

    results_df = evaluate_all_detectors(detectors, X_scaled, y_noise, contamination=contam)
    print(format_results_table(rank_detectors(results_df)).to_string())

    out = outdir(key)
    results_df.to_csv(os.path.join(out, "results.csv"))

    scores, preds = {}, {}
    for det in detectors:
        scores[det.name] = det.score_samples(X_scaled)
        preds[det.name]= det.predict(X_scaled)

    savefig(plot_scatter_predictions(X_2d, y_noise, preds, dataset_name=key), os.path.join(out, "scatter.png"))
    savefig(plot_roc_curves(y_noise, scores, title=f"ROC - {key}"), os.path.join(out, "roc.png"))
    savefig(plot_pr_curves(y_noise, scores, title=f"PR - {key}"), os.path.join(out, "pr.png"))
    savefig(plot_metric_bars(results_df, title=f"Metric Comparison - {key}"), os.path.join(out, "metric_bars.png"))
    savefig(plot_metric_heatmap(results_df, title=f"Heatmap - {key}"), os.path.join(out, "heatmap.png"))
    savefig(plot_score_distributions(y_noise, scores, title=f"Score Distributions - {key}"), os.path.join(out, "score_dist.png"))

    all_results[key] = results_df

def main() -> None:
    print("Loading ODDS datasets …")
    odds_data = convert_mat_to_csv()

    print("\nLoading clustering-benchmark datasets …")
    clust_data = load_clustering_benchmark()

    all_results: dict = {}

    for name, data in odds_data.items():
        try:
            run_odds_dataset(name, data, all_results)
        except Exception as exc:
            print(f"  [ERROR] {name}: {exc}")

    for key, data in clust_data.items():
        try:
            run_clustering_dataset(key, data, all_results)
        except Exception as exc:
            print(f"  [ERROR] {key}: {exc}")

    if all_results:
        print("\n" + "="*60)
        print("Cross-dataset summary")
        print("="*60)

        for metric in ("auc_roc", "auc_pr", "f1"):
            try:
                fig = plot_multi_dataset_summary(all_results, metric=metric)
                savefig(fig, os.path.join(PERF_DIR, f"cross_dataset_{metric}.png"))
            except Exception as exc:
                print(f"  [WARN] cross-dataset {metric}: {exc}")

        agg_rows = []
        for ds, df in all_results.items():
            for det in df.index:
                row = {"dataset": ds, "detector": det}
                row.update(df.loc[det].to_dict())
                agg_rows.append(row)

        agg = pd.DataFrame(agg_rows)
        mean_perf = (agg.groupby("detector")[["auc_roc", "auc_pr", "f1"]].mean().sort_values("auc_roc", ascending=False))
        print("\nMean performance across all datasets:")
        print(mean_perf.to_string(float_format="{:.4f}".format))
        mean_perf.to_csv(os.path.join(PERF_DIR, "mean_performance.csv"))

    print("\nDone. All results saved to ./results/performance/")


if __name__ == "__main__":
    main()
