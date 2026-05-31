import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
import clustbench

MAT_DIR = "data/odds/mat_files"
ODDS_CSV_DIR = "data/odds"
CLUSTERING_BENCHMARK_CSV_DIR = "data/clustering_benchmark"
DATA_URL = "https://github.com/gagolews/clustering-data-v1/raw/v1.1.0"

os.makedirs(ODDS_CSV_DIR, exist_ok=True)
os.makedirs(CLUSTERING_BENCHMARK_CSV_DIR, exist_ok=True)

CLUSTERING_BENCHMARK_DATASETS = [
    ("graves", "ring_outliers",   1),
    ("graves", "zigzag_outliers", 1),
    ("graves", "fuzzyx",          1),
    ("wut",    "x2",              1),
    ("fcps",   "target",          1),
    ("sipu",   "flame",           1),
]

def convert_mat_to_csv():
    datasets = {}
    mat_files = [f for f in os.listdir(MAT_DIR) if f.endswith(".mat")]

    if not mat_files:
        print("No .mat files in folder.")
        return datasets

    print(f"Found {len(mat_files)} .mat files.\n")

    for filename in sorted(mat_files):
        name = filename.replace(".mat", "")
        mat_path = os.path.join(MAT_DIR, filename)
        csv_path = os.path.join(ODDS_CSV_DIR, f"{name}.csv")

        try:
            mat = loadmat(mat_path)
            X = mat["X"].astype(float)
            y = mat["y"].ravel().astype(int)

            cols = [f"f{i}" for i in range(X.shape[1])]
            df = pd.DataFrame(X, columns=cols)
            df["y"] = y
            df.to_csv(csv_path, index=False)

            datasets[name] = {"X": X, "y": y}
            print(f"{name} n={X.shape[0]}, d={X.shape[1]}, "
                  f"outliers={y.sum()} ({y.mean()*100:.1f}%)")

        except Exception as e:
            print(f"{name:<15} Error: {e}")

    return datasets


def load_clustering_benchmark():
    datasets = {}
    failed = []

    for battery, name, labels_idx in CLUSTERING_BENCHMARK_DATASETS:
        key = f"{battery}/{name}"
        csv_path = os.path.join(CLUSTERING_BENCHMARK_CSV_DIR, f"{battery}_{name}.csv")

        if not os.path.exists(csv_path):
            print(f"Loading {key}...", end=" ")
            try:
                b = clustbench.load_dataset(battery, name, url=DATA_URL)
                X = b.data
                labels = b.labels[labels_idx]

                df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
                df["cluster"] = labels
                df["is_noise"] = (labels == 0).astype(int)
                df.to_csv(csv_path, index=False)

                n_noise = (labels == 0).sum()
                print(f"n={X.shape[0]}, d={X.shape[1]}, "
                      f"k={b.n_clusters[labels_idx]}, noise={n_noise}")

                datasets[key] = {
                    "X": X,
                    "labels": labels,
                    "is_noise": (labels == 0).astype(int),
                    "n_clusters": b.n_clusters[labels_idx],
                    "battery": battery,
                    "name": name,
                }

            except Exception as e:
                print(f"Error: {e}")
                failed.append(key)

        else:
            df = pd.read_csv(csv_path)
            X = df.drop(columns=["cluster", "is_noise"]).values
            labels = df["cluster"].values

            datasets[key] = {
                "X": X,
                "labels": labels,
                "is_noise": df["is_noise"].values,
                "battery": battery,
                "name": name,
            }
            print(f"{key} already exists.")

    if failed:
        print(f"\nFailed to download: {failed}")

    return datasets


if __name__ == "__main__":
    odds_data = convert_mat_to_csv()
    gagolewski_data = load_clustering_benchmark()