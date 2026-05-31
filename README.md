# anomaly-detection-benchmark

## Repository Structure

```
src/
├── hyperparameters/          # Hyperparameter sweep experiments
│   ├── svm/experiment.py
│   ├── isolation_forest/experiment.py
│   ├── local_outlier_factor/experiment.py
│   ├── dbscan/experiment.py
│   ├── ECOD/experiment.py
│   └── hp_utils.py           # Shared sweep utilities
├── data_loader.py            # Dataset loading and splitting
├── detectors.py              # Detector classes (OC-SVM, IsoForest, LOF, DBSCAN, ECOD)
├── main.py                   # Run full performance evaluation
├── metrics.py                # AUC-ROC, AUC-PR, F1, Precision, Recall
├── preprocessing.py          # Scaling, train/test split
├── utils.py                  # Plotting helpers, load_results_csv
└── visualization.py          # Per-dataset plots (ROC, score dist, scatter)

01_performance_evaluation.ipynb   # Section 3 — compare all methods across datasets
02_hyperparameter_analysis.ipynb  # Section 4 — hyperparameter sweeps
03_robustness_analysis.ipynb      # Section 5 — dimensionality, sample size, local vs global
04_discussion.ipynb               # Section 7 — strengths, weaknesses, conclusions
05_verification_task.ipynb        # Section 8 — ensemble prediction on test_data.csv

results/
├── performance/<dataset>/        # results.csv + plots per dataset
└── hyperparameters/<algo>/       # sweep CSVs per algorithm

test_data.csv                     # Unlabelled test set (3443 × 21)
test_labels.csv                   # Final predictions (class: 0/1)
```

---

## How to Run

### 1. Install dependencies
```bash
pip install numpy pandas scikit-learn matplotlib seaborn pyod
```

### 2. Generate performance results
Runs all detectors on all datasets and saves CSVs + plots to `results/performance/`.
```bash
python src/main.py
```

### 3. Generate hyperparameter sweep results
Run each experiment separately — results saved to `results/hyperparameters/<algo>/`.
```bash
python src/hyperparameters/svm/experiment.py
python src/hyperparameters/isolation_forest/experiment.py
python src/hyperparameters/local_outlier_factor/experiment.py
python src/hyperparameters/dbscan/experiment.py
python src/hyperparameters/ECOD/experiment.py
```

### 4. Open notebooks in order
All notebooks load from pre-computed `results/` — run steps 2 and 3 first.

| Notebook | Content |
|---|---|
| `01_performance_evaluation.ipynb` | AUC-ROC/F1 heatmaps, mean ranking, per-dataset summaries |
| `02_hyperparameter_analysis.ipynb` | Sensitivity plots for every parameter of every algorithm |
| `03_robustness_analysis.ipynb` | Dimensionality, sample size, local vs global outlier analysis |
| `04_discussion.ipynb` | Overall discussion and conclusions |
| `05_verification_task.ipynb` | Ensemble prediction on `test_data.csv` -> `test_labels.csv` |

---

## Key Files

| File | Description |
|---|---|
| `src/detectors.py` | All detector classes with a unified `fit / predict / score_samples` interface |
| `src/utils.py` | `load_results_csv`, plotting helpers used across all notebooks |
| `test_labels.csv` | Final submission — binary column `class` (1 = outlier, 0 = inlier) |
