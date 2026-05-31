from __future__ import annotations
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN

class BaseOutlierDetector:
    """Minimal interface shared by all detectors."""

    name: str = "BaseDetector"

    def fit(self, X: np.ndarray) -> "BaseOutlierDetector":
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary labels: 1 = outlier, 0 = inlier."""
        raise NotImplementedError

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores (higher = more anomalous)."""
        raise NotImplementedError

    def fit_predict(self, X: np.ndarray):
        """Convenience: fit then predict on the same data."""
        self.fit(X)
        return self.predict(X)


class OneClassSVMDetector(BaseOutlierDetector):
    """
    One-Class SVM with RBF kernel.

    Parameters
    nu : float
        Upper bound on the fraction of outliers (and lower bound on the fraction of support vectors).  Defaults to the dataset 
        contamination rate when supplied via `fit`.
    kernel : str
        Kernel type
    gamma : str or float
        Kernel coefficient; 'scale' = 1/(n_features * X.var()).
    """

    name = "One-Class SVM"

    def __init__(self, nu: float = 0.1, kernel: str = "rbf", gamma: str = "scale"):
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self._model: OneClassSVM | None = None

    def fit(self, X: np.ndarray, contamination: float | None = None) -> "OneClassSVMDetector":
        nu = contamination if contamination is not None else self.nu
        # nu must be in (0, 1]
        nu = float(np.clip(nu, 1e-4, 1.0 - 1e-4))
        self._model = OneClassSVM(nu=nu, kernel=self.kernel, gamma=self.gamma)
        self._model.fit(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # sklearn: +1 = inlier, -1 = outlier -> flip to 0/1
        raw = self._model.predict(X)
        return ((raw == -1).astype(int))

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        # decision_function: higher = more inlier -> negate for anomaly score
        return -self._model.decision_function(X)


class IsolationForestDetector(BaseOutlierDetector):
    """
    Isolation Forest.

    Parameters
    n_estimators : int
        Number of isolation trees.
    contamination : float or 'auto'
        Proportion of outliers used to set the decision threshold.
    random_state : int
        Reproducibility seed.
    """

    name = "Isolation Forest"

    def __init__(self, n_estimators: int = 200, contamination: float | str = "auto", random_state: int = 42):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self._model: IsolationForest | None = None

    def fit(self, X: np.ndarray, contamination: float | None = None) -> "IsolationForestDetector":
        contam = contamination if contamination is not None else self.contamination
        self._model = IsolationForest(n_estimators=self.n_estimators, contamination=contam, random_state=self.random_state, n_jobs=-1)
        self._model.fit(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        raw = self._model.predict(X) # +1 / -1
        return (raw == -1).astype(int)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        return -self._model.score_samples(X)  #negate: higher = more anomalous
    
class LOFDetector(BaseOutlierDetector):
    """
    Local Outlier Factor.
    
    Parameters
    n_neighbors : int
        Number of neighbours for LOF computation.
    contamination : float or 'auto'
        Fraction of outliers for thresholding.
    """

    name = "Local Outlier Factor"

    def __init__(self, n_neighbors: int = 20, contamination: float | str = "auto"):
        self.n_neighbors   = n_neighbors
        self.contamination = contamination
        self._model: LocalOutlierFactor | None = None

    def fit(self, X: np.ndarray, contamination: float | None = None) -> "LOFDetector":
        contam = contamination if contamination is not None else self.contamination
        self._model = LocalOutlierFactor(
            n_neighbors=min(self.n_neighbors, len(X) - 1),
            contamination=contam,
            novelty=True,
            n_jobs=-1,
        )
        self._model.fit(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self._model.predict(X) == -1).astype(int)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        return -self._model.score_samples(X)
        
class DBSCANDetector(BaseOutlierDetector):
    """
    DBSCAN-based outlier detector.

    Points labeled -1 by DBSCAN (noise) are treated as outliers. Anomaly score = negative of the distance to the nearest core point
    (so isolated points get a high score).  Falls back to distance to nearest training neighbour when no core points exist in a region.

    Parameters
    eps : float
        Maximum distance between two samples for neighbourhood membership.
    min_samples : int
        Minimum points to form a dense region (core point).
    metric : str
        Distance metric for DBSCAN.
    """

    name = "DBSCAN"

    def __init__(self, eps: float = 0.5, min_samples: int = 5, metric: str = "euclidean"):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self._model: DBSCAN | None = None
        self._X_train: np.ndarray | None = None
        self._labels_train: np.ndarray | None = None

    def fit(self, X: np.ndarray, contamination: float | None = None) -> "DBSCANDetector":
        self._model = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric=self.metric, n_jobs=-1)
        self._labels_train = self._model.fit_predict(X)
        self._X_train = X.copy()
        return self

    def _is_outlier(self, X: np.ndarray) -> np.ndarray:
        """A point is an inlier if it falls within eps of any core point."""
        from sklearn.metrics import pairwise_distances_argmin_min
        core_indices = self._model.core_sample_indices_
        if len(core_indices) == 0:
            return np.ones(len(X), dtype=int)
        core_points = self._X_train[core_indices]
        _, dist = pairwise_distances_argmin_min(X, core_points)
        return (dist > self.eps).astype(int)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if np.array_equal(X, self._X_train):
            return (self._labels_train == -1).astype(int)
        return self._is_outlier(X)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Anomaly score = min distance to any core point. Points far from all core points receive high scores.
        """
        from sklearn.metrics import pairwise_distances_argmin_min

        core_indices = self._model.core_sample_indices_
        if len(core_indices) == 0:
            # Degenerate: all points are noise -> uniform score
            return np.ones(len(X))

        core_points = self._X_train[core_indices]
        _, dist_to_core = pairwise_distances_argmin_min(X, core_points)
        return dist_to_core  # higher = farther from any core point = more anomalous



class ECODDetector(BaseOutlierDetector):
    """
    ECOD - Empirical Cumulative-distribution-based Outlier Detection.

    A parameter-free, interpretable method that models the tail probability of each feature independently and combines them into an outlier score.
    Reference: Li et al., "ECOD: Unsupervised Outlier Detection Using Empirical Cumulative Distribution Functions", TKDE 2022.

    Requires: pip install pyod

    Parameters
    contamination : float
        Expected fraction of outliers; used to set the decision threshold.
    """

    name = "ECOD"

    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self._model = None

    def fit(self, X: np.ndarray, contamination: float | None = None) -> "ECODDetector":
        try:
            from pyod.models.ecod import ECOD
        except ImportError as exc:
            raise ImportError("pyod is required for ECODDetector: pip install pyod") from exc

        contam = contamination if contamination is not None else self.contamination
        contam = float(np.clip(contam, 1e-4, 0.5))
        self._model = ECOD(contamination=contam)
        self._model.fit(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X).astype(int) # pyod already returns 0/1

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        return self._model.decision_function(X) # higher = more anomalous
    

def get_all_detectors(contamination: float = 0.1) -> list[BaseOutlierDetector]:
    """
    Return one instance of every detector, pre-configured with the given contamination rate.

    Parameters
    contamination : float
        Dataset-specific outlier fraction from the ground-truth labels.

    Returns
    list[BaseOutlierDetector]
    """
    return [
        OneClassSVMDetector(nu=contamination),
        IsolationForestDetector(contamination=contamination),
        LOFDetector(contamination=contamination),
        DBSCANDetector(), # eps/min_samples tuned externally
        ECODDetector(contamination=contamination),
    ]