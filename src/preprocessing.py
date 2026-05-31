import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def scale(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def handle_missing(X):
    col_medians = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X_clean = X.copy()
    X_clean[inds] = np.take(col_medians, inds[1])
    return X_clean

def reduce_for_visualization(X_scaled, n_components=2, method="pca"):
    if method == "pca":
        reducer = PCA(n_components=n_components, random_state=42)
        X_2d = reducer.fit_transform(X_scaled)
        var_explained = reducer.explained_variance_ratio_.sum()
        print(f"PCA variance explained: {var_explained*100:.1f}%")
        return X_2d, reducer

    else:
        raise ValueError(f"Unknown reduction method: {method}")
    
def get_contamination(y):
    return float(y.mean())

def preprocess_odds(X, y):
    X_clean = handle_missing(X)
    X_scaled, scaler = scale(X_clean)
    contam = get_contamination(y)
    return X_scaled, contam

def preprocess_test(X, scaler=None):
    X_clean = handle_missing(X)
    if scaler is not None:
        X_scaled = scaler.transform(X_clean)
    else:
        X_scaled, _ = scale(X_clean)
    return X_scaled