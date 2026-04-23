"""Classical ML baselines for EEG seizure prediction.

Uses DWT feature vectors (200-dim, from :func:`auras.data.preprocess.dwt_features`)
as input to sklearn-compatible classifiers.  These models act as scientific
benchmarks — demonstrating how far hand-crafted wavelet features get before
deep learning is needed.

Paper coverage:
    Paper 22 (Dokare & Gupta — DWT-SVM):
        SVM-RBF with DWT Db4 features.  C=100, gamma=0.001, class_weight='balanced'.
        Achieved 97.7% accuracy, 86.7% sensitivity on CHB-MIT with 4-second windows.
    Paper 07 (Dash et al. — TF-Wearable):
        Compared SVM, RF, XGBoost, LightGBM, KNN with wavelet features.
    Paper 11 (Djemal et al. — XGBoost-CS):
        XGBoost with compressive-sensing features; n_estimators=200, max_depth=8.
    Paper 05 (GAT):
        RF n_estimators=100 as comparison baseline.
    Paper 14 (Manzouri et al. — EE-Implantable):
        RF max_depth=10, class_weight='balanced' for implantable constraint.
    Paper 17 (Sánchez-Reyes et al. — PCA+DWT+SVM):
        KNN k=3, weights='distance' after DWT+PCA feature reduction.

All classifiers support the sklearn interface: fit(X, y) / predict(X) / predict_proba(X).
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


# ---------------------------------------------------------------------------
# Best-paper hyperparameter presets
# ---------------------------------------------------------------------------

#: Default hyperparameters derived from paper best-results survey.
BEST_PARAMS: Dict[str, Dict[str, Any]] = {
    "svm_rbf": {
        # Paper 22 (Dokare & Gupta — DWT-SVM): C=100, gamma=0.001, balanced
        "C": 100,
        "gamma": 0.001,
        "kernel": "rbf",
        "class_weight": "balanced",
        "probability": True,      # needed for predict_proba / AUC-ROC
        "cache_size": 500,
    },
    "svm_linear": {
        # Paper 07 (Dash et al. — TF-Wearable): linear SVM comparison
        "C": 1.0,
        "kernel": "linear",
        "class_weight": "balanced",
        "probability": True,
        "max_iter": 5000,
    },
    "random_forest": {
        # Paper 05 (GAT) / Paper 14 (Manzouri et al. — EE-Implantable):
        # n_estimators=100, max_depth=10, balanced
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": 42,
    },
    "xgboost": {
        # Paper 11 (Djemal et al. — XGBoost-CS): n_estimators=200, max_depth=8, lr=0.1
        "n_estimators": 200,
        "max_depth": 8,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": None,   # set dynamically from class imbalance ratio
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
    },
    "lightgbm": {
        # Paper 07 (Dash et al. — TF-Wearable): LightGBM comparison baseline
        "n_estimators": 200,
        "num_leaves": 63,
        "max_depth": -1,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    },
    "knn": {
        # Paper 17 (Sánchez-Reyes et al. — PCA+DWT+SVM): KNN k=3, distance-weighted
        "n_neighbors": 3,
        "weights": "distance",
        "metric": "euclidean",
        "n_jobs": -1,
    },
}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_classical_model(name: str, class_ratio: float = 1.0):
    """Instantiate a classical ML classifier with paper-derived hyperparameters.

    Parameters
    ----------
    name : str
        One of: ``'svm_rbf'``, ``'svm_linear'``, ``'random_forest'``,
        ``'xgboost'``, ``'lightgbm'``, ``'knn'``.
    class_ratio : float
        ``neg_count / pos_count`` — used to set ``scale_pos_weight`` for
        XGBoost (imbalanced-class correction, equivalent to class_weight='balanced').

    Returns
    -------
    sklearn-compatible estimator
    """
    if name not in BEST_PARAMS:
        raise ValueError(
            f"Unknown classical model '{name}'. "
            f"Available: {sorted(BEST_PARAMS.keys())}"
        )

    params = dict(BEST_PARAMS[name])  # copy

    if name == "svm_rbf" or name == "svm_linear":
        from sklearn.svm import SVC
        return SVC(**params)

    elif name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**params)

    elif name == "xgboost":
        from xgboost import XGBClassifier
        params["scale_pos_weight"] = class_ratio  # neg/pos handles imbalance
        return XGBClassifier(**params)

    elif name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**params)

    elif name == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(**params)

    else:
        raise ValueError(f"Unhandled model: {name}")


def list_classical_models() -> list:
    """Return names of all available classical ML models."""
    return sorted(BEST_PARAMS.keys())


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------

def train_and_evaluate(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Train one classical ML model and return evaluation metrics.

    Parameters
    ----------
    model_name : str
        Model identifier (see :func:`build_classical_model`).
    X_train : (N_train, F) float array
        DWT feature vectors for training.
    y_train : (N_train,) int array
        Training labels.
    X_test : (N_test, F) float array
        DWT feature vectors for testing.
    y_test : (N_test,) int array
        Test labels.

    Returns
    -------
    dict with keys: model_name, accuracy, recall, precision, specificity,
                    f1, auc_roc, fpr, n_train, n_test, n_pos_train, n_pos_test
    """
    from auras.training.metrics import compute_metrics

    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    ratio = n_neg / max(n_pos, 1)

    clf = build_classical_model(model_name, class_ratio=ratio)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob: np.ndarray | None = None
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_prob)
    result = {"model_name": model_name}
    result.update(metrics.to_dict())
    result.update({
        "n_train": len(y_train),
        "n_test": len(y_test),
        "n_pos_train": n_pos,
        "n_neg_train": n_neg,
        "n_pos_test": int(y_test.sum()),
        "n_neg_test": int(len(y_test) - y_test.sum()),
    })
    return result
