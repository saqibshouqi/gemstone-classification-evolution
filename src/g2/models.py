"""G2 model grids and in-fold class balancing."""

from __future__ import annotations

from typing import Any

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


MODEL_NAMES = ("rf", "lr", "lda", "knn", "dt", "nb", "svm")


def require_imblearn() -> Any:
    try:
        import imblearn
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "The G2 anchor requires imbalanced-learn. Install it with "
            "'python -m pip install -e \".[g2]\"' or 'python -m pip install -r requirements-g2.txt'."
        ) from exc
    return imblearn, SMOTE, Pipeline


def make_pipeline(model: str, *, seed: int, smote_neighbors: int = 4):
    """Build the 2022 classifier and parameter grid behind a SMOTE pipeline."""
    _imblearn, SMOTE, Pipeline = require_imblearn()
    if model not in MODEL_NAMES:
        raise ValueError(f"Unknown model {model!r}; choose from {', '.join(MODEL_NAMES)}.")

    if model == "rf":
        classifier = RandomForestClassifier(random_state=seed, n_jobs=1)
        grid = {
            "classifier__n_estimators": [50, 100],
            "classifier__max_depth": [3, 5, 7, 9],
            "classifier__min_samples_leaf": [3, 5, 7, 9],
        }
    elif model == "lr":
        classifier = LogisticRegression(max_iter=3000, random_state=seed)
        grid = {"classifier__C": [0.001, 0.01, 0.1, 1, 10]}
    elif model == "lda":
        classifier = LinearDiscriminantAnalysis(solver="lsqr")
        grid = {"classifier__shrinkage": [0.0, 0.5, 1.0]}
    elif model == "knn":
        classifier = KNeighborsClassifier()
        grid = {"classifier__n_neighbors": [3, 5, 7, 9]}
    elif model == "dt":
        classifier = DecisionTreeClassifier(random_state=seed)
        grid = {
            "classifier__max_depth": [10, None],
            "classifier__max_features": [3, 5, 7, 9],
            "classifier__min_samples_leaf": [3, 5, 7, 9],
        }
    elif model == "nb":
        classifier = GaussianNB()
        grid = {"classifier__var_smoothing": [1e-9]}
    else:
        classifier = OneVsRestClassifier(
            SVC(probability=True, random_state=seed, cache_size=512)
        )
        grid = {
            "classifier__estimator__kernel": ["linear", "poly", "rbf", "sigmoid"],
            "classifier__estimator__C": [1, 10, 100],
            "classifier__estimator__gamma": [0.1, 0.01],
        }

    pipeline = Pipeline([
        ("smote", SMOTE(k_neighbors=smote_neighbors, random_state=seed)),
        ("classifier", classifier),
    ])
    return pipeline, grid


def aligned_probabilities(estimator, features, class_count: int):
    """Return a finite probability matrix in canonical class-index order."""
    import numpy as np

    pipeline_classifier = estimator.named_steps["classifier"]
    raw = np.asarray(estimator.predict_proba(features), dtype=np.float64)
    estimator_classes = np.asarray(pipeline_classifier.classes_, dtype=np.int64)
    if raw.ndim != 2 or raw.shape[1] != len(estimator_classes):
        raise ValueError("Classifier probability columns do not match its class labels.")
    output = np.zeros((len(raw), class_count), dtype=np.float64)
    output[:, estimator_classes] = raw
    output = np.clip(output, 0.0, 1.0)
    row_sums = output.sum(axis=1, keepdims=True)
    if np.any(~np.isfinite(row_sums)) or np.any(row_sums <= 0):
        raise ValueError("Classifier produced an invalid probability row.")
    output /= row_sums
    return output
