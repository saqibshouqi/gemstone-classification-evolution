"""Small metric helpers so G1 can be checked before package A's evaluate.py exists.

The official numbers should come from src/common/evaluate.py; these follow the same
definitions (top-1, top-5, macro-F1, 95% bootstrap interval over test images).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def top1(proba: np.ndarray, y: np.ndarray) -> float:
    return float((proba.argmax(1) == y).mean())


def topk(proba: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    """Top-k accuracy. The true class must also have a non-zero score: vote shares
    contain many zeros, and padding the top-k with arbitrary zero-score classes would
    inflate the number."""
    idx = np.argsort(-proba, axis=1, kind="stable")[:, :k]
    hit = (idx == y[:, None]).any(1) & (proba[np.arange(len(y)), y] > 0)
    return float(hit.mean())


def macro_f1(proba: np.ndarray, y: np.ndarray) -> float:
    return float(f1_score(y, proba.argmax(1), labels=np.unique(y), average="macro", zero_division=0))


def log_loss(proba: np.ndarray, y: np.ndarray, eps: float = 1e-3) -> float:
    """Log-loss with additive smoothing (vote shares can be exactly 0)."""
    k = proba.shape[1]
    p = (proba + eps) / (1.0 + k * eps)
    return float(-np.log(p[np.arange(len(y)), y]).mean())


def bootstrap_ci(correct: np.ndarray, n_boot: int = 1000, seed: int = 0):
    """95% bootstrap interval of the mean of a 0/1 vector (resampling test images)."""
    rng = np.random.default_rng(seed)
    n = len(correct)
    means = correct[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summary(proba: np.ndarray, y: np.ndarray) -> dict:
    return dict(top1=top1(proba, y), top5=topk(proba, y, 5), macro_f1=macro_f1(proba, y),
                logloss=log_loss(proba, y))
