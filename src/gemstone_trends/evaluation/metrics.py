"""Metrics shared by all model arms."""

from __future__ import annotations

import numpy as np
from scipy.stats import binomtest


def classification_metrics(
    true_indices: np.ndarray,
    probabilities: np.ndarray,
    *,
    class_count: int,
    ece_bins: int = 15,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    truth = np.asarray(true_indices, dtype=np.int64)
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 2 or probs.shape != (len(truth), class_count):
        raise ValueError("Probability matrix shape does not match labels/classes.")
    if len(truth) == 0:
        raise ValueError("Cannot score an empty prediction set.")
    if ece_bins < 1:
        raise ValueError("ECE needs at least one confidence bin.")
    if np.any(truth < 0) or np.any(truth >= class_count):
        raise ValueError("Ground-truth class index is outside the fixed class order.")

    ranked = np.argsort(-probs, axis=1, kind="stable")
    predicted = ranked[:, 0]
    correct = predicted == truth
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    np.add.at(confusion, (truth, predicted), 1)

    support = confusion.sum(axis=1)
    predicted_count = confusion.sum(axis=0)
    true_positive = np.diag(confusion).astype(np.float64)
    recall = np.divide(true_positive, support, out=np.zeros(class_count), where=support > 0)
    precision = np.divide(
        true_positive, predicted_count, out=np.zeros(class_count), where=predicted_count > 0
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(class_count),
        where=(precision + recall) > 0,
    )
    top_five = np.any(ranked[:, : min(5, class_count)] == truth[:, None], axis=1)

    confidence = probs[np.arange(len(truth)), predicted]
    bin_index = np.minimum((confidence * ece_bins).astype(int), ece_bins - 1)
    ece = 0.0
    for index in range(ece_bins):
        members = bin_index == index
        if np.any(members):
            ece += float(np.mean(members)) * abs(
                float(np.mean(correct[members])) - float(np.mean(confidence[members]))
            )

    metrics = {
        "n": float(len(truth)),
        "top1_accuracy": float(np.mean(correct)),
        "top5_accuracy": float(np.mean(top_five)),
        "macro_f1": float(np.mean(f1)),
        "ece": float(ece),
        "multiclass_brier": float(np.mean(np.sum((probs - np.eye(class_count)[truth]) ** 2, axis=1))),
    }
    return metrics, confusion, recall


def bootstrap_accuracy_interval(
    true_indices: np.ndarray,
    probabilities: np.ndarray,
    *,
    resamples: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    truth = np.asarray(true_indices, dtype=np.int64)
    predicted = np.argmax(probabilities, axis=1)
    correct = (predicted == truth).astype(np.float64)
    if resamples <= 0 or len(correct) == 0:
        raise ValueError("Bootstrap needs a positive resample count and non-empty data.")
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(correct), size=(resamples, len(correct)))
    estimates = correct[sample_indices].mean(axis=1)
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return float(lower), float(upper)


def risk_coverage_rows(
    true_indices: np.ndarray, probabilities: np.ndarray
) -> list[dict[str, float | int]]:
    truth = np.asarray(true_indices, dtype=np.int64)
    probs = np.asarray(probabilities, dtype=np.float64)
    predicted = np.argmax(probs, axis=1)
    confidence = probs[np.arange(len(truth)), predicted]
    error = (predicted != truth).astype(np.float64)
    order = np.argsort(-confidence, kind="stable")
    cumulative_errors = np.cumsum(error[order])
    rows = []
    for rank, index in enumerate(order, start=1):
        rows.append({
            "rank": rank,
            "coverage": rank / len(order),
            "risk": float(cumulative_errors[rank - 1] / rank),
            "confidence_threshold": float(confidence[index]),
        })
    return rows


def exact_mcnemar(
    correct_a: np.ndarray, correct_b: np.ndarray
) -> tuple[int, int, float]:
    left = np.asarray(correct_a, dtype=bool)
    right = np.asarray(correct_b, dtype=bool)
    if left.shape != right.shape:
        raise ValueError("McNemar inputs must be paired and have equal lengths.")
    a_only = int(np.count_nonzero(left & ~right))
    b_only = int(np.count_nonzero(~left & right))
    discordant = a_only + b_only
    p_value = 1.0 if discordant == 0 else float(
        binomtest(min(a_only, b_only), n=discordant, p=0.5, alternative="two-sided").pvalue
    )
    return a_only, b_only, p_value
