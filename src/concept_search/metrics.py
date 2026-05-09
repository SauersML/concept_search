"""Evaluation metrics for Phase-A replay runs."""

from __future__ import annotations

import numpy as np


def best_observed_curve(observed_mean: list[float]) -> np.ndarray:
    """Running max of observed scores. [n_obs] array."""
    return np.maximum.accumulate(np.asarray(observed_mean, dtype=np.float64))


def recall_at_k(true_top_set: set[int], recovered_order: list[int], k: int) -> float:
    """Fraction of true_top_set recovered in the first k positions of recovered_order."""
    if not true_top_set:
        return 0.0
    return len(true_top_set & set(recovered_order[:k])) / len(true_top_set)


def mean_top_k_score(
    posterior_mean: np.ndarray,
    truth: np.ndarray,
    k: int,
) -> float:
    """Average true score of the top-k features under the posterior ranking."""
    order = np.argsort(-posterior_mean)
    return float(truth[order[:k]].mean())
