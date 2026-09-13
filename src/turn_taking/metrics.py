"""Turn-taking specific metrics.

Plain accuracy hides the two failure modes that actually matter in a spoken
dialogue system, so this module reports them directly (mirrors the ChatGPT
discussion this project started from):

* **false_stop_rate**   -- P(model says yield | user did not want the floor).
  The AI interrupts itself for no reason: annoying, breaks flow.
* **missed_interrupt_rate** -- P(model says continue | user wanted the floor).
  The AI talks over the user: worse, feels unresponsive/rude.
* **stop_latency**       -- for clips the model correctly flags, how many
  hops (at ``hop_sec`` each) it took from the true onset until the decision
  crossed the yield threshold. Only meaningful with hop-level scores, so it
  is computed separately in ``evaluate_latency``.

All rate functions accept binary arrays where 1 = "user wants the floor".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class TurnTakingMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    false_stop_rate: float
    missed_interrupt_rate: float
    n_samples: int
    n_positive: int
    n_negative: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "false_stop_rate": self.false_stop_rate,
            "missed_interrupt_rate": self.missed_interrupt_rate,
            "n_samples": self.n_samples,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
        }

    def format(self) -> str:
        return (
            f"accuracy={self.accuracy:.3f}  precision={self.precision:.3f}  "
            f"recall={self.recall:.3f}  f1={self.f1:.3f}\n"
            f"false_stop_rate={self.false_stop_rate:.3f}  "
            f"(AI yields when the user did NOT want the floor)\n"
            f"missed_interrupt_rate={self.missed_interrupt_rate:.3f}  "
            f"(AI keeps talking when the user DID want the floor)\n"
            f"n={self.n_samples} (positive={self.n_positive}, negative={self.n_negative})"
        )


def _rate(numerator_mask: NDArray[np.bool_], denominator_mask: NDArray[np.bool_]) -> float:
    denom = int(denominator_mask.sum())
    if denom == 0:
        return 0.0
    return float((numerator_mask & denominator_mask).sum()) / denom


def compute_metrics(y_true: NDArray[np.int64], y_pred: NDArray[np.int64]) -> TurnTakingMetrics:
    """Compute accuracy/precision/recall/F1 plus the two turn-taking error rates.

    Raises
    ------
    ValueError
        If the arrays are empty, differently shaped, or not binary.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("cannot compute metrics on empty arrays")
    if not set(np.unique(y_true)) <= {0, 1} or not set(np.unique(y_pred)) <= {0, 1}:
        raise ValueError("compute_metrics expects binary {0, 1} arrays")

    true_pos_mask = y_true == 1
    true_neg_mask = y_true == 0
    pred_pos_mask = y_pred == 1
    pred_neg_mask = y_pred == 0

    tp = int((true_pos_mask & pred_pos_mask).sum())
    fp = int((true_neg_mask & pred_pos_mask).sum())
    fn = int((true_pos_mask & pred_neg_mask).sum())
    tn = int((true_neg_mask & pred_neg_mask).sum())

    accuracy = (tp + tn) / y_true.size
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return TurnTakingMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        false_stop_rate=_rate(pred_pos_mask, true_neg_mask),
        missed_interrupt_rate=_rate(pred_neg_mask, true_pos_mask),
        n_samples=int(y_true.size),
        n_positive=int(true_pos_mask.sum()),
        n_negative=int(true_neg_mask.sum()),
    )


def stop_latency_hops(
    hop_probabilities: NDArray[np.float64], onset_hop: int, threshold: float
) -> int | None:
    """Hops from ``onset_hop`` until ``hop_probabilities`` first crosses ``threshold``.

    Returns ``None`` if the threshold is never crossed after onset (a missed
    interruption -- report it separately, it is not a latency of 0 or infinity).
    """
    hop_probabilities = np.asarray(hop_probabilities).reshape(-1)
    if onset_hop < 0 or onset_hop >= hop_probabilities.size:
        raise ValueError(f"onset_hop {onset_hop} out of range for {hop_probabilities.size} hops")
    tail = hop_probabilities[onset_hop:]
    crossed = np.flatnonzero(tail >= threshold)
    if crossed.size == 0:
        return None
    return int(crossed[0])
