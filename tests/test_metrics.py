from __future__ import annotations

import numpy as np
import pytest

from turn_taking.metrics import compute_metrics, stop_latency_hops


def test_perfect_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    metrics = compute_metrics(y_true, y_pred)
    assert metrics.accuracy == 1.0
    assert metrics.false_stop_rate == 0.0
    assert metrics.missed_interrupt_rate == 0.0
    assert metrics.f1 == 1.0


def test_false_stop_rate_isolated():
    # Two true negatives, one incorrectly predicted positive -> false stop.
    y_true = np.array([0, 0, 1])
    y_pred = np.array([1, 0, 1])
    metrics = compute_metrics(y_true, y_pred)
    assert metrics.false_stop_rate == pytest.approx(0.5)
    assert metrics.missed_interrupt_rate == 0.0


def test_missed_interrupt_rate_isolated():
    # Two true positives, one incorrectly predicted negative -> missed interrupt.
    y_true = np.array([1, 1, 0])
    y_pred = np.array([0, 1, 0])
    metrics = compute_metrics(y_true, y_pred)
    assert metrics.missed_interrupt_rate == pytest.approx(0.5)
    assert metrics.false_stop_rate == 0.0


def test_compute_metrics_rejects_empty():
    with pytest.raises(ValueError):
        compute_metrics(np.array([]), np.array([]))


def test_compute_metrics_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        compute_metrics(np.array([0, 1]), np.array([0, 1, 1]))


def test_compute_metrics_rejects_non_binary():
    with pytest.raises(ValueError):
        compute_metrics(np.array([0, 2]), np.array([0, 1]))


def test_stop_latency_hops_finds_crossing():
    probs = np.array([0.1, 0.2, 0.5, 0.85, 0.9])
    assert stop_latency_hops(probs, onset_hop=1, threshold=0.8) == 2


def test_stop_latency_hops_returns_none_when_never_crossed():
    probs = np.array([0.1, 0.2, 0.3])
    assert stop_latency_hops(probs, onset_hop=0, threshold=0.8) is None


def test_stop_latency_hops_rejects_out_of_range_onset():
    probs = np.array([0.1, 0.2])
    with pytest.raises(ValueError):
        stop_latency_hops(probs, onset_hop=5, threshold=0.5)
