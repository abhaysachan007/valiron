"""TDD — RED phase. Tests for valiron.evaluate.metrics (module not yet created)."""

import numpy as np
import pytest

from valiron.evaluate.metrics import (
    compute_metrics,
    MetricsResult,
    bootstrap_ci,
)


@pytest.fixture
def perfect_binary():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    return y_true, y_true.copy()


@pytest.fixture
def random_binary():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=500)
    y_pred = rng.integers(0, 2, size=500)
    y_prob = rng.random(500)
    return y_true, y_pred, y_prob


@pytest.fixture
def imbalanced():
    y_true = np.array([0] * 90 + [1] * 10)
    y_pred = np.zeros(100, dtype=int)
    y_prob = np.zeros(100)
    return y_true, y_pred, y_prob


class TestComputeMetrics:
    def test_returns_metrics_result(self, random_binary):
        y_true, y_pred, y_prob = random_binary
        assert isinstance(compute_metrics(y_true, y_pred, y_prob), MetricsResult)

    def test_perfect_accuracy(self, perfect_binary):
        y_true, y_pred = perfect_binary
        assert compute_metrics(y_true, y_pred).accuracy == pytest.approx(1.0)

    def test_accuracy_range(self, random_binary):
        y_true, y_pred, y_prob = random_binary
        r = compute_metrics(y_true, y_pred, y_prob)
        assert 0.0 <= r.accuracy <= 1.0

    def test_f1_range(self, random_binary):
        y_true, y_pred, y_prob = random_binary
        r = compute_metrics(y_true, y_pred, y_prob)
        assert 0.0 <= r.f1 <= 1.0

    def test_sensitivity_range(self, random_binary):
        y_true, y_pred, y_prob = random_binary
        r = compute_metrics(y_true, y_pred, y_prob)
        assert 0.0 <= r.sensitivity <= 1.0

    def test_specificity_range(self, random_binary):
        y_true, y_pred, y_prob = random_binary
        r = compute_metrics(y_true, y_pred, y_prob)
        assert 0.0 <= r.specificity <= 1.0

    def test_auc_none_without_proba(self, random_binary):
        y_true, y_pred, _ = random_binary
        assert compute_metrics(y_true, y_pred).auc is None

    def test_auc_range_with_proba(self, random_binary):
        y_true, y_pred, y_prob = random_binary
        r = compute_metrics(y_true, y_pred, y_prob)
        assert r.auc is not None and 0.0 <= r.auc <= 1.0

    def test_perfect_sensitivity_specificity(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        r = compute_metrics(y_true, y_true.copy())
        assert r.sensitivity == pytest.approx(1.0)
        assert r.specificity == pytest.approx(1.0)

    def test_imbalanced(self, imbalanced):
        y_true, y_pred, y_prob = imbalanced
        r = compute_metrics(y_true, y_pred, y_prob)
        assert r.accuracy == pytest.approx(0.9)
        assert r.sensitivity == pytest.approx(0.0)

    def test_ci_present_with_bootstrap(self, random_binary):
        y_true, y_pred, y_prob = random_binary
        r = compute_metrics(y_true, y_pred, y_prob, bootstrap_n=200)
        assert r.accuracy_ci is not None
        assert r.auc_ci is not None
        lo, hi = r.accuracy_ci
        assert lo <= r.accuracy <= hi

    def test_ci_none_without_bootstrap(self, random_binary):
        y_true, y_pred, y_prob = random_binary
        r = compute_metrics(y_true, y_pred, y_prob, bootstrap_n=0)
        assert r.accuracy_ci is None

    def test_raises_on_non_binary(self):
        y = np.array([0, 1, 2, 1, 0])
        with pytest.raises(ValueError, match="binary"):
            compute_metrics(y, y.copy())


class TestBootstrapCI:
    def test_returns_ordered_tuple(self):
        rng = np.random.default_rng(1)
        y_true = rng.integers(0, 2, size=100)
        y_pred = rng.integers(0, 2, size=100)
        lo, hi = bootstrap_ci(y_true, y_pred, metric="accuracy", n=200, seed=0)
        assert lo <= hi

    def test_positive_ci_width(self):
        rng = np.random.default_rng(2)
        y_true = rng.integers(0, 2, size=200)
        y_pred = rng.integers(0, 2, size=200)
        lo, hi = bootstrap_ci(y_true, y_pred, metric="accuracy", n=200, seed=1)
        assert hi > lo

    def test_unknown_metric_raises(self):
        y = np.array([0, 1])
        with pytest.raises(ValueError, match="metric"):
            bootstrap_ci(y, y, metric="bogus", n=10, seed=0)
