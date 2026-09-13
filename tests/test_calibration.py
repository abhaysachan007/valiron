"""Tests for calibration checker."""

import numpy as np

from valiron.calibration.checker import check_calibration, CalibrationResult


def test_perfect_calibration():
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9])
    result = check_calibration(y_true, y_prob)
    assert isinstance(result, CalibrationResult)
    assert result.well_calibrated is True


def test_poor_calibration():
    y_true = np.zeros(100)
    y_prob = np.ones(100) * 0.9
    result = check_calibration(y_true, y_prob)
    assert result.expected_calibration_error > 0.5
    assert result.well_calibrated is False


def test_ece_range():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=200)
    y_prob = rng.random(200)
    result = check_calibration(y_true, y_prob)
    assert 0.0 <= result.expected_calibration_error <= 1.0
    assert 0.0 <= result.max_calibration_error <= 1.0
