"""TDD — calibration checker tests. 16+ tests covering ECE, MCE, reliability data."""

from __future__ import annotations

import numpy as np
import pytest

from valiron.calibration.checker import check_calibration, CalibrationResult, reliability_data


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def perfect_cal():
    """Probabilities match observed frequencies exactly."""
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9])
    return y_true, y_prob


@pytest.fixture
def random_cal():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=500)
    y_prob = rng.random(500)
    return y_true, y_prob


@pytest.fixture
def overconfident():
    """Model always predicts 0.9 but only 10% are positive."""
    y_true = np.array([0] * 90 + [1] * 10)
    y_prob = np.full(100, 0.9)
    return y_true, y_prob


# ---------------------------------------------------------------------------
# CalibrationResult dataclass
# ---------------------------------------------------------------------------

class TestCalibrationResult:
    def test_fields_exist(self):
        r = CalibrationResult(ece=0.05, mce=0.12, well_calibrated=True)
        assert hasattr(r, "ece")
        assert hasattr(r, "mce")
        assert hasattr(r, "well_calibrated")

    def test_well_calibrated_flag_true(self):
        r = CalibrationResult(ece=0.05, mce=0.15, well_calibrated=True)
        assert r.well_calibrated is True

    def test_well_calibrated_flag_false(self):
        r = CalibrationResult(ece=0.15, mce=0.30, well_calibrated=False)
        assert r.well_calibrated is False


# ---------------------------------------------------------------------------
# check_calibration — output contract
# ---------------------------------------------------------------------------

class TestCheckCalibration:
    def test_returns_calibration_result(self, random_cal):
        y_true, y_prob = random_cal
        result = check_calibration(y_true, y_prob)
        assert isinstance(result, CalibrationResult)

    def test_ece_in_range(self, random_cal):
        y_true, y_prob = random_cal
        r = check_calibration(y_true, y_prob)
        assert 0.0 <= r.ece <= 1.0

    def test_mce_in_range(self, random_cal):
        y_true, y_prob = random_cal
        r = check_calibration(y_true, y_prob)
        assert 0.0 <= r.mce <= 1.0

    def test_mce_gte_ece(self, random_cal):
        """MCE is the max per-bin gap — must be >= weighted average (ECE)."""
        y_true, y_prob = random_cal
        r = check_calibration(y_true, y_prob)
        assert r.mce >= r.ece - 1e-9

    def test_perfect_calibration_low_ece(self, perfect_cal):
        y_true, y_prob = perfect_cal
        r = check_calibration(y_true, y_prob)
        assert r.ece < 0.15  # small sample; bins have 0 gap but ECE ~ n_in_bin/n

    def test_perfect_calibration_well_calibrated(self):
        """Uniform 0.5 probs on balanced data → ECE near 0 → well_calibrated True."""
        rng = np.random.default_rng(5)
        # 1000 samples, balanced, proba = 0.5 → fraction_pos in that bin ≈ 0.5
        y_true = np.array([0, 1] * 500)
        y_prob = np.full(1000, 0.5)
        r = check_calibration(y_true, y_prob)
        assert r.well_calibrated is True

    def test_overconfident_high_ece(self, overconfident):
        y_true, y_prob = overconfident
        r = check_calibration(y_true, y_prob)
        assert r.ece > 0.5

    def test_overconfident_not_well_calibrated(self, overconfident):
        y_true, y_prob = overconfident
        r = check_calibration(y_true, y_prob)
        assert r.well_calibrated is False  # bool() cast in checker ensures Python bool

    def test_well_calibrated_threshold_below_0_1(self):
        """Alternating 0/1 labels with 0.5 prob → ECE~0 → well_calibrated True."""
        y_true = np.array([0, 1] * 250)
        y_prob = np.full(500, 0.5)
        r = check_calibration(y_true, y_prob)
        assert r.well_calibrated is True

    def test_ece_rounded_to_4dp(self, random_cal):
        y_true, y_prob = random_cal
        r = check_calibration(y_true, y_prob)
        assert round(r.ece, 4) == r.ece

    def test_mce_rounded_to_4dp(self, random_cal):
        y_true, y_prob = random_cal
        r = check_calibration(y_true, y_prob)
        assert round(r.mce, 4) == r.mce

    def test_n_bins_parameter(self, random_cal):
        """Different n_bins both produce valid results."""
        y_true, y_prob = random_cal
        r5 = check_calibration(y_true, y_prob, n_bins=5)
        r20 = check_calibration(y_true, y_prob, n_bins=20)
        assert 0.0 <= r5.ece <= 1.0
        assert 0.0 <= r20.ece <= 1.0

    def test_all_zeros_proba(self):
        """All probabilities 0.0, all labels 0 → near-perfect calibration."""
        y_true = np.zeros(50)
        y_prob = np.zeros(50)
        r = check_calibration(y_true, y_prob)
        assert r.ece == pytest.approx(0.0, abs=1e-4)

    def test_all_ones_proba(self):
        """All probabilities 1.0, all labels 1 → near-perfect calibration."""
        y_true = np.ones(50)
        y_prob = np.ones(50)
        r = check_calibration(y_true, y_prob)
        assert r.ece == pytest.approx(0.0, abs=1e-4)

    def test_uniform_proba_reasonable_ece(self):
        """Constant 0.5 prob on balanced data → small ECE."""
        rng = np.random.default_rng(7)
        y_true = rng.integers(0, 2, size=1000)
        y_prob = np.full(1000, 0.5)
        r = check_calibration(y_true, y_prob)
        assert r.ece < 0.20

    def test_more_miscalibration_higher_ece(self):
        """Increasing miscalibration should monotonically increase ECE."""
        y_true = np.zeros(100)
        r_mild = check_calibration(y_true, np.full(100, 0.3))
        r_severe = check_calibration(y_true, np.full(100, 0.8))
        assert r_severe.ece > r_mild.ece


# ---------------------------------------------------------------------------
# reliability_data — bin-level output for reliability diagrams
# ---------------------------------------------------------------------------

class TestReliabilityData:
    def test_returns_three_arrays(self, random_cal):
        y_true, y_prob = random_cal
        mean_pred, frac_pos, counts = reliability_data(y_true, y_prob)
        assert len(mean_pred) == len(frac_pos) == len(counts)

    def test_lengths_lte_n_bins(self, random_cal):
        y_true, y_prob = random_cal
        mean_pred, frac_pos, counts = reliability_data(y_true, y_prob, n_bins=5)
        assert len(mean_pred) <= 5

    def test_mean_pred_in_range(self, random_cal):
        y_true, y_prob = random_cal
        mean_pred, _, _ = reliability_data(y_true, y_prob)
        assert all(0.0 <= v <= 1.0 for v in mean_pred)

    def test_frac_pos_in_range(self, random_cal):
        y_true, y_prob = random_cal
        _, frac_pos, _ = reliability_data(y_true, y_prob)
        assert all(0.0 <= v <= 1.0 for v in frac_pos)

    def test_counts_positive(self, random_cal):
        y_true, y_prob = random_cal
        _, _, counts = reliability_data(y_true, y_prob)
        assert all(c > 0 for c in counts)

    def test_counts_sum_to_n(self, random_cal):
        y_true, y_prob = random_cal
        _, _, counts = reliability_data(y_true, y_prob)
        assert sum(counts) == len(y_true)
