"""TDD — subgroup fairness analysis tests."""

import numpy as np
import pandas as pd
import pytest

from valiron.subgroups.analyzer import analyze_subgroups, SubgroupResult, disparity_report


@pytest.fixture
def binary_data():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    y_pred = rng.integers(0, 2, size=200)
    sensitive = pd.DataFrame({
        "gender": rng.choice(["M", "F"], size=200),
        "age_group": rng.choice(["young", "middle", "senior"], size=200),
    })
    return y_true, y_pred, sensitive


@pytest.fixture
def perfect_data():
    y = np.array([0, 0, 1, 1, 0, 1])
    sensitive = pd.DataFrame({"group": ["A", "A", "A", "B", "B", "B"]})
    return y, y.copy(), sensitive


@pytest.fixture
def biased_data():
    """Group A: perfect predictions. Group B: all wrong."""
    y_true = np.array([0, 0, 1, 1,   0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1,   1, 1, 0, 0])
    sensitive = pd.DataFrame({"group": ["A", "A", "A", "A", "B", "B", "B", "B"]})
    return y_true, y_pred, sensitive


class TestAnalyzeSubgroups:
    def test_returns_dict(self, binary_data):
        y_true, y_pred, sensitive = binary_data
        result = analyze_subgroups(y_true, y_pred, sensitive)
        assert isinstance(result, dict)

    def test_keys_match_columns(self, binary_data):
        y_true, y_pred, sensitive = binary_data
        result = analyze_subgroups(y_true, y_pred, sensitive)
        assert set(result.keys()) == {"gender", "age_group"}

    def test_subgroup_values_in_range(self, binary_data):
        y_true, y_pred, sensitive = binary_data
        result = analyze_subgroups(y_true, y_pred, sensitive)
        for col, groups in result.items():
            for val, metrics in groups.items():
                assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_perfect_group_accuracy_1(self, perfect_data):
        y_true, y_pred, sensitive = perfect_data
        result = analyze_subgroups(y_true, y_pred, sensitive)
        assert result["group"]["A"]["accuracy"] == pytest.approx(1.0)
        assert result["group"]["B"]["accuracy"] == pytest.approx(1.0)

    def test_biased_group_accuracy(self, biased_data):
        y_true, y_pred, sensitive = biased_data
        result = analyze_subgroups(y_true, y_pred, sensitive)
        assert result["group"]["A"]["accuracy"] == pytest.approx(1.0)
        assert result["group"]["B"]["accuracy"] == pytest.approx(0.0)

    def test_features_filter(self, binary_data):
        y_true, y_pred, sensitive = binary_data
        result = analyze_subgroups(y_true, y_pred, sensitive, features=["gender"])
        assert "gender" in result
        assert "age_group" not in result

    def test_subgroup_result_has_metrics(self, binary_data):
        y_true, y_pred, sensitive = binary_data
        result = analyze_subgroups(y_true, y_pred, sensitive)
        for col, groups in result.items():
            for val, metrics in groups.items():
                assert "accuracy" in metrics
                assert "sensitivity" in metrics
                assert "specificity" in metrics
                assert "f1" in metrics
                assert "n" in metrics

    def test_n_counts_correct(self, perfect_data):
        y_true, y_pred, sensitive = perfect_data
        result = analyze_subgroups(y_true, y_pred, sensitive)
        assert result["group"]["A"]["n"] == 3
        assert result["group"]["B"]["n"] == 3

    def test_empty_feature_list_returns_all(self, binary_data):
        y_true, y_pred, sensitive = binary_data
        result = analyze_subgroups(y_true, y_pred, sensitive, features=None)
        assert set(result.keys()) == set(sensitive.columns)

    def test_single_class_group_no_crash(self):
        """Group with only one class — sensitivity/specificity degenerate, should not raise."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 1, 1])
        sensitive = pd.DataFrame({"group": ["A", "A", "A", "B", "B", "B"]})
        result = analyze_subgroups(y_true, y_pred, sensitive)
        assert "group" in result


class TestSubgroupResult:
    def test_is_dataclass(self):
        sr = SubgroupResult(accuracy=1.0, f1=1.0, sensitivity=1.0, specificity=1.0, n=10)
        assert sr.accuracy == 1.0
        assert sr.n == 10

    def test_fields_in_range(self):
        sr = SubgroupResult(accuracy=0.75, f1=0.8, sensitivity=0.7, specificity=0.9, n=50)
        assert 0.0 <= sr.accuracy <= 1.0


class TestDisparityReport:
    def test_returns_dict(self, biased_data):
        y_true, y_pred, sensitive = biased_data
        report = disparity_report(y_true, y_pred, sensitive)
        assert isinstance(report, dict)

    def test_max_disparity_detected(self, biased_data):
        y_true, y_pred, sensitive = biased_data
        report = disparity_report(y_true, y_pred, sensitive)
        assert report["group"]["accuracy_disparity"] == pytest.approx(1.0)

    def test_zero_disparity_perfect(self, perfect_data):
        y_true, y_pred, sensitive = perfect_data
        report = disparity_report(y_true, y_pred, sensitive)
        assert report["group"]["accuracy_disparity"] == pytest.approx(0.0)

    def test_worst_group_key(self, biased_data):
        y_true, y_pred, sensitive = biased_data
        report = disparity_report(y_true, y_pred, sensitive)
        assert report["group"]["worst_group"] == "B"
