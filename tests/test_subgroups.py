"""Tests for subgroup fairness analysis."""

import numpy as np
import pandas as pd

from valiron.subgroups.analyzer import analyze_subgroups


def test_basic_subgroup_analysis():
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0, 0, 1])
    df = pd.DataFrame({"gender": ["M", "M", "F", "F", "M", "F"]})
    result = analyze_subgroups(y_true, y_pred, df)
    assert "gender" in result
    assert "M" in result["gender"]
    assert "F" in result["gender"]


def test_accuracy_in_range():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 1])
    df = pd.DataFrame({"group": ["A", "A", "B", "B"]})
    result = analyze_subgroups(y_true, y_pred, df)
    for val in result["group"].values():
        assert 0.0 <= val <= 1.0


def test_feature_subset():
    df = pd.DataFrame({"age": ["young", "old", "young"], "gender": ["M", "F", "M"]})
    y = np.array([1, 0, 1])
    result = analyze_subgroups(y, y, df, features=["age"])
    assert "age" in result
    assert "gender" not in result
