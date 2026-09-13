"""Tests for valiron.evaluate()."""

import pytest
from sklearn.dummy import DummyClassifier
from sklearn.datasets import make_classification

import valiron
from valiron.evaluate.runner import EvaluationResult, SUPPORTED_REGULATIONS


@pytest.fixture
def simple_model():
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    model = DummyClassifier(strategy="most_frequent")
    model.fit(X[:100], y[:100])
    return model, X[100:], y[100:]


def test_evaluate_returns_result(simple_model):
    model, X_test, y_test = simple_model
    result = valiron.evaluate(model, X_test, y_test, "eu_ai_act", "generic classification")
    assert isinstance(result, EvaluationResult)


def test_evaluate_score_range(simple_model):
    model, X_test, y_test = simple_model
    result = valiron.evaluate(model, X_test, y_test, "rbi_ml_risk", "credit scoring")
    assert 0.0 <= result.score <= 1.0


def test_evaluate_unknown_regulation_raises(simple_model):
    model, X_test, y_test = simple_model
    with pytest.raises(ValueError, match="Unknown regulation"):
        valiron.evaluate(model, X_test, y_test, "unknown_reg", "test")


def test_all_regulations_run(simple_model):
    model, X_test, y_test = simple_model
    for reg in SUPPORTED_REGULATIONS:
        result = valiron.evaluate(model, X_test, y_test, reg, "test case")
        assert result.regulation == reg


def test_compliant_flag_consistent(simple_model):
    model, X_test, y_test = simple_model
    result = valiron.evaluate(model, X_test, y_test, "cdsco_mdsw", "diagnostic aid")
    assert result.compliant == (len(result.failing_checks) == 0)
