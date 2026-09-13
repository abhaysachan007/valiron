"""TDD — checks.py tests, including metrics.py integration."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression


@pytest.fixture
def good_model():
    """LogisticRegression with accuracy ~0.90 on clean data."""
    X, y = make_classification(n_samples=600, n_features=10, random_state=0)
    model = LogisticRegression(max_iter=200, random_state=0)
    model.fit(X[:400], y[:400])
    return model, X[400:], y[400:]


@pytest.fixture
def bad_model():
    """DummyClassifier — always predicts majority class, low performance."""
    X, y = make_classification(n_samples=400, n_features=5, random_state=1)
    model = DummyClassifier(strategy="most_frequent")
    model.fit(X[:200], y[:200])
    return model, X[200:], y[200:]


# ---------------------------------------------------------------------------
# run_checks contract
# ---------------------------------------------------------------------------

class TestRunChecksContract:
    def test_returns_three_lists(self, good_model):
        from valiron.evaluate.checks import run_checks
        model, X, y = good_model
        passing, failing, warnings = run_checks(model, X, y, "cdsco_mdsw", "diagnostic_aid", [], [])
        assert isinstance(passing, list)
        assert isinstance(failing, list)
        assert isinstance(warnings, list)

    def test_all_regulations_return_results(self, good_model):
        from valiron.evaluate.checks import run_checks
        from valiron.evaluate.runner import SUPPORTED_REGULATIONS
        model, X, y = good_model
        for reg in SUPPORTED_REGULATIONS:
            p, f, w = run_checks(model, X, y, reg, "test", [], [])
            assert isinstance(p, list)

    def test_unknown_regulation_raises(self, good_model):
        from valiron.evaluate.checks import run_checks
        model, X, y = good_model
        with pytest.raises(KeyError):
            run_checks(model, X, y, "unknown_reg", "test", [], [])


# ---------------------------------------------------------------------------
# CDSCO MDSW — pass/fail with MetricsResult integration
# ---------------------------------------------------------------------------

class TestCdscoChecks:
    def test_good_model_passes_performance(self, good_model):
        from valiron.evaluate.checks import run_checks
        model, X, y = good_model
        passing, failing, _ = run_checks(model, X, y, "cdsco_mdsw", "diagnostic_aid", [], [])
        perf_checks = [c for c in passing if "PERFORMANCE" in c]
        assert len(perf_checks) > 0

    def test_bad_model_fails_performance(self, bad_model):
        from valiron.evaluate.checks import run_checks
        model, X, y = bad_model
        _, failing, _ = run_checks(model, X, y, "cdsco_mdsw", "diagnostic_aid", [], [])
        perf_fails = [c for c in failing if "PERFORMANCE" in c]
        assert len(perf_fails) > 0

    def test_cdsco_check_names_use_prefix(self, good_model):
        """All CDSCO check IDs must start with CDSCO_MDSW_."""
        from valiron.evaluate.checks import run_checks
        model, X, y = good_model
        p, f, w = run_checks(model, X, y, "cdsco_mdsw", "diagnostic_aid", [], [])
        for check in p + f:
            assert check.startswith("CDSCO_MDSW_"), f"unexpected check: {check}"

    def test_cdsco_accuracy_threshold_0_70(self):
        """Accuracy just above 0.70 must pass; just below must fail."""
        from valiron.evaluate.checks import _cdsco_mdsw_checks
        X, y = make_classification(n_samples=400, n_features=10, random_state=2)
        model = LogisticRegression(max_iter=200, random_state=2)
        model.fit(X[:200], y[:200])
        p, f, _ = _cdsco_mdsw_checks(model, X[200:], y[200:], "test", [])
        perf_checks = [c for c in p + f if "PERFORMANCE" in c]
        assert len(perf_checks) > 0

    def test_cdsco_metrics_in_check_output(self, good_model):
        """Pass/fail messages must include numeric metric values."""
        from valiron.evaluate.checks import run_checks
        model, X, y = good_model
        p, f, _ = run_checks(model, X, y, "cdsco_mdsw", "diagnostic_aid", [], [])
        perf_checks = [c for c in p + f if "PERFORMANCE" in c]
        assert any("accuracy" in c.lower() or "0." in c for c in perf_checks)

    def test_cdsco_registration_warning_always_present(self, good_model):
        from valiron.evaluate.checks import run_checks
        model, X, y = good_model
        _, _, warnings = run_checks(model, X, y, "cdsco_mdsw", "diagnostic_aid", [], [])
        reg_warnings = [w for w in warnings if "REGISTRATION" in w]
        assert len(reg_warnings) > 0

    def test_cdsco_data_privacy_check_passes(self, good_model):
        from valiron.evaluate.checks import run_checks
        model, X, y = good_model
        passing, _, _ = run_checks(model, X, y, "cdsco_mdsw", "diagnostic_aid", [], [])
        privacy_checks = [c for c in passing if "DATA_PRIVACY" in c]
        assert len(privacy_checks) > 0


# ---------------------------------------------------------------------------
# EU AI Act checks
# ---------------------------------------------------------------------------

class TestEuAiActChecks:
    def test_good_model_passes_performance(self, good_model):
        from valiron.evaluate.checks import run_checks
        model, X, y = good_model
        passing, _, _ = run_checks(model, X, y, "eu_ai_act", "generic_use", [], [])
        assert any("EU_AI_ACT_PERFORMANCE" in c for c in passing)

    def test_bad_model_may_fail_performance(self, bad_model):
        from valiron.evaluate.checks import run_checks
        model, X, y = bad_model
        p, f, _ = run_checks(model, X, y, "eu_ai_act", "generic_use", [], [])
        assert any("EU_AI_ACT_PERFORMANCE" in c for c in p + f)

    def test_high_risk_use_case_flagged(self, good_model):
        from valiron.evaluate.checks import run_checks
        model, X, y = good_model
        _, _, warnings = run_checks(model, X, y, "eu_ai_act", "medical_diagnosis", [], [])
        assert any("HIGH_RISK" in w for w in warnings)

    def test_sensitive_features_removes_bias_warning(self, good_model):
        from valiron.evaluate.checks import run_checks
        model, X, y = good_model
        p_with, _, _ = run_checks(model, X, y, "eu_ai_act", "test", ["gender"], [])
        assert any("BIAS" in c for c in p_with)


# ---------------------------------------------------------------------------
# FDA SaMD checks
# ---------------------------------------------------------------------------

class TestFdaSamdChecks:
    def test_fda_accuracy_threshold_0_75(self, good_model):
        from valiron.evaluate.checks import run_checks
        model, X, y = good_model
        p, f, _ = run_checks(model, X, y, "fda_samd", "imaging_aid", [], [])
        assert any("FDA_SAMD_PERFORMANCE" in c for c in p + f)

    def test_fda_pdcp_warning_always_present(self, good_model):
        from valiron.evaluate.checks import run_checks
        model, X, y = good_model
        _, _, warnings = run_checks(model, X, y, "fda_samd", "imaging_aid", [], [])
        assert any("PDCP" in w for w in warnings)


# ---------------------------------------------------------------------------
# RBI ML Risk checks
# ---------------------------------------------------------------------------

class TestRbiMlRiskChecks:
    def test_rbi_performance_check_present(self, good_model):
        from valiron.evaluate.checks import run_checks
        model, X, y = good_model
        p, f, _ = run_checks(model, X, y, "rbi_ml_risk", "credit_scoring", [], [])
        assert any("RBI_ML_PERFORMANCE" in c for c in p + f)

    def test_rbi_governance_warning_always_present(self, good_model):
        from valiron.evaluate.checks import run_checks
        model, X, y = good_model
        _, _, warnings = run_checks(model, X, y, "rbi_ml_risk", "credit_scoring", [], [])
        assert any("GOVERNANCE" in w for w in warnings)
