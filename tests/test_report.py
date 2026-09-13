"""TDD — report builder tests (RED phase)."""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from valiron.evaluate.runner import EvaluationResult
from valiron.evaluate.metrics import MetricsResult
from valiron.subgroups.analyzer import analyze_subgroups
from valiron.report.builder import report, ReportInput


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def eval_result():
    return EvaluationResult(
        regulation="cdsco_mdsw",
        use_case="diagnostic_imaging_aid",
        compliant=True,
        score=0.85,
        passing_checks=["CDSCO_MDSW_PERFORMANCE: accuracy=0.870", "CDSCO_MDSW_DATA_PRIVACY: local evaluation"],
        failing_checks=[],
        warnings=["CDSCO_MDSW_REGISTRATION: MDSW registration with CDSCO required"],
    )


@pytest.fixture
def metrics():
    return MetricsResult(
        accuracy=0.87,
        f1=0.83,
        sensitivity=0.80,
        specificity=0.91,
        auc=0.92,
        accuracy_ci=(0.84, 0.90),
        f1_ci=(0.79, 0.87),
        sensitivity_ci=(0.75, 0.85),
        specificity_ci=(0.88, 0.94),
        auc_ci=(0.89, 0.95),
    )


@pytest.fixture
def subgroup_data():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=100)
    y_pred = rng.integers(0, 2, size=100)
    sensitive = pd.DataFrame({"gender": rng.choice(["M", "F"], size=100)})
    return analyze_subgroups(y_true, y_pred, sensitive)


@pytest.fixture
def report_input(eval_result, metrics, subgroup_data):
    return ReportInput(
        eval_result=eval_result,
        metrics=metrics,
        subgroups=subgroup_data,
    )


# ---------------------------------------------------------------------------
# ReportInput dataclass
# ---------------------------------------------------------------------------

class TestReportInput:
    def test_can_construct(self, eval_result, metrics, subgroup_data):
        ri = ReportInput(eval_result=eval_result, metrics=metrics, subgroups=subgroup_data)
        assert ri.eval_result.regulation == "cdsco_mdsw"

    def test_subgroups_optional(self, eval_result, metrics):
        ri = ReportInput(eval_result=eval_result, metrics=metrics)
        assert ri.subgroups is None

    def test_metrics_optional(self, eval_result):
        ri = ReportInput(eval_result=eval_result)
        assert ri.metrics is None


# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------

class TestHtmlReport:
    def test_returns_str(self, report_input):
        html = report(report_input, format="html")
        assert isinstance(html, str)

    def test_is_valid_html(self, report_input):
        html = report(report_input, format="html")
        assert "<!DOCTYPE html>" in html or "<!doctype html>" in html.lower()

    def test_contains_regulation(self, report_input):
        html = report(report_input, format="html")
        assert "cdsco_mdsw" in html.lower() or "CDSCO" in html

    def test_contains_use_case(self, report_input):
        html = report(report_input, format="html")
        assert "diagnostic_imaging_aid" in html

    def test_contains_compliance_status(self, report_input):
        html = report(report_input, format="html")
        assert "COMPLIANT" in html

    def test_contains_score(self, report_input):
        html = report(report_input, format="html")
        assert "85" in html  # score=0.85 → 85%

    # CDSCO section headers per actual MDSW guidance document
    def test_cdsco_section_performance_evaluation(self, report_input):
        html = report(report_input, format="html")
        assert "Performance Evaluation" in html

    def test_cdsco_section_clinical_evaluation(self, report_input):
        html = report(report_input, format="html")
        assert "Clinical Evaluation" in html

    def test_cdsco_section_post_market_surveillance(self, report_input):
        html = report(report_input, format="html")
        assert "Post-Market Surveillance" in html

    def test_cdsco_section_registration(self, report_input):
        html = report(report_input, format="html")
        assert "Registration" in html

    def test_metrics_table_present(self, report_input):
        html = report(report_input, format="html")
        assert "Accuracy" in html
        assert "0.87" in html

    def test_auc_in_report(self, report_input):
        html = report(report_input, format="html")
        assert "AUC" in html
        assert "0.92" in html

    def test_ci_present(self, report_input):
        html = report(report_input, format="html")
        assert "0.84" in html or "95%" in html

    def test_subgroup_bias_table_present(self, report_input):
        html = report(report_input, format="html")
        assert "gender" in html.lower() or "Subgroup" in html

    def test_subgroup_groups_present(self, report_input):
        html = report(report_input, format="html")
        assert "M" in html and "F" in html

    def test_passing_checks_listed(self, report_input):
        html = report(report_input, format="html")
        assert "CDSCO_MDSW_PERFORMANCE" in html

    def test_warnings_listed(self, report_input):
        html = report(report_input, format="html")
        assert "CDSCO_MDSW_REGISTRATION" in html

    def test_write_to_file(self, report_input, tmp_path):
        out = str(tmp_path / "report.html")
        returned = report(report_input, format="html", output=out)
        assert returned == out
        content = open(out).read()
        assert "COMPLIANT" in content


# ---------------------------------------------------------------------------
# Non-CDSCO regulation — generic template fallback
# ---------------------------------------------------------------------------

class TestNonCdscoReport:
    def test_eu_ai_act_report(self, metrics):
        eu_result = EvaluationResult(
            regulation="eu_ai_act",
            use_case="recruitment_screening",
            compliant=False,
            score=0.6,
            passing_checks=["EU_AI_ACT_PERFORMANCE: accuracy=0.710"],
            failing_checks=["EU_AI_ACT_BIAS: sensitive features missing"],
            warnings=[],
        )
        ri = ReportInput(eval_result=eu_result, metrics=metrics)
        html = report(ri, format="html")
        assert "eu_ai_act" in html.lower() or "EU AI Act" in html
        assert "NON-COMPLIANT" in html

    def test_invalid_format_raises(self, report_input):
        with pytest.raises(ValueError, match="format"):
            report(report_input, format="docx")


# ---------------------------------------------------------------------------
# Calibration section (optional — present only when calibration data passed)
# ---------------------------------------------------------------------------

class TestCalibrationSection:
    def test_calibration_section_absent_when_not_provided(self, report_input):
        html = report(report_input, format="html")
        assert "Expected Calibration Error" not in html

    def test_calibration_section_present_when_provided(self, eval_result, metrics):
        from valiron.calibration.checker import CalibrationResult
        cal = CalibrationResult(ece=0.05, mce=0.12, well_calibrated=True)
        ri = ReportInput(eval_result=eval_result, metrics=metrics, calibration=cal)
        html = report(ri, format="html")
        assert "Calibration" in html
        assert "0.05" in html


# ---------------------------------------------------------------------------
# PDF export
# ---------------------------------------------------------------------------

class TestPdfExport:
    def test_pdf_import_error_without_weasyprint(self, report_input, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "weasyprint":
                raise ImportError("No module named 'weasyprint'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="weasyprint"):
            report(report_input, format="pdf", output="/tmp/out.pdf")
