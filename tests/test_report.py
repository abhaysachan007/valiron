"""Tests for report builder."""

import pytest

from valiron.evaluate.runner import EvaluationResult
from valiron.report.builder import report


def _make_result(**kwargs):
    defaults = dict(
        regulation="eu_ai_act", use_case="test", compliant=True, score=0.9,
        passing_checks=["CHECK_A"], failing_checks=[], warnings=["WARN_1"],
    )
    defaults.update(kwargs)
    return EvaluationResult(**defaults)


def test_report_returns_html_string():
    html = report(_make_result())
    assert isinstance(html, str)
    assert "<html" in html.lower()


def test_report_contains_regulation():
    html = report(_make_result(regulation="cdsco_mdsw"))
    assert "cdsco_mdsw" in html


def test_report_shows_compliant():
    assert "COMPLIANT" in report(_make_result(compliant=True))


def test_report_shows_non_compliant():
    html = report(_make_result(compliant=False, failing_checks=["FAIL_X"]))
    assert "NON-COMPLIANT" in html
    assert "FAIL_X" in html


def test_report_unsupported_format_raises():
    with pytest.raises(ValueError, match="Unsupported format"):
        report(_make_result(), format="xml")
