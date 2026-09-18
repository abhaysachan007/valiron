"""Tests for EU AI Act report template. Written FIRST (TDD)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from valiron.evaluate.metrics import MetricsResult
from valiron.calibration.checker import CalibrationResult
from valiron.report.builder import ReportInput, report


def _fake_eval(regulation="eu_ai_act", compliant=True):
    from valiron.evaluate.runner import EvaluationResult
    return EvaluationResult(
        regulation=regulation,
        use_case="risk_scoring",
        compliant=compliant,
        score=0.85,
        passing_checks=["EU_ACC: accuracy=0.85"],
        failing_checks=[],
        warnings=["EU_TRANSPARENCY: provide audit trail"],
        metadata={},
    )


def _fake_metrics():
    return MetricsResult(
        accuracy=0.85, f1=0.83, sensitivity=0.80, specificity=0.88
    )


def _fake_subgroups():
    return {
        "gender": {
            "male": {"accuracy": 0.87, "f1": 0.85, "sensitivity": 0.82, "specificity": 0.90, "n": 200},
            "female": {"accuracy": 0.83, "f1": 0.81, "sensitivity": 0.78, "specificity": 0.86, "n": 150},
        }
    }


class TestEUAIActTemplate:
    def test_eu_regulation_uses_eu_template(self):
        ri = ReportInput(eval_result=_fake_eval(), metrics=_fake_metrics())
        html = report(ri, format="html")
        assert "EU AI Act" in html

    def test_has_article_9_risk_management(self):
        ri = ReportInput(eval_result=_fake_eval(), metrics=_fake_metrics())
        html = report(ri, format="html")
        assert "Article 9" in html or "Risk Management" in html

    def test_has_article_10_data_governance(self):
        ri = ReportInput(eval_result=_fake_eval(), metrics=_fake_metrics())
        html = report(ri, format="html")
        assert "Article 10" in html or "Data Governance" in html

    def test_has_article_13_transparency(self):
        ri = ReportInput(eval_result=_fake_eval(), metrics=_fake_metrics())
        html = report(ri, format="html")
        assert "Article 13" in html or "Transparency" in html

    def test_has_article_14_human_oversight(self):
        ri = ReportInput(eval_result=_fake_eval(), metrics=_fake_metrics())
        html = report(ri, format="html")
        assert "Article 14" in html or "Human Oversight" in html

    def test_has_article_15_accuracy_robustness(self):
        ri = ReportInput(eval_result=_fake_eval(), metrics=_fake_metrics())
        html = report(ri, format="html")
        assert "Article 15" in html or "Accuracy" in html

    def test_has_annex_iii_badge(self):
        ri = ReportInput(eval_result=_fake_eval(), metrics=_fake_metrics())
        html = report(ri, format="html")
        assert "Annex III" in html or "High Risk" in html

    def test_subgroups_rendered_when_present(self):
        ri = ReportInput(eval_result=_fake_eval(), metrics=_fake_metrics(),
                         subgroups=_fake_subgroups())
        html = report(ri, format="html")
        assert "gender" in html.lower() or "subgroup" in html.lower()

    def test_non_eu_regulation_does_not_use_eu_template(self):
        ri = ReportInput(eval_result=_fake_eval(regulation="cdsco_mdsw"))
        html = report(ri, format="html")
        assert "CDSCO" in html or "§ 5" in html
