"""Tests for valiron.acp — Algorithm Change Protocol. Written FIRST (TDD)."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from valiron.evaluate.metrics import MetricsResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_metrics(accuracy=0.85, f1=0.84, sensitivity=0.80, specificity=0.88):
    return MetricsResult(
        accuracy=accuracy, f1=f1, sensitivity=sensitivity, specificity=specificity
    )


def _make_model():
    m = MagicMock()
    m.predict.return_value = np.array([0, 1, 1, 0, 1])
    return m


# ---------------------------------------------------------------------------
# valiron.acp.version
# ---------------------------------------------------------------------------

class TestModelVersion:
    def test_dataclass_fields(self):
        from valiron.acp.version import ModelVersion
        mv = ModelVersion(
            id="v1",
            name="my-model",
            version_string="1.0.0",
            created_at="2026-01-01T00:00:00",
            metrics=_make_metrics(),
            subgroups={},
            regulation="eu_ai_act",
            notes="initial",
        )
        assert mv.id == "v1"
        assert mv.name == "my-model"
        assert mv.version_string == "1.0.0"
        assert mv.regulation == "eu_ai_act"

    def test_subgroups_default_empty(self):
        from valiron.acp.version import ModelVersion
        mv = ModelVersion(
            id="v2", name="m", version_string="1.0", created_at="2026-01-01",
            metrics=_make_metrics(), subgroups={}, regulation="cdsco_mdsw", notes="",
        )
        assert mv.subgroups == {}

    def test_notes_optional_empty_string(self):
        from valiron.acp.version import ModelVersion
        mv = ModelVersion(
            id="x", name="x", version_string="1", created_at="t",
            metrics=_make_metrics(), subgroups={}, regulation="eu_ai_act", notes="",
        )
        assert mv.notes == ""


class TestSaveLoadVersion:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self._orig_cwd = Path.cwd()

    def teardown_method(self):
        import os
        os.chdir(self._orig_cwd)
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _chdir(self):
        import os
        os.chdir(self._tmpdir)

    def test_save_creates_valiron_dir(self):
        self._chdir()
        from valiron.acp.version import save_version
        model = _make_model()
        save_version(model, {"id": "v1", "name": "m", "version_string": "1.0",
                              "metrics": _make_metrics(), "subgroups": {},
                              "regulation": "eu_ai_act", "notes": ""})
        assert (Path(self._tmpdir) / ".valiron").is_dir()

    def test_save_returns_model_version(self):
        self._chdir()
        from valiron.acp.version import save_version, ModelVersion
        mv = save_version(_make_model(), {
            "id": "v1", "name": "m", "version_string": "1.0",
            "metrics": _make_metrics(), "subgroups": {}, "regulation": "eu_ai_act", "notes": "",
        })
        assert isinstance(mv, ModelVersion)

    def test_save_creates_json_file(self):
        self._chdir()
        from valiron.acp.version import save_version
        save_version(_make_model(), {
            "id": "abc123", "name": "m", "version_string": "1.0",
            "metrics": _make_metrics(), "subgroups": {}, "regulation": "eu_ai_act", "notes": "",
        })
        assert (Path(self._tmpdir) / ".valiron" / "abc123.json").exists()

    def test_roundtrip_load(self):
        self._chdir()
        from valiron.acp.version import save_version, load_version
        mv = save_version(_make_model(), {
            "id": "rt1", "name": "roundtrip", "version_string": "2.0",
            "metrics": _make_metrics(0.9, 0.88, 0.85, 0.91),
            "subgroups": {}, "regulation": "cdsco_mdsw", "notes": "test",
        })
        loaded = load_version("rt1")
        assert loaded.id == mv.id
        assert loaded.name == mv.name
        assert abs(loaded.metrics.accuracy - 0.9) < 1e-6

    def test_load_nonexistent_raises(self):
        self._chdir()
        from valiron.acp.version import load_version
        with pytest.raises(FileNotFoundError):
            load_version("does-not-exist")

    def test_list_versions_empty(self):
        self._chdir()
        from valiron.acp.version import list_versions
        result = list_versions()
        assert result == []

    def test_list_versions_returns_all(self):
        self._chdir()
        from valiron.acp.version import save_version, list_versions
        for i in range(3):
            save_version(_make_model(), {
                "id": f"v{i}", "name": "m", "version_string": f"{i}.0",
                "metrics": _make_metrics(), "subgroups": {}, "regulation": "eu_ai_act", "notes": "",
            })
        versions = list_versions()
        assert len(versions) == 3

    def test_save_preserves_subgroups(self):
        self._chdir()
        from valiron.acp.version import save_version, load_version
        sg = {"gender": {"male": {"accuracy": 0.9}, "female": {"accuracy": 0.85}}}
        save_version(_make_model(), {
            "id": "sg1", "name": "m", "version_string": "1.0",
            "metrics": _make_metrics(), "subgroups": sg,
            "regulation": "eu_ai_act", "notes": "",
        })
        loaded = load_version("sg1")
        assert loaded.subgroups == sg

    def test_created_at_auto_set(self):
        self._chdir()
        from valiron.acp.version import save_version
        mv = save_version(_make_model(), {
            "id": "ca1", "name": "m", "version_string": "1.0",
            "metrics": _make_metrics(), "subgroups": {}, "regulation": "eu_ai_act", "notes": "",
        })
        assert mv.created_at  # non-empty


# ---------------------------------------------------------------------------
# valiron.acp.diff
# ---------------------------------------------------------------------------

class TestVersionDiff:
    def test_dataclass_fields(self):
        from valiron.acp.diff import VersionDiff
        d = VersionDiff(
            baseline="v1", candidate="v2",
            metric_changes={"accuracy": 0.01},
            subgroup_changes={},
            regression_detected=False,
            regression_details=[],
        )
        assert d.baseline == "v1"
        assert d.regression_detected is False


class TestDiffVersions:
    def _make_mv(self, vid, acc=0.85, f1=0.84, sens=0.80, spec=0.88):
        from valiron.acp.version import ModelVersion
        return ModelVersion(
            id=vid, name="m", version_string="1.0", created_at="2026-01-01",
            metrics=_make_metrics(acc, f1, sens, spec),
            subgroups={}, regulation="eu_ai_act", notes="",
        )

    def test_no_regression_when_metrics_improve(self):
        from valiron.acp.diff import diff_versions
        v1 = self._make_mv("v1", acc=0.80)
        v2 = self._make_mv("v2", acc=0.85)
        d = diff_versions(v1, v2)
        assert d.regression_detected is False

    def test_regression_detected_when_accuracy_drops_over_2pct(self):
        from valiron.acp.diff import diff_versions
        v1 = self._make_mv("v1", acc=0.90)
        v2 = self._make_mv("v2", acc=0.87)  # drops 3.3%
        d = diff_versions(v1, v2)
        assert d.regression_detected is True

    def test_no_regression_when_drop_within_2pct(self):
        from valiron.acp.diff import diff_versions
        v1 = self._make_mv("v1", acc=0.90)
        v2 = self._make_mv("v2", acc=0.884)  # drops ~1.8%, clearly within 2%
        d = diff_versions(v1, v2)
        assert d.regression_detected is False

    def test_metric_changes_keys(self):
        from valiron.acp.diff import diff_versions
        v1 = self._make_mv("v1")
        v2 = self._make_mv("v2")
        d = diff_versions(v1, v2)
        assert "accuracy" in d.metric_changes
        assert "f1" in d.metric_changes

    def test_metric_changes_values_correct(self):
        from valiron.acp.diff import diff_versions
        v1 = self._make_mv("v1", acc=0.80)
        v2 = self._make_mv("v2", acc=0.85)
        d = diff_versions(v1, v2)
        assert abs(d.metric_changes["accuracy"] - 0.05) < 1e-6

    def test_regression_details_lists_dropped_metrics(self):
        from valiron.acp.diff import diff_versions
        v1 = self._make_mv("v1", acc=0.90, f1=0.88)
        v2 = self._make_mv("v2", acc=0.86, f1=0.85)
        d = diff_versions(v1, v2)
        assert any("accuracy" in s for s in d.regression_details)

    def test_baseline_candidate_in_diff(self):
        from valiron.acp.diff import diff_versions
        v1 = self._make_mv("base")
        v2 = self._make_mv("cand")
        d = diff_versions(v1, v2)
        assert d.baseline == "base"
        assert d.candidate == "cand"


class TestGenerateACPDocument:
    def _make_diff(self, regression=False, details=None):
        from valiron.acp.diff import VersionDiff
        return VersionDiff(
            baseline="v1", candidate="v2",
            metric_changes={"accuracy": 0.02, "f1": -0.01, "sensitivity": 0.01, "specificity": 0.00},
            subgroup_changes={},
            regression_detected=regression,
            regression_details=details or [],
        )

    def test_returns_string(self):
        from valiron.acp.diff import generate_acp_document
        doc = generate_acp_document(self._make_diff())
        assert isinstance(doc, str)

    def test_has_change_description_section(self):
        from valiron.acp.diff import generate_acp_document
        doc = generate_acp_document(self._make_diff())
        assert "Change Description" in doc

    def test_has_performance_comparison_section(self):
        from valiron.acp.diff import generate_acp_document
        doc = generate_acp_document(self._make_diff())
        assert "Performance Comparison" in doc

    def test_has_subgroup_impact_section(self):
        from valiron.acp.diff import generate_acp_document
        doc = generate_acp_document(self._make_diff())
        assert "Subgroup Impact" in doc

    def test_has_regression_assessment_section(self):
        from valiron.acp.diff import generate_acp_document
        doc = generate_acp_document(self._make_diff())
        assert "Regression Assessment" in doc

    def test_has_recommendation_section(self):
        from valiron.acp.diff import generate_acp_document
        doc = generate_acp_document(self._make_diff())
        assert "Recommendation" in doc

    def test_approve_when_no_regression(self):
        from valiron.acp.diff import generate_acp_document
        doc = generate_acp_document(self._make_diff(regression=False))
        assert "APPROVE" in doc

    def test_reject_when_regression(self):
        from valiron.acp.diff import generate_acp_document
        doc = generate_acp_document(self._make_diff(
            regression=True,
            details=["accuracy dropped 3.3%"]
        ))
        assert "REJECT" in doc or "REVIEW" in doc
