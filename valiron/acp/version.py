"""ACP version storage — save, load, and list ModelVersion records."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from valiron.evaluate.metrics import MetricsResult


@dataclass
class ModelVersion:
    id: str
    name: str
    version_string: str
    created_at: str
    metrics: MetricsResult
    subgroups: Dict[str, Any]
    regulation: str
    notes: str


def _valiron_dir() -> Path:
    d = Path.cwd() / ".valiron"
    d.mkdir(exist_ok=True)
    return d


def _metrics_to_dict(m: MetricsResult) -> Dict[str, Any]:
    return {
        "accuracy": m.accuracy,
        "f1": m.f1,
        "sensitivity": m.sensitivity,
        "specificity": m.specificity,
        "auc": m.auc,
        "accuracy_ci": list(m.accuracy_ci) if m.accuracy_ci else None,
        "f1_ci": list(m.f1_ci) if m.f1_ci else None,
        "sensitivity_ci": list(m.sensitivity_ci) if m.sensitivity_ci else None,
        "specificity_ci": list(m.specificity_ci) if m.specificity_ci else None,
        "auc_ci": list(m.auc_ci) if m.auc_ci else None,
    }


def _dict_to_metrics(d: Dict[str, Any]) -> MetricsResult:
    def _ci(v):
        return tuple(v) if v else None

    return MetricsResult(
        accuracy=d["accuracy"],
        f1=d["f1"],
        sensitivity=d["sensitivity"],
        specificity=d["specificity"],
        auc=d.get("auc"),
        accuracy_ci=_ci(d.get("accuracy_ci")),
        f1_ci=_ci(d.get("f1_ci")),
        sensitivity_ci=_ci(d.get("sensitivity_ci")),
        specificity_ci=_ci(d.get("specificity_ci")),
        auc_ci=_ci(d.get("auc_ci")),
    )


def save_version(model: Any, metadata: Dict[str, Any]) -> ModelVersion:
    """Persist a model version to .valiron/<id>.json.

    Args:
        model: The trained model object (stored by reference only — not serialized).
        metadata: Dict with keys: id, name, version_string, metrics (MetricsResult),
                  subgroups, regulation, notes. created_at is auto-set if absent.

    Returns:
        ModelVersion dataclass instance.
    """
    created_at = metadata.get("created_at") or datetime.now(timezone.utc).isoformat()
    metrics = metadata["metrics"]

    mv = ModelVersion(
        id=metadata["id"],
        name=metadata["name"],
        version_string=metadata["version_string"],
        created_at=created_at,
        metrics=metrics,
        subgroups=metadata.get("subgroups", {}),
        regulation=metadata["regulation"],
        notes=metadata.get("notes", ""),
    )

    payload = {
        "id": mv.id,
        "name": mv.name,
        "version_string": mv.version_string,
        "created_at": mv.created_at,
        "metrics": _metrics_to_dict(mv.metrics),
        "subgroups": mv.subgroups,
        "regulation": mv.regulation,
        "notes": mv.notes,
    }
    path = _valiron_dir() / f"{mv.id}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return mv


def load_version(version_id: str) -> ModelVersion:
    """Load a ModelVersion from .valiron/<version_id>.json.

    Raises:
        FileNotFoundError: If the version file does not exist.
    """
    path = _valiron_dir() / f"{version_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Version '{version_id}' not found in .valiron/")
    data = json.loads(path.read_text(encoding="utf-8"))
    return ModelVersion(
        id=data["id"],
        name=data["name"],
        version_string=data["version_string"],
        created_at=data["created_at"],
        metrics=_dict_to_metrics(data["metrics"]),
        subgroups=data["subgroups"],
        regulation=data["regulation"],
        notes=data["notes"],
    )


def list_versions() -> List[ModelVersion]:
    """Return all saved ModelVersions from .valiron/, sorted by created_at."""
    d = Path.cwd() / ".valiron"
    if not d.exists():
        return []
    versions = []
    for p in sorted(d.glob("*.json")):
        try:
            versions.append(load_version(p.stem))
        except Exception:
            pass
    return sorted(versions, key=lambda v: v.created_at)
