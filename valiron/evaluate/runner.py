"""Main evaluate() entry point."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


SUPPORTED_REGULATIONS = ("eu_ai_act", "fda_samd", "cdsco_mdsw", "rbi_ml_risk")


@dataclass
class EvaluationResult:
    regulation: str
    use_case: str
    compliant: bool
    score: float
    passing_checks: List[str] = field(default_factory=list)
    failing_checks: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def evaluate(
    model: Any,
    X_test: Any,
    y_test: Any,
    regulation: str,
    use_case: str,
    sensitive_features: Optional[List[str]] = None,
    feature_names: Optional[List[str]] = None,
) -> EvaluationResult:
    """Validate model against regulation for use_case."""
    if regulation not in SUPPORTED_REGULATIONS:
        raise ValueError(
            f"Unknown regulation '{regulation}'. Supported: {SUPPORTED_REGULATIONS}"
        )

    from valiron.evaluate.checks import run_checks

    passing, failing, warnings = run_checks(
        model=model,
        X_test=X_test,
        y_test=y_test,
        regulation=regulation,
        use_case=use_case,
        sensitive_features=sensitive_features or [],
        feature_names=feature_names or [],
    )

    total = len(passing) + len(failing)
    score = len(passing) / total if total > 0 else 0.0

    return EvaluationResult(
        regulation=regulation,
        use_case=use_case,
        compliant=len(failing) == 0,
        score=round(score, 4),
        passing_checks=passing,
        failing_checks=failing,
        warnings=warnings,
    )
