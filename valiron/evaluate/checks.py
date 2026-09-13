"""Per-regulation compliance checks — integrated with metrics.py."""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np

from valiron.evaluate.metrics import compute_metrics, MetricsResult


def run_checks(
    model: Any,
    X_test: Any,
    y_test: Any,
    regulation: str,
    use_case: str,
    sensitive_features: List[str],
    feature_names: List[str],
) -> Tuple[List[str], List[str], List[str]]:
    """Run all checks for *regulation*, return (passing, failing, warnings).

    Args:
        model: Fitted model with .predict() or callable.
        X_test: Test features.
        y_test: Ground-truth binary labels.
        regulation: One of SUPPORTED_REGULATIONS.
        use_case: Free-text description of the deployment use case.
        sensitive_features: Names of sensitive demographic columns.
        feature_names: Names of input features (for explainability checks).

    Returns:
        (passing, failing, warnings) — lists of check ID strings.

    Raises:
        KeyError: If regulation is not in the supported registry.
    """
    registry = {
        "eu_ai_act": _eu_ai_act_checks,
        "fda_samd": _fda_samd_checks,
        "cdsco_mdsw": _cdsco_mdsw_checks,
        "rbi_ml_risk": _rbi_ml_risk_checks,
    }
    return registry[regulation](model, X_test, y_test, use_case, sensitive_features)


def _predict(model: Any, X: Any) -> np.ndarray:
    """Return binary predictions from model."""
    if hasattr(model, "predict"):
        return np.asarray(model.predict(X))
    if callable(model):
        return np.asarray(model(X))
    raise TypeError(f"Cannot get predictions from {type(model)}")


def _predict_proba(model: Any, X: Any) -> Any:
    """Return positive-class probabilities if available, else None."""
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1]
    return None


def _get_metrics(model: Any, X: Any, y: Any) -> MetricsResult:
    """Compute MetricsResult for (model, X, y) without bootstrap CIs."""
    y_pred = _predict(model, X)
    y_prob = _predict_proba(model, X)
    return compute_metrics(np.asarray(y), y_pred, y_prob=y_prob, bootstrap_n=0)


def _eu_ai_act_checks(
    model: Any, X_test: Any, y_test: Any, use_case: str, sensitive_features: List[str]
) -> Tuple[List[str], List[str], List[str]]:
    passing, failing, warnings = [], [], []

    try:
        m = _get_metrics(model, X_test, y_test)
        label = f"accuracy={m.accuracy:.3f}"
        (passing if m.accuracy >= 0.70 else failing).append(
            f"EU_AI_ACT_PERFORMANCE: {label} {'>=0.70' if m.accuracy >= 0.70 else '<0.70 threshold'}"
        )
    except Exception as e:
        failing.append(f"EU_AI_ACT_PERFORMANCE: {e}")

    if hasattr(model, "feature_importances_") or hasattr(model, "coef_"):
        passing.append("EU_AI_ACT_TRANSPARENCY: model exposes feature importances")
    else:
        warnings.append("EU_AI_ACT_TRANSPARENCY: model has no native explainability")

    high_risk = ("medical", "health", "diagnostic", "credit", "employment", "biometric")
    if any(kw in use_case.lower() for kw in high_risk):
        warnings.append("EU_AI_ACT_HIGH_RISK: use case may be Annex III high-risk")
    else:
        passing.append("EU_AI_ACT_HIGH_RISK: not flagged as high-risk")

    if sensitive_features:
        passing.append(f"EU_AI_ACT_BIAS: sensitive features provided {sensitive_features}")
    else:
        warnings.append("EU_AI_ACT_BIAS: no sensitive_features supplied; fairness audit skipped")

    return passing, failing, warnings


def _fda_samd_checks(
    model: Any, X_test: Any, y_test: Any, use_case: str, sensitive_features: List[str]
) -> Tuple[List[str], List[str], List[str]]:
    passing, failing, warnings = [], [], []

    try:
        m = _get_metrics(model, X_test, y_test)
        label = f"accuracy={m.accuracy:.3f}"
        (passing if m.accuracy >= 0.75 else failing).append(
            f"FDA_SAMD_PERFORMANCE: {label} {'>=0.75' if m.accuracy >= 0.75 else '<0.75 threshold'}"
        )
    except Exception as e:
        failing.append(f"FDA_SAMD_PERFORMANCE: {e}")

    warnings.append("FDA_SAMD_PDCP: predetermined change control plan required")
    warnings.append("FDA_SAMD_MONITORING: real-world performance monitoring required")
    return passing, failing, warnings


def _cdsco_mdsw_checks(
    model: Any, X_test: Any, y_test: Any, use_case: str, sensitive_features: List[str]
) -> Tuple[List[str], List[str], List[str]]:
    """CDSCO MDSW compliance checks.

    Thresholds (CDSCO MDSW Guidance § 5 Performance Evaluation):
        accuracy    >= 0.70
        sensitivity >= 0.60
        specificity >= 0.60
    """
    passing, failing, warnings = [], [], []

    try:
        m = _get_metrics(model, X_test, y_test)

        (passing if m.accuracy >= 0.70 else failing).append(
            f"CDSCO_MDSW_PERFORMANCE: accuracy={m.accuracy:.3f} "
            f"{'>=0.70' if m.accuracy >= 0.70 else '<0.70 threshold'}"
        )
        (passing if m.sensitivity >= 0.60 else failing).append(
            f"CDSCO_MDSW_SENSITIVITY: sensitivity={m.sensitivity:.3f} "
            f"{'>=0.60' if m.sensitivity >= 0.60 else '<0.60 threshold'}"
        )
        (passing if m.specificity >= 0.60 else failing).append(
            f"CDSCO_MDSW_SPECIFICITY: specificity={m.specificity:.3f} "
            f"{'>=0.60' if m.specificity >= 0.60 else '<0.60 threshold'}"
        )

    except Exception as e:
        failing.append(f"CDSCO_MDSW_PERFORMANCE: {e}")

    warnings.append("CDSCO_MDSW_REGISTRATION: MDSW registration with CDSCO required")
    passing.append("CDSCO_MDSW_DATA_PRIVACY: local evaluation — no patient data transmitted")
    return passing, failing, warnings


def _rbi_ml_risk_checks(
    model: Any, X_test: Any, y_test: Any, use_case: str, sensitive_features: List[str]
) -> Tuple[List[str], List[str], List[str]]:
    passing, failing, warnings = [], [], []

    try:
        m = _get_metrics(model, X_test, y_test)
        label = f"accuracy={m.accuracy:.3f}"
        (passing if m.accuracy >= 0.65 else failing).append(
            f"RBI_ML_PERFORMANCE: {label} {'>=0.65' if m.accuracy >= 0.65 else '<0.65 threshold'}"
        )
    except Exception as e:
        failing.append(f"RBI_ML_PERFORMANCE: {e}")

    if hasattr(model, "feature_importances_") or hasattr(model, "coef_"):
        passing.append("RBI_ML_EXPLAINABILITY: model supports explainability")
    else:
        failing.append("RBI_ML_EXPLAINABILITY: model must support explainability")

    warnings.append("RBI_ML_GOVERNANCE: model risk management documentation required")
    return passing, failing, warnings
