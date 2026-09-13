"""Per-regulation compliance checks."""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np


def run_checks(
    model: Any,
    X_test: Any,
    y_test: Any,
    regulation: str,
    use_case: str,
    sensitive_features: List[str],
    feature_names: List[str],
) -> Tuple[List[str], List[str], List[str]]:
    """Run all checks for *regulation*, return (passing, failing, warnings)."""
    registry = {
        "eu_ai_act": _eu_ai_act_checks,
        "fda_samd": _fda_samd_checks,
        "cdsco_mdsw": _cdsco_mdsw_checks,
        "rbi_ml_risk": _rbi_ml_risk_checks,
    }
    return registry[regulation](model, X_test, y_test, use_case, sensitive_features)


def _predict(model: Any, X: Any) -> Any:
    if hasattr(model, "predict"):
        return model.predict(X)
    if callable(model):
        return model(X)
    raise TypeError(f"Cannot get predictions from {type(model)}")


def _accuracy(y_true: Any, y_pred: Any) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def _eu_ai_act_checks(model, X_test, y_test, use_case, sensitive_features):
    passing, failing, warnings = [], [], []
    try:
        acc = _accuracy(y_test, _predict(model, X_test))
        if acc >= 0.7:
            passing.append(f"EU_AI_ACT_PERFORMANCE: accuracy={acc:.3f} >= 0.70")
        else:
            failing.append(f"EU_AI_ACT_PERFORMANCE: accuracy={acc:.3f} < 0.70")
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


def _fda_samd_checks(model, X_test, y_test, use_case, sensitive_features):
    passing, failing, warnings = [], [], []
    try:
        acc = _accuracy(y_test, _predict(model, X_test))
        (passing if acc >= 0.75 else failing).append(
            f"FDA_SAMD_PERFORMANCE: accuracy={acc:.3f} {'>=0.75' if acc >= 0.75 else '<0.75'}"
        )
    except Exception as e:
        failing.append(f"FDA_SAMD_PERFORMANCE: {e}")
    warnings.append("FDA_SAMD_PDCP: predetermined change control plan required")
    warnings.append("FDA_SAMD_MONITORING: real-world performance monitoring required")
    return passing, failing, warnings


def _cdsco_mdsw_checks(model, X_test, y_test, use_case, sensitive_features):
    passing, failing, warnings = [], [], []
    try:
        acc = _accuracy(y_test, _predict(model, X_test))
        (passing if acc >= 0.70 else failing).append(
            f"CDSCO_MDSW_PERFORMANCE: accuracy={acc:.3f}"
        )
    except Exception as e:
        failing.append(f"CDSCO_MDSW_PERFORMANCE: {e}")
    warnings.append("CDSCO_MDSW_REGISTRATION: MDSW registration with CDSCO required")
    passing.append("CDSCO_MDSW_DATA_PRIVACY: local evaluation — no patient data transmitted")
    return passing, failing, warnings


def _rbi_ml_risk_checks(model, X_test, y_test, use_case, sensitive_features):
    passing, failing, warnings = [], [], []
    try:
        acc = _accuracy(y_test, _predict(model, X_test))
        (passing if acc >= 0.65 else failing).append(
            f"RBI_ML_PERFORMANCE: accuracy={acc:.3f}"
        )
    except Exception as e:
        failing.append(f"RBI_ML_PERFORMANCE: {e}")

    if hasattr(model, "feature_importances_") or hasattr(model, "coef_"):
        passing.append("RBI_ML_EXPLAINABILITY: model supports explainability")
    else:
        failing.append("RBI_ML_EXPLAINABILITY: model must support explainability")

    warnings.append("RBI_ML_GOVERNANCE: model risk management documentation required")
    return passing, failing, warnings
